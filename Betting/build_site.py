"""Generate site/data.json — the single payload the website reads.

The site is a rendering layer, nothing more. Every number it shows is produced
here from the pipeline's own output and the results database, so the page can
never drift from what the model actually did.

Usage:
    python Betting/build_site.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

OUT_DIR = PROJECT_ROOT / "Betting" / "output"
SITE_DIR = PROJECT_ROOT / "site"

# Measured in Model_1 / Model_2 and restated on the site. Sourced from the
# findings CSVs where they exist so the page cannot quietly go stale.
MODEL_1_DATA = PROJECT_ROOT / "Modeling" / "Model_1" / "data" / "03_primary.csv"
MODEL_2_EDGE = PROJECT_ROOT / "Modeling" / "Model_2" / "data" / "07_edge_yield.csv"


def read_weeks() -> list[dict]:
    weeks = []
    for f in sorted(OUT_DIR.glob("picks_*.json")):
        try:
            p = json.loads(f.read_text())
        except Exception:
            continue
        weeks.append({
            "season": p["season"], "week": p["week"],
            "games": p.get("games_evaluated", 0),
            "picks": p.get("qualifying_picks", 0),
            "status": p.get("status", "unknown"),
            "file": f.name,
        })
    return sorted(weeks, key=lambda w: (w["season"], w["week"]))


def latest_week() -> dict | None:
    weeks = read_weeks()
    if not weeks:
        return None
    newest = weeks[-1]
    p = json.loads((OUT_DIR / newest["file"]).read_text())
    return {
        "season": p["season"], "week": p["week"],
        "status": p["status"],
        "generated_at": p.get("generated_at"),
        "games_evaluated": p.get("games_evaluated"),
        "qualifying_picks": p.get("qualifying_picks", 0),
        "insufficient_history": p.get("insufficient_history", False),
        "notes": p.get("notes", ""),
        "history_source_counts": p.get("history_source_counts", {}),
        "picks": p.get("picks", []),
        "games": [
            {k: g.get(k) for k in
             ("team", "opponent", "model_prob", "market_prob_devig", "edge", "reason")}
            for g in p.get("all_games", [])
        ],
    }


def model_metrics() -> dict:
    out = {}
    if MODEL_1_DATA.exists():
        d = pd.read_csv(MODEL_1_DATA)
        rows = []
        for _, r in d.iterrows():
            rows.append({
                "model": r["model"], "role": r["role"],
                "auc": round(float(r["holdout_auc"]), 4),
                "ci": [round(float(r["auc_ci_lo"]), 4), round(float(r["auc_ci_hi"]), 4)],
                "accuracy": round(float(r["accuracy"]), 4),
                "brier": round(float(r["brier"]), 4),
                "ece": round(float(r["ece"]), 4),
            })
        out["holdout"] = rows
        mk = d[d["model"].str.startswith("MARKET")]
        out["benchmark_auc"] = round(float(mk["holdout_auc"].iloc[0]), 4) if len(mk) else None
        prod = d[d["role"] == "PRODUCTION"]
        out["production_auc"] = round(float(prod["holdout_auc"].iloc[0]), 4) if len(prod) else None
    if MODEL_2_EDGE.exists():
        e = pd.read_csv(MODEL_2_EDGE).iloc[0]
        out["edge_test"] = {
            "picks": int(e["qualifying_picks"]),
            "accuracy": None if pd.isna(e["pick_accuracy"]) else round(float(e["pick_accuracy"]), 4),
            "mean_edge": None if pd.isna(e["mean_edge"]) else round(float(e["mean_edge"]), 4),
            "team_games": int(e["team_games"]),
        }
    return out


def graded_summary() -> dict:
    """Live results, once any week has been graded. Empty until then."""
    try:
        from src.db import get_engine
        d = pd.read_sql(
            "SELECT r.won, r.is_pick, r.payout, r.bankroll_after "
            "FROM pick_results r", get_engine())
    except Exception:
        return {"graded": 0, "picks": 0}
    if d.empty:
        return {"graded": 0, "picks": 0}
    picks = d[d["is_pick"] == True]  # noqa: E712
    return {
        "graded": int(len(d)),
        "picks": int(len(picks)),
        "record": (f"{int(picks['won'].sum())}-{int((1 - picks['won']).sum())}"
                   if len(picks) else None),
        "net": round(float(picks["payout"].sum()), 2) if len(picks) else 0.0,
    }


def main() -> int:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest": latest_week(),
        "weeks": read_weeks(),
        "metrics": model_metrics(),
        "live": graded_summary(),
    }
    path = SITE_DIR / "data.json"
    path.write_text(json.dumps(payload, indent=2))
    n = len(payload["weeks"])
    latest = payload["latest"]
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"  {n} weeks indexed")
    if latest:
        print(f"  latest: {latest['season']} wk{latest['week']} — "
              f"{latest['qualifying_picks']} picks ({latest['status']})")
    m = payload["metrics"]
    if m.get("benchmark_auc"):
        print(f"  benchmark {m['benchmark_auc']} vs production {m['production_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
