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


def current_week() -> tuple[int, int] | None:
    """The week the site should be showing right now: the earliest week of the
    most recent season that is NOT yet fully complete.

    Deliberately NOT "whichever picks_*.json file was generated most recently" —
    that surfaced a week-3 testing artifact as "current" while week 2 games were
    still in progress, before week 2 had a single result in. A week only becomes
    "current" by playing out, never by a file existing for it.
    """
    try:
        from src.db import get_engine
        d = pd.read_sql("""
            SELECT season, week,
                   count(*) AS games,
                   count(result) AS completed
            FROM games
            WHERE game_type = 'REG'
            GROUP BY season, week
            ORDER BY season, week
        """, get_engine())
    except Exception:
        return None
    if d.empty:
        return None

    season = int(d["season"].max())
    s = d[d["season"] == season]
    incomplete = s[s["completed"] < s["games"]]
    if len(incomplete):
        row = incomplete.iloc[0]  # earliest not-yet-finished week
    else:
        row = s.iloc[-1]  # season fully played out — show its last week
    return season, int(row["week"])


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


def _pair_games(all_games: list[dict]) -> list[dict]:
    """Each team-game is its own row (both sides of a matchup appear separately).
    Pair them into one row per real game so a public list doesn't show the same
    matchup twice."""
    by_id: dict[str, list[dict]] = {}
    for g in all_games:
        gid = g.get("game_id")
        if gid:
            by_id.setdefault(gid, []).append(g)

    games = []
    for gid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: r.get("team") or "")
        a = rows[0]
        b = rows[1] if len(rows) > 1 else {"team": a.get("opponent")}
        pa, pb = a.get("model_prob"), b.get("model_prob")

        # pa and pb come from two INDEPENDENT model calls (each team's own feature
        # row through the ensemble), so nothing forces pa + pb == 1 — it was
        # observed to range 0.949-1.094 on real weeks. Renormalize to a proper
        # two-outcome distribution for display; the raw values are kept alongside
        # so the site can still show how internally consistent a call was.
        if pa is not None and pb is not None and (pa + pb) > 0:
            total = pa + pb
            norm_a, norm_b = pa / total, pb / total
        else:
            norm_a, norm_b = pa, pb

        games.append({
            "game_id": gid,
            "team_a": a.get("team"), "prob_a": norm_a, "prob_a_raw": pa,
            "team_b": b.get("team"), "prob_b": norm_b, "prob_b_raw": pb,
        })
    return sorted(games, key=lambda g: g["team_a"] or "")


def latest_week() -> dict | None:
    cw = current_week()
    weeks = read_weeks()
    if not weeks:
        return None

    if cw is not None:
        season, week = cw
        match = next((w for w in weeks
                     if w["season"] == season and w["week"] == week), None)
        if match is None:
            # The actually-current week hasn't had predictions generated yet.
            # Say so rather than silently showing a different week.
            return {
                "season": season, "week": week, "status": "not_yet_generated",
                "games_evaluated": 0, "qualifying_picks": 0,
                "notes": (f"Predictions for {season} week {week} haven't been "
                         f"generated yet. Run weekly_picks.py --season {season} "
                         f"--week {week}."),
                "history_source_counts": {}, "picks": [], "games": [],
            }
        newest = match
    else:
        # No DB connection to determine the real current week (e.g. building the
        # site without Postgres running) — fall back to the newest generated file.
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
        "picks": p.get("picks", []),          # kept for later — not rendered yet
        "games": _pair_games(p.get("all_games", [])),
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
