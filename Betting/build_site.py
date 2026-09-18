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


def week_id(season: int, week: int) -> str:
    return f"{season}_wk{week:02d}"


def actual_winners() -> dict[str, str]:
    """game_id -> winning team code, for every completed REG game.

    Ties are naturally absent: the model panel excludes result=0 upstream (see
    src/features.py's _PLAYED filter), so tie games never appear in picks_*.json
    in the first place — nothing extra to filter here.
    """
    try:
        from src.db import get_engine
        d = pd.read_sql("""
            SELECT game_id, home_team, away_team, result
            FROM games
            WHERE game_type = 'REG' AND result IS NOT NULL AND result <> 0
        """, get_engine())
    except Exception:
        return {}
    return {r.game_id: (r.home_team if r.result > 0 else r.away_team)
            for r in d.itertuples()}


def _pair_games(all_games: list[dict], winners: dict[str, str] | None = None) -> list[dict]:
    """Each team-game is its own row (both sides of a matchup appear separately).
    Pair them into one row per real game so a public list doesn't show the same
    matchup twice."""
    winners = winners or {}
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

        market_a = a.get("market_prob_devig")
        market_b = b.get("market_prob_devig")
        # edge_b is always exactly -edge_a (both prob_a+prob_b and market_a+market_b
        # sum to 1), so either side is the complete number — but which side reads
        # naturally depends on who the model actually favors.
        edge_a = (norm_a - market_a) if (norm_a is not None and market_a is not None) else None

        favorite_team, edge_favorite = None, None
        if norm_a is not None and norm_b is not None and edge_a is not None:
            if norm_a >= norm_b:
                favorite_team, edge_favorite = a.get("team"), edge_a
            else:
                favorite_team, edge_favorite = b.get("team"), -edge_a

        winner = winners.get(gid)
        # Correctness is judged against the SAME favorite used for the edge column
        # (whichever of norm_a/norm_b is larger) — not a naive "model_prob > 0.5"
        # per team-row, which can disagree with itself when the two independent
        # raw probabilities both land the same side of 0.5 (observed: their sum
        # ranges 0.949-1.094, so that happens). Comparing the pair directly is the
        # one definition that's always well-formed. None means not yet played.
        model_correct = (favorite_team == winner) if (favorite_team and winner) else None

        games.append({
            "game_id": gid,
            "team_a": a.get("team"), "prob_a": norm_a, "prob_a_raw": pa,
            "team_b": b.get("team"), "prob_b": norm_b, "prob_b_raw": pb,
            # De-vigged market probability — already sums to 1 across the two
            # sides by construction (Shin de-vig), so no renormalization needed.
            "market_a": market_a, "market_b": market_b,
            # The model's favorite (higher of prob_a/prob_b) and ITS edge vs the
            # market — the number that reads naturally, not always team_a's.
            "favorite_team": favorite_team, "edge_favorite": edge_favorite,
            "actual_winner": winner, "model_correct": model_correct,
        })
    return sorted(games, key=lambda g: g["team_a"] or "")


def _not_yet_generated(season: int, week: int) -> dict:
    return {
        "season": season, "week": week, "status": "not_yet_generated",
        "games_evaluated": 0, "qualifying_picks": 0,
        "notes": (f"Predictions for {season} week {week} haven't been generated "
                 f"yet. Run weekly_picks.py --season {season} --week {week}."),
        "history_source_counts": {}, "picks": [], "games": [],
    }


def week_payload(p: dict, winners: dict[str, str]) -> dict:
    """The shape both site/data.json's "latest" and each per-week file share."""
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
        "games": _pair_games(p.get("all_games", []), winners),
    }


def write_week_files(weeks: list[dict], winners: dict[str, str]) -> list[dict]:
    """One small JSON per week under site/weeks/, so the archive can link to a
    real, independently-loadable page of that week's games instead of a chip that
    goes nowhere. Returns `weeks` with a "path" added for each file actually
    written — a week with no file gets no path, and the frontend omits it rather
    than rendering a dead link."""
    weeks_dir = SITE_DIR / "weeks"
    weeks_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for w in weeks:
        try:
            p = json.loads((OUT_DIR / w["file"]).read_text())
            payload = week_payload(p, winners)
        except Exception:
            out.append({**w, "path": None})
            continue
        wid = week_id(w["season"], w["week"])
        (weeks_dir / f"{wid}.json").write_text(json.dumps(payload, indent=2))
        # Prefixed with "site/": the frontend fetches this path directly (no other
        # prefixing), resolved relative to index.html at the repo root — so it must
        # be the real location, not a path relative to site/data.json's own folder.
        out.append({**w, "path": f"site/weeks/{wid}.json"})
    return out


def latest_week(winners: dict[str, str]) -> dict | None:
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
            return _not_yet_generated(season, week)
        newest = match
    else:
        # No DB connection to determine the real current week (e.g. building the
        # site without Postgres running) — fall back to the newest generated file.
        newest = weeks[-1]

    p = json.loads((OUT_DIR / newest["file"]).read_text())
    return week_payload(p, winners)


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


MIN_GRADED_FOR_CONFIDENCE = 30  # below this, accuracy is a coin flip's width of noise


def all_paired_games(winners: dict[str, str]) -> list[dict]:
    """Every game across every picks_*.json, paired and graded against real
    results where known. One definition of "correct" (from _pair_games), reused
    by both per-week row highlighting and the accuracy tab below — not two
    slightly different notions of correctness computed two different ways."""
    out = []
    for f in sorted(OUT_DIR.glob("picks_*.json")):
        try:
            p = json.loads(f.read_text())
        except Exception:
            continue
        for g in _pair_games(p.get("all_games", []), winners):
            out.append({"season": p["season"], "week": p["week"], **g})
    return out


def accuracy_summary(paired: list[dict]) -> dict:
    """Real-world model accuracy, for every week the model has actually predicted
    (not gated on grade_week.py — this reads straight from picks_*.json plus real
    game results, which covers the full 2025 backtest and 2026 the moment a week's
    games finish, with nothing extra to run). Empty and honest when nothing is
    graded yet; no placeholder numbers.

    Grain is one row per GAME (not per team-row) — "did the model's favorite win",
    matching the same favorite/edge definition shown on every week's page.
    """
    graded = [g for g in paired if g.get("model_correct") is not None]
    if not graded:
        out = {"graded": 0}
    else:
        d = pd.DataFrame(graded)
        d["correct"] = d["model_correct"].astype(int)
        by_week = (d.groupby(["season", "week"], as_index=False)
                    .agg(games=("correct", "size"), correct=("correct", "sum")))
        by_week["accuracy"] = (by_week["correct"] / by_week["games"]).round(4)
        by_week = by_week.sort_values(["season", "week"])
        out = {
            "graded": int(len(d)),
            "correct": int(d["correct"].sum()),
            "accuracy": round(float(d["correct"].mean()), 4),
            "record": f"{int(d['correct'].sum())}-{int(len(d) - d['correct'].sum())}",
            "by_week": by_week.to_dict(orient="records"),
        }
        if out["graded"] < MIN_GRADED_FOR_CONFIDENCE:
            out["sample_note"] = (
                f"Only {out['graded']} graded games so far — too few for this "
                f"number to mean much. Treat it as a running count, not a "
                f"result, until it clears {MIN_GRADED_FOR_CONFIDENCE}.")

    # Actual-money picks tracking is a separate concern (stakes, payout) and still
    # needs grade_week.py to have run against the `predictions`/`pick_results`
    # tables — kept optional so its absence never blanks the accuracy numbers above.
    try:
        from src.db import get_engine
        pr = pd.read_sql("SELECT r.won, r.is_pick, r.payout FROM pick_results r",
                         get_engine())
        picks = pr[pr["is_pick"] == True]  # noqa: E712
        out["picks_graded"] = int(len(picks))
        out["picks_record"] = (f"{int(picks['won'].sum())}-{int((1 - picks['won']).sum())}"
                               if len(picks) else None)
        out["picks_net"] = round(float(picks["payout"].sum()), 2) if len(picks) else 0.0
    except Exception:
        out["picks_graded"], out["picks_record"], out["picks_net"] = 0, None, 0.0
    return out


def main() -> int:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    winners = actual_winners()
    weeks = write_week_files(read_weeks(), winners)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest": latest_week(winners),
        "weeks": weeks,
        "metrics": model_metrics(),
        "accuracy": accuracy_summary(all_paired_games(winners)),
    }
    path = SITE_DIR / "data.json"
    path.write_text(json.dumps(payload, indent=2))
    n_ok = sum(1 for w in weeks if w.get("path"))
    latest = payload["latest"]
    acc = payload["accuracy"]
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"  {n_ok} of {len(weeks)} weeks have a browsable page (site/weeks/)")
    if latest:
        print(f"  latest: {latest['season']} wk{latest['week']} — "
              f"{latest['qualifying_picks']} picks ({latest['status']})")
    print(f"  accuracy: {acc['graded']} graded"
          + (f", {acc['accuracy']:.1%} correct" if acc["graded"] else " (none yet)"))
    m = payload["metrics"]
    if m.get("benchmark_auc"):
        print(f"  benchmark {m['benchmark_auc']} vs production {m['production_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
