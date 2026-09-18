"""Grade a week's predictions once final scores exist.

Writes to `pick_results`, never mutating `predictions` — a re-grade can therefore
never corrupt the original forecast record. Idempotent: running twice produces the
same table.

Bankroll is recomputed from scratch over all graded picks in chronological order,
so it stays correct no matter what order weeks are graded in.

Usage:
    python Betting/grade_week.py --season 2026 --week 1
    python Betting/grade_week.py --season 2025 --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.db import get_engine  # noqa: E402
from src.odds import american_to_decimal  # noqa: E402

STARTING_BANKROLL = 1000.0


def load_outcomes() -> pd.DataFrame:
    """Actual result per team-game, from completed games only."""
    return pd.read_sql("""
        SELECT game_id, season, week, home_team AS team,
               CASE WHEN result > 0 THEN 1 ELSE 0 END AS won
        FROM games WHERE game_type='REG' AND result IS NOT NULL AND result <> 0
        UNION ALL
        SELECT game_id, season, week, away_team,
               CASE WHEN result < 0 THEN 1 ELSE 0 END
        FROM games WHERE game_type='REG' AND result IS NOT NULL AND result <> 0
    """, get_engine())


def grade(season: int, week: int | None) -> int:
    eng = get_engine()
    q = "SELECT * FROM predictions WHERE season = %(s)s"
    params = {"s": season}
    if week is not None:
        q += " AND week = %(w)s"
        params["w"] = week
    preds = pd.read_sql(q, eng, params=params)
    if preds.empty:
        print(f"no predictions stored for season {season}"
              + (f" week {week}" if week else ""))
        return 0

    out = preds.merge(load_outcomes(), on=["game_id", "team", "season", "week"],
                      how="inner")
    if out.empty:
        print("no completed games among those predictions yet — nothing to grade")
        return 0

    # Payout only applies to actual picks; non-picks are graded for calibration
    # tracking but never move the bankroll.
    def payout(r):
        if not r["is_pick"] or pd.isna(r["team_moneyline"]):
            return 0.0
        stake = float(r["stake"] or 0.0)
        if r["won"] == 1:
            return (american_to_decimal(r["team_moneyline"]) - 1.0) * stake
        return -stake

    out["payout"] = out.apply(payout, axis=1)

    with eng.begin() as conn:
        for r in out.to_dict(orient="records"):
            conn.execute(text("""
                INSERT INTO pick_results
                    (game_id, team, season, week, won, is_pick, stake, payout,
                     bankroll_after)
                VALUES (:game_id, :team, :season, :week, :won, :is_pick, :stake,
                        :payout, NULL)
                ON CONFLICT (game_id, team) DO UPDATE SET
                    won = EXCLUDED.won, is_pick = EXCLUDED.is_pick,
                    stake = EXCLUDED.stake, payout = EXCLUDED.payout,
                    graded_at = now()
            """), {k: r[k] for k in
                   ("game_id", "team", "season", "week", "won", "is_pick",
                    "stake", "payout")})

    _recompute_bankroll()
    n_picks = int(out["is_pick"].sum())
    print(f"graded {len(out)} team-games ({n_picks} picks) for season {season}"
          + (f" week {week}" if week else " (all weeks)"))
    if n_picks:
        p = out[out["is_pick"]]
        print(f"  pick record: {int(p['won'].sum())}-{int((1 - p['won']).sum())}, "
              f"net ${p['payout'].sum():+.2f}")
    return len(out)


def _recompute_bankroll() -> None:
    """Rebuild the running bankroll over all graded picks, chronologically."""
    eng = get_engine()
    d = pd.read_sql("""
        SELECT r.game_id, r.team, r.season, r.week, r.payout, r.is_pick, g.gameday
        FROM pick_results r JOIN games g USING (game_id)
        ORDER BY g.gameday, r.game_id, r.team
    """, eng)
    if d.empty:
        return
    bal = STARTING_BANKROLL
    updates = []
    for r in d.itertuples():
        if r.is_pick:
            bal += float(r.payout or 0.0)
        updates.append({"g": r.game_id, "t": r.team, "b": bal})
    with eng.begin() as conn:
        for u in updates:
            conn.execute(text("UPDATE pick_results SET bankroll_after = :b "
                              "WHERE game_id = :g AND team = :t"), u)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int)
    ap.add_argument("--all", action="store_true", help="grade every stored week")
    args = ap.parse_args()
    if not args.all and args.week is None:
        ap.error("pass --week N or --all")
    grade(args.season, None if args.all else args.week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
