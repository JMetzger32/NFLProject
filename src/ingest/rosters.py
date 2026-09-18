"""Week-by-week rosters.

This is the table that makes roster churn measurable: one row per player per week
with the team he was on and his status that week, so trades, signings, releases
and IR stints are recoverable by diffing consecutive weeks.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_rosters_weekly(list(seasons)))
    df = df.rename(columns={"gsis_id": "player_id"})
    return load_frame(
        df,
        "weekly_rosters",
        key_cols=["player_id", "season", "week", "team"],
        index_cols=[["player_id", "season"], ["team", "season", "week"], ["status"]],
        source_fn="load_rosters_weekly",
    )
