"""Team-level box score and EPA aggregates, per team per week (~138 columns).

Pre-aggregated by nflverse, so team-strength features don't have to be rebuilt
from play-by-play every time.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_team_stats(list(seasons), summary_level="week"))
    return load_frame(
        df,
        "team_weekly_stats",
        key_cols=["team", "season", "week", "season_type"],
        index_cols=[["team", "season"], ["season", "week"], ["game_id"]],
        source_fn="load_team_stats(week)",
    )
