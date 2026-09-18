"""Per-player, per-week statistics (~150 columns).

Every row carries `team`, so a player who changes teams has his production
attributed to the right team week by week.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_player_stats(list(seasons), summary_level="week"))
    return load_frame(
        df,
        "weekly_player_stats",
        key_cols=["player_id", "season", "week", "season_type"],
        index_cols=[["player_id"], ["team", "season"], ["season", "week"], ["game_id"]],
        source_fn="load_player_stats(week)",
    )
