"""Per-player season totals, regular season plus postseason."""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_player_stats(list(seasons), summary_level="reg+post"))
    return load_frame(
        df,
        "seasonal_player_stats",
        key_cols=["player_id", "season", "season_type"],
        index_cols=[["player_id"], ["season"], ["team", "season"]],
        source_fn="load_player_stats(reg+post)",
    )
