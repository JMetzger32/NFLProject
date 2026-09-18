"""Next Gen Stats: player-tracking-derived passing, rushing and receiving metrics.

Three feeds with different column sets, so three tables.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas

STAT_TYPES = ["passing", "rushing", "receiving"]


def run(seasons) -> int:
    total = 0
    for stat_type in STAT_TYPES:
        df = to_pandas(nfl.load_nextgen_stats(list(seasons), stat_type=stat_type))
        df = df.rename(columns={"player_gsis_id": "player_id"})
        total += load_frame(
            df,
            f"ngs_{stat_type}",
            key_cols=["player_id", "season", "week", "season_type"],
            index_cols=[["player_id", "season"], ["team_abbr", "season"]],
            source_fn=f"load_nextgen_stats({stat_type})",
        )
    return total
