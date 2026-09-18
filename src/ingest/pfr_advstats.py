"""Pro Football Reference advanced stats, per player per game.

This is the PFR-only charting layer that the standard box score does not carry:
pressures, hurries, blitzes, bad throws, drops, broken tackles, coverage targets.
Four stat types with different column sets, so four tables.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas

STAT_TYPES = ["pass", "rush", "rec", "def"]


def run(seasons) -> int:
    total = 0
    for stat_type in STAT_TYPES:
        df = to_pandas(
            nfl.load_pfr_advstats(list(seasons), stat_type=stat_type, summary_level="week")
        )
        total += load_frame(
            df,
            f"pfr_adv_{stat_type}",
            key_cols=["pfr_player_id", "game_id"],
            index_cols=[["game_id"], ["team", "season"], ["season", "week"]],
            source_fn=f"load_pfr_advstats({stat_type})",
        )
    return total
