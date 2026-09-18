"""Per-game snap counts and snap share.

The quantitative counterpart to the depth chart: how much a backup actually played.
Keyed on pfr_player_id, which joins to players.pfr_id.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_snap_counts(list(seasons)))
    return load_frame(
        df,
        "snap_counts",
        key_cols=["pfr_player_id", "game_id"],
        index_cols=[["game_id"], ["team", "season"], ["season", "week"]],
        source_fn="load_snap_counts",
    )
