"""Trade history: who moved where, and what draft capital moved with them.

Season-independent full reload; the source ships the complete history in one frame.
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons=None) -> int:
    df = to_pandas(nfl.load_trades())
    return load_frame(
        df,
        "trades",
        key_cols=(),
        season_col=None,
        index_cols=[["season"], ["pfr_id"], ["trade_id"]],
        source_fn="load_trades",
    )
