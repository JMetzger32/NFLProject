"""Weekly injury reports: practice participation and game-status designations."""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    df = to_pandas(nfl.load_injuries(list(seasons)))
    df = df.rename(columns={"gsis_id": "player_id"})
    return load_frame(
        df,
        "injuries",
        key_cols=["player_id", "season", "week", "team"],
        index_cols=[["player_id", "season"], ["team", "season", "week"], ["report_status"]],
        source_fn="load_injuries",
    )
