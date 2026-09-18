"""Depth charts — where a backup sits, and when he gets promoted.

`depth_rank` within a team + position group (1 = starter) is how a backup moving
into a starting role becomes visible before the box score shows it.

nflverse changed this feed's format for 2025. Both eras are normalized into one
table so a query does not have to know which era it is reading:

  2021-2024  weekly rows        -> season, week, club_code, depth_team, formation
  2025+      dated snapshots    -> dt, team, pos_rank (no week; several per week)

`week` is null for snapshot-era rows and `snapshot_dt` is null for weekly-era rows;
`source_format` says which is which. Downsampling the snapshot era to one row per
week is left to feature engineering rather than baked in here.
"""

import nflreadpy as nfl
import pandas as pd

from src.db import load_frame, to_pandas

WEEKLY_RENAMES = {
    "gsis_id": "player_id",
    "club_code": "team",
    "depth_team": "depth_rank",
    "depth_position": "position",
    "full_name": "player_name",
}

SNAPSHOT_RENAMES = {
    "gsis_id": "player_id",
    "pos_rank": "depth_rank",
    "pos_abb": "position",
    "pos_grp": "position_group",
    "dt": "snapshot_dt",
}


def _normalize(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if "dt" in df.columns:
        df = df.rename(columns=SNAPSHOT_RENAMES)
        df["source_format"] = "snapshot"
        df["week"] = pd.NA
    else:
        df = df.rename(columns=WEEKLY_RENAMES)
        df["source_format"] = "weekly"
        df["snapshot_dt"] = pd.NaT
    df["season"] = season
    df["depth_rank"] = pd.to_numeric(df["depth_rank"], errors="coerce")
    return df


def run(seasons) -> int:
    total = 0
    for season in seasons:
        df = to_pandas(nfl.load_depth_charts(season))
        if df.empty:
            continue
        df = _normalize(df, season)
        total += load_frame(
            df,
            "depth_charts",
            # No unique key: the natural grain differs between the two eras, and
            # half the candidate key is null in each. Idempotency comes from the
            # season-scoped delete-and-reload that load_frame does anyway.
            key_cols=(),
            index_cols=[
                ["player_id", "season"],
                ["team", "season"],
                ["position", "depth_rank"],
            ],
            source_fn="load_depth_charts",
        )
    return total
