"""Player master table.

Keyed on gsis_id, the same id used by weekly stats, rosters, injuries and
play-by-play. That shared key is what makes a single player followable across
team changes. Also carries pfr_id and espn_id so the PFR-sourced tables
(snap_counts, pfr_advstats) can be joined back to the same player.
"""

import nflreadpy as nfl

from src.db import to_pandas, upsert_frame


def run(seasons=None) -> int:
    df = to_pandas(nfl.load_players())
    df = df.rename(columns={"gsis_id": "player_id"})
    df = df[df["player_id"].notna()]
    return upsert_frame(df, "players", ["player_id"], source_fn="load_players")
