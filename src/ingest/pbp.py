"""Play-by-play: the widest and largest table (~380 columns, ~50k rows per season).

Loaded one season at a time to keep memory bounded, with the full nflverse column
set preserved (EPA, win probability, air yards, CPOE, drive and situation context).
"""

import nflreadpy as nfl

from src.db import load_frame, to_pandas


def run(seasons) -> int:
    total = 0
    for season in seasons:
        df = to_pandas(nfl.load_pbp(season))
        rows = load_frame(
            df,
            "plays",
            key_cols=["game_id", "play_id"],
            index_cols=[
                ["game_id"],
                ["season", "week"],
                ["posteam", "season"],
                ["passer_player_id"],
                ["rusher_player_id"],
                ["receiver_player_id"],
            ],
            source_fn="load_pbp",
        )
        total += rows
        print(f"    season {season}: {rows:,} plays")
    return total
