"""Games and the closing Vegas lines attached to them.

nflverse ships both in one schedules frame; they are split across two tables here
so additional books or line snapshots can be added to betting_lines later without
touching games.
"""

import nflreadpy as nfl

from src.db import to_pandas, upsert_frame

LINE_COLS = [
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
]


def run(seasons) -> int:
    df = to_pandas(nfl.load_schedules(list(seasons)))
    games = upsert_frame(df, "games", ["game_id"], source_fn="load_schedules")

    cols = ["game_id", "season", "week"] + [c for c in LINE_COLS if c in df.columns]
    lines = df[cols].copy()
    lines["source"] = "nflverse_closing"
    upsert_frame(lines, "betting_lines", ["game_id", "source"], source_fn="load_schedules")
    return games
