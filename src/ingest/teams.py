"""Team lookup table. Season-independent."""

import nflreadpy as nfl

from src.db import to_pandas, upsert_frame


def run(seasons=None) -> int:
    df = to_pandas(nfl.load_teams())
    return upsert_frame(df, "teams", ["team_abbr"], source_fn="load_teams")
