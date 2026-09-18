"""Backfill every table for the given seasons.

    python scripts/backfill_all.py                # default seasons (2021-2025)
    python scripts/backfill_all.py 2025           # one season, for a smoke test
    python scripts/backfill_all.py 2021 2022      # a subset
    python scripts/backfill_all.py --skip pbp     # everything except play-by-play
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_SEASONS  # noqa: E402
from src.db import init_schema, row_count  # noqa: E402
from src.ingest import PIPELINE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seasons", nargs="*", type=int, default=None)
    parser.add_argument("--skip", nargs="*", default=[], help="module names to skip")
    parser.add_argument("--only", nargs="*", default=[], help="module names to run")
    args = parser.parse_args()

    seasons = args.seasons or DEFAULT_SEASONS
    print(f"Seasons: {seasons}\n")

    init_schema()

    failures = []
    for label, module in PIPELINE:
        name = module.__name__.rsplit(".", 1)[-1]
        if name in args.skip or (args.only and name not in args.only):
            print(f"-- {label}: skipped")
            continue
        start = time.time()
        print(f"-- {label} ({name})")
        try:
            rows = module.run(seasons)
            print(f"   {rows:,} rows in {time.time() - start:.1f}s")
        except Exception as exc:
            failures.append((name, exc))
            print(f"   FAILED: {exc}")
            traceback.print_exc()

    print("\nRow counts:")
    for table in [
        "teams",
        "players",
        "games",
        "betting_lines",
        "weekly_player_stats",
        "seasonal_player_stats",
        "team_weekly_stats",
        "weekly_rosters",
        "depth_charts",
        "snap_counts",
        "injuries",
        "trades",
        "ngs_passing",
        "ngs_rushing",
        "ngs_receiving",
        "pfr_adv_pass",
        "pfr_adv_rush",
        "pfr_adv_rec",
        "pfr_adv_def",
        "plays",
    ]:
        print(f"  {table:<24} {row_count(table):>10,}")

    if failures:
        print(f"\n{len(failures)} module(s) failed: {[f[0] for f in failures]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
