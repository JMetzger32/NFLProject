"""Ingest modules. Each exposes run(seasons) -> number of rows loaded."""

from src.ingest import (
    depth_charts,
    injuries,
    ngs,
    pbp,
    pfr_advstats,
    players,
    rosters,
    schedules,
    seasonal_stats,
    snap_counts,
    team_stats,
    teams,
    trades,
    weekly_stats,
)

# Dependency order: backbone entities first, then everything that references them.
# Play-by-play is last because it is by far the slowest.
PIPELINE = [
    ("teams", teams),
    ("players", players),
    ("games + betting_lines", schedules),
    ("weekly_player_stats", weekly_stats),
    ("seasonal_player_stats", seasonal_stats),
    ("team_weekly_stats", team_stats),
    ("weekly_rosters", rosters),
    ("depth_charts", depth_charts),
    ("snap_counts", snap_counts),
    ("injuries", injuries),
    ("trades", trades),
    ("next_gen_stats", ngs),
    ("pfr_advanced_stats", pfr_advstats),
    ("plays", pbp),
]
