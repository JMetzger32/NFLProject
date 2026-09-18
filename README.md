# NFLProject

Project to try and predict nfl games and track stats throughout the year.


Historical NFL statistics in PostgreSQL, built as the data foundation for modeling
NFL games and finding edges against the betting market.

## Data source

All data comes from **[nflverse](https://github.com/nflverse)** via the
[`nflreadpy`](https://github.com/nflverse/nflreadpy) client.

Why nflverse:

- **Free, no API key, no rate limit.** It reads static Parquet releases published on
  GitHub, so a five-season backfill takes minutes and can be re-run at will.
- **It already contains Pro Football Reference data.** PFR is one of nflverse's
  upstream sources, and `pfr_adv_*` exposes PFR's advanced charting directly
  (pressures, hurries, blitzes, bad throws, drops, broken tackles). Scraping
  pro-football-reference.com separately would duplicate this and add a fragile,
  rate-limited dependency. Keep PFR as a manual lookup for one-off stats only.
- **It ships the betting lines too.** The schedules feed carries closing spread,
  total and moneyline per game, which is what the market-edge work needs.

Paid alternatives (SportsDataIO, Sportradar) start around $500/month and add value
only for live in-game feeds — not for historical backtesting.

> Note: `nfl_data_py`, nflverse's older Python client, pins `pandas<2.0` and
> `numpy<2.0`. That would block modern scikit-learn/xgboost later, so this project
> uses `nflreadpy` (polars-based, no such pins). It is the maintained successor and
> exposes strictly more data.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Start Postgres. Either the container:

```bash
docker compose up -d
```

...or the local Homebrew cluster currently in use on this machine:

```bash
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
pg_ctl -D pgdata -o "-p 5433 -k /tmp" -l pgdata/server.log start   # stop: pg_ctl -D pgdata stop
```

Then create the tables and load the data:

```bash
./venv/bin/python scripts/init_db.py
./venv/bin/python scripts/backfill_all.py            # 2021-2025
./venv/bin/python scripts/backfill_all.py 2025       # one season
./venv/bin/python scripts/backfill_all.py --skip pbp # everything but play-by-play
```

Backfills are idempotent — rows for the seasons being loaded are deleted and
replaced, so re-running never duplicates data.

Connect with `psql "postgresql://nfl@localhost:5433/nfl"`.

## Schema

Two kinds of table.

**Backbone** (declared in `src/schema.sql`, explicitly typed, with primary and
foreign keys). These are small, stable, and are what everything joins against:

| Table | Grain |
|---|---|
| `teams` | one row per franchise |
| `players` | one row per player, keyed on nflverse `gsis_id` (= `player_id`) |
| `games` | one row per game |
| `betting_lines` | one row per (game, source) — closing spread, total, moneylines |
| `ingest_log` | provenance for every load |

**Statistical** (created directly from the source frames by `src/db.py:load_frame`).
Every column the source ships is preserved, and new columns in future nflverse
releases are added automatically rather than silently dropped:

| Table | Grain | Notes |
|---|---|---|
| `weekly_player_stats` | player × week | ~150 cols; carries `team` per week |
| `seasonal_player_stats` | player × season | reg + post |
| `team_weekly_stats` | team × week | ~138 cols, pre-aggregated EPA/box score |
| `weekly_rosters` | player × week | team + status; diff weeks to detect roster moves |
| `depth_charts` | player × week (2021-24) or × snapshot (2025+) | `depth_rank` = 1 is the starter; see note below |
| `snap_counts` | player × game | snap share; keyed on `pfr_player_id` |
| `injuries` | player × week | practice participation + game status |
| `trades` | trade leg | full history, includes draft capital |
| `ngs_passing/rushing/receiving` | player × week | Next Gen Stats tracking metrics |
| `pfr_adv_pass/rush/rec/def` | player × game | PFR advanced charting |
| `plays` | play | 372 columns: EPA, WP, air yards, CPOE, drive context |

### Design principle

Everything is keyed on a small set of stable identifiers — `season`, `week`,
`game_id`, `team`, `player_id` — so a new feature category becomes a **new table**
joined on those keys, never new columns bolted onto an existing table. The feature
set is deliberately undecided; this schema is built to absorb that.

### Tracking players across teams, and backups

Three tables make roster churn measurable, all joinable on `player_id`:

- `weekly_rosters` — which team a player was on each week, and his status.
  Diffing consecutive weeks surfaces trades, signings, releases and IR stints.
- `depth_charts` — `pos_rank` within a position group, so a backup being promoted
  is visible *before* it shows up in a box score.
- `snap_counts` — how much he actually played (`offense_pct` / `defense_pct`).

`weekly_player_stats` carries `team` on every row, so production follows the player
while remaining attributable to the right team.

PFR-sourced tables (`snap_counts`, `pfr_adv_*`) key on `pfr_player_id`; join them
back through `players.pfr_id`.

> **Depth chart caveat:** nflverse changed this feed's format in 2025. 2021-2024 are
> weekly rows; 2025+ are dated snapshots (several per week). Both are normalized into
> `depth_charts` — use `week` for the older era, `snapshot_dt` for the newer, and
> `source_format` to tell them apart.

## Currently loaded (seasons 2021-2025)

1,424 games with 100% betting-line coverage · 247,284 plays (372 columns) ·
94,738 player-weeks · 231,859 roster-weeks · 704,121 depth chart rows ·
132,616 snap-count rows · 29,149 injury rows · 78,205 PFR advanced rows.

## Layout

```
src/config.py        .env loading, DATABASE_URL, default seasons
src/db.py            engine, schema init, load_frame / upsert_frame loaders
src/schema.sql       backbone tables
src/ingest/*.py      one module per source feed, each exposing run(seasons)
scripts/init_db.py   apply the schema
scripts/backfill_all.py  run the whole pipeline
```
