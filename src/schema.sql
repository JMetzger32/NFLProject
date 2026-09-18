-- NFLProject core schema.
--
-- Design split:
--   * The "backbone" entity tables below are declared explicitly. They are small,
--     stable, and are what everything else joins against.
--   * The wide statistical tables (player_weekly_stats, plays, rosters, ngs_*,
--     snap_counts, injuries, depth_charts, ...) are NOT declared here. They are
--     created directly from the nflverse source frames by src/db.py:load_frame so
--     that every column the source ships is preserved, and new columns in future
--     nflverse releases are added automatically instead of being silently dropped.
--     Their keys and indexes are declared in src/ingest/specs.py.
--
-- Safe to run repeatedly.

CREATE TABLE IF NOT EXISTS teams (
    team_abbr          TEXT PRIMARY KEY,
    team_name          TEXT,
    team_id            TEXT,
    team_nick          TEXT,
    team_conf          TEXT,
    team_division      TEXT,
    team_color         TEXT,
    team_color2        TEXT,
    team_logo_espn     TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id          TEXT PRIMARY KEY,   -- nflverse gsis_id
    display_name       TEXT,
    first_name         TEXT,
    last_name          TEXT,
    position           TEXT,
    position_group     TEXT,
    birth_date         DATE,
    height             NUMERIC,
    weight             NUMERIC,
    college_name       TEXT,
    rookie_season      INTEGER,
    last_season        INTEGER,
    draft_club         TEXT,
    draft_number       INTEGER,
    esb_id             TEXT,
    pfr_id             TEXT,
    espn_id            TEXT,
    status             TEXT
);

CREATE INDEX IF NOT EXISTS idx_players_position ON players (position);
CREATE INDEX IF NOT EXISTS idx_players_display_name ON players (display_name);

CREATE TABLE IF NOT EXISTS games (
    game_id            TEXT PRIMARY KEY,
    season             INTEGER NOT NULL,
    game_type          TEXT,                -- REG / WC / DIV / CON / SB
    week               INTEGER,
    gameday            DATE,
    weekday            TEXT,
    gametime           TEXT,
    home_team          TEXT,
    away_team          TEXT,
    home_score         INTEGER,
    away_score         INTEGER,
    result             NUMERIC,             -- home_score - away_score
    total              NUMERIC,             -- home_score + away_score
    overtime           BOOLEAN,
    location           TEXT,
    roof               TEXT,
    surface            TEXT,
    temp               NUMERIC,
    wind               NUMERIC,
    stadium_id         TEXT,
    stadium            TEXT,
    div_game           BOOLEAN,
    home_rest          INTEGER,
    away_rest          INTEGER,
    home_coach         TEXT,
    away_coach         TEXT,
    referee            TEXT,
    home_qb_id         TEXT,
    away_qb_id         TEXT,
    home_qb_name       TEXT,
    away_qb_name       TEXT,
    old_game_id        TEXT,
    pfr                TEXT,
    espn               TEXT
);

CREATE INDEX IF NOT EXISTS idx_games_season_week ON games (season, week);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games (home_team, season);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games (away_team, season);
CREATE INDEX IF NOT EXISTS idx_games_gameday ON games (gameday);

-- One row per (game, source). Kept separate from games so additional books or
-- line snapshots can be added later without altering games.
CREATE TABLE IF NOT EXISTS betting_lines (
    game_id            TEXT NOT NULL REFERENCES games (game_id) ON DELETE CASCADE,
    source             TEXT NOT NULL DEFAULT 'nflverse_closing',
    season             INTEGER,
    week               INTEGER,
    spread_line        NUMERIC,             -- positive favors the home team
    total_line         NUMERIC,
    home_moneyline     INTEGER,
    away_moneyline     INTEGER,
    home_spread_odds   INTEGER,
    away_spread_odds   INTEGER,
    over_odds          INTEGER,
    under_odds         INTEGER,
    captured_at        TIMESTAMPTZ,
    PRIMARY KEY (game_id, source)
);

CREATE INDEX IF NOT EXISTS idx_betting_lines_season_week ON betting_lines (season, week);

-- One row per team-game the picks pipeline evaluated — PICK OR NOT.
-- Storing only picks would make calibration unmeasurable: you could never tell
-- whether the model is well-calibrated across the probability range, only whether
-- the filtered subset happened to win.
CREATE TABLE IF NOT EXISTS predictions (
    game_id              TEXT NOT NULL,
    team                 TEXT NOT NULL,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    opponent             TEXT,
    is_home              BOOLEAN,
    model_prob           NUMERIC,      -- bagged bootstrap point estimate
    ci_lo                NUMERIC,      -- 2.5th percentile across resamples
    ci_hi                NUMERIC,      -- 97.5th percentile
    market_prob_raw      NUMERIC,      -- still contains vig; diagnostic only
    market_prob_devig    NUMERIC,      -- what the model is actually compared against
    devig_method         TEXT,
    overround            NUMERIC,
    team_moneyline       INTEGER,
    opp_moneyline        INTEGER,
    spread_line          NUMERIC,
    edge                 NUMERIC,      -- model_prob - market_prob_devig
    ev_dollars           NUMERIC,
    kelly_full           NUMERIC,      -- diagnostic; staking is flat
    is_pick              BOOLEAN NOT NULL DEFAULT FALSE,
    stake                NUMERIC DEFAULT 0,
    confidence_label     TEXT,
    -- CLV: columns defined, deliberately NULL. nflverse publishes one line per
    -- game with no timestamp, so line-at-pick-time cannot be recovered. Populating
    -- these requires a capture job that does not exist yet.
    line_at_pick         NUMERIC,
    closing_line         NUMERIC,
    clv                  NUMERIC,
    model_version        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, team)
);

CREATE INDEX IF NOT EXISTS idx_predictions_season_week ON predictions (season, week);
CREATE INDEX IF NOT EXISTS idx_predictions_is_pick ON predictions (is_pick);

-- Grading, written once final scores exist. Separate table so re-grading never
-- risks mutating the original prediction record.
CREATE TABLE IF NOT EXISTS pick_results (
    game_id              TEXT NOT NULL,
    team                 TEXT NOT NULL,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    won                  INTEGER,       -- 1/0 actual outcome
    is_pick              BOOLEAN,
    stake                NUMERIC,
    payout               NUMERIC,       -- net profit/loss on the stake
    bankroll_after       NUMERIC,
    graded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, team),
    FOREIGN KEY (game_id, team) REFERENCES predictions (game_id, team) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pick_results_season_week ON pick_results (season, week);

-- Provenance for every ingest run, so a partial/failed backfill is diagnosable.
CREATE TABLE IF NOT EXISTS ingest_log (
    id                 BIGSERIAL PRIMARY KEY,
    table_name         TEXT NOT NULL,
    seasons            INTEGER[],
    row_count          INTEGER,
    source_fn          TEXT,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ,
    status             TEXT,
    message            TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_log_table ON ingest_log (table_name, started_at DESC);
