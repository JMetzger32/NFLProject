"""Team-game panel, rolling features, and the leakage guard.

One row per team-game (not per game), so QB and defense features can be asymmetric
between the two sides of a matchup.

THE LEAKAGE RULE
----------------
Anything measured *during* a game may only become a predictor for a *later* game.
The build order enforces this and must not be short-circuited:

  1. raw_team_game_stats()  same-game measurements. NEVER predictors.
  2. add_rolling()          the only way step 1 becomes a feature: shift(1) then roll.
  3. build_model_frame()    joins lagged features to the target.

Pre-game information (injury reports, depth charts, rest, travel, the closing line)
is known before kickoff and is used unlagged on purpose. It is listed in
PREGAME_COLUMNS so the leakage checker does not flag it.

Usage:
    python -m src.features --verify-leakage
    python -m src.features --summary
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.db import get_engine
from src.stadiums import timezone_shift, travel_miles

# Rolling windows. 4 games ~= a month of form, 8 ~= half a season of signal.
ROLLING_WINDOWS = (4, 8)

# Injury report positions that matter most, grouped. OL and EDGE are the positions
# whose absence most changes pass protection and pass rush.
POSITION_GROUPS = {
    "qb": ("QB",),
    "ol": ("T", "G", "C", "OT", "OG"),
    "cb": ("CB",),
    "edge": ("DE", "OLB"),
}

# Known before kickoff, so intentionally not lagged.
PREGAME_COLUMNS = {
    "is_home", "rest_days", "short_week", "div_game", "is_dome", "temp", "wind",
    "wind_missing", "travel_miles", "tz_shift", "team_spread", "total_line",
    "team_ml_implied_prob", "week", "season", "games_of_history",
    "inj_qb_out", "inj_ol_out", "inj_cb_out", "inj_edge_out", "inj_total_out",
    # EDA_2: continuity / coaching, also known before kickoff.
    "roster_continuity_score", "coach_tenure_weeks", "coach_tenure_censored",
    "division_familiarity",
}

# Top-N offensive-snap players compared week-to-week for roster_continuity_score.
CONTINUITY_TOP_N = 11
CONTINUITY_LOOKBACK = 8

_RAW_SQL = """
WITH base AS (
    SELECT game_id, season, week, gameday, home_team, away_team,
           home_score, away_score, result, div_game, roof, temp, wind,
           home_rest, away_rest, location, home_coach, away_coach
    FROM games
    WHERE game_type = 'REG' AND ({result_filter})
),
panel AS (
    SELECT game_id, season, week, gameday, home_team AS team, away_team AS opponent,
           TRUE AS is_home, home_score AS points_for, away_score AS points_against,
           home_rest AS rest_days, div_game, roof, temp, wind, location,
           home_coach AS coach
    FROM base
    UNION ALL
    SELECT game_id, season, week, gameday, away_team, home_team,
           FALSE, away_score, home_score,
           away_rest, div_game, roof, temp, wind, location,
           away_coach
    FROM base
),
off AS (
    SELECT p.game_id, p.posteam AS team,
        count(*) FILTER (WHERE p.play_type IN ('pass','run'))            AS off_plays,
        avg(p.epa) FILTER (WHERE p.play_type IN ('pass','run'))          AS off_epa_play,
        avg(p.epa) FILTER (WHERE p.pass = 1)                             AS off_pass_epa,
        avg(p.epa) FILTER (WHERE p.rush = 1)                             AS off_rush_epa,
        avg(p.success) FILTER (WHERE p.play_type IN ('pass','run'))      AS off_success,
        avg(p.cpoe)                                                      AS off_cpoe,
        avg(p.air_yards) FILTER (WHERE p.pass_attempt = 1)               AS off_adot,
        sum(p.qb_dropback)                                               AS off_dropbacks,
        sum(p.sack)                                                      AS off_sacks_taken,
        sum(p.qb_scramble)                                               AS off_scrambles,
        sum(p.third_down_converted)                                      AS off_3d_conv,
        sum(p.third_down_converted) + sum(p.third_down_failed)           AS off_3d_att,
        count(DISTINCT p.drive)                                          AS off_drives,
        sum(CASE WHEN p.pass = 1 THEN 1 ELSE 0 END)                      AS off_pass_plays,
        sum(CASE WHEN p.rush = 1 THEN 1 ELSE 0 END)                      AS off_rush_plays,
        sum(p.yards_gained) FILTER (WHERE p.play_type IN ('pass','run')) AS off_total_yards,
        count(*) FILTER (WHERE (p.pass = 1 AND p.yards_gained >= 20)
                             OR (p.rush = 1 AND p.yards_gained >= 10))   AS off_explosive_plays,
        count(DISTINCT CASE WHEN p.drive_inside20 = 1
                             THEN p.drive END)                           AS off_rz_trips,
        count(DISTINCT CASE WHEN p.drive_inside20 = 1
                              AND p.fixed_drive_result = 'Touchdown'
                             THEN p.drive END)                           AS off_rz_tds,
        count(*) FILTER (WHERE p.interception = 1 OR p.fumble_lost = 1) AS off_turnovers
    FROM plays p
    WHERE p.posteam IS NOT NULL AND p.season_type = 'REG'
    GROUP BY 1, 2
),
def AS (
    SELECT p.game_id, p.defteam AS team,
        count(*) FILTER (WHERE p.play_type IN ('pass','run'))            AS def_plays,
        avg(p.epa) FILTER (WHERE p.play_type IN ('pass','run'))          AS def_epa_allowed,
        avg(p.epa) FILTER (WHERE p.pass = 1)                             AS def_pass_epa_allowed,
        avg(p.epa) FILTER (WHERE p.rush = 1)                             AS def_rush_epa_allowed,
        avg(p.success) FILTER (WHERE p.play_type IN ('pass','run'))      AS def_success_allowed,
        sum(p.sack)                                                      AS def_sacks,
        sum(p.qb_hit)                                                    AS def_qb_hits,
        sum(COALESCE(p.tackled_for_loss,0) + COALESCE(p.interception,0)
            + COALESCE(p.fumble_forced,0))                               AS def_havoc_events,
        sum(p.third_down_converted)                                      AS def_3d_conv_allowed,
        sum(p.third_down_converted) + sum(p.third_down_failed)           AS def_3d_att,
        sum(p.qb_dropback)                                               AS def_dropbacks_faced,
        count(DISTINCT p.drive)                                          AS def_drives_faced,
        count(*) FILTER (WHERE (p.pass = 1 AND p.yards_gained >= 20)
                             OR (p.rush = 1 AND p.yards_gained >= 10))   AS def_explosive_plays_allowed,
        count(DISTINCT CASE WHEN p.drive_inside20 = 1
                             THEN p.drive END)                           AS def_rz_trips_allowed,
        count(DISTINCT CASE WHEN p.drive_inside20 = 1
                              AND p.fixed_drive_result = 'Touchdown'
                             THEN p.drive END)                           AS def_rz_tds_allowed,
        count(*) FILTER (WHERE p.interception = 1 OR p.fumble_lost = 1) AS def_takeaways
    FROM plays p
    WHERE p.defteam IS NOT NULL AND p.season_type = 'REG'
    GROUP BY 1, 2
),
-- Drive-level field position (yardline_100: distance from the opponent's end
-- zone, so lower = better). `drive_start_yard_line` is a text field ("DET 14"),
-- not usable numerically, so this takes yardline_100 on each drive's first play
-- instead, then averages over DISTINCT drives so a long drive isn't overweighted
-- relative to a three-and-out.
drive_fp AS (
    SELECT game_id, team, avg(start_yardline_100) AS off_avg_start_field_pos
    FROM (
        SELECT DISTINCT ON (p.game_id, p.posteam, p.drive)
               p.game_id, p.posteam AS team, p.drive, p.yardline_100 AS start_yardline_100
        FROM plays p
        WHERE p.posteam IS NOT NULL AND p.season_type = 'REG'
        ORDER BY p.game_id, p.posteam, p.drive, p.play_id ASC
    ) d
    GROUP BY 1, 2
),
-- Penalties keyed by who committed them, split by whether they were on offense or
-- defense at the time (posteam/defteam context on the penalized play).
penalty AS (
    SELECT game_id, team,
           sum(penalty_yards) FILTER (WHERE team = posteam) AS off_penalty_yards,
           sum(penalty_yards) FILTER (WHERE team = defteam) AS def_penalty_yards
    FROM (
        SELECT game_id, penalty_team AS team, penalty_yards, posteam, defteam
        FROM plays WHERE penalty = 1 AND penalty_team IS NOT NULL AND season_type = 'REG'
    ) x
    GROUP BY 1, 2
),
-- Special-teams EPA from both sides of the ball, signed to the team: value added
-- kicking/punting/covering (posteam) plus value added returning (defteam, sign
-- flipped since EPA is stored from the possession team's perspective).
st AS (
    SELECT game_id, team, avg(epa) AS st_epa
    FROM (
        SELECT game_id, posteam AS team, epa
        FROM plays
        WHERE play_type IN ('kickoff','punt','extra_point','field_goal')
          AND posteam IS NOT NULL AND season_type = 'REG'
        UNION ALL
        SELECT game_id, defteam AS team, -epa
        FROM plays
        WHERE play_type IN ('kickoff','punt','extra_point','field_goal')
          AND defteam IS NOT NULL AND season_type = 'REG'
    ) u
    GROUP BY 1, 2
),
-- The starter's Next Gen Stats time-to-throw. NGS has no game_id, so the join key
-- is (season, week, team) instead; the busiest passer is picked the same way the
-- main `qb` CTE picks the starter.
ngs_qb AS (
    SELECT DISTINCT ON (season, week, team_abbr)
           season, week, team_abbr AS team, avg_time_to_throw AS qb_time_to_throw
    FROM ngs_passing
    WHERE season_type = 'REG'
    ORDER BY season, week, team_abbr, attempts DESC NULLS LAST
),
qb_by_player AS (
    SELECT p.game_id, p.posteam AS team, p.passer_player_id AS qb_player_id,
           count(*)                                          AS qb_plays,
           avg(p.qb_epa)                                     AS qb_epa,
           avg(p.cpoe)                                       AS qb_cpoe,
           avg(p.success)                                    AS qb_success,
           avg(p.air_yards) FILTER (WHERE p.pass_attempt = 1) AS qb_adot,
           sum(p.sack)                                       AS qb_sacks_taken
    FROM plays p
    WHERE p.passer_player_id IS NOT NULL AND p.posteam IS NOT NULL
      AND p.season_type = 'REG'
    GROUP BY 1, 2, 3
),
qb AS (
    -- The starter is whoever threw the most passes; backups who mop up don't displace him.
    SELECT DISTINCT ON (game_id, team)
           game_id, team, qb_player_id, qb_plays, qb_epa, qb_cpoe,
           qb_success, qb_adot, qb_sacks_taken
    FROM qb_by_player
    ORDER BY game_id, team, qb_plays DESC
),
prs_def AS (
    SELECT game_id, team,
           sum(def_pressures)         AS def_pressures,
           sum(def_times_blitzed)     AS def_blitzes,
           sum(def_missed_tackles)    AS def_missed_tackles,
           sum(def_tackles_combined)  AS def_tackles_combined
    FROM pfr_adv_def GROUP BY 1, 2
),
prs_off AS (
    SELECT game_id, team,
           sum(passing_bad_throws) AS qb_bad_throws,
           sum(times_pressured)    AS off_pressures_faced,
           sum(passing_drops)      AS off_drops
    FROM pfr_adv_pass GROUP BY 1, 2
)
SELECT pa.*,
       o.off_plays, o.off_epa_play, o.off_pass_epa, o.off_rush_epa, o.off_success,
       o.off_cpoe, o.off_adot, o.off_dropbacks, o.off_sacks_taken,
       o.off_3d_conv, o.off_3d_att, o.off_drives, o.off_pass_plays, o.off_rush_plays,
       o.off_scrambles, o.off_total_yards, o.off_explosive_plays, o.off_rz_trips,
       o.off_rz_tds, o.off_turnovers,
       d.def_plays, d.def_epa_allowed, d.def_pass_epa_allowed, d.def_rush_epa_allowed,
       d.def_success_allowed, d.def_sacks, d.def_qb_hits, d.def_havoc_events,
       d.def_3d_conv_allowed, d.def_3d_att, d.def_dropbacks_faced, d.def_drives_faced,
       d.def_explosive_plays_allowed, d.def_rz_trips_allowed, d.def_rz_tds_allowed,
       d.def_takeaways,
       q.qb_player_id, q.qb_plays, q.qb_epa, q.qb_cpoe, q.qb_success, q.qb_adot,
       q.qb_sacks_taken,
       pd.def_pressures, pd.def_blitzes, pd.def_missed_tackles, pd.def_tackles_combined,
       po.qb_bad_throws, po.off_pressures_faced, po.off_drops,
       fp.off_avg_start_field_pos,
       pen.off_penalty_yards, pen.def_penalty_yards,
       st.st_epa,
       ngs.qb_time_to_throw
FROM panel pa
LEFT JOIN off    o   ON o.game_id   = pa.game_id AND o.team   = pa.team
LEFT JOIN def    d   ON d.game_id   = pa.game_id AND d.team   = pa.team
LEFT JOIN qb     q   ON q.game_id   = pa.game_id AND q.team   = pa.team
LEFT JOIN prs_def pd ON pd.game_id  = pa.game_id AND pd.team  = pa.team
LEFT JOIN prs_off po ON po.game_id  = pa.game_id AND po.team  = pa.team
LEFT JOIN drive_fp fp ON fp.game_id = pa.game_id AND fp.team  = pa.team
LEFT JOIN penalty pen ON pen.game_id = pa.game_id AND pen.team = pa.team
LEFT JOIN st      st  ON st.game_id  = pa.game_id AND st.team  = pa.team
LEFT JOIN ngs_qb  ngs ON ngs.season = pa.season AND ngs.week = pa.week AND ngs.team = pa.team
ORDER BY pa.season, pa.week, pa.game_id, pa.team
"""

_MARKET_SQL = """
SELECT g.game_id, b.spread_line, b.total_line, b.home_moneyline, b.away_moneyline
FROM games g JOIN betting_lines b USING (game_id)
WHERE g.game_type = 'REG' AND ({result_filter})
"""

# `{p}` is a table-qualifier placeholder: empty in _RAW_SQL (which selects straight
# FROM games) and "g." in _MARKET_SQL (where games is joined to betting_lines and
# season/week would otherwise be ambiguous).
# Completed games only — the default everywhere: training, backtesting, EDA.
_PLAYED = "{p}result IS NOT NULL AND {p}result <> 0"
# Completed games PLUS one upcoming week, for live prediction. Ties stay excluded;
# unplayed rows arrive with result NULL and therefore `won` NULL.
_PLAYED_PLUS_UPCOMING = (
    "(({p}result IS NOT NULL AND {p}result <> 0) "
    "OR ({p}result IS NULL AND {p}season = {season} AND {p}week = {week}))"
)

_INJURY_SQL = """
SELECT season, week, team, position, count(*) AS n
FROM injuries
WHERE report_status IN ('Out', 'Doubtful')
GROUP BY 1, 2, 3, 4
"""


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Per-attempt rate, guarding against divide-by-zero on odd game scripts."""
    den = den.replace(0, np.nan)
    return num / den


def _result_filter(upcoming: tuple[int, int] | None, prefix: str = "") -> str:
    if upcoming is None:
        return _PLAYED.format(p=prefix)
    season, week = upcoming
    return _PLAYED_PLUS_UPCOMING.format(p=prefix, season=int(season), week=int(week))


def raw_team_game_stats(upcoming: tuple[int, int] | None = None) -> pd.DataFrame:
    """Step 1: one row per team-game of SAME-GAME measurements. Never predictors.

    `upcoming=(season, week)` additionally includes that week's UNPLAYED games, for
    live prediction. Those rows carry `won` = NaN and no same-game stats; their
    rolling features come from prior completed games via the usual shift(1), so the
    forward path introduces no new leakage surface.
    """
    eng = get_engine()
    df = pd.read_sql(
        _RAW_SQL.format(result_filter=_result_filter(upcoming)), eng)
    market = pd.read_sql(
        _MARKET_SQL.format(result_filter=_result_filter(upcoming, prefix="g.")), eng)
    inj = pd.read_sql(_INJURY_SQL, eng)

    played = df["points_for"].notna() & df["points_against"].notna()
    df["won"] = np.where(
        played, (df["points_for"] > df["points_against"]).astype(float), np.nan)
    df["is_upcoming"] = ~played
    df["gameday"] = pd.to_datetime(df["gameday"])

    # Derived same-game rates.
    df["off_sack_rate"] = _rate(df["off_sacks_taken"], df["off_dropbacks"])
    df["off_3d_rate"] = _rate(df["off_3d_conv"], df["off_3d_att"])
    df["off_pass_rate"] = _rate(df["off_pass_plays"], df["off_plays"])
    df["off_points_per_drive"] = _rate(df["points_for"], df["off_drives"])
    # QB mobility: scrambles per dropback. Drives the pressure-defense interaction.
    df["qb_scramble_rate"] = _rate(df["off_scrambles"], df["off_dropbacks"])
    df["def_points_per_drive"] = _rate(df["points_against"], df["def_drives_faced"])
    df["def_3d_stop_rate"] = 1 - _rate(df["def_3d_conv_allowed"], df["def_3d_att"])
    df["def_havoc_rate"] = _rate(df["def_havoc_events"], df["def_plays"])
    df["def_pressure_rate"] = _rate(df["def_pressures"], df["def_dropbacks_faced"])
    df["def_blitz_rate"] = _rate(df["def_blitzes"], df["def_dropbacks_faced"])
    df["def_pressure_to_sack"] = _rate(df["def_sacks"], df["def_pressures"])
    df["qb_sack_rate"] = _rate(df["qb_sacks_taken"], df["qb_plays"])
    df["qb_bad_throw_rate"] = _rate(df["qb_bad_throws"], df["off_dropbacks"])
    df["qb_pressured_rate"] = _rate(df["off_pressures_faced"], df["off_dropbacks"])
    df["off_rush_rate"] = _rate(df["off_rush_plays"], df["off_plays"])

    # EDA_2: drive efficiency, turnovers, special teams, PFR depth.
    df["off_red_zone_td_rate"] = _rate(df["off_rz_tds"], df["off_rz_trips"])
    df["off_yards_per_drive"] = _rate(df["off_total_yards"], df["off_drives"])
    df["off_plays_per_drive"] = _rate(df["off_plays"], df["off_drives"])
    df["off_explosive_play_rate"] = _rate(df["off_explosive_plays"], df["off_plays"])
    df["off_turnover_rate"] = _rate(df["off_turnovers"], df["off_drives"])
    df["off_penalty_yards_per_game"] = df["off_penalty_yards"].fillna(0)

    df["def_red_zone_td_rate_allowed"] = _rate(df["def_rz_tds_allowed"], df["def_rz_trips_allowed"])
    df["def_explosive_play_rate_allowed"] = _rate(df["def_explosive_plays_allowed"], df["def_plays"])
    df["def_takeaway_rate"] = _rate(df["def_takeaways"], df["def_drives_faced"])
    df["def_penalty_yards_per_game"] = df["def_penalty_yards"].fillna(0)
    df["def_broken_tackle_rate_allowed"] = _rate(df["def_missed_tackles"], df["def_tackles_combined"])

    # Built as one differenced series (takeaways minus giveaways) rather than lagging
    # the two rates separately and subtracting afterward — a difference-then-lag
    # avoids compounding the rolling noise of two independently-estimated rates.
    df["turnover_margin_rate"] = df["def_takeaway_rate"] - df["off_turnover_rate"]

    df["off_drop_rate"] = _rate(df["off_drops"], df["off_dropbacks"])
    # qb_time_to_throw is already in seconds from NGS — no rate conversion needed.

    # Market, oriented to this team rather than to the home side.
    market["home_implied"] = _implied_prob(market["home_moneyline"])
    market["away_implied"] = _implied_prob(market["away_moneyline"])
    df = df.merge(market, on="game_id", how="left")
    df["team_spread"] = np.where(df["is_home"], df["spread_line"], -df["spread_line"])
    df["team_ml_implied_prob"] = np.where(
        df["is_home"], df["home_implied"], df["away_implied"]
    )

    # Pre-game context.
    df["short_week"] = (df["rest_days"] < 7).astype(int)
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    # Outdoor wind is genuinely missing for many 2022-23 games, so the model gets an
    # explicit indicator rather than a silently imputed zero.
    df["wind_missing"] = (df["wind"].isna() & (df["is_dome"] == 0)).astype(int)
    df["wind"] = np.where(df["is_dome"] == 1, 0.0, df["wind"])
    df["temp"] = np.where(df["is_dome"] == 1, 70.0, df["temp"])
    host = np.where(df["is_home"], df["team"], df["opponent"])
    df["travel_miles"] = [travel_miles(t, h) for t, h in zip(df["team"], host)]
    df["tz_shift"] = [timezone_shift(t, h) for t, h in zip(df["team"], host)]

    df = _add_injury_counts(df, inj)
    return df


def _implied_prob(ml: pd.Series) -> pd.Series:
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def _add_injury_counts(df: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Injury reports are published before kickoff, so they are used unlagged."""
    for group, positions in POSITION_GROUPS.items():
        sub = (
            inj[inj["position"].isin(positions)]
            .groupby(["season", "week", "team"], as_index=False)["n"]
            .sum()
            .rename(columns={"n": f"inj_{group}_out"})
        )
        df = df.merge(sub, on=["season", "week", "team"], how="left")
        df[f"inj_{group}_out"] = df[f"inj_{group}_out"].fillna(0).astype(int)

    total = inj.groupby(["season", "week", "team"], as_index=False)["n"].sum().rename(
        columns={"n": "inj_total_out"}
    )
    df = df.merge(total, on=["season", "week", "team"], how="left")
    df["inj_total_out"] = df["inj_total_out"].fillna(0).astype(int)
    return df


def _trailing_mean_skipna(s: pd.Series, w: int) -> pd.Series:
    """Rolling mean over the last `w` NON-NULL values, reaching back as far as
    needed to find them — not `w` positional rows with nulls averaged out of a
    truncated window.

    Plain `Series.rolling(w).mean()` does NOT do this: it opens a fixed window of
    the last `w` ROWS and skips NaNs found inside that window, so a single sparse
    game (e.g. zero red-zone trips) silently shrinks the effective window instead
    of being replaced by reaching one game further back. For a rate stat that can
    legitimately be 0/0 for a given game (red-zone rate, turnover rate, NGS
    coverage gaps), that mismatch is not cosmetic — verified by
    assert_no_leakage's independent naive recomputation, which encodes the
    reach-back definition and caught this as a real discrepancy, not a rounding
    difference.
    """
    compact = s.dropna()
    rolled = compact.rolling(w, min_periods=1).mean()
    return rolled.reindex(s.index).ffill()


def add_rolling(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    also_season_to_date: bool = True,
) -> pd.DataFrame:
    """Step 2: THE leakage-safe transform. Every derived feature routes through here.

    shift(1) comes before the window, so a row never sees its own game. Grouping by
    (team, season) means form does not bleed across the offseason.

    min_periods=1 keeps weeks 1-4 rather than dropping them; `games_of_history`
    records how thin each window actually is so a model can discount it. The `_r{w}`
    windows reach back over sparse/null values (see _trailing_mean_skipna); the
    `_std` season-to-date columns use expanding(), which already skips NaN
    correctly on its own.
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    grp = df.groupby(["team", "season"], sort=False)

    df["games_of_history"] = grp.cumcount()

    for col in value_cols:
        prior = grp[col].shift(1)
        keyed = prior.groupby([df["team"], df["season"]], sort=False)
        for w in windows:
            df[f"{col}_r{w}"] = keyed.transform(lambda s, w=w: _trailing_mean_skipna(s, w))
        if also_season_to_date:
            df[f"{col}_std"] = keyed.expanding(min_periods=1).mean().reset_index(
                level=[0, 1], drop=True
            )
    return df.copy()  # de-fragment: dozens of piecemeal inserts above


# ------------------------------------------------- cold-start prior blending

# roster_continuity_score is deliberately excluded from blending. It is defined as
# an 8-game within-season lookback, so carrying a prior-season value would defeat
# its purpose. (Note: it was retired from the model feature set in Model_1 rev 3,
# so this exclusion is currently moot — kept because the rule still holds if it
# ever returns.)
NO_BLEND = {"roster_continuity_score"}
QB_PREFIX = "qb_"


def _final_window_mean(raw: pd.DataFrame, col: str, w: int,
                       key: str = "team") -> pd.DataFrame:
    """End-of-season form: mean of the last `w` observed values per (key, season).

    Everything here comes from a season that finished before the one being
    predicted, so it is available pre-kickoff and introduces no leakage.
    """
    d = raw[[key, "season", "week", col]].dropna(subset=[col, key])
    d = d.sort_values([key, "season", "week"])
    out = (d.groupby([key, "season"])[col]
             .apply(lambda s: s.tail(w).mean())
             .rename("prior_value").reset_index())
    out["season"] = out["season"] + 1  # attach to the season it will prime
    return out


def _prior_league_mean(raw: pd.DataFrame, col: str) -> pd.DataFrame:
    """League-wide mean of `col` over seasons strictly before each season.

    The fallback when no team/player prior exists. For 2021 (earliest loaded
    season) there is no prior season at all, so this is NaN and those rows keep
    their current-season value — flagged as league_average_fallback.
    """
    d = raw[["season", col]].dropna(subset=[col])
    per_season = d.groupby("season")[col].mean().sort_index()
    rows = []
    for s in per_season.index:
        earlier = per_season[per_season.index < s]
        rows.append({"season": int(s),
                     "league_value": float(earlier.mean()) if len(earlier) else np.nan})
    return pd.DataFrame(rows)


def _expected_starter(df: pd.DataFrame) -> pd.Series:
    """Who is expected to start, using only pre-kickoff information.

    Priority: the depth chart's QB1 for that team/season (published before the
    game), else the most recent prior starter. The current game's own
    `qb_player_id` is NOT used for upcoming games, where it does not exist.
    """
    prior_starter = df.groupby(["team", "season"], sort=False)["qb_player_id"].shift(1)
    # Week 1 has no in-season predecessor: fall back to last season's final starter.
    last_of_prev = (df.dropna(subset=["qb_player_id"])
                      .sort_values(["team", "season", "week"])
                      .groupby(["team", "season"])["qb_player_id"].last()
                      .rename("prev_final").reset_index())
    last_of_prev["season"] = last_of_prev["season"] + 1
    merged = df[["team", "season"]].merge(last_of_prev, on=["team", "season"], how="left")
    merged.index = df.index

    try:
        dc = pd.read_sql(
            "SELECT DISTINCT ON (season, team) season, team, player_id AS dc_qb1 "
            "FROM depth_charts WHERE position = 'QB' AND depth_rank = 1 "
            "AND player_id IS NOT NULL "
            "ORDER BY season, team, snapshot_dt DESC NULLS LAST, week DESC NULLS LAST",
            get_engine())
        dcm = df[["team", "season"]].merge(dc, on=["team", "season"], how="left")
        dcm.index = df.index
        dc_qb1 = dcm["dc_qb1"]
    except Exception:
        dc_qb1 = pd.Series(np.nan, index=df.index)

    return prior_starter.fillna(dc_qb1).fillna(merged["prev_final"])


def blend_prior_season(df: pd.DataFrame, raw: pd.DataFrame, source_cols: list[str],
                       windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Bayesian-style blend of early-season rolling values toward a prior.

        weight  = min(games_of_history / window, 1)
        blended = weight * current_season_rolling + (1 - weight) * prior

    The prior is the team's end-of-last-season rolling value. For QB columns it is
    carried at the PLAYER level, so a starter who finished last season and opens
    this one brings his own form rather than his team's; when the starter changed,
    it falls back to the prior-season league average for that stat.

    Weight reaches 1 once a team has `window` games, so mid- and late-season rows
    are mathematically unchanged — only weeks 1-4 move.

    Applied globally (training and prediction) on purpose: fitting coefficients on
    noisy 1-game averages and then serving blended values would mean the model
    under-uses a signal it was never trained to trust.
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    starter = _expected_starter(df)
    prior_available = pd.Series(False, index=df.index)
    league_available = pd.Series(False, index=df.index)

    for col in source_cols:
        if col in NO_BLEND or col not in raw.columns:
            continue
        is_qb = col.startswith(QB_PREFIX)
        league = _prior_league_mean(raw, col)
        lg = df[["season"]].merge(league, on="season", how="left")["league_value"]
        lg.index = df.index

        if is_qb:
            pri = _final_window_mean(raw, col, max(windows), key="qb_player_id")
            key = pd.DataFrame({"qb_player_id": starter, "season": df["season"]})
            pv = key.merge(pri, on=["qb_player_id", "season"], how="left")["prior_value"]
        else:
            pri = _final_window_mean(raw, col, max(windows), key="team")
            pv = df[["team", "season"]].merge(
                pri, on=["team", "season"], how="left")["prior_value"]
        pv.index = df.index
        prior_available |= pv.notna()
        league_available |= lg.notna()

        base = pv.fillna(lg)
        for w in windows:
            rc = f"{col}_r{w}"
            if rc not in df.columns:
                continue
            weight = (df["games_of_history"] / float(w)).clip(upper=1.0)
            blended = weight * df[rc].fillna(base) + (1.0 - weight) * base
            # Only touch rows short of a full window. `games_of_history` counts
            # GAMES, but a sparse column (NGS time-to-throw, say) can still be null
            # at full history — filling that would silently rewrite a mid-season
            # value the blend has no business touching.
            df[rc] = df[rc].where(weight >= 1.0, blended)

    # blend_weight is reported against the LONGEST window, the most conservative
    # reading: at 4 games the r4 features are fully current while r8 is still half
    # prior, so the row is genuinely only half self-supported.
    full = df["games_of_history"] >= max(windows)
    df["blend_weight"] = (df["games_of_history"] / float(max(windows))).clip(upper=1.0)
    df["history_source"] = np.where(
        full, "current_season",
        np.where(prior_available, "blended_prior_season",
                 np.where(league_available, "league_average_fallback",
                          "no_prior_available")))
    return df


def _add_qb_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """QB continuity, built from who started the PREVIOUS games only.

    `qb_player_id` for the current row is same-game information, so everything here
    is derived from the shifted series.
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    grp = df.groupby(["team", "season"], sort=False)

    prev_qb = grp["qb_player_id"].shift(1)
    prev_qb2 = grp["qb_player_id"].shift(2)

    # Whether last week's starter differed from the week before — an observed change
    # going into this game, using no current-game data.
    df["qb_changed_last_week"] = (
        (prev_qb.notna() & prev_qb2.notna() & (prev_qb != prev_qb2)).astype(int)
    )

    streak = []
    last_team_season = None
    current_qb, count = None, 0
    for row in df[["team", "season", "qb_player_id"]].itertuples(index=False):
        key = (row.team, row.season)
        if key != last_team_season:
            current_qb, count = None, 0
            last_team_season = key
        streak.append(count)  # appended BEFORE this game is counted
        if row.qb_player_id == current_qb and pd.notna(row.qb_player_id):
            count += 1
        else:
            current_qb, count = row.qb_player_id, 1
    df["qb_consecutive_starts_prior"] = streak
    df["qb_is_new_starter"] = (df["qb_consecutive_starts_prior"] == 0).astype(int)
    return df


def _add_rookie_flag(df: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_sql(
        "SELECT player_id, rookie_season FROM players WHERE player_id IS NOT NULL",
        get_engine(),
    )
    prior_qb = df.groupby(["team", "season"], sort=False)["qb_player_id"].shift(1)
    lookup = players.set_index("player_id")["rookie_season"]
    rookie = prior_qb.map(lookup)
    df["qb_is_rookie"] = (rookie == df["season"]).astype(int)
    return df


def _add_coach_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive team-games (not calendar weeks — byes aren't rows) with the
    current head coach, counted from his first appearance IN THIS DATASET.

    The dataset starts in 2021, so a coach already in place before then (Andy Reid
    at KC, for example) reads as tenure=0 at 2021 week 1 despite having coached
    since 2013. `coach_tenure_censored` flags every row where the current coach is
    still the one first observed for that team, so the left-censoring is visible
    rather than silently producing an understated tenure number.
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    first_coach = df.groupby("team")["coach"].transform("first")

    tenure = []
    cur_coach: dict[str, object] = {}
    cur_count: dict[str, int] = {}
    for team, coach in zip(df["team"], df["coach"]):
        if cur_coach.get(team) == coach and pd.notna(coach):
            cur_count[team] += 1
        else:
            cur_coach[team] = coach
            cur_count[team] = 1
        tenure.append(cur_count[team] - 1)  # weeks WITH this coach prior to this game

    df["coach_tenure_weeks"] = tenure
    df["coach_tenure_censored"] = (df["coach"] == first_coach).astype(int)
    return df


def _add_division_familiarity(df: pd.DataFrame) -> pd.DataFrame:
    """Prior meetings between this exact pair of teams, this season + last.

    Computed for every matchup, not only divisional ones — a nonzero count for a
    same-season non-divisional pair would indicate a scheduling-data bug (only
    division rivals meet twice within a season), which is a useful sanity check in
    the EDA driver. Cross-season repeats are legitimate for any pair via the normal
    scheduling rotation.
    """
    df = df.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    history: dict[tuple, list[tuple[int, int]]] = {}
    registered: set[tuple] = set()
    counts = []
    for row in df.itertuples(index=False):
        pair = tuple(sorted((row.team, row.opponent)))
        hist = history.get(pair, [])
        counts.append(sum(
            1 for (s, w) in hist
            if s in (row.season, row.season - 1) and (s < row.season or w < row.week)
        ))
        key = (pair, row.game_id)
        if key not in registered:
            registered.add(key)
            history.setdefault(pair, []).append((row.season, row.week))
    df["division_familiarity"] = counts
    return df


def _add_roster_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """Snap-share overlap of this week's top-N offensive players against the same
    team CONTINUITY_LOOKBACK games prior, within-season only.

    An offseason roster overhaul is real turnover, not noise to smooth over, so
    continuity is not compared across a season boundary (same convention as
    add_rolling's (team, season) grouping).
    """
    snaps = pd.read_sql(
        "SELECT game_id, team, pfr_player_id, offense_snaps FROM snap_counts "
        "WHERE pfr_player_id IS NOT NULL AND offense_snaps > 0",
        get_engine(),
    )
    top_sets = (
        snaps.sort_values("offense_snaps", ascending=False)
        .groupby(["game_id", "team"])["pfr_player_id"]
        .apply(lambda s: frozenset(s.head(CONTINUITY_TOP_N)))
    )

    df = df.sort_values(["team", "season", "week"]).copy()
    cur = pd.Series(list(zip(df["game_id"], df["team"])), index=df.index).map(top_sets)
    prior = cur.groupby([df["team"], df["season"]], sort=False).shift(CONTINUITY_LOOKBACK)

    def _overlap(c, p):
        if not isinstance(c, frozenset) or not isinstance(p, frozenset) or len(c) == 0:
            return np.nan
        return len(c & p) / len(c)

    df["roster_continuity_score"] = [_overlap(c, p) for c, p in zip(cur, prior)]
    return df


def _zscore(s: pd.Series, ref: pd.Series | None = None) -> pd.Series:
    """Standardize `s`, taking mean/std from `ref` when given.

    The reference matters: standardizing against the frame being transformed makes
    every row's value depend on which other rows happen to be present.
    """
    ref = s if ref is None else ref
    sd = ref.std()
    return (s - ref.mean()) / (sd if sd and np.isfinite(sd) else 1.0)


def _add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous interaction terms, replacing EDA_1's binned/terciled versions.

    Z-score statistics are taken from COMPLETED games only. Standardizing against
    the whole frame made these values shift retroactively whenever rows were added
    — caught by verify_forward_path() when an unplayed week changed 2,742 existing
    rows. The same flaw also meant the scaling absorbed test-set statistics. Both
    are fixed by pinning the reference to completed rows.

    (These three features were DROPped in EDA_2 and are not in the Model_1 feature
    set, so no published result depended on the old behaviour — but the transform
    had to be made order-independent regardless.)
    """
    pairs = {
        "rush_matchup_z": ("off_rush_rate_r4", "opp_def_rush_epa_allowed_r4"),
        "pass_matchup_z": ("off_pass_rate_r4", "opp_def_pass_epa_allowed_r4"),
        "mobility_pressure_z": ("qb_scramble_rate_r4", "opp_def_pressure_rate_r4"),
    }
    completed = df["won"].notna() if "won" in df.columns else pd.Series(True, index=df.index)
    for name, (a, b) in pairs.items():
        if a in df.columns and b in df.columns:
            df[name] = (_zscore(df[a], df.loc[completed, a])
                        * _zscore(df[b], df.loc[completed, b]))
    return df


# Same-game measurements that become features only after add_rolling().
ROLLING_SOURCE_COLS = [
    # Offense
    "off_epa_play", "off_pass_epa", "off_rush_epa", "off_success", "off_cpoe",
    "off_adot", "off_sack_rate", "off_3d_rate", "off_pass_rate", "off_rush_rate",
    "off_points_per_drive",
    # Defense
    "def_epa_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed",
    "def_success_allowed", "def_havoc_rate", "def_pressure_rate", "def_blitz_rate",
    "def_pressure_to_sack", "def_3d_stop_rate", "def_points_per_drive",
    # QB
    "qb_epa", "qb_cpoe", "qb_success", "qb_adot", "qb_sack_rate",
    "qb_bad_throw_rate", "qb_pressured_rate", "qb_scramble_rate",
    # EDA_2: drive efficiency
    "off_red_zone_td_rate", "off_yards_per_drive", "off_plays_per_drive",
    "off_explosive_play_rate", "off_penalty_yards_per_game",
    "def_red_zone_td_rate_allowed", "def_explosive_play_rate_allowed",
    "def_penalty_yards_per_game",
    # EDA_2: turnovers
    "off_turnover_rate", "def_takeaway_rate", "turnover_margin_rate",
    # EDA_2: special teams
    "st_epa", "off_avg_start_field_pos",
    # EDA_2: PFR depth
    "qb_time_to_throw", "off_drop_rate", "def_broken_tackle_rate_allowed",
]


def build_team_game_panel(upcoming: tuple[int, int] | None = None,
                          blend_prior: bool = False) -> pd.DataFrame:
    """Steps 1-3: raw stats -> lagged rolling features -> opponent-joined model frame.

    `upcoming=(season, week)` includes that week's unplayed games for prediction.
    `blend_prior` applies the cold-start prior blend (see blend_prior_season).
    DEFAULT OFF: it was built, verified leakage-safe, and measured — and it does not
    help. On the 2025 holdout it moved weeks 1-4 AUC by -0.0168 (bootstrap CI
    [-0.0388, +0.0047], worse in 93.9% of resamples). Enable it explicitly to
    reproduce that measurement or to re-test on more seasons.
    """
    raw = raw_team_game_stats(upcoming)
    panel = add_rolling(raw, ROLLING_SOURCE_COLS)
    if blend_prior:
        panel = blend_prior_season(panel, raw, ROLLING_SOURCE_COLS)
    else:
        panel["blend_weight"] = np.nan
        panel["history_source"] = "current_season"
    panel = _add_qb_continuity(panel)
    panel = _add_rookie_flag(panel)
    panel = _add_coach_tenure(panel)
    panel = _add_division_familiarity(panel)
    panel = _add_roster_continuity(panel)
    panel = _attach_opponent_features(panel)  # must precede interactions: they need opp_ cols
    panel = _add_interaction_terms(panel)
    return panel.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


# Features worth comparing against the opponent's equivalent.
DIFF_BASES = [
    "qb_epa_r4", "qb_epa_r8", "qb_cpoe_r4", "off_epa_play_r4", "off_epa_play_r8",
    "def_epa_allowed_r4", "def_epa_allowed_r8", "def_pressure_rate_r4",
    "def_havoc_rate_r4", "off_points_per_drive_r4", "def_points_per_drive_r4",
    # EDA_2: needed as opponent context for the continuous interaction terms below.
    "def_rush_epa_allowed_r4", "def_pass_epa_allowed_r4",
]


def _attach_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its opponent's lagged features and build the deltas.

    The headline feature — `qb_epa_r4_diff` — lives here.
    """
    cols = [c for c in DIFF_BASES if c in df.columns]
    opp = df[["game_id", "team"] + cols].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in cols}}
    )
    df = df.merge(opp, on=["game_id", "opponent"], how="left")
    for c in cols:
        df[f"{c}_diff"] = df[c] - df[f"opp_{c}"]
    return df


def assert_no_leakage(panel: pd.DataFrame, sample: int = 250, seed: int = 42) -> None:
    """Independently recompute rolling features and assert they match.

    Deliberately naive: for a sample of rows, filter the panel down to the same
    team and season with a STRICTLY earlier week, take the trailing window, and
    average it. A vectorised shift/roll that is off by one will disagree with this
    loop, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    checks = [c for c in (
        "qb_epa", "off_epa_play", "def_epa_allowed",
        "off_red_zone_td_rate", "turnover_margin_rate", "st_epa",
        "qb_time_to_throw", "off_drop_rate", "def_broken_tackle_rate_allowed",
    ) if c in panel]
    idx = rng.choice(panel.index, size=min(sample, len(panel)), replace=False)

    failures: list[str] = []
    for i in idx:
        row = panel.loc[i]
        prior = panel[
            (panel["team"] == row["team"])
            & (panel["season"] == row["season"])
            & (panel["week"] < row["week"])
        ].sort_values("week")
        for col in checks:
            for w in ROLLING_WINDOWS:
                got = row[f"{col}_r{w}"]
                vals = prior[col].dropna().tail(w)
                want = vals.mean() if len(vals) else np.nan
                if pd.isna(got) and pd.isna(want):
                    continue
                if pd.isna(got) != pd.isna(want) or not np.isclose(
                    got, want, rtol=1e-9, atol=1e-9, equal_nan=True
                ):
                    failures.append(
                        f"{row['team']} {row['season']}wk{row['week']} "
                        f"{col}_r{w}: got {got!r} want {want!r}"
                    )

    if failures:
        raise AssertionError(
            f"LEAKAGE CHECK FAILED ({len(failures)} mismatches):\n"
            + "\n".join(failures[:15])
        )

    # A feature must never be perfectly explained by the current game's own outcome.
    for col in ("qb_epa_r4", "off_epa_play_r4"):
        if col in panel:
            sub = panel[[col, "won"]].dropna()
            if len(sub) > 100 and abs(sub[col].corr(sub["won"])) > 0.5:
                raise AssertionError(
                    f"{col} correlates {sub[col].corr(sub['won']):.3f} with the "
                    "current result — suspiciously high for a lagged feature."
                )
    print(f"Leakage check PASSED on {len(idx)} sampled rows x {len(checks)} features.")


def verify_blend(sample: int = 200, seed: int = 42) -> None:
    """The prior blend may only look BACKWARD, into already-completed seasons.

    assert_no_leakage checks add_rolling's within-season property and is run
    against an unblended panel, because the blend intentionally changes week-1
    values from NaN to a prior-season estimate — that is the feature, not a leak.
    This function checks the property the blend itself must satisfy.

    Two assertions:
      1. Rows with a full window (weight = 1) are byte-identical blended or not —
         the blend must not touch mid/late season at all.
      2. A row's blended value depends only on seasons STRICTLY EARLIER than its
         own: rebuilding with every later season deleted must not change it.
    """
    rng = np.random.default_rng(seed)
    blended = build_team_game_panel(blend_prior=True)
    plain = build_team_game_panel(blend_prior=False)

    key = ["game_id", "team"]
    b = blended.set_index(key).sort_index()
    p = plain.set_index(key).sort_index()
    # Own-team rolling columns only. `opp_*` and `*_diff` are derived from the
    # OPPONENT's row, so a full-window team facing a week-1 opponent legitimately
    # sees a blended opponent value — that is correct, not a violation.
    cols = [c for c in b.columns
            if c.endswith(("_r4", "_r8")) and c in p.columns
            and not c.startswith("opp_") and not c.endswith("_diff")
            and pd.api.types.is_numeric_dtype(b[c])]

    full = b["games_of_history"] >= max(ROLLING_WINDOWS)
    bad = [c for c in cols
           if not np.allclose(b.loc[full, c].to_numpy(dtype=float),
                              p.loc[full, c].to_numpy(dtype=float),
                              equal_nan=True, rtol=1e-12)]
    if bad:
        raise AssertionError(
            f"BLEND TOUCHED FULL-WINDOW ROWS (it must not): {bad[:8]}")
    print(f"PASS: {len(cols)} rolling columns unchanged on "
          f"{int(full.sum()):,} full-window rows.")

    # Property 2: no forward dependence across seasons.
    seasons = sorted(s for s in blended["season"].unique() if s > 2021)
    target = int(rng.choice(seasons))
    raw_all = raw_team_game_stats()
    raw_cut = raw_all[raw_all["season"] <= target]
    cut = blend_prior_season(add_rolling(raw_cut, ROLLING_SOURCE_COLS),
                             raw_cut, ROLLING_SOURCE_COLS)
    cut = cut[cut["season"] == target].set_index(key).sort_index()
    ref = blended[blended["season"] == target].set_index(key).sort_index()
    common = [c for c in cols if c in cut.columns]
    bad2 = [c for c in common
            if not np.allclose(ref.loc[cut.index, c].to_numpy(dtype=float),
                               cut[c].to_numpy(dtype=float),
                               equal_nan=True, rtol=1e-9)]
    if bad2:
        raise AssertionError(
            f"BLEND USES FUTURE SEASONS: season {target} changed when later "
            f"seasons were removed: {bad2[:8]}")
    print(f"PASS: season {target} blended values identical with all later seasons "
          f"deleted ({len(common)} columns, {len(cut):,} rows).")


def verify_forward_path(season: int, week: int) -> None:
    """Adding an UNPLAYED week must not change any already-computed feature.

    The historical guard (assert_no_leakage) only covers completed games. The live
    path is new: it pulls rows for a game that has not happened. This asserts the
    two properties that matter — every pre-existing row is byte-identical, and the
    upcoming rows carry no outcome.
    """
    base = build_team_game_panel()
    fwd = build_team_game_panel(upcoming=(season, week))

    up = fwd[fwd["is_upcoming"]]
    assert len(up) > 0, f"no upcoming rows found for {season} week {week}"
    assert up["won"].isna().all(), "upcoming rows must have no outcome"
    print(f"upcoming rows: {len(up)} for {season} wk{week}, all with won=NaN")

    key = ["game_id", "team"]
    b = base.set_index(key).sort_index()
    f = fwd[~fwd["is_upcoming"]].set_index(key).sort_index()
    assert len(b) == len(f), f"row count changed: {len(b)} -> {len(f)}"
    assert (b.index == f.index).all(), "row identity changed"

    cols = [c for c in b.columns
            if c in f.columns and pd.api.types.is_numeric_dtype(b[c])]
    bad = []
    for c in cols:
        if not np.allclose(b[c].to_numpy(dtype=float),
                           f[c].to_numpy(dtype=float), equal_nan=True, rtol=1e-12):
            bad.append(c)
    if bad:
        raise AssertionError(
            f"FORWARD PATH LEAK: {len(bad)} feature(s) changed when the upcoming "
            f"week was added: {bad[:10]}")
    print(f"PASS: all {len(cols)} numeric features identical across "
          f"{len(b):,} historical rows when the upcoming week is added.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-leakage", action="store_true")
    ap.add_argument("--verify-blend", action="store_true")
    ap.add_argument("--verify-forward", nargs=2, type=int, metavar=("SEASON", "WEEK"))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.verify_forward:
        verify_forward_path(*args.verify_forward)
        return 0
    if args.verify_blend:
        verify_blend()
        return 0

    # Unblended: this guard verifies add_rolling's within-season property. The
    # prior blend legitimately replaces week-1 NaNs with prior-season values, which
    # would read as a mismatch here; it is verified separately by --verify-blend.
    panel = build_team_game_panel(blend_prior=not args.verify_leakage)
    print(f"panel: {panel.shape[0]:,} team-game rows x {panel.shape[1]} columns")
    print(f"win rate: {panel['won'].mean():.4f}  seasons: "
          f"{panel['season'].min()}-{panel['season'].max()}")

    if args.summary:
        print(f"\nhome win rate: {panel.loc[panel['is_home'], 'won'].mean():.4f}")
        fav = panel[panel["team_spread"].notna() & (panel["team_spread"] != 0)]
        print("favourite win rate: "
              f"{fav.loc[fav['team_spread'] > 0, 'won'].mean():.4f}")
        print(f"rows with no prior games: {(panel['games_of_history'] == 0).sum()}")

    if args.verify_leakage:
        assert_no_leakage(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
