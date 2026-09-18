# CLAUDE.md — NFLProject

Goal: predict NFL games and find edges against the betting market, using five-plus
seasons of NFL statistics. Data foundation, two EDA rounds, two modeling rounds, and
a full betting pipeline are complete. Two headline results, both negative and both
load-bearing: **no model beats the closing spread**, and **the edge filter built on
top of it loses money (11-38, -$90 on 2025)**. Read the Model_2 and Model_1 sections
before proposing new features, model families, or betting logic.

## Non-negotiables

- **PostgreSQL, not SQLite.** (The sibling MLB/Stock projects use SQLite; this one
  intentionally does not.)
- **Data source is nflverse via `nflreadpy`.** Do not add a Pro-Football-Reference
  scraper — PFR is already an nflverse upstream, and `pfr_adv_*` exposes PFR's
  advanced charting directly. PFR is for manual one-off lookups only.
- **Do not switch to `nfl_data_py`.** It pins `pandas<2.0` / `numpy<2.0`, which
  would block modern scikit-learn/xgboost in the modeling phase.
- **Never deploy or push without an explicit go-ahead.**

## Schema rules

Two table classes, and the distinction matters:

1. **Backbone** — `teams`, `players`, `games`, `betting_lines`, `ingest_log`.
   Declared explicitly in `src/schema.sql` with real types and FKs. Loaded with
   `upsert_frame`. Adding a column here is a deliberate edit to `schema.sql`.

2. **Statistical** — everything else. **Not** declared in `schema.sql`. Created
   directly from the source frame by `load_frame`, which preserves every column
   nflverse ships and auto-widens the table when a future release adds columns.

New feature categories become **new tables** keyed on `season` / `week` / `game_id` /
`team` / `player_id`. Do not bolt columns onto an existing stat table.

## Loader invariants (`src/db.py`)

- `load_frame` is idempotent per season: it deletes rows for the seasons present in
  the frame, then COPYs. Re-running a backfill never duplicates.
  With no `season` column it TRUNCATEs and fully reloads.
- Both loaders coerce the frame to the **target table's existing Postgres types**
  before COPY. This is load-bearing: a column that is int64 in a one-season frame
  becomes float64 in a multi-season frame once it picks up NaNs, and COPY rejects
  `"2000.0"` for an integer column.
- Because a stat table's types are fixed by whichever frame created it, prefer
  building tables from the **full** season range. If a table was first created by a
  single-season smoke test, drop and rebuild it rather than loading over it.
- Index specs are declared optimistically and skipped when a column is absent
  (season summaries have no `team` column, for example).
- Bulk insert is `COPY`, not `to_sql` — `to_sql` is far too slow for the
  372-column, ~50k-rows-per-season `plays` table.

## Model_2 / betting pipeline findings (2026-09-18) — READ BEFORE BETTING ANYTHING

Full evidence: `Modeling/Model_2/MODEL_2_FINDINGS.md`. Pipeline lives in `Betting/`.

- **THE EDGE FILTER LOSES MONEY. Do not bet it.** 2025 holdout: 49–57 qualifying
  picks (two independent runs), record **11–38**, 22–25% accuracy, **−$90.06** at
  flat $10. Mean claimed edge on those picks was +0.10.
- **The picks are anti-signal, not marginal.** Root cause: the model is
  **under-dispersed** — its probabilities are flatter than the market's. Decile
  calibration shows it says 0.219 where reality is 0.145, and 0.779 where reality is
  0.873. A one-sided edge filter then reads that flatness as edge on every heavy
  underdog. Picks averaged model 0.366 / market 0.263 / **actual 0.250**.
- **CORRECTS Model_1's "no calibration needed" conclusion.** That was based on mean
  predicted probability matching the base rate, which says nothing about the tails.
  The mean can match while every decile is wrong — and here it is. Calibration
  doesn't matter for AUC; it matters enormously for an edge engine.
- **Calibration cannot rescue it — it makes it worse.** Platt is monotonic so it
  can't fix a shape problem (59 picks, 23.7%). Isotonic flips selection to heavy
  favorites (75 picks, 53.3% accuracy, avg moneyline −246) and loses **−$160.33,
  ROI −21.4%** — nearly double the uncalibrated loss, because break-even at −246
  needs ~71%. Higher accuracy, worse P&L: accuracy and profitability are different
  questions. The binding constraint is that the model ranks *worse* than the market
  (AUC 0.703 vs 0.720); no transform of a worse ranking beats a better one.
- **Bagging did not help** — marginally worse on every metric than a single fit
  (AUC 0.7029 vs 0.7060), and market agreement *fell* to 97.42% (from 99.45%). On
  the 14 games where the bagged model disagreed with the market it was right
  **42.9%** of the time. More disagreement is noise, not edge.
- **What IS sound**: block-bootstrap machinery (72 season-week blocks; season-level
  rejected at only 4 blocks / 31.6% drop rate), convergence (CI width change 4.89%
  → 0.28% by B=500), per-game CIs (mean width 0.1563), Shin de-vig, and the whole
  weekly/tracking pipeline. Reusable regardless of model quality.
- `team_spread` is top-5 in **100% of 500 resamples** (mean importance 0.585).
  `qb_cpoe_r4_diff` is stable (73.6% top-5), just small.

## Cold-start prior blend (2026-09-18) — BUILT, MEASURED, DEFAULT OFF

`blend_prior_season()` in `src/features.py`. Blends early-season rolling features
toward a prior: `weight = min(games_of_history/window, 1)`,
`blended = weight*current + (1-weight)*prior`. Team-level prior = end-of-last-season
rolling value; QB columns carry at the PLAYER level via the expected starter (depth
chart QB1 / last known starter), falling back to prior-season league average.

- **It does not help. `build_team_game_panel(blend_prior=False)` is the default.**
  On the 2025 holdout it moved weeks 1-4 AUC by **−0.0168** (bootstrap CI
  [−0.0388, +0.0047]; worse in **93.9%** of resamples). No evidence of benefit, mild
  evidence of harm. Do not re-enable without new evidence.
- **Why it fails:** the cold start is an *information* problem, not a
  feature-availability problem. Early-season predictions already lean on
  `team_spread` (64.7% of model importance, fully available from week 1), and NFL
  roster turnover makes a prior-season team value genuinely noisy — which is exactly
  what EDA_1 argued when it first rejected carryover. That argument is now measured
  rather than assumed.
- Weeks 1-4 are *already* the easier subset to predict (holdout AUC 0.7986 vs 0.6847
  for weeks 5+), because the market is sharper early. There was less to fix than the
  "insufficient history" label suggested.
- `roster_continuity_score` is in `NO_BLEND` — though it was retired from the feature
  set in Model_1 rev 3, so the exclusion is currently moot.
- **Guard: `python -m src.features --verify-blend`.** Asserts (1) full-window rows
  are byte-identical blended or not, and (2) a season's blended values are unchanged
  when every later season is deleted. It caught two real bugs: a sparse column
  (`qb_time_to_throw`) being back-filled at full history, and the opponent-derived
  `opp_*`/`*_diff` columns being wrongly held to the row's own window.
- `--verify-leakage` deliberately runs on the UNBLENDED panel: the blend replaces
  week-1 NaNs with prior-season values by design, which the rolling guard would read
  as a mismatch. The blend has its own guard above.
- Output carries `history_source` (current_season / blended_prior_season /
  league_average_fallback / no_prior_available) and `blend_weight`, and
  `widen_ci_for_blend()` inflates the interval by `1 + (1 - weight)` so a blended row
  must clear a higher bar. Those stay useful for flagging even with the blend off.

## Betting pipeline invariants (`Betting/`, `src/odds.py`, `src/bootstrap.py`)

- **Never compare against a raw implied probability** — always de-vig first.
  Skipping it inflates favorite probability by ~1.9 points, counted as edge on every
  bet. Default method is Shin.
- **The edge filter is ONE-SIDED on purpose**: a pick needs `ci_lo > market_prob`.
  A two-sided test flagged both sides of the same game and staked a negative-EV bet.
- **`MIN_HISTORY_FOR_PICK = 4`** is a validity gate, not a tightening. With 1 game
  of in-season history the model produced 4 "qualifying" picks off noise.
- **Log every game, not just picks** (`predictions` table). A picks-only log cannot
  measure calibration.
- **ROI is suppressed below 50 picks** in `track.py`. Do not remove that.
- **CLV columns exist but are NULL by design** — nflverse publishes one untimestamped
  line per game, so line-at-pick-time is unrecoverable. Needs a capture job.
- Commands: `Betting/show.py` (view), `weekly_picks.py` (generate),
  `grade_week.py` (grade), `track.py` (cumulative metrics).

## Modeling policy (set after the Model_1 review — follow these)

- **The reduced/cluster-representative set is THE default feature set**, not an
  alternative. Collinearity diagnostics run first and define it. The full 29/30 set
  exists only as a reference row. Reduction costs <=0.009 AUC (inside noise) for 6
  fewer features.
- **Simple average is the production ensemble. Stacking is a comparison row only** —
  the average beat stacking in both regimes across two independent runs.
- **Calibrate at final output only, never as a selection criterion, and use Platt
  (`sigmoid`), never isotonic** — the calibration slice here is ~250-300 rows, far
  too small for a non-parametric fit. Judge calibration on Brier / ECE / log-loss;
  an AUC delta is a category error (Platt is monotonic and cannot change ranking —
  verified, AUC identical to 4 dp).
  **SUPERSEDED by Model_2:** Model_1 concluded "no calibration needed" because mean
  predicted probability matched the base rate. That test was insufficient — the mean
  can match while every decile is wrong, and it is (see the Model_2 section above).
  For AUC-only work, uncalibrated is still fine. For anything comparing probability
  LEVELS against a market, the model is under-dispersed and neither Platt nor
  isotonic fixes it.
- **Borderline features need 3-of-3 gates to count as confirmed**: BH-corrected
  residual screen + positive incremental AUC + significant permutation test.
  2-of-3 is NOT enough. Anything short ships labeled UNCONFIRMED in every report.
- **ONE full-season model, no week-based split** (see the rev-3 correction below).
  Regularization: `max_depth` 2-4, `min_samples_leaf` 50-80, `max_features` <=0.7,
  RF `max_samples` <=0.9. Worst overfit gap is 0.051 with all 2,168 training rows.
  Sklearn name mapping (the XGBoost names do not exist here): `subsample` ->
  `max_samples` (RF only), `colsample_bytree` -> `max_features`,
  `min_child_weight` -> `min_samples_leaf`.
  **`HistGradientBoostingClassifier` has NO row-subsampling parameter**; if GBM
  overfitting ever needs pushing below ~0.05, that requires XGBoost/LightGBM.

## Model_1 findings (2026-09-17) — the modeling question is answered

Full evidence: `Modeling/Model_1/MODEL_1_FINDINGS.md`. Rebuild with
`python Modeling/model_1.py --all`.

- **No model beats the closing line.** Confirmed across three revisions. Current
  (rev 3, single full-season model): **0 of 10 configurations** beat the 0.7187
  benchmark; production (simple average, 22 features) scores 0.7007, i.e. −0.0180.
  Do not propose "try another model family" or "tune harder" — three families, two
  ensembling schemes, calibration, feature reduction, and an architecture change all
  landed in the same place.
- **Revision 2's "1 of 20 beat the benchmark" was a false positive produced by the
  split.** Removing the split removed it. Fragmenting 2,168 rows into two models
  generated apparent edge where none existed — a cautionary example worth
  remembering before adding any future subgroup model.
- **The market also wins on calibration**, by roughly 2x: ECE 0.0348 vs 0.0691,
  Brier 0.2123 vs 0.2188, log-loss 0.6094 vs 0.6261.
- **At a 0.5 threshold the models make essentially the same picks as the market.**
  In the rev-2 late-season run every model scored accuracy of exactly 0.609272 —
  byte-identical confusion matrices to always picking the closing favorite.
- **CORRECTED (rev 3): there is NO significant early/late regime gap.** Revisions 1–2
  claimed the market was much sharper in weeks 1–8 (0.7465) than 9–18 (0.7041) and
  called it the project's most actionable finding. **That was wrong** — it was never
  significance-tested. Bootstrapping gives a 95% CI of **[−0.042, +0.127]**, straddling
  zero (16.6% of samples show late as sharper). Do not plan work around a regime
  difference; there isn't one.
- **The EARLY/LATE split is REMOVED.** It cost training data (978/1,190 vs 2,168),
  raised the worst overfit gap from 0.051 to 0.123, and manufactured the single
  false-positive "win" that revision 2 reported. A single full-season model matches
  or beats it on AUC / accuracy / log-loss / Brier.
- **`roster_continuity_score` is RETIRED.** It was null weeks 1–8 so it only lived
  inside the LATE model; the split was its only home. It also failed the 3-of-3 gate
  (2/3) with an unstable permutation p-value (0.11 → 0.047 across specs).
- **`team_spread` dominates every model's importance**: 50-61% of Random Forest
  feature importance, and in one L1 fit 4 of 7 non-zero coefficients were
  `team_spread` and its threshold encodings. Two model families independently
  concluded the closing line is nearly the only thing that matters.
- **Stacking never beat the simple average** across three independent runs.
- **CV lesson worth keeping**: the holdout number alone will not reveal overfitting.
  Always report the train-vs-validation gap; rev-1 trees looked fine on holdout while
  carrying gaps of 0.14-0.18.

## Modeling invariants (src/model.py)

- **Train/test split is time-based**: train 2021–2024, test 2025, touched once.
- **CV is walk-forward expanding**, inside training years only: 2021→2022,
  2021-22→2023, 2021-23→2024. Never random k-fold.
- `python -m src.model --verify-split` asserts all of it (folds strictly forward in
  time, no fold validates on a training season, 2025 absent from every fold).
- **`ThresholdEncoder` must stay inside the sklearn Pipeline.** Its breakpoints are
  learned from the target, so fitting it outside the Pipeline leaks the holdout —
  verified: thresholds do change when holdout rows are present.
  `python -m src.model --verify-thresholds` demonstrates this.
- Watch for the duplicate-column trap when a loop variable can equal `team_spread`:
  `df[[f, "team_spread", "won"]]` yields a 2-column DataFrame for `df[f]` and raises
  "truth value of a Series is ambiguous". Use `dict.fromkeys` to dedupe. This bit
  twice.

## EDA_2 findings (2026-09-17) — read before proposing features

Full evidence: `EDA/EDA_2/EDA_2_FINDINGS.md`. Rebuild with `python EDA/eda_2.py --all`.
Tested 45 features (drive efficiency, PFR depth, turnovers, special teams,
continuity/coaching, 3 continuous interactions) — same gate as EDA_1.

- **Same headline as EDA_1, with one exception: `roster_continuity_score` is the
  only feature across 100 tested (both rounds combined) to improve 2025 holdout AUC
  over spread-alone** (0.7242 vs 0.7187, +0.0054). It does NOT survive
  Benjamini-Hochberg correction (nominal p=0.040 across 45 tests) — treat as a
  **lead worth a second season of replication, not a confirmed finding.** Don't
  build a model on it yet; don't dismiss it either.
- Otherwise 0/45 survive BH and 14/15 tested candidates lower holdout AUC — same
  pattern as EDA_1.
- **Found and fixed a real bug in `add_rolling`** while building this round:
  `Series.rolling(w).mean()` opens a fixed window of the last `w` positional rows
  and silently shrinks the effective sample when one is NaN, instead of reaching
  back further. Fixed via `_trailing_mean_skipna`. This matters for any sparse
  rate column (red-zone rate, anything that can be 0/0 for a game) — see the
  dedicated section below before touching `add_rolling` again.
- **The 8-game window beat the 4-game window in every case tested this round**
  (5 features) — now 6-for-6 combined with EDA_1's QB EPA finding. Treat "prefer
  r8" as a standing default, not a per-feature coincidence.
- Continuous interactions (properly z-scored, not binned) confirm EDA_1's null:
  all three underperform BOTH of their own main-effect components. Stronger
  evidence than EDA_1's binned test alone — don't re-propose these interactions.
- `division_familiarity` is AUC 0.500 by construction (symmetric across both rows
  of a matchup) — same trap as EDA_1's symmetric-feature lesson, now confirmed on
  a second feature.
- `coach_tenure_weeks`/`coach_tenure_censored` are confounded with team quality
  (established coaches simply win more) — weak AUC (~0.54) is not evidence of a
  tenure effect net of talent.
- Data gaps carried forward unresolved: no opening lines (`betting_lines.captured_at`
  is 0% populated, single source only), no market-side timestamp to pair with
  injury `date_modified`, no public-betting-percentage source anywhere. All three
  point toward the same next step EDA_1 already identified: line movement and
  market timing, not more box-score derivatives.

## EDA_1 findings (2026-09-17) — read before proposing features

Full evidence: `EDA/EDA_1/EDA_1_FINDINGS.md`. Rebuild with `python EDA/eda_1.py --all`.

- **Nothing in the box-score feature set beats the closing line.** 0 of 55 features
  survive Benjamini-Hochberg once the spread is partialled out, and every one of them
  *lowers* 2025 holdout AUC when added to a spread-only model (spread alone = 0.7187).
  Do not propose "add more rolling EPA features" as a path to edge — that was tested.
- Benchmarks: overall win rate 0.5000 (panel sanity check), home 0.5410,
  **closing favorite 0.6664**. A 60%-accurate win model is worse than the market.
- **Symmetric features have AUC exactly 0.500 by construction** in a team-game panel.
  `div_game`, `is_dome`, `wind`, `total_line` share one value across both rows of a
  game, so they can never predict the winner. They are only usable as interactions
  with an asymmetric feature. A symmetric feature scoring != 0.500 means a panel bug.
- **Differential features beat absolute ones**: `qb_epa_r4_diff` AUC 0.613 vs
  `qb_epa_r4` 0.583. Always compare against the opponent.
- **The 8-game window beats the 4-game window** for QB EPA (0.636 vs 0.613). QB play
  is noisy; longer windows estimate true quality better.
- **Biggest single effect in the data: a QB listed Out/Doubtful costs ~17pp of win
  probability** (0.346 vs 0.513). The `qb_is_new_starter` flag, by contrast, shows
  nothing — use the injury report, not the starter-change flag.
- **Complete-case analysis structurally drops new-starter rows** (no prior starts ->
  null rolling features). Evaluate churn features on the full panel, never on the
  complete-case matrix, or the effect is invisible by construction.
- `team_spread` and `team_ml_implied_prob` are near-perfectly correlated — use one.
- `travel_miles` / `tz_shift` are proxies for `is_home`; both tested interactions
  (mobile QB x pressure, run-heavy x weak run D) showed no coherent signal.

## Known source quirks

- **`depth_charts` changed format in 2025.** 2021-2024 are weekly rows
  (`season`/`week`/`club_code`/`depth_team`); 2025+ are dated snapshots
  (`dt`/`team`/`pos_rank`, several per week, ~15x the row count).
  `src/ingest/depth_charts.py` normalizes both into one table with `depth_rank`,
  `week` (null in snapshot era), `snapshot_dt` (null in weekly era) and
  `source_format`. It declares no unique key because the grain differs by era.
  Downsampling the snapshot era to weekly is a feature-engineering decision, not
  done at ingest.
- `seasonal_player_stats` has no `team` column — a season summary can span teams.
- Never run two backfills concurrently. They write the same tables and the
  season-scoped DELETE of one will race the COPY of the other.

## Player identity

`players.player_id` is the nflverse `gsis_id` and is the join key for
`weekly_player_stats`, `weekly_rosters`, `depth_charts`, `injuries`, `ngs_*` and
play-by-play. PFR-sourced tables (`snap_counts`, `pfr_adv_*`) key on
`pfr_player_id` instead — join through `players.pfr_id`.

Roster churn and backups are answerable from three tables: `weekly_rosters` (team +
status per week), `depth_charts` (`pos_rank`, 1 = starter), `snap_counts` (actual
playing time). `weekly_player_stats` carries `team` per row so production follows
the player while staying attributable to a team.

## Local environment

Docker is **not** installed on this machine. `docker-compose.yml` exists for later,
but Postgres currently runs as a project-local Homebrew cluster:

```bash
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
pg_ctl -D pgdata -o "-p 5433 -k /tmp" -l pgdata/server.log start
pg_ctl -D pgdata stop
```

`pgdata/` is gitignored. DB is `postgresql://nfl@localhost:5433/nfl` (trust auth,
local only). Python is `./venv/bin/python` (venv + pip + requirements.txt, matching
the sibling projects).

## Rolling-window semantics: reach back over nulls, don't shrink the window

`add_rolling`'s `_r{w}` columns are "last `w` games **that had a value**," not "last
`w` calendar rows with nulls averaged out." Plain `Series.rolling(w).mean()` does
the latter: it opens a fixed window of the last `w` positional rows and silently
shrinks the effective sample when one of them is NaN, instead of reaching one game
further back to compensate. For EPA-style columns (defined on every play) this
never mattered; it was a real, silently-wrong discrepancy for sparse rate columns
added in EDA_2 (red-zone rate is undefined on a game with zero red-zone trips,
NGS time-to-throw has coverage gaps). Found by `assert_no_leakage`'s independent
naive recomputation disagreeing with production on `off_red_zone_td_rate` and
`qb_time_to_throw` — fixed via `_trailing_mean_skipna` (dropna, roll, reindex +
ffill). `expanding()` (the `_std` season-to-date columns) already had correct
NaN-skipping semantics and needed no fix. If you ever touch `add_rolling` again,
re-run `--verify-leakage` with the sparse columns in its `checks` list — an EPA-only
check list would not have caught this.

## The leakage rule (src/features.py)

Every in-game stat becomes a predictor only as a lagged rolling value, always
`groupby(team, season).shift(1).rolling(w)` — shift first, then roll. Grouping by
season stops form bleeding across the offseason. Everything routes through
`add_rolling()`; do not write one-off feature code that skips it.

Pre-game information (injury reports, rest, travel, the closing line) is known
before kickoff and is deliberately *not* lagged — see `PREGAME_COLUMNS`.

`python -m src.features --verify-leakage` is the gate. It independently recomputes
rolling values with a naive loop and asserts they match. It has been verified to
catch both an injected same-game leak and a missing-shift off-by-one; if you change
the rolling code, re-confirm it still fails on a deliberate break.

## Adding a source

Add `src/ingest/<name>.py` exposing `run(seasons) -> int` that calls the nflreadpy
loader, converts with `to_pandas`, renames the player id column to `player_id` where
the source calls it `gsis_id`, and calls `load_frame`. Then register it in
`PIPELINE` in `src/ingest/__init__.py` in dependency order, and add the table to the
row-count list in `scripts/backfill_all.py`.
