"""EDA_2 — 30 candidate stats against the market gate.

Same premise as EDA_1: raw predictiveness is not the bar. A feature only matters if
it survives residualization against the closing spread (Benjamini-Hochberg
corrected) and improves 2025 holdout AUC over spread-alone (0.7187). This round
covers drive efficiency, PFR charting depth, turnover margin, special teams,
continuity/coaching, and three continuous interactions (replacing EDA_1's binned
versions). Three items (28-30, market microstructure) are documented as data gaps,
not built — see EDA_2_FINDINGS.md.

All output lands in EDA/EDA_2/: one PNG per figure, a CSV of the numbers behind it
in data/, and the section's stdout in logs/.

Usage:
    python EDA/eda_2.py --list
    python EDA/eda_2.py --sections 01,08
    python EDA/eda_2.py --all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from EDA._eda_common import (  # noqa: E402
    DIVERGING_MID, DIVERGING_NEG, DIVERGING_POS, INK_MUTED, INK_SECONDARY, LOSS_COLOR,
    SERIES, WIN_COLOR, all_sections, apply_style, auc, bh_screen, binned_rate,
    despine, new_fig, run_sections, save_table, savefig, section, set_output_dir,
)

set_output_dir("EDA_2")
from src.features import build_team_game_panel  # noqa: E402

TEST_SEASON = 2025
MARKET_ONLY_HOLDOUT_AUC = 0.7187  # from EDA_1 sec 10 — the number every candidate must beat

DRIVE_FEATURES = [
    "off_3d_rate_r4", "off_3d_rate_r8",
    "off_red_zone_td_rate_r4", "off_red_zone_td_rate_r8",
    "off_yards_per_drive_r4", "off_yards_per_drive_r8",
    "off_plays_per_drive_r4", "off_plays_per_drive_r8",
    "off_explosive_play_rate_r4", "off_explosive_play_rate_r8",
    "off_penalty_yards_per_game_r4", "off_penalty_yards_per_game_r8",
    "def_3d_stop_rate_r4", "def_3d_stop_rate_r8",
    "def_red_zone_td_rate_allowed_r4", "def_red_zone_td_rate_allowed_r8",
    "def_explosive_play_rate_allowed_r4", "def_explosive_play_rate_allowed_r8",
    "def_penalty_yards_per_game_r4", "def_penalty_yards_per_game_r8",
]
TURNOVER_ST_FEATURES = [
    "off_turnover_rate_r4", "off_turnover_rate_r8",
    "def_takeaway_rate_r4", "def_takeaway_rate_r8",
    "turnover_margin_rate_r4", "turnover_margin_rate_r8",
    "st_epa_r4", "st_epa_r8",
    "off_avg_start_field_pos_r4", "off_avg_start_field_pos_r8",
]
PFR_FEATURES = [
    "qb_bad_throw_rate_r4", "qb_bad_throw_rate_r8",
    "qb_time_to_throw_r4", "qb_time_to_throw_r8",
    "off_drop_rate_r4", "off_drop_rate_r8",
    "def_broken_tackle_rate_allowed_r4", "def_broken_tackle_rate_allowed_r8",
]
CONTINUITY_FEATURES = [
    "roster_continuity_score", "coach_tenure_weeks", "coach_tenure_censored",
    "division_familiarity",
]
INTERACTION_FEATURES = ["rush_matchup_z", "pass_matchup_z", "mobility_pressure_z"]

ALL_FEATURES = (
    DRIVE_FEATURES + TURNOVER_ST_FEATURES + PFR_FEATURES + CONTINUITY_FEATURES
    + INTERACTION_FEATURES
)

LOWER_IS_BETTER = {
    "def_red_zone_td_rate_allowed_r4", "def_red_zone_td_rate_allowed_r8",
    "def_explosive_play_rate_allowed_r4", "def_explosive_play_rate_allowed_r8",
    "def_penalty_yards_per_game_r4", "def_penalty_yards_per_game_r8",
    "off_penalty_yards_per_game_r4", "off_penalty_yards_per_game_r8",
    "off_turnover_rate_r4", "off_turnover_rate_r8",
    "off_avg_start_field_pos_r4", "off_avg_start_field_pos_r8",  # yardline_100: lower = better field position
    "qb_bad_throw_rate_r4", "qb_bad_throw_rate_r8",
    "def_broken_tackle_rate_allowed_r4", "def_broken_tackle_rate_allowed_r8",
    "off_drop_rate_r4", "off_drop_rate_r8",
}

# Interaction term -> (own-team component, opponent component)
INTERACTION_COMPONENTS = {
    "rush_matchup_z": ("off_rush_rate_r4", "opp_def_rush_epa_allowed_r4"),
    "pass_matchup_z": ("off_pass_rate_r4", "opp_def_pass_epa_allowed_r4"),
    "mobility_pressure_z": ("qb_scramble_rate_r4", "opp_def_pressure_rate_r4"),
}


@dataclass
class Ctx:
    panel: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return [f for f in ALL_FEATURES if f in self.panel.columns]


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])


def _bar_by_sign(ax, labels, values):
    colors = [DIVERGING_POS if v >= 0 else DIVERGING_NEG for v in values]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.invert_yaxis()


def _panel_grid(ctx: Ctx, cols: list[str], title: str, name: str, ncols: int = 4):
    p = ctx.panel
    cols = [c for c in cols if c in p.columns]
    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, col in zip(axes, cols):
        despine(ax)
        g = binned_rate(p, col, bins=6)
        if g.empty:
            ax.set_visible(False)
            continue
        ax.errorbar(g["center"], g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
                    fmt="o-", color=SERIES[0], ecolor=INK_MUTED, elinewidth=1, capsize=2,
                    markersize=5)
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
        a = auc(p[col], p["won"])
        note = " (lo better)" if col in LOWER_IS_BETTER else ""
        ax.set_title(f"{col}{note}\nAUC {a:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        rows.append({"feature": col, "auc": a, "lower_is_better": col in LOWER_IS_BETTER,
                     "n": int(p[col].notna().sum())})
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    savefig(fig, name)
    tbl = pd.DataFrame(rows).sort_values("auc")
    save_table(tbl, name)
    print(tbl.to_string(index=False))
    return tbl


# ----------------------------------------------------------------- sections


@section("01", "Univariate — all EDA_2 candidates")
def s01(ctx: Ctx):
    p = ctx.panel
    rows = []
    for col in ctx.features:
        d = p[[col, "won"]].dropna()
        if len(d) < 100 or d[col].std() == 0:
            continue
        w = d.loc[d["won"] == 1, col]
        l = d.loc[d["won"] == 0, col]
        a = auc(d[col], d["won"])
        rows.append({
            "feature": col, "n": len(d), "auc": a,
            "lower_is_better": col in LOWER_IS_BETTER,
            "auc_oriented": (1 - a) if col in LOWER_IS_BETTER else a,
            "std_mean_diff": (w.mean() - l.mean()) / d[col].std(),
        })
    tbl = pd.DataFrame(rows).sort_values("auc_oriented", ascending=False)
    save_table(tbl, "01_univariate")
    print(tbl.to_string(index=False))

    top = tbl.head(20).iloc[::-1]
    fig, ax = new_fig(9, 7.5)
    ax.barh(top["feature"], top["auc_oriented"] - 0.5, color=SERIES[0], height=0.68,
            left=0.5)
    ax.axvline(0.5, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_xlabel("AUC, oriented so higher = more predictive (see lower_is_better flag in CSV)")
    ax.set_title("Top 20 EDA_2 candidates by univariate AUC")
    savefig(fig, "01_univariate")


@section("02", "Drive efficiency — offense and defense")
def s02(ctx: Ctx):
    off = [c for c in DRIVE_FEATURES if c.startswith("off_")]
    deff = [c for c in DRIVE_FEATURES if c.startswith("def_")]
    _panel_grid(ctx, off, "Offensive drive efficiency vs win rate", "02_drive_offense")
    _panel_grid(ctx, deff, "Defensive drive efficiency vs win rate", "02_drive_defense")


@section("03", "PFR charting depth")
def s03(ctx: Ctx):
    _panel_grid(ctx, PFR_FEATURES, "PFR advanced charting vs win rate", "03_pfr_depth", ncols=4)


@section("04", "Turnovers and special teams")
def s04(ctx: Ctx):
    _panel_grid(ctx, TURNOVER_ST_FEATURES, "Turnovers & special teams vs win rate",
                "04_turnovers_st")


@section("05", "Continuity and coaching")
def s05(ctx: Ctx):
    p = ctx.panel
    rows = []
    for col in CONTINUITY_FEATURES:
        if col not in p or p[col].nunique() < 2:
            continue
        a = auc(p[col], p["won"])
        rows.append({"feature": col, "auc": a, "n": int(p[col].notna().sum())})
    tbl = pd.DataFrame(rows)
    save_table(tbl, "05_continuity_auc")
    print(tbl.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    despine(axes[0])
    g = binned_rate(p, "roster_continuity_score", bins=6)
    axes[0].errorbar(g["center"], g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
                      fmt="o-", color=SERIES[0], ecolor=INK_MUTED, capsize=3)
    axes[0].axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
    axes[0].set_title(f"Roster continuity\nAUC {auc(p['roster_continuity_score'], p['won']):.3f}",
                       fontsize=10)
    axes[0].set_ylabel("Win rate")

    despine(axes[1])
    g2 = binned_rate(p, "coach_tenure_weeks", bins=6)
    axes[1].errorbar(g2["center"], g2["rate"], yerr=[g2["rate"] - g2["lo"], g2["hi"] - g2["rate"]],
                      fmt="o-", color=SERIES[0], ecolor=INK_MUTED, capsize=3)
    axes[1].axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
    axes[1].set_title(f"Coach tenure (games)\nAUC {auc(p['coach_tenure_weeks'], p['won']):.3f}",
                       fontsize=10)

    # The left-censoring check: does the censored group behave differently?
    despine(axes[2])
    gc = p.groupby("coach_tenure_censored")["won"].agg(["size", "mean"]).reset_index()
    axes[2].bar(["Post-2021\nhire", "Pre-2021\n(censored)"], gc["mean"], color=SERIES[0],
                width=0.6)
    axes[2].axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
    for xi, (m, n) in enumerate(zip(gc["mean"], gc["size"])):
        axes[2].text(xi, m + 0.006, f"{m:.3f}\nn={n:,}", ha="center", fontsize=8,
                     color=INK_SECONDARY)
    axes[2].set_title("Win rate: censored vs known coach tenure", fontsize=10)
    axes[2].set_ylim(0.4, 0.6)

    fig.suptitle("Continuity and coaching vs win rate", fontsize=13)
    fig.tight_layout()
    savefig(fig, "05_continuity")

    # Division-familiarity sanity check promised in the plan: a nonzero SAME-SEASON
    # count should only occur for divisional opponents.
    same_season = p[(p["division_familiarity"] > 0)]
    print("\ndivision_familiarity>0 rows by div_game (sanity: same-season repeats "
          "should be ~all div_game=1; cross-season repeats can be either):")
    print(p.groupby(["div_game"])["division_familiarity"].agg(["size", "mean", "max"])
          .to_string())


@section("06", "Interactions — continuous, with main effects reported alongside")
def s06(ctx: Ctx):
    p = ctx.panel
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (name, (a_col, b_col)) in zip(axes, INTERACTION_COMPONENTS.items()):
        despine(ax)
        aucs = {
            a_col: auc(p[a_col], p["won"]) if a_col in p else float("nan"),
            b_col: auc(p[b_col], p["won"]) if b_col in p else float("nan"),
            name: auc(p[name], p["won"]) if name in p else float("nan"),
        }
        labels = ["own\n" + a_col.replace("_r4", ""), "opp\n" + b_col.replace("_r4", "").replace("opp_", ""),
                  "interaction"]
        vals = list(aucs.values())
        ax.bar(labels, vals, color=SERIES[:2] + [SERIES[2]], width=0.6)
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0.35, 0.65)
        rows.append({"interaction": name, "own_component": a_col, "own_auc": aucs[a_col],
                     "opp_component": b_col, "opp_auc": aucs[b_col],
                     "interaction_auc": aucs[name]})
    axes[0].set_ylabel("AUC")
    fig.suptitle("Interaction AUC vs its own main effects — does it add anything?", fontsize=12)
    fig.tight_layout()
    savefig(fig, "06_interactions")
    tbl = pd.DataFrame(rows)
    save_table(tbl, "06_interactions")
    print(tbl.to_string(index=False))
    print("\nVerdict rule: an interaction earns its keep only if interaction_auc "
          "exceeds BOTH main-effect AUCs — otherwise it's redundant with a main effect.")


@section("07", "Correlation and mutual information")
def s07(ctx: Ctx):
    p = ctx.panel
    feats = ctx.features
    d = p[feats + ["won"]].dropna()
    print(f"complete-case rows for the matrix: {len(d):,} of {len(p):,}")

    corr = d[feats].corr()
    save_table(corr.reset_index(names="feature"), "07_correlation_matrix")

    mi = mutual_info_classif(StandardScaler().fit_transform(d[feats]), d["won"], random_state=42)
    mi_tbl = pd.DataFrame({"feature": feats, "mutual_info": mi}).sort_values(
        "mutual_info", ascending=False)
    save_table(mi_tbl, "07_mutual_info")
    print(mi_tbl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap=_diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    ax.grid(False)
    ax.set_title("EDA_2 feature correlation", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    savefig(fig, "07_correlation")

    top = mi_tbl.head(20).iloc[::-1]
    fig, ax = new_fig(9, 7)
    ax.barh(top["feature"], top["mutual_info"], color=SERIES[0], height=0.68)
    ax.set_xlabel("Mutual information with win/loss")
    ax.set_title("Top 20 EDA_2 features by mutual information")
    savefig(fig, "07_mutual_info")


@section("08", "Market-independence screen — the edge test")
def s08(ctx: Ctx):
    """BH-corrected across ALL EDA_2 candidates at once, not per sub-group —
    testing 27+ features invites false positives regardless of which section they
    came from."""
    p = ctx.panel
    feats = ctx.features
    rows = []
    for f in feats:
        d = p[[f, "team_spread", "won"]].dropna()
        if len(d) < 200 or d[f].std() == 0:
            continue
        raw_r, _ = stats.pearsonr(d[f], d["won"])
        slope, intercept, *_ = stats.linregress(d["team_spread"], d[f])
        resid = d[f] - (intercept + slope * d["team_spread"])
        res_r, res_p = stats.pearsonr(resid, d["won"])
        rows.append({
            "feature": f, "n": len(d), "raw_corr": raw_r, "raw_auc": auc(d[f], d["won"]),
            "corr_with_spread": d[f].corr(d["team_spread"]),
            "residual_corr": res_r, "residual_p": res_p,
            "residual_auc": auc(resid, d["won"]),
        })
    tbl = pd.DataFrame(rows)
    tbl["survives_bh"] = bh_screen(tbl.set_index("feature")["residual_p"]).reindex(
        tbl["feature"]).values
    tbl = tbl.sort_values("residual_corr", key=abs, ascending=False)
    save_table(tbl, "08_market_independence")
    print(tbl.to_string(index=False))
    n_survive = int(tbl["survives_bh"].sum())
    print(f"\n{n_survive} of {len(tbl)} EDA_2 features survive Benjamini-Hochberg "
          "on the residual (single correction across all candidates).")

    top = tbl.head(18).iloc[::-1]
    fig, ax = new_fig(10, 7.5)
    _bar_by_sign(ax, top["feature"], top["residual_corr"])
    ax.set_xlabel("Correlation with winning, AFTER removing the closing spread")
    ax.set_title("EDA_2 market-independence screen")
    savefig(fig, "08_market_independence")

    _incremental_test(p, tbl)


def _incremental_test(p: pd.DataFrame, tbl: pd.DataFrame):
    train = p[p["season"] < TEST_SEASON]
    test = p[p["season"] == TEST_SEASON]
    base_cols = ["team_spread"]

    def fit_auc(cols):
        tr = train[cols + ["won"]].dropna()
        te = test[cols + ["won"]].dropna()
        if len(tr) < 200 or len(te) < 100:
            return float("nan")
        sc = StandardScaler().fit(tr[cols])
        m = LogisticRegression(max_iter=1000).fit(sc.transform(tr[cols]), tr["won"])
        return auc(pd.Series(m.predict_proba(sc.transform(te[cols]))[:, 1], index=te.index),
                   te["won"])

    base = fit_auc(base_cols)
    rows = [{"feature": "(closing spread only)", "holdout_auc": base, "lift": 0.0}]
    for f in tbl.head(15)["feature"]:
        rows.append({"feature": f, "holdout_auc": fit_auc(base_cols + [f]),
                     "lift": fit_auc(base_cols + [f]) - base})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    save_table(out, "08_incremental_holdout_auc")
    print(f"\nHoldout ({TEST_SEASON}) AUC, spread alone = {base:.4f} "
          f"(EDA_1 reference: {MARKET_ONLY_HOLDOUT_AUC})")
    print(out.to_string(index=False))

    beat_line = out[(out["feature"] != "(closing spread only)") & (out["lift"] > 0)]
    if len(beat_line):
        print(f"\n*** {len(beat_line)} feature(s) IMPROVED holdout AUC over spread-alone ***")
    else:
        print("\nNo EDA_2 feature improved holdout AUC over spread-alone.")

    plot = out[out["feature"] != "(closing spread only)"].head(15).iloc[::-1]
    fig, ax = new_fig(10, 6.5)
    _bar_by_sign(ax, plot["feature"], plot["lift"])
    ax.set_xlabel(f"Change in {TEST_SEASON} holdout AUC when added to the closing spread")
    ax.set_title("EDA_2: does the feature add anything the market lacks?")
    savefig(fig, "08_incremental_auc")


# ------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in all_sections():
            print(f"  {s.key}  {s.title}")
        return 0

    keys = [k.strip() for k in args.sections.split(",") if k.strip()]
    if not keys and not args.all:
        ap.error("pass --all or --sections")

    apply_style()
    print("building team-game panel...")
    ctx = Ctx(panel=build_team_game_panel())
    print(f"panel: {ctx.panel.shape[0]:,} rows x {ctx.panel.shape[1]} cols; "
          f"{len(ctx.features)} EDA_2 candidate features")

    failures = run_sections(keys, ctx)
    if failures:
        print(f"\nFAILED sections: {failures}")
        return 1
    print("\nAll sections complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
