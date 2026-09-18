"""EDA_1 — what drives NFL wins, 2021-2025 regular season.

Target is win/loss, one row per team-game. Every in-game stat enters as a lagged
rolling feature (see src/features.py); the closing line is a control, never a target.

All output lands in EDA/EDA_1/: one PNG per figure, a CSV of the numbers behind it
in data/, and the section's stdout in logs/.

Usage:
    python EDA/eda_1.py --list
    python EDA/eda_1.py --sections 01,03
    python EDA/eda_1.py --all
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from EDA._eda_common import (  # noqa: E402
    DIVERGING_MID, DIVERGING_NEG, DIVERGING_POS, INK_MUTED, INK_SECONDARY,
    LOSS_COLOR, SERIES, WIN_COLOR, all_sections, apply_style, auc, bh_screen,
    binned_rate, despine, new_fig, run_sections, save_table, savefig, section,
    set_output_dir,
)
from src.features import build_team_game_panel  # noqa: E402

set_output_dir("EDA_1")

TEST_SEASON = 2025  # time-based holdout; train is 2021-2024

QB_FEATURES = [
    "qb_epa_r4", "qb_epa_r8", "qb_cpoe_r4", "qb_success_r4", "qb_sack_rate_r4",
    "qb_adot_r4", "qb_bad_throw_rate_r4", "qb_pressured_rate_r4",
    "qb_scramble_rate_r4", "qb_consecutive_starts_prior", "qb_is_new_starter",
    "qb_is_rookie", "qb_changed_last_week",
    "qb_epa_r4_diff", "qb_epa_r8_diff", "qb_cpoe_r4_diff",
]
OFF_FEATURES = [
    "off_epa_play_r4", "off_epa_play_r8", "off_success_r4", "off_cpoe_r4",
    "off_points_per_drive_r4", "off_3d_rate_r4", "off_pass_rate_r4",
    "off_sack_rate_r4", "off_epa_play_r4_diff", "off_points_per_drive_r4_diff",
]
DEF_FEATURES = [
    "def_epa_allowed_r4", "def_epa_allowed_r8", "def_pass_epa_allowed_r4",
    "def_rush_epa_allowed_r4", "def_success_allowed_r4", "def_havoc_rate_r4",
    "def_pressure_rate_r4", "def_blitz_rate_r4", "def_pressure_to_sack_r4",
    "def_3d_stop_rate_r4", "def_points_per_drive_r4", "def_epa_allowed_r4_diff",
    "def_pressure_rate_r4_diff", "def_havoc_rate_r4_diff",
    "def_points_per_drive_r4_diff",
]
INJ_FEATURES = ["inj_qb_out", "inj_ol_out", "inj_cb_out", "inj_edge_out", "inj_total_out"]
CTX_FEATURES = [
    "is_home", "rest_days", "short_week", "div_game", "is_dome", "wind",
    "travel_miles", "tz_shift", "games_of_history",
]
MARKET_FEATURES = ["team_spread", "total_line", "team_ml_implied_prob"]

ALL_FEATURES = (
    QB_FEATURES + OFF_FEATURES + DEF_FEATURES + INJ_FEATURES + CTX_FEATURES
    + MARKET_FEATURES
)

# Lower is better for these, so a negative correlation with winning is the
# expected sign. Flagged so the writeup does not misread the direction.
LOWER_IS_BETTER = {
    "def_epa_allowed_r4", "def_epa_allowed_r8", "def_pass_epa_allowed_r4",
    "def_rush_epa_allowed_r4", "def_success_allowed_r4", "def_points_per_drive_r4",
    "def_epa_allowed_r4_diff", "def_points_per_drive_r4_diff",
    "qb_sack_rate_r4", "qb_bad_throw_rate_r4", "qb_pressured_rate_r4",
    "off_sack_rate_r4", "inj_qb_out", "inj_ol_out", "inj_cb_out", "inj_edge_out",
    "inj_total_out", "qb_is_new_starter", "qb_changed_last_week",
}


@dataclass
class Ctx:
    panel: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return [f for f in ALL_FEATURES if f in self.panel.columns]


def _diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "div", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]
    )


def _bar_by_sign(ax, labels, values, invert: set[str] | None = None):
    """Horizontal bars colored by sign using the diverging pair."""
    invert = invert or set()
    colors = [DIVERGING_POS if v >= 0 else DIVERGING_NEG for v in values]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.invert_yaxis()


# ----------------------------------------------------------------- sections


@section("01", "Target and market baselines")
def s01(ctx: Ctx):
    p = ctx.panel
    rows = []
    for season, g in p.groupby("season"):
        home = g[g["is_home"]]
        fav = g[(g["team_spread"].notna()) & (g["team_spread"] != 0)]
        rows.append({
            "season": season,
            "team_games": len(g),
            "home_win_rate": home["won"].mean(),
            "favorite_win_rate": fav.loc[fav["team_spread"] > 0, "won"].mean(),
        })
    tbl = pd.DataFrame(rows)
    overall_home = p.loc[p["is_home"], "won"].mean()
    fav_all = p[(p["team_spread"].notna()) & (p["team_spread"] != 0)]
    overall_fav = fav_all.loc[fav_all["team_spread"] > 0, "won"].mean()
    print(f"team-game rows: {len(p):,}   overall win rate: {p['won'].mean():.4f}")
    print(f"home win rate:      {overall_home:.4f}")
    print(f"favorite win rate:  {overall_fav:.4f}  <- the bar to beat")
    save_table(tbl, "01_baselines")

    fig, ax = new_fig(9, 4.6)
    x = np.arange(len(tbl))
    ax.bar(x - 0.2, tbl["home_win_rate"], width=0.38, color=SERIES[0],
           label="Home team wins")
    ax.bar(x + 0.2, tbl["favorite_win_rate"], width=0.38, color=SERIES[1],
           label="Closing favorite wins")
    ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
    ax.text(len(tbl) - 0.45, 0.508, "coin flip", color=INK_MUTED, fontsize=8, ha="right")
    for xi, (h, f) in enumerate(zip(tbl["home_win_rate"], tbl["favorite_win_rate"])):
        ax.text(xi - 0.2, h + 0.008, f"{h:.3f}", ha="center", fontsize=8,
                color=INK_SECONDARY)
        ax.text(xi + 0.2, f + 0.008, f"{f:.3f}", ha="center", fontsize=8,
                color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl["season"])
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("Win rate")
    ax.set_title("Baselines any model must beat")
    ax.legend(loc="upper left", ncols=2)
    savefig(fig, "01_baselines")


@section("02", "Univariate — feature distributions by outcome")
def s02(ctx: Ctx):
    p = ctx.panel
    show = [f for f in (
        "qb_epa_r4_diff", "off_epa_play_r4_diff", "def_epa_allowed_r4_diff",
        "qb_epa_r4", "def_epa_allowed_r4", "def_pressure_rate_r4",
        "off_points_per_drive_r4", "def_points_per_drive_r4", "qb_cpoe_r4",
        "def_havoc_rate_r4", "inj_total_out", "rest_days",
    ) if f in p.columns]

    rows = []
    fig, axes = plt.subplots(4, 3, figsize=(13, 13))
    for ax, col in zip(axes.ravel(), show):
        despine(ax)
        w = p.loc[p["won"] == 1, col].dropna()
        l = p.loc[p["won"] == 0, col].dropna()
        lo, hi = np.nanpercentile(p[col].dropna(), [1, 99])
        bins = np.linspace(lo, hi, 26)
        ax.hist(w, bins=bins, density=True, histtype="step", lw=2,
                color=WIN_COLOR, label="Win")
        ax.hist(l, bins=bins, density=True, histtype="step", lw=2,
                color=LOSS_COLOR, label="Loss")
        d = (w.mean() - l.mean()) / p[col].std()
        ax.set_title(f"{col}\nstd. mean diff {d:+.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
        rows.append({"feature": col, "mean_win": w.mean(), "mean_loss": l.mean(),
                     "std_mean_diff": d, "auc": auc(p[col], p["won"])})
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(show):]:
        ax.set_visible(False)
    fig.suptitle("Feature distributions, wins vs losses", fontsize=13, y=0.995)
    fig.tight_layout()
    savefig(fig, "02_univariate")
    save_table(pd.DataFrame(rows).sort_values("auc", ascending=False), "02_univariate")


@section("03", "QB delta — the headline feature")
def s03(ctx: Ctx):
    p = ctx.panel
    col = "qb_epa_r4_diff"
    g = binned_rate(p, col, bins=10)
    save_table(g, "03_qb_delta_bins")
    a = auc(p[col], p["won"])
    print(f"{col}: AUC {a:.4f} on {p[col].notna().sum():,} rows")
    for alt in ("qb_epa_r8_diff", "qb_epa_r4", "qb_cpoe_r4_diff", "team_spread"):
        if alt in p:
            print(f"  for comparison {alt:<22} AUC {auc(p[alt], p['won']):.4f}")

    fig, ax = new_fig(9, 5.2)
    ax.errorbar(g["center"], g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
                fmt="o-", color=SERIES[0], ecolor=INK_MUTED, elinewidth=1,
                capsize=3, markersize=8)
    ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("Team QB rolling EPA − opponent QB rolling EPA (last 4 games)")
    ax.set_ylabel("Win rate")
    ax.set_title(f"QB EPA advantage vs win rate  (AUC {a:.3f})")
    ax.text(0.02, 0.96, "95% Wilson intervals", transform=ax.transAxes,
            fontsize=8, color=INK_MUTED, va="top")
    savefig(fig, "03_qb_delta")


@section("04", "Defense")
def s04(ctx: Ctx):
    p = ctx.panel
    cols = [c for c in ("def_epa_allowed_r4_diff", "def_pressure_rate_r4",
                        "def_havoc_rate_r4", "def_3d_stop_rate_r4") if c in p]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, col in zip(axes.ravel(), cols):
        despine(ax)
        g = binned_rate(p, col, bins=8)
        ax.errorbar(g["center"], g["rate"],
                    yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
                    fmt="o-", color=SERIES[0], ecolor=INK_MUTED, elinewidth=1,
                    capsize=3)
        ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
        a = auc(p[col], p["won"])
        note = " (lower is better)" if col in LOWER_IS_BETTER else ""
        ax.set_title(f"{col}{note}\nAUC {a:.3f}", fontsize=10)
        ax.set_ylabel("Win rate")
        rows.append({"feature": col, "auc": a})
    fig.suptitle("Defensive features vs win rate", fontsize=13)
    fig.tight_layout()
    savefig(fig, "04_defense")
    tbl = pd.DataFrame([{"feature": c, "auc": auc(p[c], p["won"]),
                         "lower_is_better": c in LOWER_IS_BETTER}
                        for c in DEF_FEATURES if c in p])
    save_table(tbl.sort_values("auc"), "04_defense")
    print(tbl.sort_values("auc").to_string(index=False))


@section("05", "Injuries and roster churn")
def s05(ctx: Ctx):
    p = ctx.panel
    rows = []
    for col in INJ_FEATURES + ["qb_is_new_starter", "qb_changed_last_week",
                               "qb_is_rookie"]:
        if col not in p:
            continue
        for val, g in p.groupby(p[col].clip(upper=4) if p[col].max() > 1 else p[col]):
            if len(g) < 30:
                continue
            rows.append({"feature": col, "value": val, "n": len(g),
                         "win_rate": g["won"].mean()})
    tbl = pd.DataFrame(rows)
    save_table(tbl, "05_injuries")

    qb_out = p.groupby(p["inj_qb_out"].clip(upper=2))["won"].agg(["size", "mean"])
    print("win rate by QBs listed Out/Doubtful:")
    print(qb_out.to_string())
    print("\nwin rate by new-starter flag:")
    print(p.groupby("qb_is_new_starter")["won"].agg(["size", "mean"]).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    panels = [
        ("inj_total_out", "Players Out/Doubtful (all positions)"),
        ("inj_ol_out", "Offensive linemen Out/Doubtful"),
        ("qb_is_new_starter", "New QB starter this week"),
    ]
    for ax, (col, title) in zip(axes, panels):
        despine(ax)
        s = p[col].clip(upper=4) if p[col].max() > 1 else p[col]
        g = p.groupby(s)["won"].agg(["size", "mean"]).reset_index()
        g = g[g["size"] >= 30]
        ax.bar(g[col].astype(str), g["mean"], color=SERIES[0], width=0.62)
        ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
        for xi, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(xi, m + 0.006, f"{m:.3f}\nn={n:,}", ha="center", fontsize=8,
                    color=INK_SECONDARY)
        ax.set_ylim(0.3, 0.68)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Win rate")
    fig.suptitle("Injuries and QB churn vs win rate", fontsize=13)
    fig.tight_layout()
    savefig(fig, "05_injuries")


@section("06", "Situational context")
def s06(ctx: Ctx):
    p = ctx.panel
    rows = []
    for col in CTX_FEATURES:
        if col not in p:
            continue
        a = auc(p[col], p["won"])
        rows.append({"feature": col, "auc": a,
                     "win_rate_high": p.loc[p[col] > p[col].median(), "won"].mean(),
                     "win_rate_low": p.loc[p[col] <= p[col].median(), "won"].mean()})
    tbl = pd.DataFrame(rows).sort_values("auc", ascending=False)
    save_table(tbl, "06_situational")
    print(tbl.to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    panels = [("is_home", "Home"), ("div_game", "Divisional"),
              ("short_week", "Short week"), ("is_dome", "Dome")]
    for ax, (col, title) in zip(axes, panels):
        despine(ax)
        g = p.groupby(p[col].astype(int))["won"].agg(["size", "mean"]).reset_index()
        ax.bar(["No", "Yes"][: len(g)], g["mean"], color=SERIES[0], width=0.6)
        ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
        for xi, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(xi, m + 0.006, f"{m:.3f}\nn={n:,}", ha="center", fontsize=8,
                    color=INK_SECONDARY)
        ax.set_ylim(0.35, 0.62)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Win rate")
    fig.suptitle("Situational context vs win rate", fontsize=13)
    fig.tight_layout()
    savefig(fig, "06_situational")


@section("07", "Market baseline and calibration")
def s07(ctx: Ctx):
    p = ctx.panel.dropna(subset=["team_spread", "won"]).copy()
    g = binned_rate(p, "team_spread", bins=12)
    save_table(g, "07_market_calibration")
    print(f"closing spread AUC: {auc(p['team_spread'], p['won']):.4f}")
    if "team_ml_implied_prob" in p:
        print(f"moneyline implied prob AUC: "
              f"{auc(p['team_ml_implied_prob'], p['won']):.4f}")

    fig, ax = new_fig(9, 5.2)
    ax.errorbar(g["center"], g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
                fmt="o", color=SERIES[0], ecolor=INK_MUTED, elinewidth=1, capsize=3)
    # Logistic fit of win probability on the spread - the market's own curve.
    xs = np.linspace(g["center"].min(), g["center"].max(), 100)
    lr = LogisticRegression().fit(p[["team_spread"]], p["won"])
    ax.plot(xs, lr.predict_proba(xs.reshape(-1, 1))[:, 1], color=SERIES[1], lw=2,
            label="Logistic fit on spread")
    ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls="--")
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("Closing spread, team perspective (positive = favored)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("The market is well calibrated — this is the benchmark")
    ax.legend(loc="upper left")
    savefig(fig, "07_market_calibration")


@section("08", "Interactions")
def s08(ctx: Ctx):
    p = ctx.panel
    pairs = [
        ("qb_scramble_rate_r4", "opp_def_pressure_rate_r4",
         "Mobile QB vs pressure defense"),
        ("off_pass_rate_r4", "opp_def_rush_epa_allowed_r4",
         "Run-heavy offense vs weak run defense"),
    ]
    # Some opponent columns already exist from DIFF_BASES; re-merging them would
    # collide into _x/_y and silently drop the interaction.
    for src in ("def_pressure_rate_r4", "def_rush_epa_allowed_r4"):
        if f"opp_{src}" in p.columns:
            continue
        opp = p[["game_id", "team", src]].rename(
            columns={"team": "opponent", src: f"opp_{src}"})
        p = p.merge(opp, on=["game_id", "opponent"], how="left")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (a_col, b_col, title) in zip(axes, pairs):
        despine(ax)
        if a_col not in p or b_col not in p:
            ax.set_visible(False)
            continue
        d = p[[a_col, b_col, "won"]].dropna()
        d = d.assign(
            a=pd.qcut(d[a_col], 3, labels=["low", "mid", "high"], duplicates="drop"),
            b=pd.qcut(d[b_col], 3, labels=["low", "mid", "high"], duplicates="drop"),
        )
        grid = d.groupby(["a", "b"], observed=True)["won"].mean().unstack()
        counts = d.groupby(["a", "b"], observed=True)["won"].size().unstack()
        im = ax.imshow(grid.values, cmap=_diverging_cmap(), vmin=0.35, vmax=0.65)
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel(b_col)
        ax.set_ylabel(a_col)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid.values[i, j]:.3f}\nn={counts.values[i, j]:,}",
                        ha="center", va="center", fontsize=8, color="#0b0b0b")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Win rate")
        flat = grid.stack().reset_index()
        flat.columns = ["a_tercile", "b_tercile", "win_rate"]
        flat.insert(0, "interaction", title)
        rows.append(flat)
    fig.suptitle("Interaction checks (terciles)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "08_interactions")
    if rows:
        save_table(pd.concat(rows), "08_interactions")
        print(pd.concat(rows).to_string(index=False))


@section("09", "Correlation and mutual information")
def s09(ctx: Ctx):
    p = ctx.panel
    feats = ctx.features
    d = p[feats + ["won"]].dropna()
    print(f"complete-case rows for the matrix: {len(d):,}")

    corr = d[feats].corr()
    save_table(corr.reset_index(names="feature"), "09_correlation_matrix")

    mi = mutual_info_classif(
        StandardScaler().fit_transform(d[feats]), d["won"], random_state=42
    )
    mi_tbl = pd.DataFrame({"feature": feats, "mutual_info": mi,
                           "abs_corr_with_won": [abs(d[f].corr(d["won"])) for f in feats]})
    mi_tbl = mi_tbl.sort_values("mutual_info", ascending=False)
    save_table(mi_tbl, "09_mutual_info")
    print(mi_tbl.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap=_diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    ax.grid(False)
    ax.set_title("Feature correlation — dark blocks are redundant groups", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    savefig(fig, "09_correlation")

    top = mi_tbl.head(20).iloc[::-1]
    fig, ax = new_fig(9, 7)
    ax.barh(top["feature"], top["mutual_info"], color=SERIES[0], height=0.68)
    ax.set_xlabel("Mutual information with win/loss")
    ax.set_title("Top 20 features by mutual information")
    savefig(fig, "09_mutual_info")


@section("10", "Market-independence screen — the edge test")
def s10(ctx: Ctx):
    """A feature that only re-derives the closing line cannot produce edge.

    Each feature is regressed on the closing spread; what survives as a residual is
    the part the market has not already priced. Features are ranked by the residual's
    relationship to the outcome, not by raw predictiveness.
    """
    p = ctx.panel
    feats = [f for f in ctx.features if f not in MARKET_FEATURES]
    rows = []
    for f in feats:
        d = p[[f, "team_spread", "won"]].dropna()
        if len(d) < 200 or d[f].std() == 0:
            continue
        raw_r, raw_p = stats.pearsonr(d[f], d["won"])
        slope, intercept, *_ = stats.linregress(d["team_spread"], d[f])
        resid = d[f] - (intercept + slope * d["team_spread"])
        res_r, res_p = stats.pearsonr(resid, d["won"])
        rows.append({
            "feature": f, "n": len(d),
            "raw_corr": raw_r, "raw_auc": auc(d[f], d["won"]),
            "corr_with_spread": d[f].corr(d["team_spread"]),
            "residual_corr": res_r, "residual_p": res_p,
            "residual_auc": auc(resid, d["won"]),
        })
    tbl = pd.DataFrame(rows)
    tbl["survives_bh"] = bh_screen(tbl.set_index("feature")["residual_p"]).reindex(
        tbl["feature"]).values
    tbl = tbl.sort_values("residual_corr", key=abs, ascending=False)
    save_table(tbl, "10_market_independence")
    print(tbl.head(20).to_string(index=False))
    print(f"\n{int(tbl['survives_bh'].sum())} of {len(tbl)} features survive "
          "Benjamini-Hochberg on the residual.")

    top = tbl.head(18).iloc[::-1]
    fig, ax = new_fig(10, 7.5)
    _bar_by_sign(ax, top["feature"], top["residual_corr"])
    ax.set_xlabel("Correlation with winning, AFTER removing the closing spread")
    ax.set_title("Market-independence screen — what Vegas has not already priced")
    for yi, (v, ok) in enumerate(zip(top["residual_corr"], top["survives_bh"])):
        ax.text(v + (0.002 if v >= 0 else -0.002), yi, "✓" if ok else "",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK_SECONDARY)
    savefig(fig, "10_market_independence")

    # Does anything actually add to the market on a held-out season?
    _incremental_test(p, tbl)


def _incremental_test(p: pd.DataFrame, tbl: pd.DataFrame):
    """Holdout AUC of spread alone vs spread + each candidate. The decisive test."""
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
        return auc(pd.Series(m.predict_proba(sc.transform(te[cols]))[:, 1],
                             index=te.index), te["won"])

    base = fit_auc(base_cols)
    rows = [{"feature": "(closing spread only)", "holdout_auc": base, "lift": 0.0}]
    for f in tbl.head(15)["feature"]:
        a = fit_auc(base_cols + [f])
        rows.append({"feature": f, "holdout_auc": a, "lift": a - base})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    save_table(out, "10_incremental_holdout_auc")
    print(f"\nHoldout ({TEST_SEASON}) AUC, spread alone = {base:.4f}")
    print(out.to_string(index=False))

    plot = out[out["feature"] != "(closing spread only)"].head(15).iloc[::-1]
    fig, ax = new_fig(10, 6.5)
    _bar_by_sign(ax, plot["feature"], plot["lift"])
    ax.set_xlabel(f"Change in {TEST_SEASON} holdout AUC when added to the closing spread")
    ax.set_title("Does the feature add anything the market lacks?")
    savefig(fig, "10_incremental_auc")


# ------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="", help="comma-separated keys, e.g. 01,03")
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
          f"{len(ctx.features)} candidate features")

    failures = run_sections(keys, ctx)
    if failures:
        print(f"\nFAILED sections: {failures}")
        return 1
    print("\nAll sections complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
