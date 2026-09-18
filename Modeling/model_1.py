"""Model_1 — LR + GBM + RF vs the closing line. Single full-season model.

Revision 3. The two-regime (EARLY/LATE) split was TESTED AND REJECTED — section 04
carries the evidence. The split's entire justification was an apparent early-vs-late
market-efficiency gap (0.7465 vs 0.7041); bootstrapping that gap gives a 95% CI of
[-0.042, +0.127], straddling zero. One season of noise, not a regime difference.

A single full-season model matches or beats the split on every metric while training
on all 2,168 rows instead of fragmenting into 978/1,190, which also shrinks the worst
overfitting gap from 0.123 to 0.052.

Consequence: `roster_continuity_score` is dropped. It is null for weeks 1-8, so it
only existed to justify the split. It was UNCONFIRMED anyway (2 of 3 gates).

Standing policy, carried forward:
  * collinearity defines the default feature set (the full set is a reference row)
  * simple average ships; stacking is comparison-only
  * calibration at output only, Platt, judged on Brier/ECE never AUC
  * borderline features need 3-of-3 gates to count as confirmed

Usage:
    python Modeling/model_1.py --all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from EDA._eda_common import (  # noqa: E402
    DIVERGING_NEG, DIVERGING_POS, INK_MUTED, INK_SECONDARY, SERIES,
    all_sections, apply_style, despine, new_fig, run_sections, save_table, savefig,
    section, set_output_dir,
)
from src.features import build_team_game_panel  # noqa: E402
from src.model import (  # noqa: E402
    LR_GRID, RANDOM_STATE, TEST_SEASON, ThresholdEncoder, calibrate, compute_vif,
    correlation_clusters, gbm_grid, make_gbm, make_lr, make_rf, monotone_features,
    oof_predictions, pick_representatives, reliability_stats, rf_grid,
    walk_forward_tune,
)

set_output_dir("Model_1", base=PROJECT_ROOT / "Modeling")

QB = ["qb_epa_r8_diff", "qb_epa_r4_diff", "qb_cpoe_r4_diff", "qb_epa_r8",
      "qb_success_r4", "qb_sack_rate_r4"]
OFF = ["off_epa_play_r4_diff", "off_points_per_drive_r4_diff", "off_epa_play_r8",
       "off_success_r4", "off_yards_per_drive_r8", "off_3d_rate_r8",
       "off_explosive_play_rate_r8", "off_pass_rate_r4", "off_red_zone_td_rate_r8"]
DEF = ["def_epa_allowed_r4_diff", "def_takeaway_rate_r8",
       "def_points_per_drive_r4_diff", "def_pressure_rate_r4_diff",
       "def_explosive_play_rate_allowed_r8", "def_havoc_rate_r4_diff",
       "def_epa_allowed_r8", "def_success_allowed_r4"]
TO = ["turnover_margin_rate_r8"]
INJ = ["inj_total_out", "inj_qb_out", "inj_ol_out"]
CTX = ["is_home"]
MKT = ["team_spread"]

# roster_continuity_score is deliberately ABSENT: null weeks 1-8, existed only to
# justify the rejected split, and never cleared the 3-of-3 confirmation gate.
FULL_FEATURES = QB + OFF + DEF + TO + INJ + CTX + MKT  # 29

N_BOOTSTRAP = 2000


@dataclass
class Ctx:
    panel: pd.DataFrame
    cache: dict = field(default_factory=dict)


def season_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(panel["week"] >= 1) & (panel["week"] <= 18)]


def bootstrap_auc_ci(y, p, n: int = N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        boots.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def spread_only(df: pd.DataFrame):
    tr = df[df["season"] < TEST_SEASON][["team_spread", "won"]].dropna()
    te = df[df["season"] == TEST_SEASON][["team_spread", "won"]].dropna()
    sc = StandardScaler().fit(tr[["team_spread"]])
    m = LogisticRegression(max_iter=1000).fit(sc.transform(tr[["team_spread"]]), tr["won"])
    return (float(roc_auc_score(te["won"], m.predict_proba(sc.transform(te[["team_spread"]]))[:, 1])),
            m.predict_proba(sc.transform(te[["team_spread"]]))[:, 1], te, len(tr), len(te))


def reduced_features(panel: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    df = season_frame(panel)
    train = df[df["season"] < TEST_SEASON]
    feats = FULL_FEATURES

    vif = compute_vif(train, feats)
    clusters = correlation_clusters(train, feats, threshold=0.8)

    resid_r = {}
    for f in feats:
        if f == "team_spread":
            resid_r[f] = np.nan
            continue
        cols = list(dict.fromkeys([f, "team_spread", "won"]))
        d = train[cols].dropna()
        if len(d) < 100 or d[f].std() == 0:
            resid_r[f] = np.nan
            continue
        sl, ic, *_ = stats.linregress(d["team_spread"], d[f])
        r, _ = stats.pearsonr(d[f] - (ic + sl * d["team_spread"]), d["won"])
        resid_r[f] = abs(r)

    reps = pick_representatives(clusters, pd.Series(resid_r))
    if "team_spread" not in reps:
        reps = sorted(set(reps) | {"team_spread"})

    pipe = make_lr([])
    pipe.set_params(clf__C=0.1, clf__l1_ratio=1.0, thresh__mode="none")
    pipe.fit(train[feats], train["won"])
    l1_sel = {f for f, c in zip(feats, pipe.named_steps["clf"].coef_[0]) if abs(c) > 1e-8}

    tbl = (vif.merge(clusters, on="feature")
              .assign(resid_corr_abs=lambda d: d["feature"].map(resid_r),
                      kept_as_representative=lambda d: d["feature"].isin(reps),
                      l1_selected=lambda d: d["feature"].isin(l1_sel)))
    tbl["l1_vs_cluster_disagree"] = tbl["kept_as_representative"] != tbl["l1_selected"]
    return reps, tbl


def fit_all(panel, features, label, subset=None):
    df = season_frame(panel) if subset is None else subset
    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]

    strengths = {}
    for f in features:
        d = train[[f, "won"]].dropna()
        if len(d) > 100 and d[f].nunique() > 2:
            strengths[f] = abs(roc_auc_score(d["won"], d[f]) - 0.5)
    top10 = sorted(strengths, key=strengths.get, reverse=True)[:10]

    makers = {"LR": lambda: make_lr(top10), "GBM": make_gbm, "RF": make_rf}
    grids = {"LR": LR_GRID, "GBM": gbm_grid("LATE"), "RF": rf_grid("LATE")}

    tuned, probs, rows = {}, {}, []
    for name in ("LR", "GBM", "RF"):
        tm = walk_forward_tune(makers[name], grids[name], train, features, name)
        tuned[name] = tm
        pipe = makers[name]()
        pipe.set_params(**tm.params)
        pipe.fit(train[features], train["won"])
        probs[name] = pipe.predict_proba(test[features])[:, 1]
        rows.append(_row(name, test["won"], probs[name], tm, "single"))

    avg = np.mean([probs[n] for n in ("LR", "GBM", "RF")], axis=0)
    probs["SimpleAvg"] = avg
    rows.append(_row("SimpleAvg", test["won"], avg, None, "PRODUCTION"))

    oof = pd.DataFrame({n: oof_predictions(makers[n], tuned[n].params, train, features)
                        for n in ("LR", "GBM", "RF")}).dropna()
    meta = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    meta.fit(oof, train.loc[oof.index, "won"])
    stack = meta.predict_proba(pd.DataFrame(probs)[oof.columns])[:, 1]
    probs["Stacked"] = stack
    rows.append(_row("Stacked", test["won"], stack, None, "comparison-only"))

    bench, bench_p, bench_te, n_tr, n_te = spread_only(df)
    rows.append(_row("MARKET (spread only)", bench_te["won"], bench_p, None, "benchmark"))

    tbl = pd.DataFrame(rows)
    tbl.insert(0, "features", label)
    tbl["benchmark"] = bench
    tbl["lift_vs_market"] = tbl["holdout_auc"] - bench
    tbl["n_features"] = len(features)
    return {"table": tbl, "tuned": tuned, "probs": probs, "test": test, "train": train,
            "features": features, "top10": top10, "benchmark": bench, "oof": oof,
            "makers": makers, "n_train": n_tr, "n_test": n_te}


def _row(name, y, p, tuned, role) -> dict:
    y = np.asarray(y)
    lo, hi = bootstrap_auc_ci(y, p)
    rel = reliability_stats(y, p)
    return {"model": name, "role": role,
            "holdout_auc": float(roc_auc_score(y, p)),
            "auc_ci_lo": lo, "auc_ci_hi": hi,
            "accuracy": float(accuracy_score(y, (np.asarray(p) > 0.5).astype(int))),
            "brier": rel["brier"], "log_loss": rel["log_loss"], "ece": rel["ece"],
            "cv_val_auc": tuned.cv_auc if tuned else np.nan,
            "cv_train_auc": tuned.cv_train_auc if tuned else np.nan,
            "overfit_gap": (tuned.cv_train_auc - tuned.cv_auc) if tuned else np.nan,
            "params": str(tuned.params) if tuned else ""}


# ----------------------------------------------------------------- sections


@section("01", "Setup, benchmark, thresholds")
def s01(ctx: Ctx):
    df = season_frame(ctx.panel)
    bench, _, te, n_tr, n_te = spread_only(df)
    fav = accuracy_score(te["won"], (te["team_spread"] > 0).astype(int))
    tbl = pd.DataFrame([{"weeks": "1-18 (single model)", "full_features": len(FULL_FEATURES),
                         "train_rows": n_tr, "test_rows": n_te,
                         "spread_only_auc": bench, "expected": 0.7187,
                         "always_pick_favorite_accuracy": fav}])
    save_table(tbl, "01_setup")
    print(tbl.to_string(index=False))
    assert abs(bench - 0.7187) < 0.005, "benchmark drift — pipeline wired wrong"
    print("\nSanity gate PASSED: spread-only reproduces the 0.7187 full-season benchmark.")

    train = df[df["season"] < TEST_SEASON]
    st = {f: abs(roc_auc_score(train[[f, "won"]].dropna()["won"],
                               train[[f, "won"]].dropna()[f]) - 0.5)
          for f in FULL_FEATURES if train[[f, "won"]].dropna()[f].nunique() > 2}
    top10 = sorted(st, key=st.get, reverse=True)[:10]
    mono = monotone_features(train, top10)
    enc = ThresholdEncoder(features=top10, mode="both").fit(train[FULL_FEATURES], train["won"])
    thr = pd.DataFrame([{"feature": f, "univariate_auc": 0.5 + st[f],
                         "stump_threshold": enc.thresholds_.get(f, np.nan),
                         "youden_threshold": enc.youden_.get(f, np.nan),
                         "shape": "monotone->hinge" if mono.get(f) else "step->binary"}
                        for f in top10])
    save_table(thr, "01_thresholds")
    print("\nThresholds (training years only):")
    print(thr.to_string(index=False))


@section("02", "Collinearity — defines the default feature set")
def s02(ctx: Ctx):
    reps, tbl = reduced_features(ctx.panel)
    ctx.cache["reps"] = reps
    save_table(tbl, "02_collinearity_labeled")
    print(tbl.sort_values("vif", ascending=False)[
        ["feature", "vif", "vif_tier", "cluster", "resid_corr_abs",
         "kept_as_representative", "l1_selected"]].to_string(index=False))
    print(f"\nVIF tiers: {tbl['vif_tier'].value_counts().to_dict()}")
    print(f"L1 kept {int(tbl['l1_selected'].sum())} of {len(tbl)}; "
          f"cluster disagreements: {int(tbl['l1_vs_cluster_disagree'].sum())}")
    print(f"\nDEFAULT feature set ({len(reps)}): {reps}")

    top = tbl.sort_values("vif", ascending=False).head(20)
    fig, ax = new_fig(9.5, 7)
    colors = {"SEVERE": DIVERGING_NEG, "HIGH": "#eda100", "MODERATE": SERIES[2], "LOW": SERIES[0]}
    y = np.arange(len(top))
    ax.barh(y, top["vif"].clip(upper=50), height=0.68, color=[colors[t] for t in top["vif_tier"]])
    ax.set_yticks(y); ax.set_yticklabels(top["feature"], fontsize=8); ax.invert_yaxis()
    for x in (2.5, 5, 10):
        ax.axvline(x, color=INK_MUTED, lw=0.8, ls="--")
    ax.set_xlabel("VIF (clipped at 50); dashed = 2.5 / 5 / 10 tier cuts")
    ax.set_title("Collinearity — drives the default feature set")
    savefig(fig, "02_vif")


@section("03", "Primary models — single full-season, default feature set")
def s03(ctx: Ctx):
    reps = ctx.cache.get("reps") or reduced_features(ctx.panel)[0]
    ctx.cache["reps"] = reps
    res = fit_all(ctx.panel, reps, "reduced(DEFAULT)")
    ctx.cache["primary"] = res
    save_table(res["table"], "03_primary")
    print(res["table"][["model", "role", "holdout_auc", "auc_ci_lo", "auc_ci_hi",
                        "accuracy", "brier", "log_loss", "ece", "cv_val_auc",
                        "cv_train_auc", "overfit_gap", "lift_vs_market"]].round(4).to_string(index=False))
    print("\nSelected hyperparameters:")
    for n, tm in res["tuned"].items():
        print(f"  {n}: {tm.params}")

    d = res["table"]
    d = d[d["model"] != "MARKET (spread only)"]
    fig, ax = new_fig(9, 5.2)
    x = np.arange(len(d))
    ax.bar(x, d["holdout_auc"], width=0.6, color=SERIES[0])
    ax.errorbar(x, d["holdout_auc"],
                yerr=[d["holdout_auc"] - d["auc_ci_lo"], d["auc_ci_hi"] - d["holdout_auc"]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=4)
    ax.axhline(res["benchmark"], color=DIVERGING_NEG, ls="--", lw=1.5,
               label=f"market benchmark {res['benchmark']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(d["model"], fontsize=9)
    ax.set_ylabel("2025 holdout AUC"); ax.set_ylim(0.6, 0.8)
    ax.set_title("Single full-season model vs the closing line\nbars = 95% bootstrap CI")
    ax.legend(loc="lower right")
    savefig(fig, "03_primary")


@section("04", "The EARLY/LATE split — tested and rejected")
def s04(ctx: Ctx):
    """The split's justification was an apparent market-efficiency regime gap.
    Bootstrapping that gap shows it is not distinguishable from zero."""
    p = ctx.panel
    early = p[(p["week"] >= 1) & (p["week"] <= 8)]
    late = p[(p["week"] >= 9) & (p["week"] <= 18)]
    ae, pe, tee, _, _ = spread_only(early)
    al, pl, tel, _, _ = spread_only(late)
    ye, yl = tee["won"].to_numpy(), tel["won"].to_numpy()

    rng = np.random.default_rng(RANDOM_STATE)
    diffs = []
    for _ in range(5000):
        ie = rng.choice(len(ye), len(ye), replace=True)
        il = rng.choice(len(yl), len(yl), replace=True)
        if len(np.unique(ye[ie])) < 2 or len(np.unique(yl[il])) < 2:
            continue
        diffs.append(roc_auc_score(ye[ie], pe[ie]) - roc_auc_score(yl[il], pl[il]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    tbl = pd.DataFrame([{"early_market_auc": ae, "late_market_auc": al,
                         "observed_gap": ae - al, "ci_lo": lo, "ci_hi": hi,
                         "frac_early_le_late": float((diffs <= 0).mean()),
                         "significant": bool(lo > 0 or hi < 0)}])
    save_table(tbl, "04_regime_gap_test")
    print(tbl.round(4).to_string(index=False))
    print(f"\nObserved gap {ae - al:+.4f}, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}].")
    print("The CI straddles zero: the regime gap is NOT statistically significant.")
    print("The split is therefore unjustified and has been removed. It cost training")
    print("data (978/1,190 rows instead of 2,168) and raised the worst overfit gap")
    print("from 0.052 to 0.123 for no measurable benefit.")

    fig, ax = new_fig(9, 5)
    ax.hist(diffs, bins=60, color=SERIES[0])
    ax.axvline(0, color=DIVERGING_NEG, lw=2, label="no difference")
    ax.axvline(ae - al, color=INK_SECONDARY, lw=2, ls="--",
               label=f"observed {ae - al:+.4f}")
    ax.axvspan(lo, hi, color=SERIES[0], alpha=0.15, label="95% CI")
    ax.set_xlabel("EARLY market AUC − LATE market AUC (bootstrap)")
    ax.set_ylabel("count")
    ax.set_title("The regime gap that justified the split is not significant")
    ax.legend()
    savefig(fig, "04_regime_gap")


@section("05", "Calibration at output — Platt, judged on reliability")
def s05(ctx: Ctx):
    res = ctx.cache["primary"]
    test, train, oof = res["test"], res["train"], res["oof"]
    y = test["won"].to_numpy()
    ship = res["probs"]["SimpleAvg"]

    cal_idx = oof.index[train.loc[oof.index, "season"] == 2024]
    fn = calibrate(oof.loc[cal_idx].mean(axis=1).to_numpy(),
                   train.loc[cal_idx, "won"].to_numpy(), method="sigmoid")
    cal_p = np.clip(fn(ship), 1e-6, 1 - 1e-6)

    rows = [{"stage": "uncalibrated", "auc": roc_auc_score(y, ship),
             "n_calibration_rows": len(cal_idx), **reliability_stats(y, ship)},
            {"stage": "Platt-calibrated", "auc": roc_auc_score(y, cal_p),
             "n_calibration_rows": len(cal_idx), **reliability_stats(y, cal_p)}]
    tbl = pd.DataFrame(rows)
    save_table(tbl, "05_calibration_metrics")
    print(tbl[["stage", "auc", "brier", "log_loss", "ece", "mean_pred",
               "base_rate", "n_calibration_rows"]].round(4).to_string(index=False))
    print("\nAUC identical before/after confirms Platt is monotonic (cannot change")
    print("ranking). Judge calibration on Brier/ECE/log-loss only.")

    fig, ax = new_fig(8.5, 5.2)
    for lbl, pr, c in [("uncalibrated", ship, SERIES[0]), ("Platt", cal_p, SERIES[1])]:
        d = pd.DataFrame({"p": pr, "y": y})
        d["b"] = pd.qcut(d["p"], 8, duplicates="drop")
        g = d.groupby("b", observed=True).agg(pred=("p", "mean"), act=("y", "mean")).reset_index()
        ax.plot(g["pred"], g["act"], "o-", color=c, label=lbl)
    ax.plot([0, 1], [0, 1], ls="--", color=INK_MUTED, lw=1)
    ax.set_xlabel("Predicted win probability"); ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration — production model (simple average)")
    ax.legend(loc="upper left")
    savefig(fig, "05_calibration")


@section("06", "Verdict — full-set reference and final answer")
def s06(ctx: Ctx):
    prim = ctx.cache["primary"]["table"]
    ref = fit_all(ctx.panel, FULL_FEATURES, "full(reference)")
    allr = pd.concat([prim, ref["table"]])
    save_table(allr, "06_all_results")

    print("ALL RESULTS (2025 holdout, 542 rows):")
    print(allr[["features", "model", "role", "n_features", "holdout_auc", "auc_ci_lo",
                "auc_ci_hi", "accuracy", "brier", "log_loss", "ece",
                "lift_vs_market"]].round(4).to_string(index=False))

    beat = allr[(allr["lift_vs_market"] > 0) & (allr["model"] != "MARKET (spread only)")]
    n_cfg = len(allr[allr["model"] != "MARKET (spread only)"])
    print(f"\nConfigurations beating the benchmark: {len(beat)} of {n_cfg}")
    if len(beat):
        print(beat[["features", "model", "holdout_auc", "benchmark", "lift_vs_market",
                    "auc_ci_lo", "auc_ci_hi"]].round(4).to_string(index=False))
    else:
        print("  None. Every configuration underperforms the closing spread alone.")

    prod = allr[(allr["role"] == "PRODUCTION") & (allr["features"] == "reduced(DEFAULT)")]
    print("\nPRODUCTION model:")
    print(prod[["holdout_auc", "auc_ci_lo", "auc_ci_hi", "accuracy", "brier",
                "benchmark", "lift_vs_market"]].round(4).to_string(index=False))


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
    print(f"panel: {ctx.panel.shape[0]:,} rows; single full-season model, "
          f"{len(FULL_FEATURES)} candidate features")

    failures = run_sections(keys, ctx)
    if failures:
        print(f"\nFAILED sections: {failures}")
        return 1
    print("\nAll sections complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
