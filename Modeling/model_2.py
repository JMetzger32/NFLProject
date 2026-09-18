"""Model_2 — bagged bootstrap, per-game uncertainty, and the edge engine.

Does NOT attempt to beat the closing line. Model_1 settled that. This round asks a
narrower question: does bagging over block resamples make the model steadier and
better calibrated, and what does an honest edge filter built on that uncertainty
actually yield?

Usage:
    python Modeling/model_2.py --all
    python Modeling/model_2.py --sections 01,05
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
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from EDA._eda_common import (  # noqa: E402
    DIVERGING_NEG, DIVERGING_POS, INK_MUTED, INK_SECONDARY, SERIES,
    all_sections, apply_style, despine, new_fig, run_sections, save_table, savefig,
    section, set_output_dir,
)
from src.bootstrap import (  # noqa: E402
    DEFAULT_B, bagged_bootstrap, convergence_curve, importance_stability,
    load_model1_config, make_blocks,
)
from src.config import COMPLETED_SEASONS  # noqa: E402
from src.features import build_team_game_panel  # noqa: E402
from src.model import RANDOM_STATE, make_gbm, make_lr, make_rf, reliability_stats  # noqa: E402
from src.odds import (  # noqa: E402
    DEVIG_METHOD, MIN_HISTORY_FOR_PICK, build_edge_frame, devig, implied_prob,
    overround,
)

set_output_dir("Model_2", base=PROJECT_ROOT / "Modeling")

TEST_SEASON = 2025
MARKET_BENCHMARK = 0.7187


@dataclass
class Ctx:
    panel: pd.DataFrame
    cache: dict = field(default_factory=dict)


def _split(panel):
    p = panel[panel["season"].isin(COMPLETED_SEASONS)]
    return (p[p["season"] < TEST_SEASON].reset_index(drop=True),
            p[p["season"] == TEST_SEASON].reset_index(drop=True))


def get_boot(ctx: Ctx):
    if "boot" not in ctx.cache:
        feats, params = load_model1_config()
        train, test = _split(ctx.panel)
        print(f"  bootstrapping B={DEFAULT_B} over {len(make_blocks(train))} "
              f"season-week blocks...")
        ctx.cache["boot"] = bagged_bootstrap(train, test, feats, params,
                                             B=DEFAULT_B, verbose=False)
        ctx.cache["split"] = (train, test)
        ctx.cache["feats"] = feats
        ctx.cache["params"] = params
    return ctx.cache["boot"]


@section("01", "Bootstrap convergence")
def s01(ctx: Ctx):
    res = get_boot(ctx)
    train, _ = ctx.cache["split"]
    blocks = make_blocks(train)
    sizes = [len(b) for b in blocks]
    print(f"blocks: {len(blocks)} season-week units, median {int(np.median(sizes))} "
          f"rows (min {min(sizes)}, max {max(sizes)})")
    print("Season-level blocking was rejected: only 4 blocks, 35 distinct resample "
          "multisets,\nand a 31.6% chance of dropping an entire season per resample.")

    cur = convergence_curve(res)
    save_table(cur, "01_convergence")
    print()
    print(cur.round(5).to_string(index=False))

    fig, ax = new_fig(8.5, 5)
    ax.plot(cur["B"], cur["mean_ci_width"], "o-", color=SERIES[0], label="mean CI width")
    ax.plot(cur["B"], cur["median_ci_width"], "o--", color=SERIES[1], label="median")
    ax.set_xlabel("bootstrap resamples (B)")
    ax.set_ylabel("per-game CI width")
    ax.set_title("CI width stabilises before B=500")
    ax.legend()
    savefig(fig, "01_convergence")


@section("02", "Bagged vs single-fit")
def s02(ctx: Ctx):
    res = get_boot(ctx)
    train, test = ctx.cache["split"]
    feats, params = ctx.cache["feats"], ctx.cache["params"]
    y = test["won"].to_numpy()

    makers = {"LR": lambda: make_lr([]), "GBM": make_gbm, "RF": make_rf}
    singles = []
    for n in ("LR", "GBM", "RF"):
        pipe = makers[n]()
        pipe.set_params(**params[n])
        pipe.fit(train[feats], train["won"])
        singles.append(pipe.predict_proba(test[feats])[:, 1])
    single_avg = np.mean(singles, axis=0)

    rows = []
    for label, p in (("single-fit simple average", single_avg),
                     ("BAGGED (block bootstrap)", res.point)):
        rel = reliability_stats(y, p)
        rows.append({"model": label, "auc": roc_auc_score(y, p),
                     "accuracy": accuracy_score(y, (p > .5).astype(int)),
                     "brier": rel["brier"], "log_loss": rel["log_loss"],
                     "ece": rel["ece"]})
    mk = test["team_ml_implied_prob"].to_numpy()
    relm = reliability_stats(y, mk)
    rows.append({"model": "MARKET (moneyline implied)", "auc": roc_auc_score(y, mk),
                 "accuracy": accuracy_score(y, (mk > .5).astype(int)),
                 "brier": relm["brier"], "log_loss": relm["log_loss"],
                 "ece": relm["ece"]})
    t = pd.DataFrame(rows)
    t["vs_benchmark"] = t["auc"] - MARKET_BENCHMARK
    save_table(t, "02_bagged_vs_single")
    print(t.round(4).to_string(index=False))

    fig, ax = new_fig(9, 5)
    x = np.arange(len(t))
    ax.bar(x, t["auc"], color=[SERIES[0], SERIES[1], INK_MUTED], width=0.6)
    ax.axhline(MARKET_BENCHMARK, color=DIVERGING_NEG, ls="--", lw=1.5,
               label=f"market benchmark {MARKET_BENCHMARK}")
    ax.set_xticks(x); ax.set_xticklabels(t["model"], fontsize=8)
    ax.set_ylim(0.6, 0.78); ax.set_ylabel("2025 holdout AUC")
    ax.set_title("Bagging vs a single fit — both still under the market")
    ax.legend()
    savefig(fig, "02_bagged_vs_single")


@section("03", "Per-game uncertainty")
def s03(ctx: Ctx):
    res = get_boot(ctx)
    _, test = ctx.cache["split"]
    w = res.ci_hi - res.ci_lo
    t = pd.DataFrame([{"mean_ci_width": w.mean(), "median": np.median(w),
                       "p10": np.percentile(w, 10), "p90": np.percentile(w, 90),
                       "min": w.min(), "max": w.max(),
                       "mean_half_width": w.mean() / 2}])
    save_table(t, "03_ci_widths")
    print(t.round(4).to_string(index=False))
    print(f"\nA pick needs the CI to clear the market probability entirely, so the "
          f"edge must exceed roughly {w.mean() / 2:.3f} — that is the bar the "
          f"filter sets.")

    fig, ax = new_fig(9, 5)
    ax.hist(w, bins=40, color=SERIES[0])
    ax.axvline(w.mean(), color=DIVERGING_NEG, lw=2,
               label=f"mean {w.mean():.3f}")
    ax.set_xlabel("95% CI width on the model's win probability")
    ax.set_ylabel("games")
    ax.set_title("Per-game uncertainty (2025 holdout)")
    ax.legend()
    savefig(fig, "03_ci_widths")


@section("04", "Feature-importance stability")
def s04(ctx: Ctx):
    res = get_boot(ctx)
    t = importance_stability(res, top_n=5)
    save_table(t, "04_importance_stability")
    print(t.head(15).round(4).to_string(index=False))
    print("\nroster_continuity_score was retired in Model_1 rev 3 and is absent "
          "from this feature set, so the request's question about it is already "
          "settled — it is not re-tested here.")

    top = t.head(12).iloc[::-1]
    fig, ax = new_fig(9.5, 6.5)
    ax.barh(top["feature"], top["pct_in_top5"], color=SERIES[0], height=0.68)
    ax.set_xlabel("% of bootstrap resamples where the feature ranks top-5")
    ax.set_title("Which features are consistently important, not luckily important")
    savefig(fig, "04_importance_stability")


@section("05", "De-vig method comparison")
def s05(ctx: Ctx):
    from src.db import get_engine
    from sklearn.metrics import brier_score_loss, log_loss

    d = pd.read_sql("""
        SELECT g.game_id, g.result, b.home_moneyline, b.away_moneyline
        FROM games g JOIN betting_lines b USING (game_id)
        WHERE g.game_type='REG' AND g.result IS NOT NULL AND g.result <> 0
          AND b.home_moneyline IS NOT NULL
    """, get_engine())
    y = (d["result"] > 0).astype(int).to_numpy()
    ovr = overround(implied_prob(d["home_moneyline"]), implied_prob(d["away_moneyline"]))

    rows = []
    for name in ("proportional", "shin"):
        ph, pa = devig(d["home_moneyline"], d["away_moneyline"], method=name)
        assert np.allclose(ph + pa, 1.0, atol=1e-9)
        rows.append({"method": name, "brier": brier_score_loss(y, ph),
                     "log_loss": log_loss(y, np.clip(ph, 1e-9, 1 - 1e-9))})
    raw = np.clip(implied_prob(d["home_moneyline"]), 1e-9, 1 - 1e-9)
    rows.append({"method": "RAW (skipping de-vig)", "brier": brier_score_loss(y, raw),
                 "log_loss": log_loss(y, raw)})
    t = pd.DataFrame(rows)
    t["mean_overround"] = ovr.mean()
    save_table(t, "05_devig")
    print(f"games {len(d):,}; mean overround {ovr.mean():.4f} "
          f"({(ovr.mean() - 1) * 100:.2f}% book margin)")
    print(t.round(5).to_string(index=False))
    print(f"\nDefault in use: {DEVIG_METHOD}")


@section("06", "Market agreement, re-checked on the bagged model")
def s06(ctx: Ctx):
    res = get_boot(ctx)
    _, test = ctx.cache["split"]
    y = test["won"].to_numpy()
    mk = test["team_ml_implied_prob"].to_numpy()
    agree = (res.point > 0.5) == (mk > 0.5)
    t = pd.DataFrame([
        {"case": "bagged model agrees with market", "n": int(agree.sum()),
         "share": float(agree.mean()),
         "accuracy": accuracy_score(y[agree], (res.point[agree] > .5).astype(int))},
        {"case": "bagged model DISAGREES", "n": int((~agree).sum()),
         "share": float((~agree).mean()),
         "accuracy": (accuracy_score(y[~agree], (res.point[~agree] > .5).astype(int))
                      if (~agree).sum() else np.nan)},
    ])
    save_table(t, "06_agreement")
    print(t.round(4).to_string(index=False))
    print(f"\nModel_1 single-fit baseline: 539/542 agree (99.45%).")
    print(f"Bagged: {int(agree.sum())}/{len(agree)} ({agree.mean() * 100:.2f}%).")


@section("07", "Edge-filter yield on the 2025 holdout")
def s07(ctx: Ctx):
    res = get_boot(ctx)
    _, test = ctx.cache["split"]
    from src.db import get_engine

    ml = pd.read_sql("SELECT game_id, home_moneyline, away_moneyline, spread_line "
                     "FROM betting_lines", get_engine())
    d = test.copy()
    d["model_prob"], d["ci_lo"], d["ci_hi"] = res.point, res.ci_lo, res.ci_hi
    d = d.merge(ml, on="game_id", how="left", suffixes=("", "_bl"))
    d["team_moneyline"] = np.where(d["is_home"], d["home_moneyline"], d["away_moneyline"])
    d["opp_moneyline"] = np.where(d["is_home"], d["away_moneyline"], d["home_moneyline"])
    d = build_edge_frame(d, method=DEVIG_METHOD)

    picks = d[d["qualifies"]]
    t = pd.DataFrame([{
        "team_games": len(d), "with_odds": int(d["team_moneyline"].notna().sum()),
        "passed_edge_test": int(d["edge_test_passed"].sum()),
        "passed_history_gate": int(d["history_ok"].sum()),
        "qualifying_picks": len(picks),
        "pick_rate_pct": round(len(picks) / len(d) * 100, 3),
        "pick_accuracy": round(float(picks["won"].mean()), 4) if len(picks) else None,
        "mean_edge": round(float(picks["edge"].mean()), 4) if len(picks) else None,
    }])
    save_table(t, "07_edge_yield")
    print(t.to_string(index=False))

    if len(picks):
        from src.odds import american_to_decimal
        profit = sum((american_to_decimal(r.team_moneyline) - 1) * 10 if r.won == 1
                     else -10 for r in picks.itertuples())
        print(f"\npicks: {len(picks)}  record "
              f"{int(picks['won'].sum())}-{int((1 - picks['won']).sum())}  "
              f"net ${profit:+.2f} at flat $10")
        print(picks[["team", "opponent", "week", "model_prob", "market_prob_devig",
                     "edge", "ev_dollars", "won"]].round(4).to_string(index=False))
    else:
        print("\nNo qualifying picks on the entire 2025 season.")

    fig, ax = new_fig(9, 5)
    ax.scatter(d["market_prob_devig"], d["model_prob"], s=14, alpha=0.5,
               color=SERIES[0], label="all team-games")
    if len(picks):
        ax.scatter(picks["market_prob_devig"], picks["model_prob"], s=45,
                   color=DIVERGING_NEG, label="qualifying picks", zorder=5)
    ax.plot([0, 1], [0, 1], ls="--", color=INK_MUTED, lw=1)
    ax.set_xlabel("de-vigged market probability")
    ax.set_ylabel("bagged model probability")
    ax.set_title("Model vs market — the diagonal is agreement")
    ax.legend()
    savefig(fig, "07_edge_yield")


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
    print("building panel...")
    ctx = Ctx(panel=build_team_game_panel())
    failures = run_sections(keys, ctx)
    if failures:
        print(f"\nFAILED sections: {failures}")
        return 1
    print("\nAll sections complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
