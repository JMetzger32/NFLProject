"""Cumulative tracking: model quality, betting performance, and the open hypotheses.

Two things are measured separately on purpose. A model can be accurate and still
lose money, or be mediocre and profit — accuracy and P&L are not the same question.

Small-sample discipline is enforced rather than suggested: ROI is suppressed below
MIN_PICKS_FOR_ROI, because one lucky week swings it wildly and a headline number
invites exactly the wrong conclusion.

Usage:
    python Betting/track.py
    python Betting/track.py --season 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

from src.db import get_engine  # noqa: E402
from src.model import reliability_stats  # noqa: E402

MIN_PICKS_FOR_ROI = 50
OUT_DIR = PROJECT_ROOT / "Betting" / "output"

# Confidence bands mirror the Model_1 report so the numbers stay comparable.
BANDS = [(0.0, 0.05, "coin-flip (<.05)"), (0.05, 0.10, "lean (.05-.10)"),
         (0.10, 0.20, "confident (.10-.20)"), (0.20, 1.0, "strong (>.20)")]


def load(season: int | None) -> pd.DataFrame:
    q = """
        SELECT p.*, r.won, r.payout, r.bankroll_after, g.gameday
        FROM predictions p
        JOIN pick_results r USING (game_id, team)
        JOIN games g ON g.game_id = p.game_id
    """
    if season:
        q += f" WHERE p.season = {int(season)}"
    q += " ORDER BY g.gameday, p.game_id, p.team"
    d = pd.read_sql(q, get_engine())
    for c in ("model_prob", "market_prob_devig", "edge", "payout", "stake",
              "bankroll_after", "ev_dollars"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def model_vs_market(d: pd.DataFrame) -> pd.DataFrame:
    """Same metrics, same games, model against market. Apples to apples."""
    ok = d.dropna(subset=["model_prob", "market_prob_devig", "won"])
    if len(ok) < 10:
        return pd.DataFrame()
    y = ok["won"].to_numpy()
    rows = []
    for label, p in (("model", ok["model_prob"].to_numpy()),
                     ("market", ok["market_prob_devig"].to_numpy())):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        rel = reliability_stats(y, p)
        rows.append({"who": label, "n": len(y),
                     "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                     "accuracy": accuracy_score(y, (p > 0.5).astype(int)),
                     "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
                     "ece": rel["ece"]})
    return pd.DataFrame(rows)


def by_band(d: pd.DataFrame) -> pd.DataFrame:
    ok = d.dropna(subset=["model_prob", "market_prob_devig", "won"])
    if ok.empty:
        return pd.DataFrame()
    conf = (ok["model_prob"] - 0.5).abs()
    rows = []
    for lo, hi, lbl in BANDS:
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 5:
            continue
        s = ok[m]
        rows.append({
            "band": lbl, "n": int(m.sum()), "share": round(m.mean(), 4),
            "model_accuracy": accuracy_score(s["won"], (s["model_prob"] > .5).astype(int)),
            "market_accuracy": accuracy_score(s["won"], (s["market_prob_devig"] > .5).astype(int)),
        })
    return pd.DataFrame(rows)


def agreement(d: pd.DataFrame) -> pd.DataFrame:
    """Drift against Model_1's 99.45% baseline.

    A large move means either the model found something new or something upstream
    broke — and the second is far more likely.
    """
    ok = d.dropna(subset=["model_prob", "market_prob_devig"])
    if ok.empty:
        return pd.DataFrame()
    agree = (ok["model_prob"] > 0.5) == (ok["market_prob_devig"] > 0.5)
    return pd.DataFrame([{
        "n": len(ok), "agreement_rate": round(float(agree.mean()), 4),
        "model_1_baseline": 0.9945,
        "drift": round(float(agree.mean()) - 0.9945, 4),
        "disagreements": int((~agree).sum()),
    }])


def betting_performance(d: pd.DataFrame) -> pd.DataFrame:
    picks = d[d["is_pick"] == True].dropna(subset=["won"])  # noqa: E712
    n = len(picks)
    if n == 0:
        return pd.DataFrame([{"picks": 0, "status": "no picks graded yet"}])
    staked = picks["stake"].sum()
    profit = picks["payout"].sum()
    row = {
        "picks": n,
        "record": f"{int(picks['won'].sum())}-{int((1 - picks['won']).sum())}",
        "win_rate": round(float(picks["won"].mean()), 4),
        "total_staked": round(float(staked), 2),
        "net_profit": round(float(profit), 2),
        "units": round(float(profit / 10.0), 2),
        "bankroll": round(float(picks["bankroll_after"].iloc[-1]), 2),
    }
    if n < MIN_PICKS_FOR_ROI:
        row["roi"] = None
        row["roi_status"] = (f"SUPPRESSED — {n} picks, need {MIN_PICKS_FOR_ROI}. "
                             "One week swings this wildly at this sample size.")
    else:
        row["roi"] = round(float(profit / staked), 4)
        row["roi_status"] = "interpretable"
    return pd.DataFrame([row])


def regime_hypothesis(d: pd.DataFrame) -> pd.DataFrame:
    """OPEN HYPOTHESIS, pre-registered — not an established finding.

    Model_1 tested the early/late market-efficiency gap and got a bootstrap CI of
    [-0.042, +0.127], which includes zero. It is tracked here in case more data
    changes that. The threshold is stated in advance: the bootstrap CI on the
    EARLY-minus-LATE difference must exclude zero.
    """
    ok = d.dropna(subset=["model_prob", "market_prob_devig", "won"])
    rows = []
    for lbl, m in (("EARLY (wk1-8)", ok["week"] <= 8), ("LATE (wk9-18)", ok["week"] >= 9)):
        s = ok[m]
        if len(s) < 20 or s["won"].nunique() < 2:
            rows.append({"regime": lbl, "n": len(s), "market_auc": np.nan,
                         "model_auc": np.nan})
            continue
        rows.append({"regime": lbl, "n": len(s),
                     "market_auc": roc_auc_score(s["won"], s["market_prob_devig"]),
                     "model_auc": roc_auc_score(s["won"], s["model_prob"])})
    out = pd.DataFrame(rows)
    out["status"] = "OPEN HYPOTHESIS — threshold: bootstrap CI on the gap excludes 0"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    args = ap.parse_args()

    d = load(args.season)
    if d.empty:
        print("No graded predictions yet. Run weekly_picks.py then grade_week.py.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    B = "=" * 78
    print(B); print(f"TRACKING — {len(d):,} graded team-games"
                    + (f", season {args.season}" if args.season else "")); print(B)

    for title, fn, name in (
        ("MODEL vs MARKET (same games)", model_vs_market, "track_model_vs_market"),
        ("BY CONFIDENCE BAND", by_band, "track_by_band"),
        ("MARKET AGREEMENT", agreement, "track_agreement"),
        ("BETTING PERFORMANCE", betting_performance, "track_betting"),
        ("REGIME HYPOTHESIS (open)", regime_hypothesis, "track_regime"),
    ):
        t = fn(d)
        print(f"\n--- {title}")
        print("  (insufficient data)" if t.empty else t.round(4).to_string(index=False))
        if not t.empty:
            t.to_csv(OUT_DIR / f"{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
