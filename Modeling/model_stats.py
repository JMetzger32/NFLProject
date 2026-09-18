"""Complete statistics dump for the Model_1 production model (single full-season).

Refits using the hyperparameters already selected by walk-forward CV (read from
Modeling/Model_1/data/03_primary.csv), so this is fast — no re-tuning. Reports the
full classification suite, probability quality, confusion matrices, coefficients and
feature importances, and a betting-relevant confidence-band breakdown.

Usage:
    python Modeling/model_stats.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features import build_team_game_panel  # noqa: E402
from src.model import TEST_SEASON, make_gbm, make_lr, make_rf, reliability_stats  # noqa: E402

DATA = PROJECT_ROOT / "Modeling" / "Model_1" / "data"
pd.set_option("display.width", 220)


def full_metrics(name: str, y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    yhat = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rel = reliability_stats(y, p)
    return {
        "model": name, "n": len(y),
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall_sensitivity": recall_score(y, yhat, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "log_loss": rel["log_loss"], "brier": rel["brier"], "ece": rel["ece"],
        "mean_pred": rel["mean_pred"], "base_rate": rel["base_rate"],
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def main() -> int:
    panel = build_team_game_panel()
    coll = pd.read_csv(DATA / "02_collinearity_labeled.csv")
    prim = pd.read_csv(DATA / "03_primary.csv")

    df = panel[(panel["week"] >= 1) & (panel["week"] <= 18)]
    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    feats = sorted(coll[coll.kept_as_representative]["feature"])

    strengths = {}
    for f in feats:
        d = train[[f, "won"]].dropna()
        if len(d) > 100 and d[f].nunique() > 2:
            strengths[f] = abs(roc_auc_score(d["won"], d[f]) - 0.5)
    top10 = sorted(strengths, key=strengths.get, reverse=True)[:10]

    makers = {"LR": lambda: make_lr(top10), "GBM": make_gbm, "RF": make_rf}
    y = test["won"].to_numpy()
    rows, coef_rows, probs = [], [], {}

    for name in ("LR", "GBM", "RF"):
        params = ast.literal_eval(prim[prim.model == name].iloc[0]["params"])
        pipe = makers[name]()
        pipe.set_params(**params)
        pipe.fit(train[feats], train["won"])
        p = pipe.predict_proba(test[feats])[:, 1]
        probs[name] = p
        m = full_metrics(name, y, p); m["role"] = "single"; m["params"] = str(params)
        rows.append(m)

        clf = pipe.named_steps["clf"]
        if name == "LR":
            names = list(pipe.named_steps["thresh"].transform(train[feats]).columns)
            for nm, c in zip(names, clf.coef_[0]):
                coef_rows.append({"model": "LR", "feature": nm,
                                  "value": float(c), "abs": abs(float(c))})
        elif name == "RF":
            for nm, imp in zip(feats, clf.feature_importances_):
                coef_rows.append({"model": "RF", "feature": nm,
                                  "value": float(imp), "abs": float(imp)})

    avg = np.mean([probs[k] for k in ("LR", "GBM", "RF")], axis=0)
    m = full_metrics("SimpleAvg (PRODUCTION)", y, avg); m["role"] = "PRODUCTION"; m["params"] = ""
    rows.append(m)

    sc = StandardScaler().fit(train[["team_spread"]])
    mk = LogisticRegression(max_iter=1000).fit(sc.transform(train[["team_spread"]]), train["won"])
    mp = mk.predict_proba(sc.transform(test[["team_spread"]]))[:, 1]
    m = full_metrics("MARKET (spread only)", y, mp); m["role"] = "benchmark"; m["params"] = ""
    rows.append(m)

    # Coin-flip reference, so every metric has an absolute floor.
    m = full_metrics("Coin flip (p=0.5)", y, np.full(len(y), 0.5)); m["role"] = "floor"; m["params"] = ""
    rows.append(m)

    res = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)

    # Confidence-band breakdown, production vs market.
    conf = np.abs(avg - 0.5)
    bands = []
    for lo, hi, lbl in [(0.0, 0.05, "coin-flip (<.05)"), (0.05, 0.10, "lean (.05-.10)"),
                        (0.10, 0.20, "confident (.10-.20)"), (0.20, 1.0, "strong (>.20)")]:
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() < 10:
            continue
        bands.append({"band": lbl, "n": int(msk.sum()), "share": msk.mean(),
                      "model_accuracy": accuracy_score(y[msk], (avg[msk] > .5).astype(int)),
                      "market_accuracy": accuracy_score(y[msk], (mp[msk] > .5).astype(int))})
    bands = pd.DataFrame(bands)

    # Agreement between production model and market.
    agree = ((avg > 0.5) == (mp > 0.5))
    agree_tbl = pd.DataFrame([
        {"case": "model agrees with market", "n": int(agree.sum()),
         "share": float(agree.mean()),
         "accuracy": accuracy_score(y[agree], (avg[agree] > .5).astype(int))},
        {"case": "model DISAGREES with market", "n": int((~agree).sum()),
         "share": float((~agree).mean()),
         "accuracy": accuracy_score(y[~agree], (avg[~agree] > .5).astype(int))},
    ])

    res.to_csv(DATA / "08_full_metrics.csv", index=False)
    coefs.to_csv(DATA / "08_coefficients.csv", index=False)
    bands.to_csv(DATA / "08_confidence_bands.csv", index=False)
    agree_tbl.to_csv(DATA / "08_model_vs_market_agreement.csv", index=False)

    B = "=" * 104
    print(B); print("CLASSIFICATION METRICS — 2025 holdout, threshold 0.5"); print(B)
    print(res[["model", "n", "auc", "accuracy", "precision", "recall_sensitivity",
               "specificity", "f1"]].round(4).to_string(index=False))
    print("\n" + B); print("PROBABILITY QUALITY — lower is better"); print(B)
    print(res[["model", "log_loss", "brier", "ece", "mean_pred", "base_rate"]].round(4).to_string(index=False))
    print("\n" + B); print("CONFUSION MATRICES"); print(B)
    print(res[["model", "TP", "FP", "TN", "FN"]].to_string(index=False))
    print("\n" + B); print("CROSS-VALIDATION (walk-forward) + HOLDOUT CI"); print(B)
    print(prim[["model", "role", "cv_val_auc", "cv_train_auc", "overfit_gap",
                "holdout_auc", "auc_ci_lo", "auc_ci_hi", "lift_vs_market"]].round(4).to_string(index=False))
    print("\n" + B); print("ACCURACY BY CONFIDENCE BAND — production vs market"); print(B)
    print(bands.round(4).to_string(index=False))
    print("\n" + B); print("WHERE THE MODEL DISAGREES WITH THE MARKET"); print(B)
    print(agree_tbl.round(4).to_string(index=False))
    print("\n" + B); print("LR COEFFICIENTS — non-zero, by magnitude"); print(B)
    lr = coefs[coefs.model == "LR"]
    print(f"{int((lr['abs'] > 1e-8).sum())} of {len(lr)} non-zero")
    print(lr[lr["abs"] > 1e-8].sort_values("abs", ascending=False)[
        ["feature", "value"]].round(4).to_string(index=False))
    print("\n" + B); print("RANDOM FOREST FEATURE IMPORTANCE — top 15"); print(B)
    rf = coefs[coefs.model == "RF"].sort_values("abs", ascending=False)
    print(rf.head(15)[["feature", "value"]].round(4).to_string(index=False))
    print(f"\nteam_spread share of total RF importance: "
          f"{rf[rf.feature == 'team_spread']['value'].iloc[0] / rf['value'].sum():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
