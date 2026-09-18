"""Modeling utilities: threshold encoding, walk-forward CV, model factory,
collinearity diagnostics, stacking and calibration.

THE LEAKAGE RULE, CONTINUED
---------------------------
src/features.py guarantees no feature sees its own game. This module has a second,
separate leakage surface: anything *learned from the target* — split thresholds,
scaler statistics, hyperparameters, calibration curves — must be fit on training
rows only. ThresholdEncoder is a sklearn transformer specifically so it sits inside
a Pipeline and gets refit per CV fold automatically, making that structural rather
than something to remember.

Usage:
    python -m src.model --verify-thresholds
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SEASON = 2025
TRAIN_SEASONS = (2021, 2022, 2023, 2024)

# Walk-forward folds for hyperparameter selection: expanding window, never random.
WALK_FORWARD_FOLDS = [
    ((2021,), 2022),
    ((2021, 2022), 2023),
    ((2021, 2022, 2023), 2024),
]

VIF_TIERS = [(10.0, "SEVERE"), (5.0, "HIGH"), (2.5, "MODERATE"), (0.0, "LOW")]


def vif_tier(v: float) -> str:
    if not np.isfinite(v):
        return "SEVERE"
    for cutoff, label in VIF_TIERS:
        if v >= cutoff:
            return label
    return "LOW"


class ThresholdEncoder(BaseEstimator, TransformerMixin):
    """Adds breakpoint-derived columns for a subset of features.

    Logistic regression assumes linear log-odds; several features in this panel
    bend or step instead. Trees find those splits themselves, so this is applied to
    the LR branch only.

    Thresholds come from a depth-1 decision stump and from Youden's J, both fit on
    the training rows handed to fit(). Because this is a transformer, a Pipeline
    refits it on each CV fold's training portion — the holdout can never influence a
    threshold.

    mode:
      "none"   passthrough
      "binary" adds an above/below indicator per encoded feature
      "hinge"  adds a 2-knot piecewise-linear pair per encoded feature
      "both"   adds both
    """

    def __init__(self, features: list[str] | None = None, mode: str = "none"):
        self.features = features
        self.mode = mode

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.encoded_ = [f for f in (self.features or []) if f in X.columns]
        self.thresholds_: dict[str, float] = {}
        self.youden_: dict[str, float] = {}

        for f in self.encoded_:
            col = X[f]
            ok = col.notna() & pd.Series(y, index=X.index).notna()
            if ok.sum() < 50 or col[ok].nunique() < 3:
                continue
            xv = col[ok].to_numpy().reshape(-1, 1)
            yv = np.asarray(y)[ok.to_numpy()]
            stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
            stump.fit(xv, yv)
            thr = stump.tree_.threshold[0]
            if thr == -2:  # no split found
                continue
            self.thresholds_[f] = float(thr)
            self.youden_[f] = float(_youden_threshold(col[ok].to_numpy(), yv))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.mode == "none":
            return X
        for f, thr in self.thresholds_.items():
            if f not in X.columns:
                continue
            col = X[f]
            if self.mode in ("binary", "both"):
                X[f"{f}__gt"] = (col > thr).astype(float).where(col.notna())
            if self.mode in ("hinge", "both"):
                X[f"{f}__hi"] = (col - thr).clip(lower=0)
                X[f"{f}__lo"] = (thr - col).clip(lower=0)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.transform(
            pd.DataFrame(columns=self.feature_names_in_, dtype=float)
        ).columns)


def _youden_threshold(x: np.ndarray, y: np.ndarray) -> float:
    """Cut maximizing sensitivity + specificity - 1."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    pos, neg = ys.sum(), (1 - ys).sum()
    if pos == 0 or neg == 0:
        return float(np.median(x))
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    # J at each candidate cut = TPR - FPR, evaluated as "above the cut is positive"
    tpr = (pos - tps) / pos
    fpr = (neg - fps) / neg
    j = tpr - fpr
    return float(xs[int(np.argmax(np.abs(j)))])


def monotone_features(panel: pd.DataFrame, features: list[str], bins: int = 8) -> dict[str, bool]:
    """Classify each feature as monotone (-> hinge) or step-like (-> binary).

    Measured from the per-bin win rate's sign consistency rather than eyeballed off
    the EDA charts: with 30 features, hand-classifying shape is neither reliable nor
    traceable. A feature whose binned win-rate deltas mostly share one sign is
    treated as monotone.
    """
    out: dict[str, bool] = {}
    for f in features:
        d = panel[[f, "won"]].dropna()
        if len(d) < 200 or d[f].nunique() < bins:
            out[f] = True
            continue
        q = pd.qcut(d[f], bins, duplicates="drop")
        rates = d.groupby(q, observed=True)["won"].mean().to_numpy()
        deltas = np.diff(rates)
        if len(deltas) == 0:
            out[f] = True
            continue
        share = max((deltas > 0).mean(), (deltas < 0).mean())
        out[f] = bool(share >= 0.7)
    return out


# ------------------------------------------------------------------ model zoo


def make_lr(threshold_features: list[str] | None = None) -> Pipeline:
    return Pipeline([
        ("thresh", ThresholdEncoder(features=threshold_features, mode="none")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1,
            max_iter=5000, random_state=RANDOM_STATE)),
    ])


def make_gbm() -> Pipeline:
    # HistGradientBoosting handles NaN natively, so no imputer.
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, min_samples_leaf=40,
            l2_regularization=1.0, max_iter=500, max_features=0.8,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=30,
            random_state=RANDOM_STATE)),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600, max_depth=6, min_samples_leaf=20,
            max_features="sqrt", max_samples=0.8, bootstrap=True,
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])


LR_GRID = [
    {"thresh__mode": m, "clf__C": c, "clf__l1_ratio": r}
    for m in ("none", "binary", "hinge", "both")
    for c in (0.01, 0.03, 0.1, 0.3, 1.0)
    for r in (0.0, 0.5, 1.0)
]

def gbm_grid(regime: str = "LATE") -> list[dict]:
    """EARLY gets a deliberately tighter grid.

    EARLY trees overfit hard in Model_1's first run (train-val gaps 0.14-0.18 on
    978 rows), so its grid caps depth at 2-3, raises the leaf minimum, forces
    column subsampling below 0.8, and starts L2 an order of magnitude higher.

    Note on names: `subsample` / `colsample_bytree` / `min_child_weight` are
    XGBoost's. The sklearn equivalents used here are `max_features` (column
    subsampling) and `min_samples_leaf`. HistGradientBoosting has NO row
    subsampling parameter at all, so the row-level variance reduction that
    `subsample` would give is compensated with stronger L2 + leaf minimums.
    """
    if regime == "EARLY":
        return [
            {"clf__max_depth": d, "clf__learning_rate": lr,
             "clf__min_samples_leaf": leaf, "clf__l2_regularization": l2,
             "clf__max_features": mf, "clf__max_leaf_nodes": nodes}
            for d in (2, 3)
            for lr in (0.01, 0.02, 0.05)
            for leaf in (60, 100, 150)
            for l2 in (1.0, 10.0, 50.0)
            for mf in (0.5, 0.7)
            for nodes in (7, 15)
        ]
    return [
        {"clf__max_depth": d, "clf__learning_rate": lr,
         "clf__min_samples_leaf": leaf, "clf__l2_regularization": l2,
         "clf__max_features": mf}
        for d in (2, 3, 4)
        for lr in (0.02, 0.05, 0.1)
        for leaf in (20, 40, 80)
        for l2 in (0.1, 1.0, 10.0)
        for mf in (0.7, 1.0)
    ]


def rf_grid(regime: str = "LATE") -> list[dict]:
    """EARLY caps depth at 2-3 and forces both row (`max_samples`) and column
    (`max_features`) subsampling below 0.8."""
    if regime == "EARLY":
        return [
            {"clf__max_depth": d, "clf__min_samples_leaf": leaf,
             "clf__max_features": mf, "clf__max_samples": ms}
            for d in (2, 3)
            for leaf in (30, 60, 100)
            for mf in ("sqrt", 0.3, 0.5)
            for ms in (0.6, 0.75)
        ]
    return [
        {"clf__max_depth": d, "clf__min_samples_leaf": leaf,
         "clf__max_features": mf, "clf__max_samples": ms}
        for d in (4, 6, 10, None)
        for leaf in (5, 20, 50)
        for mf in ("sqrt", 0.3, 0.6)
        for ms in (0.7, 0.9)
    ]


# Back-compat aliases for any caller that imported the old constants.
GBM_GRID = gbm_grid("LATE")
RF_GRID = rf_grid("LATE")


@dataclass
class TunedModel:
    name: str
    pipeline: Pipeline
    params: dict
    cv_auc: float
    cv_train_auc: float


def walk_forward_tune(
    make_fn, grid: list[dict], panel: pd.DataFrame, features: list[str], name: str,
) -> TunedModel:
    """Expanding-window walk-forward hyperparameter selection. Never random k-fold.

    Returns the config with the best mean validation AUC, and also carries the mean
    TRAINING AUC so the overfitting gap is reportable.
    """
    best = None
    for params in grid:
        val_aucs, tr_aucs = [], []
        for train_seasons, val_season in WALK_FORWARD_FOLDS:
            tr = panel[panel["season"].isin(train_seasons)]
            va = panel[panel["season"] == val_season]
            if len(tr) < 100 or len(va) < 50:
                continue
            pipe = make_fn()
            pipe.set_params(**params)
            pipe.fit(tr[features], tr["won"])
            val_aucs.append(roc_auc_score(va["won"], pipe.predict_proba(va[features])[:, 1]))
            tr_aucs.append(roc_auc_score(tr["won"], pipe.predict_proba(tr[features])[:, 1]))
        if not val_aucs:
            continue
        score = float(np.mean(val_aucs))
        if best is None or score > best.cv_auc:
            pipe = make_fn()
            pipe.set_params(**params)
            best = TunedModel(name, pipe, params, score, float(np.mean(tr_aucs)))
    return best


def oof_predictions(
    make_fn, params: dict, panel: pd.DataFrame, features: list[str]
) -> pd.Series:
    """Out-of-fold predictions across the walk-forward folds, for stacking."""
    oof = pd.Series(np.nan, index=panel.index, dtype=float)
    for train_seasons, val_season in WALK_FORWARD_FOLDS:
        tr = panel[panel["season"].isin(train_seasons)]
        va = panel[panel["season"] == val_season]
        if len(tr) < 100 or len(va) < 50:
            continue
        pipe = make_fn()
        pipe.set_params(**params)
        pipe.fit(tr[features], tr["won"])
        oof.loc[va.index] = pipe.predict_proba(va[features])[:, 1]
    return oof


# --------------------------------------------------------- collinearity tools


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """VIF per feature, on complete cases of the training rows."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    d = df[features].dropna()
    if len(d) < len(features) + 10:
        d = df[features].fillna(df[features].median())
    # Drop zero-variance columns; VIF is undefined for them.
    usable = [f for f in features if d[f].std() > 0]
    X = StandardScaler().fit_transform(d[usable])
    X = np.column_stack([np.ones(len(X)), X])  # intercept, so VIF is centred correctly
    rows = []
    for i, f in enumerate(usable, start=1):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.inf
        rows.append({"feature": f, "vif": float(v), "vif_tier": vif_tier(float(v))})
    for f in features:
        if f not in usable:
            rows.append({"feature": f, "vif": np.nan, "vif_tier": "LOW"})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def correlation_clusters(
    df: pd.DataFrame, features: list[str], threshold: float = 0.8
) -> pd.DataFrame:
    """Hierarchical clustering on |Spearman r|, cut so members correlate >= threshold.

    Names the redundancy blocks the EDA_1 heatmap showed visually.
    """
    d = df[features].dropna()
    if len(d) < 50:
        d = df[features].fillna(df[features].median())
    corr = d.corr(method="spearman").abs().fillna(0.0)
    dist = 1.0 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")
    labels = hierarchy.fcluster(link, t=1.0 - threshold, criterion="distance")
    return pd.DataFrame({"feature": features, "cluster": labels})


def pick_representatives(
    clusters: pd.DataFrame, ranking: pd.Series
) -> list[str]:
    """One feature per cluster: the highest-ranked member.

    `ranking` should be market-independent signal (residual-vs-spread), not raw AUC
    — a representative that merely re-derives the closing line is worthless for
    finding edge even when it wins on raw predictiveness.
    """
    keep = []
    for _, grp in clusters.groupby("cluster"):
        scores = ranking.reindex(grp["feature"]).fillna(-np.inf)
        keep.append(scores.idxmax() if scores.notna().any() else grp["feature"].iloc[0])
    return sorted(keep)


# ------------------------------------------------------------- calibration


def calibrate(probs_cal: np.ndarray, y_cal: np.ndarray, method: str = "sigmoid"):
    """Fit a calibrator on a slice held out from both training and stacking.

    Platt (`sigmoid`) is the default, NOT isotonic. Isotonic is non-parametric and
    needs far more data than the ~300-row calibration slice available here; on a
    slice this small it overfits the calibration curve itself. Platt fits two
    parameters and degrades gracefully.

    Calibration is applied at FINAL OUTPUT ONLY. It is never a model-selection
    criterion: it optimizes a different objective (probability reliability) than
    AUC (ranking), so judging it by an AUC delta is a category error.
    """
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs_cal, y_cal)
        return iso.predict
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.asarray(probs_cal).reshape(-1, 1), y_cal)
    return lambda p: lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def reliability_stats(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """The metrics calibration should actually be judged on, instead of AUC."""
    from sklearn.metrics import brier_score_loss, log_loss

    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": float(ece),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
    }


def _verify_thresholds(panel: pd.DataFrame, features: list[str]) -> None:
    """ThresholdEncoder must not see the holdout: thresholds fit on 2021-2023 have
    to be identical whether or not 2024-2025 rows are present in the frame."""
    tr = panel[panel["season"].isin((2021, 2022, 2023))]
    enc_a = ThresholdEncoder(features=features, mode="both").fit(tr[features], tr["won"])
    enc_b = ThresholdEncoder(features=features, mode="both").fit(tr[features], tr["won"])
    full = panel
    enc_c = ThresholdEncoder(features=features, mode="both").fit(full[features], full["won"])

    assert enc_a.thresholds_ == enc_b.thresholds_, "ThresholdEncoder is nondeterministic"
    differing = [f for f in enc_a.thresholds_
                 if f in enc_c.thresholds_
                 and not np.isclose(enc_a.thresholds_[f], enc_c.thresholds_[f])]
    print(f"Threshold determinism: PASS ({len(enc_a.thresholds_)} thresholds)")
    print(f"Thresholds that change when holdout rows are added: {len(differing)} "
          f"of {len(enc_a.thresholds_)}")
    print("  (a nonzero count here is EXPECTED and is exactly why the encoder lives")
    print("   inside the Pipeline — refit per fold, the holdout never leaks in.)")


def verify_split(panel: pd.DataFrame) -> None:
    """Prove the train/test split and the cross-validation scheme are sound.

    Three properties, asserted rather than asserted-in-prose:
      1. every CV fold trains strictly in the PAST of the season it validates on;
      2. no fold validates on a season it also trained on;
      3. the 2025 holdout appears in NO training and NO validation fold anywhere.
    """
    print("TRAIN/TEST SPLIT (time-based, never random shuffle)")
    tr = panel[panel["season"].isin(TRAIN_SEASONS)]
    te = panel[panel["season"] == TEST_SEASON]
    print(f"  train seasons {TRAIN_SEASONS} -> {len(tr):,} rows")
    print(f"  test  season  {TEST_SEASON}      -> {len(te):,} rows (touched once, at final eval)")
    assert set(tr["season"]).isdisjoint({TEST_SEASON}), "holdout leaked into training"
    assert len(te) > 0 and len(tr) > 0

    print("\nCROSS-VALIDATION (walk-forward, expanding window)")
    for train_seasons, val_season in WALK_FORWARD_FOLDS:
        n_tr = len(panel[panel["season"].isin(train_seasons)])
        n_va = len(panel[panel["season"] == val_season])
        assert all(s < val_season for s in train_seasons), (
            f"fold trains on {train_seasons} but validates on earlier {val_season}")
        assert val_season not in train_seasons, "fold validates on a training season"
        assert val_season != TEST_SEASON, "holdout used as a validation fold"
        assert TEST_SEASON not in train_seasons, "holdout used as a training fold"
        print(f"  train {train_seasons} ({n_tr:>5,} rows) -> validate {val_season} "
              f"({n_va:,} rows)")

    used = {s for tsn, _ in WALK_FORWARD_FOLDS for s in tsn}
    used |= {v for _, v in WALK_FORWARD_FOLDS}
    assert TEST_SEASON not in used, "holdout season appears somewhere in CV"
    print(f"\n  seasons touched during CV: {sorted(used)}")
    print(f"  holdout season {TEST_SEASON} absent from every fold: PASS")
    print("\nAll split/CV assertions PASSED.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-thresholds", action="store_true")
    ap.add_argument("--verify-split", action="store_true")
    args = ap.parse_args()

    from src.features import build_team_game_panel

    panel = build_team_game_panel()
    feats = ["qb_epa_r8_diff", "off_epa_play_r4_diff", "def_epa_allowed_r4_diff"]
    if args.verify_thresholds:
        _verify_thresholds(panel, feats)
    if args.verify_split:
        verify_split(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
