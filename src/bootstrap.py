"""Block bootstrap and bagged ensemble.

WHY BLOCK, NOT ROW-LEVEL
------------------------
Team-game rows inside a single week are correlated: shared opponents, a common
injury wave, one market environment. Row-level iid resampling shatters that
structure and understates true uncertainty, producing confidence intervals that
look reassuringly tight and are wrong.

WHY SEASON-WEEK, NOT SEASON
---------------------------
Season blocks would be the most conservative choice, but there are only 4 training
seasons: 35 distinct resample multisets, and a 31.6% chance of dropping an entire
season from any given resample. Measured, not assumed. Season-week gives 72 blocks
(median 32 rows each) while still preserving the within-week correlation that
matters.

WHAT THIS CAN AND CANNOT DO
---------------------------
Bagging reduces variance and sharpens calibration. It cannot create predictive
information that is not in the features. Model_1 found 99.45% agreement with the
closing line; the expected result here is a steadier model that still tracks the
market.

Usage:
    python -m src.bootstrap --convergence
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import RANDOM_STATE, make_gbm, make_lr, make_rf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL1_PARAMS = PROJECT_ROOT / "Modeling" / "Model_1" / "data" / "03_primary.csv"
MODEL1_COLL = PROJECT_ROOT / "Modeling" / "Model_1" / "data" / "02_collinearity_labeled.csv"

DEFAULT_B = 500


def load_model1_config() -> tuple[list[str], dict]:
    """Feature set and hyperparameters selected by Model_1's walk-forward CV.

    Hyperparameters are NOT re-tuned inside the bootstrap — tuning on resamples
    would be fitting the resampling procedure itself.
    """
    coll = pd.read_csv(MODEL1_COLL)
    feats = sorted(coll[coll.kept_as_representative]["feature"])
    prim = pd.read_csv(MODEL1_PARAMS)
    params = {n: ast.literal_eval(prim[prim.model == n].iloc[0]["params"])
              for n in ("LR", "GBM", "RF")}
    return feats, params


def make_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    """One block per (season, week), as arrays of row indices."""
    return [g.index.to_numpy() for _, g in df.groupby(["season", "week"], sort=True)]


@dataclass
class BootstrapResult:
    predictions: np.ndarray        # (B, n_games) per-resample predicted probability
    point: np.ndarray              # bagged mean prediction per game
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    importances: pd.DataFrame      # per-resample RF importance
    coefficients: pd.DataFrame     # per-resample LR coefficient
    n_resamples: int


def bagged_bootstrap(
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    features: list[str],
    params: dict,
    B: int = DEFAULT_B,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> BootstrapResult:
    """Refit LR/GBM/RF on B block-resamples; return bagged predictions + CIs."""
    rng = np.random.default_rng(seed)
    blocks = make_blocks(train)
    n_blocks = len(blocks)
    makers = {"LR": lambda: make_lr([]), "GBM": make_gbm, "RF": make_rf}

    preds, imps, coefs = [], [], []
    for b in range(B):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[i] for i in pick])
        sample = train.loc[idx]
        if sample["won"].nunique() < 2:
            continue

        member = []
        for name in ("LR", "GBM", "RF"):
            pipe = makers[name]()
            pipe.set_params(**params[name])
            try:
                pipe.fit(sample[features], sample["won"])
            except Exception:
                continue
            member.append(pipe.predict_proba(predict_on[features])[:, 1])

            clf = pipe.named_steps["clf"]
            if name == "RF":
                imps.append(dict(zip(features, clf.feature_importances_)))
            elif name == "LR":
                coefs.append(dict(zip(features, clf.coef_[0])))
        if member:
            preds.append(np.mean(member, axis=0))
        if verbose and (b + 1) % 100 == 0:
            print(f"    bootstrap {b + 1}/{B}")

    P = np.vstack(preds)
    return BootstrapResult(
        predictions=P,
        point=P.mean(axis=0),
        ci_lo=np.percentile(P, 2.5, axis=0),
        ci_hi=np.percentile(P, 97.5, axis=0),
        importances=pd.DataFrame(imps),
        coefficients=pd.DataFrame(coefs),
        n_resamples=len(P),
    )


def convergence_curve(res: BootstrapResult,
                      checkpoints=(50, 100, 200, 300, 400, 500)) -> pd.DataFrame:
    """Mean CI width as resamples accumulate.

    The output is only trustworthy once this flattens. Reported as evidence, not
    asserted — if it is still shrinking at B, raise B.
    """
    rows = []
    for b in checkpoints:
        if b > res.n_resamples:
            continue
        sub = res.predictions[:b]
        lo = np.percentile(sub, 2.5, axis=0)
        hi = np.percentile(sub, 97.5, axis=0)
        rows.append({"B": b, "mean_ci_width": float(np.mean(hi - lo)),
                     "median_ci_width": float(np.median(hi - lo))})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        out["pct_change_vs_prev"] = out["mean_ci_width"].pct_change() * 100
    return out


def importance_stability(res: BootstrapResult, top_n: int = 5) -> pd.DataFrame:
    """How often each feature lands in the top-N across resamples.

    Separates genuinely stable contributors from features that only look important
    because of one lucky resample.
    """
    imp = res.importances
    if imp.empty:
        return pd.DataFrame()
    ranks = imp.rank(axis=1, ascending=False)
    return (pd.DataFrame({
        "feature": imp.columns,
        "mean_importance": imp.mean().to_numpy(),
        "std_importance": imp.std().to_numpy(),
        f"pct_in_top{top_n}": (ranks <= top_n).mean().to_numpy() * 100,
        "mean_rank": ranks.mean().to_numpy(),
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True))


def _convergence_demo() -> None:
    from src.config import COMPLETED_SEASONS
    from src.features import build_team_game_panel

    feats, params = load_model1_config()
    panel = build_team_game_panel()
    panel = panel[panel["season"].isin(COMPLETED_SEASONS)]
    train = panel[panel["season"] < 2025].reset_index(drop=True)
    test = panel[panel["season"] == 2025].reset_index(drop=True)

    blocks = make_blocks(train)
    print(f"features: {len(feats)}   train rows: {len(train):,}   "
          f"blocks: {len(blocks)}   test rows: {len(test):,}")
    res = bagged_bootstrap(train, test, feats, params, B=DEFAULT_B)
    print(f"\ncompleted {res.n_resamples} resamples")
    print("\nconvergence of mean CI width:")
    print(convergence_curve(res).round(5).to_string(index=False))
    print(f"\nmean per-game CI width at B={res.n_resamples}: "
          f"{np.mean(res.ci_hi - res.ci_lo):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convergence", action="store_true")
    args = ap.parse_args()
    if args.convergence:
        _convergence_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
