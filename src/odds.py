"""Odds engine: de-vigging, edge, expected value, and stake sizing.

THE DE-VIG RULE
---------------
Raw two-way implied probabilities always sum to more than 1 — that excess is the
book's margin (the "vig" or "overround"). Comparing a model probability against a
RAW implied probability systematically overstates edge by roughly half the
overround on every bet. Every comparison in this project uses a de-vigged
probability. There is no code path that skips it.

Two methods are implemented:
  * proportional — divide each side by the overround. Simple, but it removes vig
    evenly, which assumes the book prices both sides with the same margin.
  * Shin — solves for an insider-trading parameter z, removing proportionally MORE
    margin from longshots. Closer to how books actually price when
    favorite-longshot bias is present.

Usage:
    python -m src.odds --compare-devig
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Flat stake. The plan called for fractional Kelly; the user chose a flat $10 per
# bet. Kelly is still computed and reported as a diagnostic so the two can be
# compared, but FLAT_STAKE is what gets recommended and logged.
# Shin fit this book's lines marginally better than proportional on 1,371 completed
# games (log-loss 0.61073 vs 0.61098, Brier 0.21171 vs 0.21176) — consistent with
# favorite-longshot bias. The gap is small at a 3.69% average overround, but Shin
# is never worse, so it is the default.
DEVIG_METHOD = "shin"

FLAT_STAKE = 10.0
KELLY_FRACTIONS = {"quarter_kelly": 0.25, "half_kelly": 0.5}
MAX_STAKE_FRACTION = 0.02  # hard cap on the Kelly diagnostic, as a bankroll share


def american_to_decimal(ml: float) -> float:
    """American moneyline -> decimal odds (total return per 1 staked)."""
    ml = float(ml)
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / abs(ml))


def implied_prob(ml) -> np.ndarray:
    """Raw implied probability from an American moneyline. Still contains vig."""
    ml = np.asarray(ml, dtype=float)
    # np.where evaluates both branches, so guard the denominators to avoid a
    # spurious divide-by-zero warning at ml = -100 / ml = 0.
    neg = np.where(ml < 0, ml, -100.0)
    pos = np.where(ml >= 0, ml, 100.0)
    return np.where(ml < 0, -neg / (-neg + 100.0), 100.0 / (pos + 100.0))


def overround(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Sum of raw two-way implied probabilities. 1.045 means a 4.5% margin."""
    return np.asarray(p_a, dtype=float) + np.asarray(p_b, dtype=float)


def devig_proportional(p_a, p_b) -> tuple[np.ndarray, np.ndarray]:
    """Scale both sides by the overround so they sum to exactly 1."""
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    tot = p_a + p_b
    return p_a / tot, p_b / tot


def _shin_prob(p: np.ndarray, z: float, tot: float) -> np.ndarray:
    inner = np.clip(z * z + 4.0 * (1.0 - z) * p * p / tot, 0.0, None)
    return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))


def devig_shin(p_a, p_b) -> tuple[np.ndarray, np.ndarray]:
    """Shin (1993) de-vig for a two-way market.

    Solves for z — the assumed share of insider money — such that the implied
    probabilities sum to exactly 1. Removes relatively more margin from the
    longshot side, which is the empirically observed shape of book pricing.

    z is found by numerical root-find rather than a closed form. An earlier
    closed-form attempt degenerated to z≈0, which (after renormalisation) makes
    Shin identical to proportional — silently defeating the whole point of
    offering two methods. Brentq is cheap at this scale and cannot degenerate
    unnoticed: if no root is bracketed the function falls back explicitly.
    """
    from scipy.optimize import brentq

    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    scalar = p_a.ndim == 0
    p_a, p_b = np.atleast_1d(p_a), np.atleast_1d(p_b)

    out_a = np.empty_like(p_a)
    out_b = np.empty_like(p_b)
    for i in range(len(p_a)):
        tot = p_a[i] + p_b[i]
        if not np.isfinite(tot) or tot <= 1.0:
            out_a[i], out_b[i] = devig_proportional(p_a[i], p_b[i])
            continue

        def f(z, i=i, tot=tot):
            return (_shin_prob(p_a[i], z, tot) + _shin_prob(p_b[i], z, tot)) - 1.0

        try:
            lo, hi = 1e-12, 0.5
            if f(lo) * f(hi) > 0:  # no sign change -> no root in range
                raise ValueError
            z = brentq(f, lo, hi, xtol=1e-12, maxiter=200)
        except (ValueError, RuntimeError):
            out_a[i], out_b[i] = devig_proportional(p_a[i], p_b[i])
            continue
        out_a[i] = _shin_prob(p_a[i], z, tot)
        out_b[i] = _shin_prob(p_b[i], z, tot)

    if scalar:
        return float(out_a[0]), float(out_b[0])
    return out_a, out_b


def devig(ml_a, ml_b, method: str = "shin") -> tuple[np.ndarray, np.ndarray]:
    p_a, p_b = implied_prob(ml_a), implied_prob(ml_b)
    if method == "proportional":
        return devig_proportional(p_a, p_b)
    if method == "shin":
        return devig_shin(p_a, p_b)
    raise ValueError(f"unknown de-vig method: {method}")


def expected_value(model_prob, ml, stake: float = FLAT_STAKE) -> np.ndarray:
    """EV at the ACTUAL offered odds, not the fair de-vigged price.

    You bet at the real number, so EV must be computed at the real number — using
    fair odds here would quietly inflate every EV by the vig.
    """
    model_prob = np.asarray(model_prob, dtype=float)
    dec = np.array([american_to_decimal(m) for m in np.atleast_1d(ml)], dtype=float)
    profit = (dec - 1.0) * stake
    return model_prob * profit - (1.0 - model_prob) * stake


def kelly_fraction(model_prob, ml) -> np.ndarray:
    """Full-Kelly bankroll fraction. Reported as a DIAGNOSTIC only — actual staking
    is flat. Negative means no bet."""
    model_prob = np.asarray(model_prob, dtype=float)
    dec = np.array([american_to_decimal(m) for m in np.atleast_1d(ml)], dtype=float)
    b = dec - 1.0
    return np.clip((model_prob * b - (1.0 - model_prob)) / b, 0.0, 1.0)


def qualifies(ci_lo, ci_hi, market_prob) -> np.ndarray:
    """The edge must survive the model's OWN uncertainty — and point the right way.

    ONE-SIDED on purpose. A bet on this team is only sensible when the model thinks
    the team is UNDERVALUED, i.e. the whole CI sits above the market probability.
    A CI entirely BELOW the market means the team is overvalued — that is not a bet
    on this team, it is a bet on the opponent, which is already represented by the
    opponent's own row.

    A two-sided test (the first implementation) flagged both sides of the same
    game and staked negative-EV bets: SF at edge -0.0762 / EV -$1.08 was recommended
    alongside MIA at +0.0794 in 2026 week 2. Caught on the first live run.
    """
    ci_lo = np.asarray(ci_lo, dtype=float)
    market_prob = np.asarray(market_prob, dtype=float)
    return ci_lo > market_prob


def confidence_label(edge: float, qualified: bool) -> str:
    if not qualified:
        return "no edge"
    a = abs(edge)
    if a >= 0.10:
        return "strong edge"
    if a >= 0.05:
        return "moderate edge"
    return "slight edge"


# Below this many in-season games, rolling features rest mostly on a prior-season
# blend rather than on what this team has actually done. Kept as a validity gate
# even after the cold-start blend: the blend makes early-season features *defined*
# and more stable, but it does not make them informative about this season's team.
MIN_HISTORY_FOR_PICK = 4

# Bootstrap resamples the training data; it does not resample the prior-season
# value a blended row leans on, so it understates uncertainty exactly where the
# blend is doing the most work. This inflates the interval in proportion to how
# much of the row is prior rather than present: no change at weight 1, doubled at
# weight 0. A heuristic, not a derived quantity — flagged as such in the output.
BLEND_CI_INFLATION = 1.0


def widen_ci_for_blend(ci_lo, ci_hi, blend_weight):
    """Widen the interval around its midpoint for prior-blended rows."""
    ci_lo = np.asarray(ci_lo, dtype=float)
    ci_hi = np.asarray(ci_hi, dtype=float)
    w = np.nan_to_num(np.asarray(blend_weight, dtype=float), nan=1.0)
    factor = 1.0 + BLEND_CI_INFLATION * (1.0 - np.clip(w, 0.0, 1.0))
    mid = (ci_lo + ci_hi) / 2.0
    half = (ci_hi - ci_lo) / 2.0 * factor
    return np.clip(mid - half, 0.0, 1.0), np.clip(mid + half, 0.0, 1.0)


def build_edge_frame(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Attach de-vigged market probability, edge, EV and stake to a team-game frame.

    Expects: team_moneyline, opp_moneyline, model_prob, ci_lo, ci_hi, and
    games_of_history when the history gate should apply.
    """
    out = df.copy()
    # Widen the interval before any edge test, so a blended row has to clear a
    # correspondingly higher bar.
    if "blend_weight" in out.columns:
        lo, hi = widen_ci_for_blend(out["ci_lo"], out["ci_hi"], out["blend_weight"])
        out["ci_lo_raw"], out["ci_hi_raw"] = out["ci_lo"], out["ci_hi"]
        out["ci_lo"], out["ci_hi"] = lo, hi
        out["ci_widened"] = (out["ci_hi"] - out["ci_lo"]) > (
            out["ci_hi_raw"] - out["ci_lo_raw"] + 1e-12)
    fair_team, _ = devig(out["team_moneyline"], out["opp_moneyline"], method=method)
    out["market_prob_raw"] = implied_prob(out["team_moneyline"])
    out["market_prob_devig"] = fair_team
    out["overround"] = overround(implied_prob(out["team_moneyline"]),
                                 implied_prob(out["opp_moneyline"]))
    out["edge"] = out["model_prob"] - out["market_prob_devig"]
    out["ev_dollars"] = expected_value(out["model_prob"], out["team_moneyline"])
    out["kelly_full"] = kelly_fraction(out["model_prob"], out["team_moneyline"])
    for name, frac in KELLY_FRACTIONS.items():
        out[name] = np.minimum(out["kelly_full"] * frac, MAX_STAKE_FRACTION)
    edge_ok = qualifies(out["ci_lo"], out["ci_hi"], out["market_prob_devig"])
    if "games_of_history" in out.columns:
        hist_ok = out["games_of_history"].fillna(0) >= MIN_HISTORY_FOR_PICK
    else:
        hist_ok = pd.Series(True, index=out.index)
    has_odds = out["team_moneyline"].notna() & out["opp_moneyline"].notna()
    out["edge_test_passed"] = edge_ok
    out["history_ok"] = hist_ok
    out["qualifies"] = edge_ok & hist_ok & has_odds
    # EV is computed at real odds, so a positive-edge bet can still price badly.
    out["qualifies"] &= out["ev_dollars"] > 0
    out["stake"] = np.where(out["qualifies"], FLAT_STAKE, 0.0)
    out["confidence_label"] = [
        confidence_label(e, q) for e, q in zip(out["edge"], out["qualifies"])
    ]
    return out


def _compare_devig() -> None:
    """Which de-vig method better describes this book's lines?

    Test: de-vigged favorite probability vs actual favorite win rate, on completed
    games. The better method is the better-calibrated one.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    from src.db import get_engine

    q = """
    SELECT g.game_id, g.season, g.result, b.home_moneyline, b.away_moneyline
    FROM games g JOIN betting_lines b USING (game_id)
    WHERE g.game_type='REG' AND g.result <> 0
      AND b.home_moneyline IS NOT NULL AND b.away_moneyline IS NOT NULL
    """
    d = pd.read_sql(q, get_engine())
    y = (d["result"] > 0).astype(int).to_numpy()  # home win

    raw_h = implied_prob(d["home_moneyline"])
    raw_a = implied_prob(d["away_moneyline"])
    ovr = overround(raw_h, raw_a)
    print(f"games: {len(d):,}")
    print(f"overround: mean {ovr.mean():.4f}  median {np.median(ovr):.4f}  "
          f"min {ovr.min():.4f}  max {ovr.max():.4f}")
    print(f"  -> book margin averages {(ovr.mean() - 1) * 100:.2f}%")

    rows = []
    for name in ("proportional", "shin"):
        ph, pa = devig(d["home_moneyline"], d["away_moneyline"], method=name)
        assert np.allclose(ph + pa, 1.0, atol=1e-9), f"{name} does not sum to 1"
        rows.append({"method": name, "brier": brier_score_loss(y, ph),
                     "log_loss": log_loss(y, np.clip(ph, 1e-9, 1 - 1e-9)),
                     "mean_fav_prob": float(np.maximum(ph, pa).mean())})
    # Raw (undevigged) shown to quantify what skipping the step would cost.
    rows.append({"method": "RAW (not de-vigged)",
                 "brier": brier_score_loss(y, np.clip(raw_h, 0, 1)),
                 "log_loss": log_loss(y, np.clip(raw_h, 1e-9, 1 - 1e-9)),
                 "mean_fav_prob": float(np.maximum(raw_h, raw_a).mean())})
    res = pd.DataFrame(rows)
    print()
    print(res.round(5).to_string(index=False))
    best = res[res.method != "RAW (not de-vigged)"].sort_values("log_loss").iloc[0]
    print(f"\nBetter fit for this book: {best['method']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-devig", action="store_true")
    args = ap.parse_args()
    if args.compare_devig:
        _compare_devig()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
