"""Weekly picks pipeline: features -> bagged predictions -> de-vig -> edge -> picks.

Produces a stable-schema JSON + CSV per week so a website is later just a rendering
layer. Every game the pipeline evaluates is written to the `predictions` table,
pick or not — a picks-only log cannot measure calibration.

EXPECT ZERO PICKS MOST WEEKS. The bootstrap CI on a model probability averages
~0.156 wide, and the model agreed with the closing line on 539 of 542 games in the
2025 holdout. A week with no qualifying edge is the filter working, not a failure.

Usage:
    python Betting/weekly_picks.py --season 2026 --week 2
    python Betting/weekly_picks.py --season 2026 --week 2 --bootstrap 200
    python Betting/weekly_picks.py --backtest 2025          # full-season backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.bootstrap import bagged_bootstrap, load_model1_config  # noqa: E402
from src.config import COMPLETED_SEASONS  # noqa: E402
from src.db import get_engine  # noqa: E402
from src.features import build_team_game_panel  # noqa: E402
from src.odds import DEVIG_METHOD, MIN_HISTORY_FOR_PICK, build_edge_frame  # noqa: E402

OUT_DIR = PROJECT_ROOT / "Betting" / "output"
MODEL_VERSION = "model_1_bagged_v1"
MIN_HISTORY = MIN_HISTORY_FOR_PICK  # shared with the odds engine's validity gate


def _market_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach this team's and the opponent's moneyline to each team-game row."""
    eng = get_engine()
    ml = pd.read_sql(
        "SELECT game_id, home_moneyline, away_moneyline, spread_line "
        "FROM betting_lines", eng)
    out = panel.merge(ml, on="game_id", how="left", suffixes=("", "_bl"))
    out["team_moneyline"] = np.where(out["is_home"], out["home_moneyline"],
                                     out["away_moneyline"])
    out["opp_moneyline"] = np.where(out["is_home"], out["away_moneyline"],
                                    out["home_moneyline"])
    return out


def predict_week(season: int, week: int, B: int, backtest: bool = False,
                 panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """`panel` lets a caller reuse one build across many weeks — the backtest would
    otherwise rebuild the same frame 18 times."""
    feats, params = load_model1_config()

    if panel is None:
        panel = (build_team_game_panel() if backtest
                 else build_team_game_panel(upcoming=(season, week)))
    target = panel[(panel["season"] == season) & (panel["week"] == week)]

    # Train on completed seasons strictly before the target season.
    train = panel[panel["season"].isin([s for s in COMPLETED_SEASONS if s < season])]
    if len(train) == 0:
        train = panel[panel["season"].isin(COMPLETED_SEASONS)]
    train = train.dropna(subset=["won"]).reset_index(drop=True)
    target = target.reset_index(drop=True)
    if target.empty:
        return target

    print(f"  train {len(train):,} rows ({sorted(train.season.unique())}) -> "
          f"predict {len(target)} rows")
    res = bagged_bootstrap(train, target, feats, params, B=B, verbose=False)

    out = target.copy()
    out["model_prob"] = res.point
    out["ci_lo"] = res.ci_lo
    out["ci_hi"] = res.ci_hi
    out["ci_width"] = res.ci_hi - res.ci_lo
    out = _market_columns(out)
    out = build_edge_frame(out, method=DEVIG_METHOD)
    out["model_version"] = MODEL_VERSION
    out["n_resamples"] = res.n_resamples
    return out


def _reason(row) -> str:
    if pd.isna(row["team_moneyline"]) or pd.isna(row["opp_moneyline"]):
        return "no market odds available"
    if row["qualifies"]:
        return "qualifies: CI entirely above market probability"
    if not row.get("history_ok", True):
        return (f"insufficient history ({int(row.get('games_of_history', 0))} "
                f"games, need {MIN_HISTORY_FOR_PICK})")
    if row.get("edge_test_passed") and row["ev_dollars"] <= 0:
        return "edge positive but EV negative at the offered price"
    if row["ci_hi"] < row["market_prob_devig"]:
        return "model rates team BELOW market (bet is on the opponent, not here)"
    return (f"CI [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] contains market "
            f"{row['market_prob_devig']:.3f}")


def write_outputs(df: pd.DataFrame, season: int, week: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = df[df["qualifies"] & df["team_moneyline"].notna()].sort_values(
        "ev_dollars", ascending=False)

    thin = df["games_of_history"].max() < MIN_HISTORY if len(df) else False
    payload = {
        "season": int(season), "week": int(week),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "devig_method": DEVIG_METHOD,
        "status": "picks_available" if len(picks) else "no_qualifying_edge",
        "games_evaluated": int(len(df) // 2),
        "team_rows_evaluated": int(len(df)),
        "qualifying_picks": int(len(picks)),
        "insufficient_history": bool(thin),
        "history_source_counts": {k: int(v) for k, v in
                                  df["history_source"].value_counts().items()},
        "notes": (
            f"Only {int(df['games_of_history'].max())} game(s) of in-season history; "
            "rolling features are near-empty this early and no edge should be "
            "expected." if thin else
            "No game's model probability CI excluded the de-vigged market "
            "probability." if not len(picks) else ""
        ),
        "picks": [
            {
                "game_id": r.game_id, "team": r.team, "opponent": r.opponent,
                "is_home": bool(r.is_home),
                "model_prob": round(float(r.model_prob), 4),
                "ci": [round(float(r.ci_lo), 4), round(float(r.ci_hi), 4)],
                "history_source": r.history_source,
                "blend_weight": round(float(r.blend_weight), 3),
                "ci_widened_for_blend": bool(getattr(r, "ci_widened", False)),
                "market_prob_devig": round(float(r.market_prob_devig), 4),
                "edge": round(float(r.edge), 4),
                "ev_dollars": round(float(r.ev_dollars), 2),
                "moneyline": int(r.team_moneyline),
                "stake": float(r.stake),
                "confidence": r.confidence_label,
            }
            for r in picks.itertuples()
        ],
        "all_games": [
            {
                "game_id": r.game_id, "team": r.team, "opponent": r.opponent,
                "model_prob": None if pd.isna(r.model_prob) else round(float(r.model_prob), 4),
                "market_prob_devig": None if pd.isna(r.market_prob_devig) else round(float(r.market_prob_devig), 4),
                "edge": None if pd.isna(r.edge) else round(float(r.edge), 4),
                "qualifies": bool(r.qualifies),
                "history_source": r.history_source,
                "reason": _reason(r._asdict()),
            }
            for r in df.itertuples()
        ],
    }

    jpath = OUT_DIR / f"picks_{season}_wk{week:02d}.json"
    jpath.write_text(json.dumps(payload, indent=2))
    cpath = OUT_DIR / f"picks_{season}_wk{week:02d}.csv"
    cols = ["game_id", "season", "week", "team", "opponent", "is_home", "model_prob",
            "ci_lo", "ci_hi", "ci_lo_raw", "ci_hi_raw", "ci_widened",
            "history_source", "blend_weight", "games_of_history",
            "market_prob_devig", "overround", "edge", "ev_dollars",
            "team_moneyline", "qualifies", "stake", "confidence_label"]
    df[[c for c in cols if c in df.columns]].to_csv(cpath, index=False)
    print(f"  wrote {jpath.name} and {cpath.name}")
    return jpath


def persist(df: pd.DataFrame) -> int:
    """Upsert every evaluated team-game into `predictions` — picks and non-picks."""
    rows = df.copy()
    rows["is_pick"] = rows["qualifies"] & rows["team_moneyline"].notna()
    rows["devig_method"] = DEVIG_METHOD
    cols = ["game_id", "team", "season", "week", "opponent", "is_home", "model_prob",
            "ci_lo", "ci_hi", "market_prob_raw", "market_prob_devig", "devig_method",
            "overround", "team_moneyline", "opp_moneyline", "spread_line", "edge",
            "ev_dollars", "kelly_full", "is_pick", "stake", "confidence_label",
            "model_version"]
    rows = rows[[c for c in cols if c in rows.columns]].replace({np.nan: None})

    eng = get_engine()
    n = 0
    with eng.begin() as conn:
        for rec in rows.to_dict(orient="records"):
            keys = list(rec)
            setters = ", ".join(f"{k} = EXCLUDED.{k}" for k in keys
                                if k not in ("game_id", "team"))
            conn.execute(text(
                f"INSERT INTO predictions ({', '.join(keys)}) "
                f"VALUES ({', '.join(':' + k for k in keys)}) "
                f"ON CONFLICT (game_id, team) DO UPDATE SET {setters}"), rec)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="resamples; 200 is adequate for weekly use (CI width has "
                         "converged by ~300), 500 for the formal report")
    ap.add_argument("--backtest", type=int, metavar="SEASON",
                    help="run every week of a completed season")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    if args.backtest:
        season = args.backtest
        allw = []
        shared = build_team_game_panel()
        for wk in range(1, 19):
            print(f"week {wk}:")
            df = predict_week(season, wk, args.bootstrap, backtest=True, panel=shared)
            if df.empty:
                continue
            allw.append(df)
            write_outputs(df, season, wk)
            if not args.no_persist:
                persist(df)
        full = pd.concat(allw)
        picks = full[full["qualifies"] & full["team_moneyline"].notna()]
        print(f"\nBACKTEST {season}: {len(full)} team-games, "
              f"{len(picks)} qualifying picks ({len(picks) / len(full) * 100:.2f}%)")
        if len(picks):
            hit = picks["won"].mean()
            print(f"  pick accuracy: {hit:.4f} on n={len(picks)}")
            print(f"  mean edge: {picks['edge'].mean():+.4f}")
        return 0

    if args.season is None or args.week is None:
        ap.error("pass --season and --week, or --backtest SEASON")

    print(f"generating picks for {args.season} week {args.week} "
          f"(B={args.bootstrap})")
    df = predict_week(args.season, args.week, args.bootstrap)
    if df.empty:
        print("  no games found for that week")
        return 1
    write_outputs(df, args.season, args.week)
    if not args.no_persist:
        print(f"  persisted {persist(df)} rows to predictions")

    picks = df[df["qualifies"] & df["team_moneyline"].notna()]
    print(f"\n{len(df) // 2} games evaluated, {len(picks)} qualifying picks")
    if not len(picks):
        print("  NO QUALIFYING EDGE — expected behaviour, not a failure.")
        thin = df["games_of_history"].max()
        if thin < MIN_HISTORY:
            print(f"  (only {int(thin)} game(s) of in-season history this week)")
    else:
        print(picks[["team", "opponent", "model_prob", "market_prob_devig", "edge",
                     "ev_dollars", "stake"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
