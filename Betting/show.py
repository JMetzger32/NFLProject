"""Show predictions for a week. The one command to run to see output.

Usage:
    python Betting/show.py                      # latest week generated
    python Betting/show.py --season 2026 --week 2
    python Betting/show.py --picks-only
    python Betting/show.py --list               # what's been generated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

OUT_DIR = PROJECT_ROOT / "Betting" / "output"


def available() -> list[Path]:
    return sorted(OUT_DIR.glob("picks_*.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--picks-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    files = available()
    if not files:
        print("No predictions generated yet. Run:")
        print("  python Betting/weekly_picks.py --season 2026 --week 2")
        return 1

    if args.list:
        print(f"{len(files)} week(s) generated:\n")
        for f in files:
            p = json.loads(f.read_text())
            flag = "  <-- PICKS" if p["qualifying_picks"] else ""
            print(f"  {p['season']} wk{p['week']:>2}  "
                  f"{p['games_evaluated']:>2} games  "
                  f"{p['qualifying_picks']} picks  {p['status']}{flag}")
        return 0

    if args.season and args.week:
        path = OUT_DIR / f"picks_{args.season}_wk{args.week:02d}.json"
        if not path.exists():
            print(f"No output for {args.season} week {args.week}. Generate it with:")
            print(f"  python Betting/weekly_picks.py --season {args.season} "
                  f"--week {args.week}")
            return 1
    else:
        path = files[-1]

    p = json.loads(path.read_text())
    bar = "=" * 74
    print(bar)
    print(f"  {p['season']}  WEEK {p['week']}   ({p['games_evaluated']} games)")
    print(f"  model {p['model_version']} · de-vig {p['devig_method']} · "
          f"generated {p['generated_at'][:16].replace('T', ' ')} UTC")
    print(bar)

    if p["qualifying_picks"]:
        print(f"\n{p['qualifying_picks']} QUALIFYING PICK(S)\n")
        pk = pd.DataFrame(p["picks"])
        pk["ci"] = pk["ci"].apply(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        print(pk[["team", "opponent", "model_prob", "ci", "market_prob_devig",
                  "edge", "moneyline", "ev_dollars", "stake",
                  "confidence"]].to_string(index=False))
        print(f"\n  total staked: ${pk['stake'].sum():.2f}")
    else:
        print(f"\n  NO QUALIFYING PICKS  —  status: {p['status']}")
        if p.get("notes"):
            print(f"  {p['notes']}")
        print("\n  This is the filter working as designed, not an error. A pick")
        print("  requires the model's confidence interval to sit entirely above")
        print("  the de-vigged market probability.")

    if not args.picks_only:
        d = pd.DataFrame(p["all_games"])
        print(f"\n{'-' * 74}\nALL {len(d)} TEAM-GAMES EVALUATED\n")
        d = d.rename(columns={"model_prob": "model", "market_prob_devig": "market"})
        print(d[["team", "opponent", "model", "market", "edge",
                 "reason"]].to_string(index=False))

    print(f"\nfiles: {path.name} · {path.with_suffix('.csv').name}  (in Betting/output/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
