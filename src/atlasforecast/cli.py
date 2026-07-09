from __future__ import annotations

import argparse
import json

from . import pipeline
from .backtest import run as backtest_run
from .data import generate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="atlasforecast")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("backtest", help="Rolling-origin backtest + model ranking.")
    pb.add_argument("--skus", type=int, default=6)
    pb.add_argument("--horizon", type=int, default=14)

    pp = sub.add_parser("plan", help="Full pipeline: forecast + order recommendations.")
    pp.add_argument("--skus", type=int, default=6)
    pp.add_argument("--horizon", type=int, default=14)

    args = p.parse_args(argv)
    if args.cmd == "backtest":
        series = generate(n_skus=args.skus)
        res = backtest_run(series, horizon=args.horizon)
        print(json.dumps([{"sku": r.sku, "best": r.best_model, "metrics": r.per_model}
                          for r in res], indent=2))
    elif args.cmd == "plan":
        print(json.dumps(pipeline.run(n_skus=args.skus, horizon=args.horizon), indent=2))


if __name__ == "__main__":
    main()
