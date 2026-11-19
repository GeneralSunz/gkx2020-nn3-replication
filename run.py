"""CLI entry point.

    python run.py prep
    python run.py diag
    python run.py ols
    python run.py nn [--years 1987 1988 ...] [--seeds 1] [--epochs 5] [--tag smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config, log  # noqa: E402


def cmd_prep(_args):
    from src.features import build_design
    from src.prep import build_panel

    cfg = load_config()
    panel = build_panel(cfg, force=True)
    build_design(cfg, panel, force=True)


def cmd_diag(_args):
    import numpy as np
    import pandas as pd

    from src.common import load_config
    from src.features import load_meta

    cfg = load_config()
    cache = Path(cfg["paths"]["cache_dir"])
    with open(cache / "panel_diag.json", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False))

    meta = load_meta(cfg)
    print("\nmeta:", meta.shape, "years", meta["ym"].min(), "->", meta["ym"].max())
    print("\nrows per year:")
    print(meta.groupby(meta["ym"] // 100).size().to_string())
    print("\ncharacteristics sanity (ranked, should be within [-1,1]):")
    print(meta[["mvel1", "y1", "y12"]].describe().to_string())
    print("\nshare y12 missing by year (last years should be non-zero):")
    print(meta.assign(yr=meta["ym"] // 100).groupby("yr")["y12"].apply(lambda s: s.isna().mean()).tail(8).to_string())


def cmd_ols(args):
    from src.train import run_ols

    cfg = load_config()
    years = args.years or list(range(1987, 2017))
    run_ols(cfg, years, tag=args.tag or "ols")


def cmd_nn(args):
    from src.train import run_nn

    cfg = load_config()
    years = args.years or list(range(1987, 2017))
    run_nn(cfg, years, seeds=args.seeds, max_epochs=args.epochs, tag=args.tag or "nn3")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prep").set_defaults(func=cmd_prep)
    sub.add_parser("diag").set_defaults(func=cmd_diag)

    p_ols = sub.add_parser("ols")
    p_ols.add_argument("--years", nargs="*", type=int, default=None)
    p_ols.add_argument("--tag", default="ols")
    p_ols.set_defaults(func=cmd_ols)

    p_nn = sub.add_parser("nn")
    p_nn.add_argument("--years", nargs="*", type=int, default=None)
    p_nn.add_argument("--seeds", nargs="*", type=int, default=None)
    p_nn.add_argument("--epochs", type=int, default=None)
    p_nn.add_argument("--tag", default="nn3")
    p_nn.set_defaults(func=cmd_nn)

    args = ap.parse_args()
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
