"""Fast ablation harness for the NN training recipe.

    python probe.py --years 1995 2002 --features 94 --epochs 60 --lr 3e-3 --bs 20000

Prints R2_oos and the cross-sectional spread of the predictions, which is the
diagnostic that matters: a model whose predictions have sd far below the target's
sd cannot produce a positive R2_oos no matter how well it fits in-sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.train import run_nn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--features", type=int, default=None, help="94 = chars only, 920 = full")
    ap.add_argument("--hidden", nargs="+", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--l1", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    cfg = load_config()
    m = cfg["model"]
    if a.features:
        m["n_features"] = a.features
    if a.hidden:
        m["hidden"] = a.hidden
    if a.lr is not None:
        m["lr"] = a.lr
    if a.l1 is not None:
        m["l1"] = a.l1
    if a.bs is not None:
        m["batch_size"] = a.bs
    if a.epochs is not None:
        m["max_epochs"] = a.epochs
    if a.patience is not None:
        m["patience"] = a.patience

    print("[config] features=%s hidden=%s lr=%s l1=%s bs=%s epochs=%s patience=%s"
          % (m.get("n_features", 920), m["hidden"], m["lr"], m["l1"],
             m["batch_size"], m["max_epochs"], m["patience"]), flush=True)

    res = run_nn(cfg, list(a.years), seeds=[a.seed],
                 max_epochs=m["max_epochs"], tag=a.tag)
    print("\n[result] All=%+.2f%%  Top=%+.2f%%  Bottom=%+.2f%%  n=%d"
          % (res["All"], res["Top"], res["Bottom"], res["n_obs"]))
    for k, v in res.get("diag", {}).items():
        if k.endswith("pred_sd") or k.endswith("y_sd"):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
