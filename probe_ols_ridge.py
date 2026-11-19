"""Does a lightly regularised OLS reproduce -34.86%?

The plain lstsq solution explodes because a few characteristics are nearly
constant inside 1957-1974 (e.g. securedind, sd=4e-5) but common later, so their
coefficients run to ~1e4.  A small ridge on the standardised design tames exactly
that pathology.  This sweeps lambda and reports which value, if any, lands on the
paper's Table-2 OLS number.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config, log  # noqa: E402
from src.features import load_meta, year_path  # noqa: E402
from src.train import target_y, year_bounds  # noqa: E402

TY, TRAIN_END = 1987, 1974
TARGET_R2 = -34.86


def main():
    cfg = load_config()
    meta = load_meta(cfg)
    bounds = year_bounds(meta)
    c = json.loads(json.dumps(cfg))
    c["data"]["y_clip_quantiles"] = None
    y_all = target_y(meta, c)

    for ncol in (920,):
        XtX = None
        Xty = None
        n = 0
        for y in range(1957, TRAIN_END + 1):
            a, b = bounds[y]
            X = np.load(year_path(cfg, y))[:, :ncol].astype(np.float64)
            yv = y_all[a:b]
            ok = np.isfinite(yv)
            X = X[ok]
            yv = yv[ok]
            X = np.hstack([X, np.ones((len(X), 1))])
            if XtX is None:
                k = X.shape[1]
                XtX = np.zeros((k, k))
                Xty = np.zeros(k)
            XtX += X.T @ X
            Xty += X.T @ yv
            n += len(X)
            del X
        log(f"design {n:,} x {k}")

        a, b = bounds[TY]
        Xt = np.load(year_path(cfg, TY))[:, :ncol].astype(np.float64)
        Xt = np.hstack([Xt, np.ones((len(Xt), 1))])
        yt = y_all[a:b]
        ok = np.isfinite(yt)
        Xt = Xt[ok]
        yt = yt[ok]

        # scale-aware ridge: lambda relative to the average diagonal of X'X/n
        diag = np.trace(XtX) / k / n
        print(f"\n{'lambda':>10} {'pred_sd':>10} {'R2':>14}")
        for lam in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            A = XtX / n + lam * diag * np.eye(k)
            beta = np.linalg.solve(A, Xty / n)
            pred = Xt @ beta
            r2 = 100 * (1 - ((yt - pred) ** 2).sum() / (yt ** 2).sum())
            mark = "  <== paper" if abs(r2 - TARGET_R2) < 8 else ""
            print(f"{lam:10.0e} {pred.std():10.3f} {r2:+14.2f}%{mark}")


if __name__ == "__main__":
    main()
