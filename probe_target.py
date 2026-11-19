"""Alternative annual targets: compounded vs summed vs log-cumulative.

Compounding turns one +452% month into a x5.5 multiplier and manufactures the
+51,870% annual outliers that dominate sum(y^2).  A simple 12-month sum (or a
log-cumulative return) keeps the same information but with a far tamer tail, and
is the more common construction of an "h-month ahead" target in this literature.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import load_meta  # noqa: E402
from src.train import year_bounds  # noqa: E402

PANEL = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\cache_crsp\panel.parquet"
H = 12


def fwd_sum(v, h):
    t = len(v)
    out = np.full(t, np.nan)
    if t < h:
        return out
    cs = np.concatenate(([0.0], np.cumsum(v)))
    out[: t - h + 1] = cs[h:] - cs[: t - h + 1]
    return out


def fwd_prod(v, h):
    t = len(v)
    out = np.full(t, np.nan)
    if t < h:
        return out
    cs = np.concatenate(([0.0], np.cumsum(np.log1p(v))))
    out[: t - h + 1] = np.exp(cs[h:] - cs[: t - h + 1]) - 1.0
    return out


def main():
    d = pd.read_parquet(PANEL, columns=["permno", "ym", "ret", "rf"])
    d = d.sort_values(["permno", "ym"], kind="mergesort").reset_index(drop=True)
    d["ex"] = d["ret"] - d["rf"]
    perm = d["permno"].to_numpy()
    bounds = np.flatnonzero(np.diff(perm)) + 1
    seg = list(zip(np.r_[0, bounds], np.r_[bounds, len(d)]))

    n = len(d)
    y_sum = np.full(n, np.nan)
    y_prod = np.full(n, np.nan)
    ex = d["ex"].to_numpy(dtype=np.float64)
    for a, b in seg:
        y_sum[a:b] = fwd_sum(ex[a:b], H)
        y_prod[a:b] = fwd_prod(ex[a:b], H)
    y_log = np.log1p(np.clip(y_prod, -0.999999, None))

    cfg = load_config()
    meta = load_meta(cfg)
    yb = year_bounds(meta)
    z = np.load(Path(cfg["paths"]["out_dir"]) / "nn3_f94_ens_preds.npz")

    print(f"{'target':>12} {'E[y^2]':>9} {'sd':>8} {'max':>12}  R2")
    for name, yv in (("compounded", y_prod), ("sum", y_sum), ("log", y_log)):
        d2 = d.assign(y=yv)
        sse, sy2, cnt = 0.0, 0.0, 0
        for ty in sorted(int(k) for k in z.files):
            a, b = yb[ty]
            yy = d2["y"].to_numpy()[a:b]
            pp = z[str(ty)]
            ok = np.isfinite(yy)
            sse += float(((yy[ok] - pp[ok]) ** 2).sum())
            sy2 += float((yy[ok] ** 2).sum())
            cnt += int(ok.sum())
        print(f"{name:>12} {sy2/cnt:9.4f} {np.sqrt(sy2/cnt):8.4f} "
              f"{np.nanmax(yv):12.2f}  {100*(1 - sse/sy2):+7.2f}%")


if __name__ == "__main__":
    main()
