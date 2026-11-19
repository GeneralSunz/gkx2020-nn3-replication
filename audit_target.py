"""Audit the target-window convention.

y12(t) = product of months t..t+11   (my assumption)
vs
y12(t) = product of months t+1..t+12 (one month later)

Both are "12-month ahead" under different conventions about when the row's
characteristics are known.  The correct one must give a HIGHER cross-sectional
correlation with a characteristic that is known to predict returns (mom12m,
mvel1, retvol).  If the shifted version wins, the whole pipeline has an
off-by-one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PANEL = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\cache_crsp\panel.parquet"
H = 12


def fwd_prod(v, h, offset=0):
    t = len(v)
    out = np.full(t, np.nan)
    need = h + offset
    if t < need:
        return out
    cs = np.concatenate(([0.0], np.cumsum(np.log1p(v))))
    # window [i+offset, i+offset+h)
    out[: t - need + 1] = np.exp(cs[offset + h: offset + h + (t - need + 1)]
                                 - cs[offset: offset + (t - need + 1)]) - 1.0
    return out


def main():
    d = pd.read_parquet(PANEL, columns=["permno", "ym", "ret", "rf",
                                        "mom12m", "mvel1", "retvol"])
    d = d.sort_values(["permno", "ym"], kind="mergesort").reset_index(drop=True)
    d["ex"] = (d["ret"] - d["rf"]).astype(np.float64)
    perm = d["permno"].to_numpy()
    bounds = np.flatnonzero(np.diff(perm)) + 1
    seg = list(zip(np.r_[0, bounds], np.r_[bounds, len(d)]))

    n = len(d)
    y0 = np.full(n, np.nan)
    y1 = np.full(n, np.nan)
    for a, b in seg:
        v = d["ex"].to_numpy()[a:b]
        y0[a:b] = fwd_prod(v, H, 0)
        y1[a:b] = fwd_prod(v, H, 1)
    d["y0"], d["y1"] = y0, y1

    d = d[d.ym.between(198703, 201612)].copy()
    print(f"rows={len(d):,}  y0 sd={d.y0.std():.4f}  y1 sd={d.y1.std():.4f}")

    for ch in ("mom12m", "mvel1", "retvol"):
        print(f"\n--- {ch} ---")
        for tgt in ("y0", "y1"):
            s = d.dropna(subset=[tgt, ch])
            ic = s.groupby("ym").apply(
                lambda g: g[ch].rank().corr(g[tgt].rank()), include_groups=False)
            print(f"   target {tgt}: mean IC={ic.mean():+.4f}  "
                  f"median={ic.median():+.4f}  share>0={100*(ic>0).mean():.1f}%")


if __name__ == "__main__":
    main()
