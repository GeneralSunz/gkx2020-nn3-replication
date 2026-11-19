"""Bottom-up annual forecast from the monthly model.

If the annual signal is persistent, summing the next 12 monthly forecasts should
beat a model trained directly on the annual target, because the monthly model sees
12x more observations.  This builds that aggregate and scores it against y12.
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

cfg = load_config()
meta = load_meta(cfg)
bounds = year_bounds(meta)
y12 = meta["y12"].to_numpy().astype(np.float64)
out = Path(cfg["paths"]["out_dir"])

z = np.load(out / "nn3_m1_preds.npz")
years = sorted(int(k) for k in z.files)
m1 = np.full(len(meta), np.nan)
for ty in years:
    a, b = bounds[ty]
    p = z[str(ty)]
    m1[a:b] = p[: b - a]

# sum the next 12 monthly forecasts, per permno
d = pd.DataFrame({"permno": meta["permno"].to_numpy(),
                  "ym": meta["ym"].to_numpy(),
                  "p": m1})
d = d.sort_values(["permno", "ym"], kind="mergesort").reset_index(drop=True)
perm = d["permno"].to_numpy()
cut = np.flatnonzero(np.diff(perm)) + 1
segs = list(zip(np.r_[0, cut], np.r_[cut, len(d)]))

annual = np.full(len(d), np.nan)
for a, b in segs:
    v = d["p"].to_numpy()[a:b]
    n = len(v)
    if n < 12:
        continue
    cs = np.concatenate(([0.0], np.cumsum(v)))
    annual[a: b - 11] = cs[12: 12 + (b - 11 - a)] - cs[: (b - 11 - a)]
d["annual"] = annual
d = d.sort_values(["ym", "permno"], kind="mergesort")
back = d.set_index(["ym", "permno"])["annual"]

key = pd.MultiIndex.from_arrays([meta["ym"].to_numpy(), meta["permno"].to_numpy()])
pred = back.reindex(key).to_numpy(dtype=np.float64)

print(f"{'tag':>14} {'R2':>9}   (evaluated against raw y12, 1987-2016)")
sse = sy2 = 0.0
n = 0
for ty in years:
    a, b = bounds[ty]
    t = y12[a:b]
    p = pred[a:b]
    ok = np.isfinite(t) & np.isfinite(p)
    sse += float(((t[ok] - p[ok]) ** 2).sum())
    sy2 += float((t[ok] ** 2).sum())
    n += int(ok.sum())
print(f"{'bottom-up':>14} {100*(1-sse/sy2):+8.2f}%   n={n:,}")
print(f"{'direct annual':>14} {1.32:+8.2f}%   (nn3_raw)")
print(f"{'paper NN3':>14} {3.40:+8.2f}%")
