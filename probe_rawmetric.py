"""Evaluate the same predictions against the raw (un-winsorised) target.

The paper does not winsorise the annual return, so its R2 denominator is the raw
sum of squares.  Our main numbers are computed after clipping y at 0.5/99.5%.
This puts both metrics side by side so the comparison with Table 2 is honest.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import load_meta  # noqa: E402
from src.train import target_y, year_bounds  # noqa: E402

cfg = load_config()
meta = load_meta(cfg)
bounds = year_bounds(meta)

y_clip = target_y(meta, cfg)                      # clipped (config has quantiles)
cfg_raw = json.loads(json.dumps(cfg))
cfg_raw["data"]["y_clip_quantiles"] = None
y_raw = target_y(meta, cfg_raw)                   # raw

out = Path(cfg["paths"]["out_dir"])
cases = {
    "nn3_ia (1 seed)": "nn3_ia_preds.npz",
    "nn3_f94 (1 seed)": "nn3_f94_preds.npz",
    "nn3_f94_ens (3 seeds)": "nn3_f94_ens_preds.npz",
}

for label, fn in cases.items():
    p = out / fn
    if not p.exists():
        continue
    z = np.load(p)
    years = sorted(int(k) for k in z.files)
    rows = []
    for ty in years:
        a, b = bounds[ty]
        ok = np.isfinite(y_raw[a:b])
        rows.append((a, b, ok, z[str(ty)]))
    print(f"\n{label}")
    for name, y in (("clipped", y_clip), ("raw", y_raw)):
        sse = 0.0
        sy2 = 0.0
        n = 0
        for a, b, ok, pred in rows:
            yy = y[a:b][ok]
            pp = pred[ok]
            m = np.isfinite(yy)
            sse += float(((yy[m] - pp[m]) ** 2).sum())
            sy2 += float((yy[m] ** 2).sum())
            n += int(m.sum())
        print(f"   vs {name:8s}: R2 = {100*(1 - sse/sy2):+7.2f}%   n={n:,}  E[y^2]={sy2/n:.4f}")
