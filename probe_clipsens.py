"""How sensitive is R2 to the tail treatment of the target?

The model does not change here -- only the winsorisation level applied to y when
computing R2.  If R2 climbs to ~3.4% as the tails are trimmed, then the gap to the
paper is a target-tail question rather than a model-quality one.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import load_meta  # noqa: E402
from src.train import year_bounds  # noqa: E402

cfg = load_config()
meta = load_meta(cfg)
bounds = year_bounds(meta)
y = meta["y12"].to_numpy().astype(np.float64)
out = Path(cfg["paths"]["out_dir"])

for tag in ("nn3_f94_ens", "nn3_raw"):
    f = out / f"{tag}_preds.npz"
    if not f.exists():
        continue
    z = np.load(f)
    years = sorted(int(k) for k in z.files)
    print(f"\n{tag}   (model unchanged; only the evaluation clip moves)")
    print(f"{'clip':>8} {'E[y^2]':>9} {'R2':>9}")
    for q in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        if q > 0:
            lo, hi = np.nanquantile(y, [q, 1 - q])
            yy = np.clip(y, lo, hi)
        else:
            yy = y
        sse = sy2 = 0.0
        n = 0
        for ty in years:
            a, b = bounds[ty]
            t = yy[a:b]
            p = z[str(ty)]
            ok = np.isfinite(t)
            sse += float(((t[ok] - p[ok]) ** 2).sum())
            sy2 += float((t[ok] ** 2).sum())
            n += int(ok.sum())
        print(f"{q:8.3f} {sy2/n:9.4f} {100*(1-sse/sy2):+8.2f}%")
