"""Metrics needed for the 'our conclusions vs the paper's' comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import load_meta  # noqa: E402
from src.train import evaluate, target_y, year_bounds  # noqa: E402

cfg = load_config()
meta = load_meta(cfg)
bounds = year_bounds(meta)
y_all = target_y(meta, cfg)
out = Path(cfg["paths"]["out_dir"])
z = np.load(out / "nn3_f94_ens_preds.npz")
years = sorted(int(k) for k in z.files)

rows = []
for ty in years:
    a, b = bounds[ty]
    y = y_all[a:b]
    ok = np.isfinite(y)
    rows.append(pd.DataFrame({"ym": meta["ym"].to_numpy()[a:b][ok],
                              "y": y[ok], "p": z[str(ty)][ok]}))
d = pd.concat(rows, ignore_index=True)
ic = d.groupby("ym").apply(lambda g: g.p.rank().corr(g.y.rank()), include_groups=False)
print(f"NN3 ensemble  monthly rank IC: mean={ic.mean():+.4f}  median={ic.median():+.4f}  "
      f"share>0={100*(ic>0).mean():.1f}%")

E = float((d.y ** 2).mean())
print(f"E[y^2]={E:.4f}  n={len(d):,}")

# R2 of the level-only benchmark on this same sample
for nm, pred in (("level only", np.full(len(d), d.y.mean())),):
    print(f"{nm}: R2={100*(1-((d.y-pred)**2).mean()/E):+.2f}%")

# decile spread of realised annual returns sorted on predictions (no trading costs)
d["dec"] = d.groupby("ym")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
sp = d.groupby("dec")["y"].mean()
print("\nrealised annual excess return by predicted decile (equal-weighted):")
for k in range(10):
    print(f"  D{k+1}: {sp[k]:+.4f}")
print(f"  D10-D1 spread = {sp[9]-sp[0]:+.4f}")
