"""Does the monthly signal persist? IC of month-t forecasts at horizons 1..12.

If the signal decays within a few months, summing monthly forecasts cannot build an
annual forecast, and the paper's large annual R2 must come from a different
(persistent) component.  If it persists, our annual model is leaving money on the
table and the fix is in how the annual target is learned.
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

cfg = load_config()
meta = load_meta(cfg)
bounds = year_bounds(meta)
out = Path(cfg["paths"]["out_dir"])

z = np.load(out / "nn3_m1_preds.npz")
m1 = np.full(len(meta), np.nan)
for ty in sorted(int(k) for k in z.files):
    a, b = bounds[ty]
    m1[a:b] = z[str(ty)][: b - a]

d = pd.DataFrame({"permno": meta["permno"].to_numpy(),
                  "ym": meta["ym"].to_numpy(),
                  "p": m1})
d = d.sort_values(["ym", "permno"], kind="mergesort")

# realised excess return per permno-month
p = pd.read_parquet(PANEL, columns=["permno", "ym", "ret_ex"])
p = p.rename(columns={"ret_ex": "r"})
d = d.merge(p, on=["permno", "ym"], how="left")
d = d.sort_values(["permno", "ym"], kind="mergesort")

grp = d.groupby("permno")
print("horizon   mean IC (rank)   share>0")
for k in range(0, 13):
    rr = grp["r"].shift(-k)
    tmp = pd.DataFrame({"ym": d["ym"], "p": d["p"], "r": rr}).dropna(subset=["r"])
    ic = tmp.groupby("ym").apply(
        lambda g: g.p.rank().corr(g.r.rank()) if len(g) > 30 else np.nan,
        include_groups=False).dropna()
    print(f"   k={k:2d}    {ic.mean():+.4f}        {100*(ic>0).mean():.0f}%")
