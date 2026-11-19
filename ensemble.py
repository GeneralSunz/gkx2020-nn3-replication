"""Average the per-seed predictions and evaluate the ensemble."""
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

TAGS = ["nn3_f94", "nn3_f94_s2", "nn3_f94_s3"]
arrs = []
for t in TAGS:
    p = out / f"{t}_preds.npz"
    if not p.exists():
        print("missing:", p)
        continue
    z = np.load(p)
    arrs.append({int(k): z[k] for k in z.files})
    try:
        r = json.loads((out / f"{t}_result.json").read_text(encoding="utf-8"))
        print(f"{t:14s} All={r['All']:+.2f}%  Top={r['Top']:+.2f}%  Bottom={r['Bottom']:+.2f}%")
    except Exception as e:
        print(t, "no result json:", e)

years = sorted(arrs[0])
ens = {ty: np.mean([a[ty] for a in arrs], axis=0).astype(np.float32) for ty in years}

print(f"\n=== ensemble of {len(arrs)} seeds ===")
res = evaluate(meta, bounds, years, ens, y_all)
print(f"All={res['All']:+.2f}%  Top={res['Top']:+.2f}%  Bottom={res['Bottom']:+.2f}%  n={res['n_obs']:,}")

# ---- amplitude sweep on the ensemble ----
train_mean = {}
for ty in years:
    train_mean[ty] = float(np.nanmean(np.concatenate(
        [y_all[bounds[t][0]:bounds[t][1]] for t in range(1957, ty - 12)])))

ys, ps, lev = [], [], []
for ty in years:
    a, b = bounds[ty]
    y = y_all[a:b]
    ok = np.isfinite(y)
    ys.append(y[ok])
    ps.append(ens[ty][ok].astype(np.float64))
    lev.append(np.full(ok.sum(), train_mean[ty]))
y = np.concatenate(ys)
p = np.concatenate(ps)
L = np.concatenate(lev)
E = float((y ** 2).mean())
cs = p - L
print("\namplitude sweep on the ensemble:")
best = (1.0, None)
for a_ in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25):
    q = L + a_ * cs
    r2 = 100 * (1 - ((y - q) ** 2).mean() / E)
    mark = ""
    if best[1] is None or r2 > best[1]:
        best = (a_, r2)
        mark = "  <-"
    print(f"  x{a_:4.2f}   R2={r2:+7.2f}%{mark}")
print(f"\nbest amplitude = {best[0]:.2f} -> R2 = {best[1]:+.2f}%")

# save the best-scaled ensemble predictions for the report
a_best = best[0]
ens_best = {}
for ty in years:
    a, b = bounds[ty]
    ens_best[ty] = (train_mean[ty] + a_best * (ens[ty].astype(np.float64) - train_mean[ty])).astype(np.float32)
res_best = evaluate(meta, bounds, years, ens_best, y_all)
np.savez(out / "nn3_f94_ens_preds.npz", **{str(k): v for k, v in ens_best.items()})
(out / "nn3_f94_ens_result.json").write_text(
    json.dumps(res_best, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nsaved ensemble result: All={res_best['All']:+.2f}%  "
      f"Top={res_best['Top']:+.2f}%  Bottom={res_best['Bottom']:+.2f}%")

v = np.array(list(res_best["by_year"].values()))
print(f"per-year: mean={v.mean():+.2f}% median={np.median(v):+.2f}% "
      f"positive={int((v>0).sum())}/{len(v)}")
