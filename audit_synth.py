"""Definitive code-correctness test: inject a KNOWN signal and recover it.

Build a synthetic target that is a linear function of two characteristics plus
noise, so the attainable out-of-sample R2 is known analytically:

    y = b1*c1 + b2*c2 + eps,      R2_max = var(b1*c1 + b2*c2) / var(y)

Then run the real walk-forward (same features, same split, same NN, same sizing)
and compare.  If the pipeline recovers roughly R2_max, the training/evaluation
code is correct and the paper gap cannot be a code bug.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import src.train as T  # noqa: E402
from src.common import load_config  # noqa: E402
from src.features import load_meta, year_path  # noqa: E402

B1, B2, NOISE = 0.030, 0.015, 0.060
YEARS = [1990, 1995, 2005, 2012]

cfg = load_config()
meta = load_meta(cfg)
bounds = T.year_bounds(meta)

# --- build the synthetic target on the design blocks ---------------------
y_parts = []
for y in sorted(bounds):
    a, b = bounds[y]
    Z = np.load(year_path(cfg, y), mmap_mode="r")
    c1 = np.asarray(Z[:, 0], dtype=np.float64)   # mvel1 (rank [-1,1])
    c2 = np.asarray(Z[:, 8], dtype=np.float64)   # mom1m
    y_parts.append(B1 * c1 + B2 * c2)
signal = np.concatenate(y_parts)
del y_parts
rng = np.random.default_rng(7)
noise = rng.normal(0, NOISE, size=len(signal))
y_synth = signal + noise

r2_max = float(signal.var() / y_synth.var())
print(f"signal sd={signal.std():.4f}  noise sd={NOISE:.4f}  "
      f"theoretical OOS R2 ceiling = {100*r2_max:+.2f}%")

y_series = pd.Series(y_synth, index=meta.index)


def fake_target_y(m, c):
    return y_series.reindex(m.index).to_numpy(dtype=np.float64)


T.target_y = fake_target_y

cfg2 = dict(cfg)
cfg2["paths"] = dict(cfg["paths"])
cfg2["paths"]["out_dir"] = str(Path(cfg["paths"]["out_dir"]) / "synth")
Path(cfg2["paths"]["out_dir"]).mkdir(parents=True, exist_ok=True)
cfg2["model"] = dict(cfg["model"])
cfg2["model"]["n_features"] = 94

res = T.run_nn(cfg2, YEARS, seeds=[1], tag="synth")
print(f"\nrecovered OOS R2 = {res['All']:+.2f}%   theoretical ceiling = {100*r2_max:+.2f}%   "
      f"recovery ratio = {res['All']/(100*r2_max):.2f}")
