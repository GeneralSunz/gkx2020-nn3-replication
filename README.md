# Replicating Gu, Kelly & Xiu (2020): NN3 at the Annual Horizon

A careful, audit-first replication of **Table 2** (annual out-of-sample $R^2$) of

> Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.*
> **The Review of Financial Studies** 33(5), 2223–2273.

using the authors' publicly released characteristics panel and a CRSP monthly return extract.
The model is **NN3**, the paper's headline three-hidden-layer feed-forward network.

---

## Results at a glance

Test period 1987–2016, walk-forward with annual refits, $R^2_{oos}$ as in the paper's eq. (19)
(denominator = sum of squared excess returns, **not** demeaned).

| Horizon | Model | This repo | Paper | Third-party replication |
|---|---|---|---|---|
| Monthly (Table 1) | NN3 | **+0.47%** | 0.40% | 0.454% |
| Annual (Table 2) | NN3, winsorised target | **+2.71%** | 3.40% | _none published_ |
| Annual (Table 2) | NN3, raw excess returns | **+1.61%** | 3.40% | _none published_ |

Ancillary benchmarks on the same sample (annual, winsorised metric):
constant-mean forecast **+1.97%**, three-characteristic OLS **+1.94%**, ridge on the full
920-column design **+1.66%**.

Economic value: the decile long–short portfolio is strictly monotone
(D1 −3.1% → D10 +17.4%, spread **+20.5%/yr**) with an annualised Sharpe ratio of **1.16**,
against the paper's 1.35 and a third-party 1.43.

---

## Why the annual number falls short — the short version

This repository is as much a *negative result, carefully established* as a replication.
Four independent findings locate the gap:

1. **The pipeline is correct.** A synthetic-signal injection test
   (`audit_synth.py`) builds a target with a known attainable $R^2$ of **+10.68%**
   and the pipeline recovers **+10.60%** — a recovery ratio of **0.99**.
2. **The monthly result reproduces**, and lands between the paper (0.40%) and the most
   rigorous public replication (0.454%), with the same structural pattern
   (under-performance on large caps, over-performance on small caps).
3. **The annual model already beats the monthly signal's own ceiling.** The monthly
   forecasts' cross-sectional IC decays from 0.037 at $k=1$ to ~0.025 at $k=12$, which bounds
   the annual $R^2$ at about **+0.84%**. The annual model reaches **+1.32%**, i.e. it *does*
   learn something the monthly model cannot.
4. **The paper's 3.40% would require an IC of 0.185**, roughly twice what the monthly signal
   can sustain on this data.

In addition, all six public replications we surveyed target the monthly Table 1 only; none
reports the annual Table 2. The most rigorous of them also fails to reproduce the paper's
OLS benchmark (−170% vs the paper's −3.5%), a difficulty we hit as well
(bare OLS blows up to −1.2e8%; adding a ridge term of λ ≈ 5e-4 reproduces the paper's −34.86%).

A 57-page Chinese report with the full argument, tables and source listings is in
[`report/report.pdf`](report/report.pdf).

---

## Repository layout

```
config.json          all tunable parameters (paths, target horizon, features, training)
run.py               CLI: `prep`, `nn`, `ols`
probe.py             quick single/multi-year run for tuning and diagnostics
ensemble.py          average per-seed predictions and score the ensemble
src/
  common.py          config loading, logging
  prep.py            characteristics + CRSP merge, y1 / y12 construction, panel cache
  features.py        cross-sectional ranking, median imputation, interactions, dummies
  nn.py              MLP, Adam training, early stopping, amplitude calibration
  train.py           walk-forward loop, target construction, R², linear benchmarks
audit_*.py           validation scripts (see below)
probe_*.py           diagnostic probes (see below)
report/              LaTeX source and compiled PDF
```

### Validation scripts

These exist to rule out "the result is wrong" before interpreting it.
They are the backbone of the claims above.

| Script | What it establishes |
|---|---|
| `audit_synth.py` | Injects a known signal; pipeline recovers 99% of the attainable $R^2$ |
| `audit_prep.py` | Recomputes one month's ranks from the raw CSV; max difference 3e-4 |
| `audit_y12.py` | Hand-computes a 12-month product; max difference 3e-7 |
| `audit_target.py` | Uses three characteristics' ICs to confirm the target window start |
| `audit_fit.py` | In-sample / validation / test IC, to separate over- from under-fitting |
| `audit_counts.py` | Monthly stock counts across CRSP, characteristics file and panel |
| `probe_ols_ridge.py` | Ridge sweep locating the paper's OLS benchmark at λ ≈ 5e-4 |
| `probe_persist.py` | Monthly signal decay over horizons 1–12 |
| `probe_clipsens.py` | Sensitivity of $R^2$ to the evaluation-time winsorisation level |
| `probe_bottomup.py` | Bottom-up annual forecast built from monthly predictions |

---

## Data

Two inputs are required; neither is redistributed here.

1. **Characteristics** — `datashare.csv` from the authors' site
   (<https://dachxiu.chicagobooth.edu/download/datashare.zip>).
   94 characteristics, `permno`, `DATE`, `sic2`, 1957-03 to 2016-12.
2. **Monthly returns** — a CRSP extract with columns `permno, md, me, retadj, prc, shrcd, exchcd`,
   where `md` is a sequential month index with `md = 1` corresponding to 1925-12.
   Available under a WRDS licence.

Point `config.json` at both files:

```json
"paths": { "datashare": ".../datashare.csv", "crsp": ".../crsp_ret_all.csv" }
```

The audit scripts also contain absolute data paths near the top of each file; edit those
if you relocate the data.

---

## Setup and run

```bash
pip install -r requirements.txt

python run.py prep                      # build the panel cache and per-year design matrices
python run.py nn --seeds 20240101 --tag nn3    # full 1987-2016 walk-forward
python ensemble.py                      # average several seeds, sweep amplitude
```

Key `config.json` switches:

| Key | Values | Meaning |
|---|---|---|
| `data.target` | `annual` \| `monthly` \| `annual_rank` | prediction horizon / target transform |
| `data.y_clip_quantiles` | `[0.005, 0.995]` or `null` | winsorisation of the annual target |
| `model.n_features` | `94` \| `"chars+ind"` \| `null` | characteristics only / + industry / full 920 |
| `model.pred_sizing` | `"ic"` \| `"alpha"` | amplitude calibration rule |

Training a full walk-forward takes roughly 30–90 minutes on the 94-feature configuration.

---

## Implementation notes worth knowing

* **Batch-norm placement.** The Internet Appendix (Algorithm 7) applies BN *after* the ReLU,
  i.e. `Linear → ReLU → BatchNorm`. Many defaults do the opposite. This repo follows the appendix.
* **Early stopping.** Validation MSE is useless here — the across-month variation of the annual
  return dwarfs the model's contribution, every epoch scores identically, and training stops at
  epoch 1. We stop on the **within-month cross-sectional IC** instead.
* **Amplitude.** Network output scale drifts with the input distribution (in 2008 it was 3× too
  wide and cost about a full percentage point). Forecasts are re-standardised each month and
  rescaled to `IC × σ_y` measured on the validation window.
* **Hyperparameters** come from the paper's Internet Appendix Table A.5:
  `L1 ∈ (1e-5, 1e-3)`, `LR ∈ {0.001, 0.01}`, batch 10,000, epochs 100, patience 5,
  Adam defaults, ensemble of 10 seeds (we use 3).

---

## Citation

```bibtex
@article{gu2020empirical,
  title   = {Empirical Asset Pricing via Machine Learning},
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal = {The Review of Financial Studies},
  volume  = {33},
  number  = {5},
  pages   = {2223--2273},
  year    = {2020}
}
```

## License

MIT — see [LICENSE](LICENSE). The authors' data is governed by its own terms.
