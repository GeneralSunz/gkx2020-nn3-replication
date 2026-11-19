"""NN3 (32-16-8 ReLU, batch-norm, Adam, early stopping, seed ensemble).

Implements Section 1.7 of Gu, Kelly & Xiu (2020) as closely as the paper's body
text allows.  Three additions beyond the paper's text, documented in README.md
because the Internet Appendix (which holds the real recipe) is not public:

  1. the target is centred on the training-window mean -- a reparameterisation that
     lets the optimiser spend its steps on the cross-sectional signal instead of
     on an intercept of 5-10%/yr;
  2. the prediction amplitude is calibrated on the validation window by a single
     scalar alpha, because an unconstrained MLP systematically mis-scales its
     forecasts (too small in calm years, too large in crises);
  3. optional Huber loss for the fat-tailed annual-return target.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .common import log


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            # Internet Appendix B.3: BN is applied to each activation AFTER the
            # ReLU transformation, i.e. Linear -> ReLU -> BatchNorm.
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, 1))  # linear output, eq. (18)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _l1(model: nn.Module) -> torch.Tensor:
    total = None
    for p in model.parameters():
        if p.dim() > 1:  # weight matrices only, not biases / BN params
            s = p.abs().sum()
            total = s if total is None else total + s
    return total if total is not None else torch.zeros(())


def _as_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True))


def _predict_array(model: nn.Module, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = _as_tensor(X[a : a + batch])
            out[a : a + batch] = model(xb).squeeze(1).numpy()
    return out


def _val_preds(model, blocks, scale):
    parts = []
    for X, y in blocks:
        ok = np.isfinite(y)
        if ok.any():
            parts.append(_predict_array(model, X[ok]) / scale)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _val_truth(blocks):
    parts = [y[np.isfinite(y)] for _, y in blocks]
    return np.concatenate(parts) if parts else np.zeros(0)


def fit(
    loader,
    y_by_year: dict[int, np.ndarray],
    train_years: list[int],
    valid_years: list[int],
    hidden: list[int],
    hp: dict,
    seed: int,
    n_in: int,
    ym_by_year: dict[int, np.ndarray] | None = None,
    eval_by_year: dict[int, np.ndarray] | None = None,
    log_prefix: str = "",
    max_epochs: int | None = None,
) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = MLP(n_in, hidden, batch_norm=hp.get("batch_norm", True))
    wd = float(hp.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=wd)
    l1 = float(hp.get("l1", 0.0))
    bs = int(hp["batch_size"])
    scale = float(hp["target_scale"])
    patience = int(hp["patience"])
    n_epochs = int(max_epochs if max_epochs is not None else hp["max_epochs"])
    loss_name = hp.get("loss", "mse")

    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=float(hp.get("huber_delta", 20.0)), reduction="mean")
    else:
        criterion = nn.MSELoss()

    # Centre the target on the training-window mean: the excess-return level is
    # 5-10%/yr while the cross-sectional signal is only ~2-3% of variance, so an
    # un-centred MLP burns its early epochs on the intercept and early stopping
    # cuts it off before it gets there.
    train_mean = float(np.nanmean(np.concatenate([y_by_year[y] for y in train_years])))
    log(f"{log_prefix} train_mean={train_mean:+.4f}  loss={loss_name}  l1={l1:g}")

    valid_blocks = [
        (np.asarray(loader(y)), y_by_year[y],
         ym_by_year[y] if ym_by_year is not None else np.arange(len(y_by_year[y])),
         (eval_by_year[y] if eval_by_year is not None else y_by_year[y]))
        for y in valid_years
    ]
    valid_idx = {y: np.flatnonzero(np.isfinite(y_by_year[y])) for y in train_years}
    best = (np.inf, None, -1)
    bad = 0
    hist = []
    t0 = time.time()

    def _within_month_ic(p, t, mm):
        """Pooled within-month correlation, i.e. the cross-sectional signal.

        Each month is demeaned before the correlation is formed, so the (huge)
        across-month variation in the annual excess return no longer dominates.
        This is the object that actually drives R2_oos; the earlier pooled
        correlation mostly rewarded getting the month level right.
        """
        codes, uniq = pd.factorize(mm, sort=False)
        n = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
        ok = n > 1
        sp = np.bincount(codes, weights=p, minlength=len(uniq))
        st = np.bincount(codes, weights=t, minlength=len(uniq))
        p_dev = p - (sp / np.maximum(n, 1))[codes]
        t_dev = t - (st / np.maximum(n, 1))[codes]
        num = np.bincount(codes, weights=p_dev * t_dev, minlength=len(uniq))
        den = np.sqrt(
            np.bincount(codes, weights=p_dev ** 2, minlength=len(uniq))
            * np.bincount(codes, weights=t_dev ** 2, minlength=len(uniq))
        )
        ic = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        return float(ic[ok].mean()) if ok.any() else 0.0

    def _collect(m):
        ps, ts, ms = [], [], []
        es = []
        for X, y, mm, ev in valid_blocks:
            ok = np.isfinite(y)
            ps.append(_predict_array(m, X[ok]) / scale)
            ts.append(y[ok] - train_mean)
            ms.append(mm[ok])
            es.append(ev[ok])
        return (np.concatenate(ps).astype(np.float64),
                np.concatenate(ts).astype(np.float64),
                np.concatenate(ms),
                np.concatenate(es).astype(np.float64))

    def evaluate_valid(m) -> float:
        """Early stopping on 1 - mean monthly cross-sectional IC (lower is better)."""
        p, t, mm, _ = _collect(m)
        if p.std() < 1e-12 or t.std() < 1e-12:
            return 1.0
        return 1.0 - _within_month_ic(p, t, mm)

    for epoch in range(n_epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for y in rng.permutation(train_years):
            X = np.asarray(loader(y))
            yv = (y_by_year[y] - train_mean) * scale
            idx = rng.permutation(valid_idx[y])
            for a in range(0, len(idx), bs):
                sel = idx[a : a + bs]
                if len(sel) < 2:  # BatchNorm needs >1 row
                    continue
                xb = _as_tensor(X[sel])
                yb = torch.from_numpy(yv[sel].astype(np.float32)).unsqueeze(1)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                if l1 > 0:
                    loss = loss + l1 * _l1(model)
                loss.backward()
                opt.step()
                ep_loss += loss.detach().item() * len(sel)
                ep_n += len(sel)
            del X
        val_mse = evaluate_valid(model)
        ep_loss /= max(ep_n, 1)
        hist.append((epoch, ep_loss, val_mse))
        improved = np.isfinite(val_mse) and val_mse < best[0] - 1e-12
        if improved:
            best = (val_mse, copy.deepcopy(model.state_dict()), epoch)
            bad = 0
        else:
            bad += 1
        if epoch % 5 == 0 or improved:
            log(f"{log_prefix} ep{epoch:3d} train={ep_loss:.5f} valid={val_mse:.6f} "
                f"best={best[0]:.6f}@{best[2]} bad={bad} ({time.time()-t0:.0f}s)")
        if bad >= patience:
            log(f"{log_prefix} early stop at epoch {epoch} (best {best[2]})")
            break

    if best[1] is not None:
        model.load_state_dict(best[1])

    # ---- amplitude calibration on the validation window -------------------
    # fit the scale within month, consistent with the cross-sectional objective
    p, _, mm, ev = _collect(model)
    t = ev - np.nanmean(ev)
    codes, uniq = pd.factorize(mm, sort=False)
    n = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
    p_dev = p - (np.bincount(codes, weights=p, minlength=len(uniq))
                 / np.maximum(n, 1))[codes]
    t_dev = t - (np.bincount(codes, weights=t, minlength=len(uniq))
                 / np.maximum(n, 1))[codes]
    var = float((p_dev ** 2).mean()) if len(p) else 0.0
    alpha = float((p_dev * t_dev).mean() / var) if var > 1e-12 else 0.0
    alpha = float(np.clip(alpha, 0.0, 3.0))
    log(f"{log_prefix} alpha={alpha:+.3f}  (val pred sd={p.std():.5f}, "
        f"val y sd={t.std():.4f}, IC={1.0 - best[0]:+.4f})")

    return model, {
        "best_valid_mse": float(best[0]),
        "best_epoch": int(best[2]),
        "train_mean": train_mean,
        "alpha": alpha,
        "val_y_sd": float(t.std()),
        "val_ic": float(1.0 - best[0]),
        "history": hist,
    }


def ensemble_predict(models: list[MLP], offsets: list[float], alphas: list[float],
                     X: np.ndarray, target_scale: float) -> np.ndarray:
    preds = np.zeros(len(X), dtype=np.float64)
    for m, off, a in zip(models, offsets, alphas):
        preds += off + a * (_predict_array(m, X) / target_scale)
    return (preds / len(models)).astype(np.float32)
