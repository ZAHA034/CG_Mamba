"""W4 pilot: native-probabilistic DL baseline (quantile-regression head).

Reviewer objection W4: the paper compared CG-Mamba's native APMD only against
*bolt-on* DL UQ (MC Dropout, ensemble-Gaussian), not against the standard
*native-probabilistic* DL heads (direct quantile regression / Gaussian-NLL).
This script trains an LSTM backbone with a direct 23-quantile pinball head
(native intervals, no post-hoc step) and reports native Cov95/WIS on
test_strict (national), to test whether the paper's claim --- "CG-Mamba is the
only DL model with native calibration" --- survives the right comparator.

HONEST EXPERIMENT (option 2): the result is reported exactly as it comes out.
  - If the quantile head under/over-covers far from 0.95 -> defends the claim.
  - If it reaches near-nominal Cov95 with competitive WIS -> narrows the claim
    (we reframe honestly; we do NOT hide it).

Eval space: z-scored target (same space CG-Mamba's APMD operates in). Cov95 is
scale-invariant; WIS is in z-score units (state the space when comparing).
Device: CPU by default (shared GPUs are saturated by other users).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.baselines.lstm import WeeklyMultiHorizonDataset  # noqa: E402
from src.data.loader import load_dataset_csv, load_norm_params  # noqa: E402
from src.eval.wis import REQUIRED_QUANTILES, wis  # noqa: E402

CSV = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM = _ROOT / "data/processed/normalization_params.json"
QLEVELS = np.asarray(REQUIRED_QUANTILES, dtype=np.float64)   # 23 FluSight levels
Q = len(QLEVELS)
LO_IDX = int(np.argmin(np.abs(QLEVELS - 0.025)))            # 0.025 -> lower 95%
HI_IDX = int(np.argmin(np.abs(QLEVELS - 0.975)))            # 0.975 -> upper 95%
TEST_STRICT_EW = 202240                                     # W40-2022


class QuantileLSTM(nn.Module):
    """LSTM encoder + direct 23-quantile head. Output [B, H, Q] (native)."""

    def __init__(self, input_dim=6, hidden=128, layers=2, H=4, n_q=Q, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, H * n_q)
        self.H, self.n_q = H, n_q

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).view(-1, self.H, self.n_q)


def pinball_loss(pred, y):
    """pred [B,H,Q], y [B,H]. Mean pinball over 23 quantiles."""
    ql = torch.tensor(QLEVELS, dtype=torch.float32, device=pred.device).view(1, 1, -1)
    err = y.unsqueeze(-1) - pred
    return torch.maximum(ql * err, (ql - 1.0) * err).mean()


def _stack(ds):
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, ys


def native_wis_cov(preds, y, mask=None):
    """preds [N,H,Q] (sorted), y [N,H]. Returns (cov95, wis_mean) over masked (n,h)."""
    N, H, _ = preds.shape
    if mask is None:
        mask = np.ones((N, H), dtype=bool)
    inside, wvals = [], []
    for h in range(H):
        m = mask[:, h]
        if not m.any():
            continue
        yh = y[m, h]
        lo, hi = preds[m, h, LO_IDX], preds[m, h, HI_IDX]
        inside.append(((yh >= lo) & (yh <= hi)).astype(np.float64))
        qf = {REQUIRED_QUANTILES[j]: preds[m, h, j] for j in range(Q)}
        wvals.append(np.asarray(wis(yh, qf), dtype=np.float64).reshape(-1))
    cov = float(np.concatenate(inside).mean())
    wis_mean = float(np.concatenate(wvals).mean())
    return cov, wis_mean


def train_eval(seed, device="cpu", hidden=128, layers=2, dropout=0.3,
               lookback=104, epochs=150, patience=25, lr=1e-3, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = load_dataset_csv(CSV)
    norm = load_norm_params(NORM)
    tr = WeeklyMultiHorizonDataset(df, "train", norm, lookback, 4)
    va = WeeklyMultiHorizonDataset(df, "val", norm, lookback, 4)
    te = WeeklyMultiHorizonDataset(df, "test", norm, lookback, 4)
    trl = DataLoader(tr, batch_size=32, shuffle=True)
    vx, vy = _stack(va)
    vx, vy_np = vx.to(device), vy.numpy()

    model = QuantileLSTM(6, hidden, layers, 4, Q, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_wis, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in trl:
            opt.zero_grad()
            loss = pinball_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp, _ = torch.sort(model(vx), dim=-1)
        _, vw = native_wis_cov(vp.cpu().numpy(), vy_np)
        if vw < best_wis - 1e-6:
            best_wis, bad = vw, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    # ---- test_strict national eval (native quantiles) ----
    tx, ty = _stack(te)
    model.eval()
    with torch.no_grad():
        tp, _ = torch.sort(model(tx.to(device)), dim=-1)
    tp, ty = tp.cpu().numpy(), ty.numpy()
    eps = df["epiweek"].to_numpy()
    ends = te.window_ends
    N, H, _ = tp.shape
    mask = np.zeros((N, H), dtype=bool)
    for i in range(N):
        for h in range(H):
            mask[i, h] = eps[ends[i] + 1 + h] >= TEST_STRICT_EW
    cov, wis_m = native_wis_cov(tp, ty, mask)
    if verbose:
        print(f"  seed {seed}: test_strict native Cov95={cov:.3f}  WIS={wis_m:.3f}  "
              f"(val-WIS={best_wis:.3f}, ep={ep + 1}, n_pairs={int(mask.sum())})")
    return dict(seed=seed, cov95=cov, wis=wis_m, val_wis=best_wis, epochs=ep + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    print(f"W4 pilot: LSTM quantile-head (native 23-q pinball) | device={args.device}")
    print(f"Compare vs CG-Mamba native APMD (national test_strict): Cov95 0.993, WIS 0.399")
    print(f"Bolt-on LSTM MC-Dropout (paper): native Cov95 0.335")
    res = [train_eval(s, args.device, args.hidden, args.layers, args.dropout,
                      epochs=args.epochs) for s in args.seeds]
    covs = np.array([r["cov95"] for r in res])
    wiss = np.array([r["wis"] for r in res])
    print(f"\n== LSTM quantile-head native (n={len(res)} seeds) ==")
    print(f"  Cov95 = {covs.mean():.3f} +/- {covs.std():.3f}")
    print(f"  WIS   = {wiss.mean():.3f} +/- {wiss.std():.3f}  (z-score space)")
    print(f"  |Cov95 - 0.95| = {abs(covs.mean() - 0.95):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
