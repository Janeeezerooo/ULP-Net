import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_MAX, N_MAX = 1000, 1000
H, W = M_MAX + 1, N_MAX + 1
BATCH_SIZE = 32
STEPS = 93750         # 31,250 * 32 = 1,000,000 full-scale 1000x1000 instances
EVAL_SAMPLES = 2000

print(f"🚀 HAKUSAN H100 Production Run: ULP-Net (L2-Sorted + ColMaxPool) | Scale {M_MAX}x{N_MAX}", flush=True)

def generate_raw_lp(force_regime=None):
    m = random.randint(50, M_MAX)
    n = random.randint(50, N_MAX)
    regime = force_regime if force_regime else random.choices(
        ["dense", "sparse", "extreme", "mixed_oblique"], weights=[0.20, 0.30, 0.20, 0.30]
    )[0]

    k = 0 if random.random() < 0.45 else random.randint(1, max(3, int(n * 0.35)))

    if regime == "sparse":
        density = random.uniform(0.005, 0.05)
        mask = (np.random.rand(m, n) < density).astype(np.float32)
        A = np.random.uniform(-5.0, 5.0, size=(m, n)).astype(np.float32) * mask
    else:
        A = np.random.uniform(-5.0, 5.0, size=(m, n)).astype(np.float32)

    b = np.random.uniform(1.0, 10.0, size=(m,)).astype(np.float32)
    c = np.random.uniform(-5.0, 1.5, size=(n,)).astype(np.float32)
    y = np.zeros((n,), dtype=np.float32)

    if k > 0:
        unb_cols = random.sample(range(n), k)
        idx = 0
        while idx < len(unb_cols):
            rem = len(unb_cols) - idx
            if regime in ("mixed_oblique", "dense", "extreme") and rem >= 3 and random.random() < 0.45:
                j1, j2, j3 = unb_cols[idx], unb_cols[idx + 1], unb_cols[idx + 2]
                col1 = np.random.uniform(-3.0, 4.0, size=(m,)).astype(np.float32)
                col2 = np.random.uniform(-3.0, 4.0, size=(m,)).astype(np.float32)
                col3 = -(col1 + col2) - np.random.uniform(0.05, 1.0, size=(m,)).astype(np.float32)
                A[:, j1], A[:, j2], A[:, j3] = col1, col2, col3
                c[j1], c[j2], c[j3] = random.uniform(1.0, 5.0), random.uniform(1.0, 5.0), random.uniform(1.0, 5.0)
                y[j1] = y[j2] = y[j3] = 1.0
                idx += 3
            elif regime in ("mixed_oblique", "dense", "extreme") and rem >= 2 and random.random() < 0.50:
                j1, j2 = unb_cols[idx], unb_cols[idx + 1]
                col1 = np.random.uniform(-4.0, 3.0, size=(m,)).astype(np.float32)
                col2 = -col1 - np.random.uniform(0.1, 1.5, size=(m,)).astype(np.float32)
                A[:, j1], A[:, j2] = col1, col2
                c[j1], c[j2] = random.uniform(1.0, 5.0), random.uniform(1.0, 5.0)
                y[j1] = y[j2] = 1.0
                idx += 2
            else:
                j = unb_cols[idx]
                A[:, j] = -np.abs(A[:, j])
                if regime == "sparse":
                    A[:, j] *= (np.random.rand(m) < 0.25).astype(np.float32)
                c[j] = random.uniform(0.5, 5.0)
                y[j] = 1.0
                idx += 1

    bounded_cols = np.where(y == 0.0)[0]
    if len(bounded_cols) > 0:
        decoys = np.random.choice(bounded_cols, size=min(5, len(bounded_cols)), replace=False)
        for hn in decoys:
            A[:, hn] = -np.abs(A[:, hn])
            rows_blk = np.random.choice(m, size=min(2, m), replace=False)
            A[rows_blk, hn] = np.random.uniform(0.2, 2.5, size=len(rows_blk)).astype(np.float32)
            c[hn] = random.uniform(0.5, 4.0)

    if regime == "extreme":
        row_scales = (10.0 ** np.random.uniform(-3.0, 4.0, size=(m, 1))).astype(np.float32)
        A *= row_scales
        b *= row_scales[:, 0]
        c *= float(10.0 ** random.uniform(-3.0, 4.0))

    return A, b, c, y

def preprocess_lp(A, b, c, y):
    m, n = A.shape
    r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
    A_hat = A / r_norm[:, None]
    b_hat = b / r_norm
    c_hat = c / max(float(np.max(np.abs(c))), 1e-8)
    y_ord = y.copy()

    # Canonical L2-Norm Ordering
    r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
    A_hat, b_hat = A_hat[r_idx, :], b_hat[r_idx]
    c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
    A_hat, c_hat, y_ord = A_hat[:, c_idx], c_hat[c_idx], y_ord[c_idx]

    X_pad = np.zeros((1, H, W), dtype=np.float32)
    X_pad[0, 0, 1:n+1] = c_hat
    X_pad[0, 1:m+1, 0] = b_hat
    X_pad[0, 1:m+1, 1:n+1] = A_hat

    y_pad = np.full((N_MAX,), 2.0, dtype=np.float32)
    y_pad[:n] = y_ord
    return X_pad, y_pad

class LPStreamDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield preprocess_lp(*generate_raw_lp())

class ULPNetFinal(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.row_conv = nn.Conv2d(1, k, kernel_size=(H, 1))
        self.col_conv = nn.Conv2d(1, k, kernel_size=(1, W))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        f_row = self.row_conv(x).expand(-1, -1, H, W)
        f_col = self.col_conv(x).expand(-1, -1, H, W)
        feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        return self.decoder(self.pool(feat))[:, 0, 0, 1:N_MAX+1]

model = ULPNetFinal().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
scaler = torch.amp.GradScaler("cuda", enabled=False)
pos_w = torch.tensor(1.5, device=DEVICE)

loader = DataLoader(LPStreamDataset(), batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
it = iter(loader)

t0 = time.time()
model.train()
for step in range(1, STEPS + 1):
    xb, yb = next(it)
    xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(xb)
        mask = (yb != 2.0)
        loss = F.binary_cross_entropy_with_logits(logits[mask], yb[mask], pos_weight=pos_w)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 5000 == 0:
        print(f"Step {step}/{STEPS} ({step*BATCH_SIZE:,} instances) | Loss={loss.item():.4f} | Time={time.time()-t0:.1f}s", flush=True)

torch.save(model.state_dict(), "ulpnet_1000_a100_3m.pt")

def eval_regime(reg_name, count=1000):
    model.eval()
    tp = fp = fn = 0
    t_inf = 0.0
    with torch.no_grad():
        for _ in range(0, count, 32):
            batch = [preprocess_lp(*generate_raw_lp(reg_name)) for _ in range(32)]
            xb = torch.tensor(np.stack([b[0] for b in batch]), device=DEVICE)
            yb = np.stack([b[1] for b in batch])
            ts = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy()
            t_inf += (time.time() - ts)
            mask = (yb != 2.0)
            p_m, y_m = preds[mask], yb[mask]
            tp += int(((p_m == 1) & (y_m == 1)).sum())
            fp += int(((p_m == 1) & (y_m == 0)).sum())
            fn += int(((p_m == 0) & (y_m == 1)).sum())
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    ms_per_lp = (t_inf / count) * 1000.0
    return f1, prec, rec, ms_per_lp

print("\n" + "="*85)
print(f"{'Regime (1000x1000)':<20} | {'F1-Score':>10} | {'Precision':>10} | {'Recall':>10} | {'Latency (ms)':>12}")
print("="*85)
for reg in [None, "dense", "sparse", "extreme", "mixed_oblique"]:
    name = "Overall_Mixed" if reg is None else reg
    f1, pr, rc, lat = eval_regime(reg, count=1024)
    print(f"{name:<20} | {f1:>10.4f} | {pr:>10.4f} | {rc:>10.4f} | {lat:>10.3f} ms")
print("="*85)
