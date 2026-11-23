import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
M_MAX, N_MAX = 100, 100
H, W = M_MAX + 1, N_MAX + 1
BATCH_SIZE = 256
STEPS = (10_000_000 // BATCH_SIZE) + 1  # 10,000,128 instances per model

print(f"🚀 10-MILLION Activation Function Ablation (ColAvgPool) on {NUM_GPUS} GPU ({DEVICE}) | Batch={BATCH_SIZE} | Steps={STEPS}", flush=True)

def generate_batch_fast(B=BATCH_SIZE, force_regime=None):
    X_batch = np.zeros((B, 1, H, W), dtype=np.float32)
    y_batch = np.full((B, N_MAX), 2.0, dtype=np.float32)

    regimes = [force_regime] * B if force_regime else random.choices(
        ["dense", "sparse", "extreme", "mixed_oblique"], weights=[0.20, 0.25, 0.20, 0.35], k=B
    )

    for idx in range(B):
        m = random.randint(15, M_MAX)
        n = random.randint(15, N_MAX)
        reg = regimes[idx]
        k = 0 if random.random() < 0.45 else random.randint(1, max(3, int(n * 0.4)))

        if reg == "sparse":
            dens = random.uniform(0.01, 0.10)
            A = (np.random.rand(m, n).astype(np.float32) < dens) * np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float32)
        else:
            A = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float32)

        b = np.random.uniform(1.0, 10.0, (m,)).astype(np.float32)
        c = np.random.uniform(-5.0, 1.5, (n,)).astype(np.float32)
        y = np.zeros((n,), dtype=np.float32)

        if k > 0:
            unb_cols = random.sample(range(n), k)
            ptr = 0
            while ptr < len(unb_cols):
                rem = len(unb_cols) - ptr
                if reg in ("mixed_oblique", "dense", "extreme") and rem >= 3 and random.random() < 0.45:
                    j1, j2, j3 = unb_cols[ptr], unb_cols[ptr+1], unb_cols[ptr+2]
                    c1 = np.random.uniform(-3.0, 4.0, (m,)).astype(np.float32)
                    c2 = np.random.uniform(-3.0, 4.0, (m,)).astype(np.float32)
                    c3 = -(c1 + c2) - np.random.uniform(0.05, 1.0, (m,)).astype(np.float32)
                    A[:, j1], A[:, j2], A[:, j3] = c1, c2, c3
                    c[j1], c[j2], c[j3] = random.uniform(1.0, 5.0), random.uniform(1.0, 5.0), random.uniform(1.0, 5.0)
                    y[j1] = y[j2] = y[j3] = 1.0
                    ptr += 3
                elif reg in ("mixed_oblique", "dense", "extreme") and rem >= 2 and random.random() < 0.50:
                    j1, j2 = unb_cols[ptr], unb_cols[ptr+1]
                    c1 = np.random.uniform(-4.0, 3.0, (m,)).astype(np.float32)
                    c2 = -c1 - np.random.uniform(0.1, 1.5, (m,)).astype(np.float32)
                    A[:, j1], A[:, j2] = c1, c2
                    c[j1], c[j2] = random.uniform(1.0, 5.0), random.uniform(1.0, 5.0)
                    y[j1] = y[j2] = 1.0
                    ptr += 2
                else:
                    j = unb_cols[ptr]
                    A[:, j] = -np.abs(A[:, j])
                    if reg == "sparse":
                        A[:, j] *= (np.random.rand(m).astype(np.float32) < 0.25)
                    c[j] = random.uniform(0.5, 5.0)
                    y[j] = 1.0
                    ptr += 1

        bounded_cols = np.where(y == 0.0)[0]
        if len(bounded_cols) > 0:
            decoys = np.random.choice(bounded_cols, size=min(4, len(bounded_cols)), replace=False)
            for hn in decoys:
                A[:, hn] = -np.abs(A[:, hn])
                rows_blk = np.random.choice(m, size=min(2, m), replace=False)
                A[rows_blk, hn] = np.random.uniform(0.2, 2.5, size=len(rows_blk)).astype(np.float32)
                c[hn] = random.uniform(0.5, 4.0)

        if reg == "extreme":
            r_sc = (10.0 ** np.random.uniform(-3.0, 4.0, (m, 1))).astype(np.float32)
            A *= r_sc
            b *= r_sc[:, 0]
            c *= float(10.0 ** random.uniform(-3.0, 4.0))

        r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
        A_hat = A / r_norm[:, None]
        b_hat = b / r_norm
        c_hat = c / max(float(np.max(np.abs(c))), 1e-8)

        r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
        A_hat, b_hat = A_hat[r_idx], b_hat[r_idx]
        c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
        A_hat, c_hat, y = A_hat[:, c_idx], c_hat[c_idx], y[c_idx]

        X_batch[idx, 0, 0, 1:n+1] = c_hat
        X_batch[idx, 0, 1:m+1, 0] = b_hat
        X_batch[idx, 0, 1:m+1, 1:n+1] = A_hat
        y_batch[idx, :n] = y

    return X_batch, y_batch

def get_act(act_type):
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "silu":
        return nn.SiLU(inplace=True)
    elif act_type == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {act_type}")

class ULPNetActVariant(nn.Module):
    def __init__(self, act_type="relu", k=32):
        super().__init__()
        self.row_conv = nn.Conv2d(1, k, kernel_size=(H, 1))
        self.col_conv = nn.Conv2d(1, k, kernel_size=(1, W))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            get_act(act_type),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            get_act(act_type),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            get_act(act_type),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        f_row = self.row_conv(x).expand(-1, -1, H, W)
        f_col = self.col_conv(x).expand(-1, -1, H, W)
        feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        return self.head(self.pool(feat))[:, 0, 0, 1:N_MAX+1]

print("📦 Building 32,768 base pool + 10,000 benchmark test set (low-CPU mode)...", flush=True)
POOL_SIZE = 32768
X_pool, y_pool = generate_batch_fast(POOL_SIZE)
X_eval_all, y_eval_all = generate_batch_fast(4000)
X_eval_sp, y_eval_sp   = generate_batch_fast(2000, "sparse")
X_eval_ex, y_eval_ex   = generate_batch_fast(2000, "extreme")
X_eval_ob, y_eval_ob   = generate_batch_fast(2000, "mixed_oblique")

X_pool_gpu = torch.from_numpy(X_pool).to(DEVICE)
y_pool_gpu = torch.from_numpy(y_pool).to(DEVICE)

def eval_f1(model, X_e, y_e):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, len(X_e), 512):
            xb = torch.from_numpy(X_e[i:i+512]).to(DEVICE)
            yb = y_e[i:i+512]
            preds = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy()
            mask = (yb != 2.0)
            p_m, y_m = preds[mask], yb[mask]
            tp += int(((p_m == 1) & (y_m == 1)).sum())
            fp += int(((p_m == 1) & (y_m == 0)).sum())
            fn += int(((p_m == 0) & (y_m == 1)).sum())
    return (2 * tp) / max(2 * tp + fp + fn, 1)

variants = [
    ("1_ReLU (Baseline)", "relu"),
    ("2_LeakyReLU (0.01)", "leaky_relu"),
    ("3_GELU (Smooth)", "gelu"),
    ("4_SiLU / Swish", "silu"),
    ("5_Tanh (Symmetric)", "tanh"),
]

results = []
scaler = torch.amp.GradScaler("cuda")
pos_w = torch.tensor(1.5, device=DEVICE)

for label, atype in variants:
    t0 = time.time()
    model = ULPNetActVariant(atype).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n▶️ Training {label} (Params: {n_params:,}) on {STEPS*BATCH_SIZE:,} instances...", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    model.train()
    for step in range(1, STEPS + 1):
        if step % 64 == 0:
            rep_idx = np.random.randint(0, POOL_SIZE, size=32)
            nx, ny = generate_batch_fast(32)
            X_pool_gpu[rep_idx] = torch.from_numpy(nx).to(DEVICE)
            y_pool_gpu[rep_idx] = torch.from_numpy(ny).to(DEVICE)

        b_idx = torch.randint(0, POOL_SIZE, (BATCH_SIZE,), device=DEVICE)
        xb = X_pool_gpu[b_idx]
        yb = y_pool_gpu[b_idx]

        if step % 2 == 0:
            xb = xb * torch.empty((BATCH_SIZE, 1, 1, 1), device=DEVICE).uniform_(0.7, 1.0)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(xb)
            mask = (yb != 2.0)
            loss = F.binary_cross_entropy_with_logits(logits[mask], yb[mask], pos_weight=pos_w)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 5000 == 0:
            print(f"   ... step {step}/{STEPS} ({step*BATCH_SIZE:,} instances) [{time.time()-t0:.1f}s]", flush=True)

    f_all = eval_f1(model, X_eval_all, y_eval_all)
    f_sp  = eval_f1(model, X_eval_sp, y_eval_sp)
    f_ex  = eval_f1(model, X_eval_ex, y_eval_ex)
    f_ob  = eval_f1(model, X_eval_ob, y_eval_ob)
    elapsed = time.time() - t0
    results.append((label, n_params, elapsed, f_all, f_sp, f_ex, f_ob))
    print(f"   ✅ {label} ({elapsed:.1f}s): Overall={f_all:.4f} | Sparse={f_sp:.4f} | Extreme={f_ex:.4f} | MixedOblique={f_ob:.4f}", flush=True)

print("\n" + "="*114)
print(f"{'Activation (10M ColAvgPool)':<28} | {'Params':>9} | {'Time':>6} | {'Overall':>8} | {'Sparse':>8} | {'Extreme':>8} | {'MixedOblique':>12}")
print("="*114)
for r in results:
    print(f"{r[0]:<28} | {r[1]:>9,} | {r[2]:>5.0f}s | {r[3]:>8.4f} | {r[4]:>8.4f} | {r[5]:>8.4f} | {r[6]:>12.4f}")
print("="*114)
