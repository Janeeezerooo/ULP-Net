import os, sys, time, random
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GROUP = sys.argv[1] if len(sys.argv) > 1 else "ALL"  # "G0" (1x,2x) or "G1" (3x,4x)
M_MAX, N_MAX = 100, 100
H, W = M_MAX + 1, N_MAX + 1
BATCH_SIZE = 256
STEPS = (10_000_000 // BATCH_SIZE) + 1

print(f"🚀 10M Depth Ablation (1-Ch + L2-Sort + ColAvgPool) | Group={GROUP} on {DEVICE}", flush=True)

def generate_batch_fast(B=BATCH_SIZE, force_regime=None):
    X_batch = np.zeros((B, 1, H, W), dtype=np.float32)
    y_batch = np.full((B, N_MAX), 2.0, dtype=np.float32)

    for idx in range(B):
        m = random.randint(25, 80)
        n = random.randint(25, 80)
        regime = force_regime if force_regime else random.choice(["Sparse", "Extreme", "MixedOblique"])

        if regime == "Sparse":
            density = random.uniform(0.05, 0.15)
            A = (np.random.rand(m, n) < density).astype(np.float32) * np.random.randn(m, n).astype(np.float32)
        elif regime == "Extreme":
            A = np.random.randn(m, n).astype(np.float32) * (10.0 ** np.random.uniform(-2, 2, size=(1, n)).astype(np.float32))
        else:
            A = np.random.randn(m, n).astype(np.float32)

        x_feas = np.abs(np.random.randn(n).astype(np.float32)) + 0.1
        b = A @ x_feas + np.abs(np.random.randn(m).astype(np.float32)) * 0.5
        c = -np.abs(np.random.randn(n).astype(np.float32)) - 0.1

        k_unb = random.randint(2, max(3, n // 6))
        unb_cols = random.sample(range(n), k_unb)
        d_ray = np.zeros(n, dtype=np.float32)
        d_ray[unb_cols] = np.random.uniform(0.5, 2.0, size=k_unb).astype(np.float32)
        Ad = A @ d_ray
        pos_rows = np.where(Ad > 0)[0]
        if len(pos_rows) > 0:
            j_cancel = unb_cols[-1]
            for r in pos_rows:
                A[r, j_cancel] -= (Ad[r] + 0.1) / d_ray[j_cancel]

        decoy_cols = [j for j in range(n) if j not in unb_cols][:max(2, n // 8)]
        for j in decoy_cols:
            r_block = random.randint(0, m - 1)
            A[r_block, j] = abs(A[r_block, j]) + 2.0

        row_norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
        A_norm = A / row_norms
        b_norm = b / row_norms.squeeze(1)
        c_norm = c / (np.linalg.norm(c) + 1e-8)

        r_order = np.argsort(-np.linalg.norm(A_norm, axis=1))
        c_order = np.argsort(-np.linalg.norm(A_norm, axis=0))
        A_s = A_norm[r_order][:, c_order]
        b_s = b_norm[r_order]
        c_s = c_norm[c_order]

        labels = np.zeros(n, dtype=np.float32)
        labels[unb_cols] = 1.0
        labels_s = labels[c_order]

        X_batch[idx, 0, 0, 1:n+1] = c_s
        X_batch[idx, 0, 1:m+1, 0] = b_s
        X_batch[idx, 0, 1:m+1, 1:n+1] = A_s
        y_batch[idx, :n] = labels_s

    return X_batch, y_batch

class ULPNetDepth(nn.Module):
    def __init__(self, depth=2, k=32):
        super().__init__()
        self.row_conv = nn.Conv2d(1, k, kernel_size=(H, 1), bias=False)
        self.col_conv = nn.Conv2d(1, k, kernel_size=(1, W), bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

        ch_configs = {
            1: [64],
            2: [64, 128],
            3: [64, 128, 128],
            4: [64, 128, 256, 256],
        }
        chs = ch_configs[depth]
        layers = []
        in_c = 2 * k
        for out_c in chs:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.01, inplace=True),
            ])
            in_c = out_c
        self.conv_blocks = nn.Sequential(*layers)
        self.col_pool = nn.AdaptiveAvgPool2d((1, W))
        self.head = nn.Conv2d(in_c, 1, kernel_size=1)

    def forward(self, x):
        r = self.act(self.row_conv(x)).expand(-1, -1, H, W)
        c = self.act(self.col_conv(x)).expand(-1, -1, H, W)
        feat = torch.cat([r, c], dim=1)
        feat = self.conv_blocks(feat)
        pooled = self.col_pool(feat)
        logits = self.head(pooled).squeeze(1).squeeze(1)[:, 1:]
        return logits

print("📦 Building 32,768 training pool + 10,000 benchmark test set...", flush=True)
POOL_N = 32768
X_pool = np.zeros((POOL_N, 1, H, W), dtype=np.float32)
y_pool = np.full((POOL_N, N_MAX), 2.0, dtype=np.float32)
for i in range(0, POOL_N, 256):
    xb, yb = generate_batch_fast(256)
    X_pool[i:i+256], y_pool[i:i+256] = xb, yb

test_sets = {}
for reg in ["Sparse", "Extreme", "MixedOblique"]:
    Xt, yt = [], []
    for _ in range(13):
        xb, yb = generate_batch_fast(256, force_regime=reg)
        Xt.append(xb); yt.append(yb)
    test_sets[reg] = (torch.from_numpy(np.concatenate(Xt)[:3300]).to(DEVICE),
                      torch.from_numpy(np.concatenate(yt)[:3300]).to(DEVICE))

def eval_model(model):
    model.eval()
    res = {}
    all_tp = all_fp = all_fn = 0.0
    with torch.no_grad():
        for reg, (Xt, yt) in test_sets.items():
            tp = fp = fn = 0.0
            for bi in range(0, Xt.size(0), 256):
                xb_e, yt_e = Xt[bi:bi+256], yt[bi:bi+256]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = (torch.sigmoid(model(xb_e)) > 0.5).float()
                mask = (yt_e != 2.0).float()
                tp += ((preds == 1) & (yt_e == 1) & (mask == 1)).sum().item()
                fp += ((preds == 1) & (yt_e == 0) & (mask == 1)).sum().item()
                fn += ((preds == 0) & (yt_e == 1) & (mask == 1)).sum().item()
            res[reg] = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            all_tp += tp; all_fp += fp; all_fn += fn
    res["Overall"] = (2 * all_tp) / (2 * all_tp + all_fp + all_fn + 1e-8)

    dummy = torch.randn(256, 1, H, W, device=DEVICE)
    for _ in range(20): _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): _ = model(dummy)
    torch.cuda.synchronize()
    res["Latency_ms"] = ((time.perf_counter() - t0) / (200 * 256)) * 1000.0
    return res

depths = [1, 2] if GROUP == "G0" else ([3, 4] if GROUP == "G1" else [1, 2, 3, 4])
summary = []

for d in depths:
    model = ULPNetDepth(depth=d).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    scaler = torch.amp.GradScaler("cuda")
    pos_w = torch.tensor(1.5, device=DEVICE)

    print(f"\n▶️ Training {d}x_ConvBlocks (Params: {params:,}) on 10,000,128 instances...", flush=True)
    t_start = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        idx = np.random.randint(0, POOL_N, size=BATCH_SIZE)
        xb = torch.from_numpy(X_pool[idx]).to(DEVICE, non_blocking=True)
        yb = torch.from_numpy(y_pool[idx]).to(DEVICE, non_blocking=True)
        mask = (yb != 2.0).float()
        yb_clean = torch.where(yb == 2.0, torch.zeros_like(yb), yb)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb_clean, pos_weight=pos_w, reduction="none")
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 10000 == 0:
            print(f"   ... step {step}/{STEPS} [{time.time() - t_start:.1f}s]", flush=True)

    elapsed = time.time() - t_start
    scores = eval_model(model)
    summary.append((d, params, elapsed, scores))
    print(f"   ✅ {d}x_ConvBlocks ({elapsed:.1f}s): Overall={scores['Overall']:.4f} | Sparse={scores['Sparse']:.4f} | Extreme={scores['Extreme']:.4f} | MixedOblique={scores['MixedOblique']:.4f} | Latency={scores['Latency_ms']:.4f} ms", flush=True)

print("\n" + "="*115)
print(f"{'Depth (1-Ch ULP-Net)':<22} | {'Params':>9} | {'Overall':>8} | {'Sparse':>8} | {'Extreme':>8} | {'MixedOblique':>12} | {'Latency/LP':>10}")
print("="*115)
for d, p, el, s in summary:
    print(f"{str(d)+'x ConvBlocks':<22} | {p:>9,} | {s['Overall']:>8.4f} | {s['Sparse']:>8.4f} | {s['Extreme']:>8.4f} | {s['MixedOblique']:>12.4f} | {s['Latency_ms']:>7.4f} ms")
print("="*115, flush=True)
