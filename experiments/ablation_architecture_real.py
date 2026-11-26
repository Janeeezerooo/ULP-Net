import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

torch.set_num_threads(2)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

M_MAX, N_MAX = 100, 100
H, W = M_MAX + 1, N_MAX + 1
BATCH_SIZE = 256
STEPS = (10_000_000 // BATCH_SIZE) + 1

print(f"🚀 10-MILLION Architecture & L2-Sort Ablation (Real Conic Generator) on {DEVICE}", flush=True)

def generate_batch_fast(B=BATCH_SIZE, force_regime=None, sort_l2=True, return_raw=False):
    X_batch = np.zeros((B, 1, H, W), dtype=np.float32)
    y_batch = np.full((B, N_MAX), 2.0, dtype=np.float32)
    raws = []

    regimes = [force_regime] * B if force_regime else random.choices(
        ["dense", "sparse", "extreme", "mixed_oblique"], weights=[0.20, 0.25, 0.20, 0.35], k=B
    )

    for idx in range(B):
        m = 60 if return_raw else random.randint(15, M_MAX)
        n = 60 if return_raw else random.randint(15, N_MAX)
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
            A *= r_sc; b *= r_sc[:, 0]
            c *= float(10.0 ** random.uniform(-3.0, 4.0))

        if return_raw:
            raws.append((A.copy(), b.copy(), c.copy()))

        r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
        A_hat, b_hat = A / r_norm[:, None], b / r_norm
        c_hat = c / max(float(np.max(np.abs(c))), 1e-8)

        if sort_l2:
            r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
            A_hat, b_hat = A_hat[r_idx], b_hat[r_idx]
            c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
            A_hat, c_hat, y = A_hat[:, c_idx], c_hat[c_idx], y[c_idx]

        X_batch[idx, 0, 0, 1:n+1] = c_hat
        X_batch[idx, 0, 1:m+1, 0] = b_hat
        X_batch[idx, 0, 1:m+1, 1:n+1] = A_hat
        y_batch[idx, :n] = y

    return (X_batch, y_batch, raws) if return_raw else (X_batch, y_batch)

class ULPNetArchVariant(nn.Module):
    def __init__(self, use_global_ctx=True, k=32):
        super().__init__()
        self.use_global_ctx = use_global_ctx
        if use_global_ctx:
            self.row_conv = nn.Conv2d(1, k, kernel_size=(H, 1))
            self.col_conv = nn.Conv2d(1, k, kernel_size=(1, W))
        else:
            self.local_in = nn.Sequential(
                nn.Conv2d(1, 2 * k, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(2 * k),
                nn.LeakyReLU(0.01, inplace=True)
            )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        if self.use_global_ctx:
            f_row = self.row_conv(x).expand(-1, -1, H, W)
            f_col = self.col_conv(x).expand(-1, -1, H, W)
            feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        else:
            feat = self.conv_blocks(self.local_in(x))
        return self.head(self.pool(feat))[:, 0, 0, 1:N_MAX+1]

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

def test_perm_consistency(model, raws, sort_l2=True, trials=10):
    rng = np.random.default_rng(777)
    cons = []
    model.eval()
    for (A, b, c) in raws:
        m, n = A.shape
        preds_trials = []
        for _ in range(trials):
            rp, cp = rng.permutation(m), rng.permutation(n)
            Ap, bp, cp_v = A[rp][:, cp], b[rp], c[cp]
            r_norm = np.maximum(np.maximum(np.max(np.abs(Ap), axis=1), np.abs(bp)), 1e-8)
            Ah, bh = Ap / r_norm[:, None], bp / r_norm
            ch = cp_v / max(float(np.max(np.abs(cp_v))), 1e-8)
            inv_cp = np.argsort(cp)
            if sort_l2:
                r_idx = np.argsort(-(np.linalg.norm(Ah, axis=1) + np.abs(bh)))
                Ah, bh = Ah[r_idx], bh[r_idx]
                c_idx = np.argsort(-(np.linalg.norm(Ah, axis=0) + np.abs(ch)))
                Ah, ch = Ah[:, c_idx], ch[c_idx]
                map_back = np.argsort(c_idx)[inv_cp]
            else:
                map_back = inv_cp
            mat = np.zeros((1, 1, H, W), dtype=np.float32)
            mat[0, 0, 0, 1:n+1], mat[0, 0, 1:m+1, 0], mat[0, 0, 1:m+1, 1:n+1] = ch, bh, Ah
            with torch.no_grad():
                p = (torch.sigmoid(model(torch.from_numpy(mat).to(DEVICE)))[0, :n].cpu().numpy() > 0.5).astype(int)
            preds_trials.append(p[map_back])
        preds_trials = np.array(preds_trials)
        cons.append(np.mean(np.all(preds_trials == preds_trials[0], axis=0)))
    return float(np.mean(cons)) * 100.0

def benchmark_highs(model, raws):
    t0 = time.time()
    for (A, b, c) in raws:
        linprog(-c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    t_highs = (time.time() - t0) * 1000.0 / len(raws)

    mats = np.zeros((len(raws), 1, H, W), dtype=np.float32)
    for idx, (A, b, c) in enumerate(raws):
        m, n = A.shape
        r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
        Ah, bh = A / r_norm[:, None], b / r_norm
        ch = c / max(float(np.max(np.abs(c))), 1e-8)
        r_idx = np.argsort(-(np.linalg.norm(Ah, axis=1) + np.abs(bh)))
        Ah, bh = Ah[r_idx], bh[r_idx]
        c_idx = np.argsort(-(np.linalg.norm(Ah, axis=0) + np.abs(ch)))
        Ah, ch = Ah[:, c_idx], ch[c_idx]
        mats[idx, 0, 0, 1:n+1], mats[idx, 0, 1:m+1, 0], mats[idx, 0, 1:m+1, 1:n+1] = ch, bh, Ah
    xb = torch.from_numpy(mats).to(DEVICE)
    model.eval()
    with torch.no_grad(): _ = model(xb[:32])
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad(): _ = model(xb)
    torch.cuda.synchronize()
    t_ulp = (time.time() - t1) * 1000.0 / len(raws)
    return t_highs, t_ulp, t_highs / max(t_ulp, 1e-6)

POOL_SIZE = 32768
X_eval_all_s, y_eval_all_s = generate_batch_fast(4000, sort_l2=True)
X_eval_sp_s,  y_eval_sp_s  = generate_batch_fast(2000, "sparse", sort_l2=True)
X_eval_ex_s,  y_eval_ex_s  = generate_batch_fast(2000, "extreme", sort_l2=True)
X_eval_ob_s,  y_eval_ob_s  = generate_batch_fast(2000, "mixed_oblique", sort_l2=True)

X_eval_all_u, y_eval_all_u = generate_batch_fast(4000, sort_l2=False)
X_eval_sp_u,  y_eval_sp_u  = generate_batch_fast(2000, "sparse", sort_l2=False)
X_eval_ex_u,  y_eval_ex_u  = generate_batch_fast(2000, "extreme", sort_l2=False)
X_eval_ob_u,  y_eval_ob_u  = generate_batch_fast(2000, "mixed_oblique", sort_l2=False)

_, _, raws_200 = generate_batch_fast(200, "mixed_oblique", sort_l2=True, return_raw=True)

configs = [
    ("1_ULPNet_Full (L2Sort + GlobalConv)", True, True),
    ("2_Ablation_Unsorted (No L2-Sort)", True, False),
    ("3_Ablation_LocalCNN (No GlobalConv)", False, True),
]

scaler = torch.amp.GradScaler("cuda")
pos_w = torch.tensor(1.5, device=DEVICE)
trained_models = {}

for label, use_g, use_s in configs:
    t0 = time.time()
    X_pool, y_pool = generate_batch_fast(POOL_SIZE, sort_l2=use_s)
    X_pool_gpu = torch.from_numpy(X_pool).to(DEVICE)
    y_pool_gpu = torch.from_numpy(y_pool).to(DEVICE)
    model = ULPNetArchVariant(use_global_ctx=use_g).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n▶️ Training {label} (Params: {n_params:,})...", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    model.train()
    for step in range(1, STEPS + 1):
        if step % 64 == 0:
            rep = np.random.randint(0, POOL_SIZE, size=32)
            nx, ny = generate_batch_fast(32, sort_l2=use_s)
            X_pool_gpu[rep] = torch.from_numpy(nx).to(DEVICE)
            y_pool_gpu[rep] = torch.from_numpy(ny).to(DEVICE)
        b_idx = torch.randint(0, POOL_SIZE, (BATCH_SIZE,), device=DEVICE)
        xb, yb = X_pool_gpu[b_idx], y_pool_gpu[b_idx]
        if step % 2 == 0:
            xb = xb * torch.empty((BATCH_SIZE, 1, 1, 1), device=DEVICE).uniform_(0.7, 1.0)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(xb)
            mask = (yb != 2.0)
            loss = F.binary_cross_entropy_with_logits(logits[mask], yb[mask], pos_weight=pos_w)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 10000 == 0:
            print(f"   ... step {step}/{STEPS} [{time.time()-t0:.1f}s]", flush=True)

    if use_s:
        fa, fs, fe, fo = eval_f1(model, X_eval_all_s, y_eval_all_s), eval_f1(model, X_eval_sp_s, y_eval_sp_s), eval_f1(model, X_eval_ex_s, y_eval_ex_s), eval_f1(model, X_eval_ob_s, y_eval_ob_s)
    else:
        fa, fs, fe, fo = eval_f1(model, X_eval_all_u, y_eval_all_u), eval_f1(model, X_eval_sp_u, y_eval_sp_u), eval_f1(model, X_eval_ex_u, y_eval_ex_u), eval_f1(model, X_eval_ob_u, y_eval_ob_u)
    trained_models[label] = model
    print(f"   ✅ {label} ({time.time()-t0:.1f}s): Overall={fa:.4f} | Sparse={fs:.4f} | Extreme={fe:.4f} | MixedOblique={fo:.4f}", flush=True)

p_sorted = test_perm_consistency(trained_models["1_ULPNet_Full (L2Sort + GlobalConv)"], raws_200, sort_l2=True)
p_unsorted = test_perm_consistency(trained_models["2_Ablation_Unsorted (No L2-Sort)"], raws_200, sort_l2=False)
th, tu, sp = benchmark_highs(trained_models["1_ULPNet_Full (L2Sort + GlobalConv)"], raws_200)

print("\n" + "="*85)
print(f"📊 Perm Consistency (L2-Sorted): {p_sorted:.2f}% vs (Unsorted): {p_unsorted:.2f}%")
print(f"⚡ Runtime (60x60 Conic LP): HiGHS = {th:.3f} ms | ULP-Net = {tu:.4f} ms | Speedup = {sp:.1f}x")
print("="*85, flush=True)
