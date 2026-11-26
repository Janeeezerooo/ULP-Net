import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ULPNetVariant(nn.Module):
    def __init__(self, k=32, use_global_ctx=True):
        super().__init__()
        self.use_global_ctx = use_global_ctx
        if use_global_ctx:
            self.row_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), nn.LeakyReLU(0.01, True))
            self.col_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), nn.LeakyReLU(0.01, True))
        else:
            self.local_in = nn.Sequential(nn.Conv2d(1, 2 * k, 3, padding=1, bias=False), nn.BatchNorm2d(2 * k), nn.LeakyReLU(0.01, True))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.01, True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.01, True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.01, True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if self.use_global_ctx:
            f_row = self.row_conv(x).mean(dim=3, keepdim=True).expand(-1, -1, h, w)
            f_col = self.col_conv(x).mean(dim=2, keepdim=True).expand(-1, -1, h, w)
            feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        else:
            feat = self.conv_blocks(self.local_in(x))
        return self.head(feat.mean(dim=2, keepdim=True))[:, 0, 0, 1:]
def generate_lp(m=60, n=60, regime="MixedOblique", sort_l2=True, rng=None):
    rng = rng or np.random.default_rng()
    if regime == "Sparse":
        A = (rng.random((m, n)) < 0.12) * rng.normal(0, 1, (m, n))
    elif regime == "Extreme":
        scale = 10.0 ** rng.uniform(-3, 4, size=(m, 1))
        A = rng.normal(0, 1, (m, n)) * scale
    else:
        A = (rng.random((m, n)) < 0.35) * rng.normal(0, 1, (m, n))
    x_feas = rng.uniform(0.1, 2.0, size=n)
    b = A @ x_feas + rng.uniform(0.5, 3.0, size=m)
    c = rng.normal(0, 1, size=n)
    y = np.zeros(n, dtype=np.float32)
    for idx in rng.choice(n, size=rng.integers(1, 5), replace=False):
        A[:, idx] = -np.abs(A[:, idx]) - 0.1
        c[idx] = -abs(c[idx]) - 1.0
        y[idx] = 1.0
    r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
    A_hat, b_hat = A / r_norm[:, None], b / r_norm
    c_hat = c / max(float(np.max(np.abs(c))), 1e-8)
    if sort_l2:
        r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
        A_hat, b_hat = A_hat[r_idx], b_hat[r_idx]
        c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
        A_hat, c_hat, y = A_hat[:, c_idx], c_hat[c_idx], y[c_idx]
    mat = np.zeros((1, m + 1, n + 1), dtype=np.float32)
    mat[0, 0, 1:], mat[0, 1:, 0], mat[0, 1:, 1:] = c_hat, b_hat, A_hat
    return mat, y, (A, b, c)

def make_dataset(num_samples, m=60, n=60, sort_l2=True, seed=42):
    rng = np.random.default_rng(seed)
    regimes = ["Sparse", "Extreme", "MixedOblique"]
    X = np.zeros((num_samples, 1, m + 1, n + 1), dtype=np.float32)
    Y = np.zeros((num_samples, n), dtype=np.float32)
    raws = []
    for i in range(num_samples):
        mat, y, raw = generate_lp(m, n, regimes[i % 3], sort_l2=sort_l2, rng=rng)
        X[i], Y[i] = mat, y
        if i < 300: raws.append(raw)
    return torch.from_numpy(X), torch.from_numpy(Y), raws

def eval_f1(model, X, Y):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append((torch.sigmoid(model(X[i:i+256].to(device))) > 0.5).cpu())
            targets.append(Y[i:i+256])
    p, t = torch.cat(preds).numpy().ravel(), torch.cat(targets).numpy().ravel()
    tp, fp, fn = np.sum((p == 1) & (t == 1)), np.sum((p == 1) & (t == 0)), np.sum((p == 0) & (t == 1))
    return (2 * tp) / max(2 * tp + fp + fn, 1)

def train_variant(name, use_global_ctx=True, sort_l2=True, steps=39063, batch_size=256):
    t0 = time.time()
    print(f"\n▶️ Training {name} on {steps * batch_size:,} instances...", flush=True)
    X_tr, Y_tr, _ = make_dataset(65536, sort_l2=sort_l2, seed=100)
    X_te, Y_te, raws = make_dataset(4096, sort_l2=sort_l2, seed=999)
    model = ULPNetVariant(k=32, use_global_ctx=use_global_ctx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(X_tr), (batch_size,))
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(X_tr[idx].to(device)), Y_tr[idx].to(device)).backward()
        opt.step()
        sched.step()
        if step % 10000 == 0:
            print(f"   ... step {step}/{steps} ({step * batch_size:,} instances) [{time.time()-t0:.1f}s]", flush=True)
    f1 = eval_f1(model, X_te, Y_te)
    print(f"   ✅ {name} finished in {time.time()-t0:.1f}s | Test F1 = {f1:.4f}", flush=True)
    return model, f1, raws
def test_permutation_invariance(model, sort_l2=True, trials=10, num_instances=200):
    rng = np.random.default_rng(777)
    consistencies = []
    model.eval()
    for _ in range(num_instances):
        _, _, (A, b, c) = generate_lp(60, 60, "MixedOblique", sort_l2=False, rng=rng)
        base_preds = []
        for _ in range(trials):
            rp, cp = rng.permutation(60), rng.permutation(60)
            Ap, bp, cp_vec = A[rp][:, cp], b[rp], c[cp]
            r_norm = np.maximum(np.maximum(np.max(np.abs(Ap), axis=1), np.abs(bp)), 1e-8)
            Ah, bh = Ap / r_norm[:, None], bp / r_norm
            ch = cp_vec / max(float(np.max(np.abs(cp_vec))), 1e-8)
            inv_cp = np.argsort(cp)
            if sort_l2:
                r_idx = np.argsort(-(np.linalg.norm(Ah, axis=1) + np.abs(bh)))
                Ah, bh = Ah[r_idx], bh[r_idx]
                c_idx = np.argsort(-(np.linalg.norm(Ah, axis=0) + np.abs(ch)))
                Ah, ch = Ah[:, c_idx], ch[c_idx]
                map_back = np.argsort(c_idx)[inv_cp]
            else:
                map_back = inv_cp
            mat = np.zeros((1, 1, 61, 61), dtype=np.float32)
            mat[0, 0, 0, 1:], mat[0, 0, 1:, 0], mat[0, 0, 1:, 1:] = ch, bh, Ah
            with torch.no_grad():
                pred = (torch.sigmoid(model(torch.from_numpy(mat).to(device)))[0].cpu().numpy() > 0.5).astype(int)
            base_preds.append(pred[map_back])
        base_preds = np.array(base_preds)
        consistencies.append(np.mean(np.all(base_preds == base_preds[0], axis=0)))
    return float(np.mean(consistencies)) * 100.0

def benchmark_speed_vs_highs(model, raws):
    subset = raws[:200]
    t0 = time.time()
    for (A, b, c) in subset:
        linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    t_highs = (time.time() - t0) * 1000.0 / len(subset)
    mats = []
    for (A, b, c) in subset:
        r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
        Ah, bh = A / r_norm[:, None], b / r_norm
        ch = c / max(float(np.max(np.abs(c))), 1e-8)
        r_idx = np.argsort(-(np.linalg.norm(Ah, axis=1) + np.abs(bh)))
        Ah, bh = Ah[r_idx], bh[r_idx]
        c_idx = np.argsort(-(np.linalg.norm(Ah, axis=0) + np.abs(ch)))
        Ah, ch = Ah[:, c_idx], ch[c_idx]
        mat = np.zeros((1, A.shape[0] + 1, A.shape[1] + 1), dtype=np.float32)
        mat[0, 0, 1:], mat[0, 1:, 0], mat[0, 1:, 1:] = ch, bh, Ah
        mats.append(mat)
    batch = torch.from_numpy(np.array(mats)).to(device)
    model.eval()
    with torch.no_grad(): _ = model(batch[:32])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad(): _ = model(batch)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_ulp = (time.time() - t1) * 1000.0 / len(subset)
    return t_highs, t_ulp, t_highs / max(t_ulp, 1e-6)

if __name__ == "__main__":
    print("🚀 Starting Final 10M-Scale JCMDS Paper Suite...", flush=True)
    m_full, f1_full, raws = train_variant("1_ULP-Net (Full + L2-Sorted)", True, True, 39063)
    m_unsort, f1_unsort, _ = train_variant("2_ULP-Net (Unsorted)", True, False, 39063)
    m_noctx, f1_noctx, _ = train_variant("3_ULP-Net (w/o Row-Col Global Conv)", False, True, 39063)
    perm_sorted = test_permutation_invariance(m_full, sort_l2=True)
    perm_unsorted = test_permutation_invariance(m_unsort, sort_l2=False)
    t_highs, t_ulp, speedup = benchmark_speed_vs_highs(m_full, raws)
    print("\n" + "=" * 75)
    print("📊 FINAL JCMDS PAPER ABLATION (10M SCALE) & BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"1. ULP-Net (Full + L2-Sorted)         | F1: {f1_full:.4f} | Perm Consistency: {perm_sorted:.2f}%")
    print(f"2. ULP-Net (Unsorted)                 | F1: {f1_unsort:.4f} | Perm Consistency: {perm_unsorted:.2f}%")
    print(f"3. ULP-Net (w/o Row-Col Global Conv)  | F1: {f1_noctx:.4f} | Perm Consistency: N/A")
    print("-" * 75)
    print(f"Runtime per LP (60x60): HiGHS = {t_highs:.3f} ms | ULP-Net = {t_ulp:.4f} ms | Speedup = {speedup:.1f}x")
    print("=" * 75, flush=True)
