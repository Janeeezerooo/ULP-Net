import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_lp(scale, regime):
    m, n = scale, scale
    if regime == "Sparse":
        density = 0.08
        A = (np.random.rand(m, n) < density).astype(np.float32) * np.random.randn(m, n).astype(np.float32)
    else:
        A = np.random.uniform(-1.0, 1.0, size=(m, n)).astype(np.float32)
        A[0, :] = np.abs(A[0, :]) + 0.5

    x_true = np.zeros(n, dtype=np.float32)
    k_active = max(3, n // 6)
    active_idx = np.random.choice(n, size=k_active, replace=False)
    x_true[active_idx] = np.random.uniform(0.1, 5.0, size=k_active).astype(np.float32)

    s_slack = np.zeros(m, dtype=np.float32)
    tight_rows = np.random.choice(m, size=min(m, k_active), replace=False)
    non_tight = np.setdiff1d(np.arange(m), tight_rows)
    s_slack[non_tight] = np.random.uniform(0.1, 3.0, size=len(non_tight)).astype(np.float32)
    b = A @ x_true + s_slack

    lam = np.zeros(m, dtype=np.float32)
    lam[tight_rows] = np.random.uniform(0.1, 3.0, size=len(tight_rows)).astype(np.float32)
    red_cost = np.zeros(n, dtype=np.float32)
    inactive_idx = np.setdiff1d(np.arange(n), active_idx)
    red_cost[inactive_idx] = np.random.uniform(0.05, 2.0, size=len(inactive_idx)).astype(np.float32)
    c = A.T @ lam + red_cost
    return A, b, c

class ResolutionAgnosticULPNet(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.row_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), nn.LeakyReLU(0.01, True))
        self.col_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), nn.LeakyReLU(0.01, True))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.01, True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.01, True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.01, True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.01, True),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        f_row = self.row_conv(x).mean(dim=3, keepdim=True).expand(-1, -1, h, w)
        f_col = self.col_conv(x).mean(dim=2, keepdim=True).expand(-1, -1, h, w)
        feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        return self.decoder(feat.mean(dim=2, keepdim=True))[:, 0, 0, 1:]

model = ResolutionAgnosticULPNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
pos_w = torch.tensor(2.5, device=DEVICE)

print("🚀 Fast Training Resolution-Agnostic ULP-Net (90s)...", flush=True)
model.train()
for step in range(1, 2501):
    sc = random.choice([60, 90, 120])
    A, b, c = generate_lp(sc, random.choice(["Sparse", "Dense"]))
    # Ground truth active set via fast solve on small scale
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if not res.success: continue
    y = (res.x > 1e-6).astype(np.float32)

    r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
    A_h, b_h, c_h = A / r_norm[:, None], b / r_norm, c / max(float(np.max(np.abs(c))), 1e-8)
    r_idx = np.argsort(-(np.linalg.norm(A_h, axis=1) + np.abs(b_h)))
    c_idx = np.argsort(-(np.linalg.norm(A_h[:, :], axis=0) + np.abs(c_h)))
    A_h, b_h, c_h, y_s = A_h[r_idx][:, c_idx], b_h[r_idx], c_h[c_idx], y[c_idx]

    X = np.zeros((1, 1, sc + 1, sc + 1), dtype=np.float32)
    X[0, 0, 0, 1:] = c_h
    X[0, 0, 1:, 0] = b_h
    X[0, 0, 1:, 1:] = A_h

    xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y_s).unsqueeze(0).to(DEVICE)
    opt.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(xt), yt, pos_weight=pos_w)
    loss.backward()
    opt.step()

model.eval()
print("\n📊 Zero-Shot Scaling Crossover Benchmark (Trained <=120x120 -> Tested up to 600x600):", flush=True)
print("=" * 112, flush=True)
print(f"{'Scale (Regime)':<20} | {'Var Reduc.':>10} | {'Exact Opt%':>10} | {'Full HiGHS':>11} | {'ULP-Net Inf':>11} | {'ULP+HiGHS':>11} | {'Net Speedup':>11}", flush=True)
print("=" * 112, flush=True)

for scale, regime in [(150, "Dense"), (300, "Dense"), (450, "Dense"), (600, "Dense"), (300, "Sparse"), (600, "Sparse")]:
    full_t, inf_t, acc_t, reds, opts = [], [], [], [], []
    for _ in range(30):
        A, b, c = generate_lp(scale, regime)
        t0 = time.perf_counter()
        res_f = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        tf = (time.perf_counter() - t0) * 1000.0
        if not res_f.success: continue

        ta0 = time.perf_counter()
        r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
        A_h, b_h, c_h = A / r_norm[:, None], b / r_norm, c / max(float(np.max(np.abs(c))), 1e-8)
        r_idx = np.argsort(-(np.linalg.norm(A_h, axis=1) + np.abs(b_h)))
        c_idx = np.argsort(-(np.linalg.norm(A_h, axis=0) + np.abs(c_h)))
        A_h, b_h, c_h = A_h[r_idx][:, c_idx], b_h[r_idx], c_h[c_idx]
        X = np.zeros((1, 1, scale + 1, scale + 1), dtype=np.float32)
        X[0, 0, 0, 1:], X[0, 0, 1:, 0], X[0, 0, 1:, 1:] = c_h, b_h, A_h

        ti0 = time.perf_counter()
        xt = torch.from_numpy(X).to(DEVICE)
        with torch.no_grad():
            probs_s = torch.sigmoid(model(xt)[0]).cpu().numpy()
        torch.cuda.synchronize(DEVICE)
        ti = (time.perf_counter() - ti0) * 1000.0

        probs = np.zeros(scale, dtype=np.float32)
        probs[c_idx] = probs_s
        keep = np.where(probs > 0.15)[0]
        if len(keep) < 5: keep = np.argsort(-probs)[:scale // 3]

        res_s = linprog(c[keep], A_ub=A[:, keep], b_ub=b, bounds=(0, None), method="highs")
        fb = True
        if res_s.success:
            y_dual = -res_s.ineqlin.marginals
            if np.min(c - A.T @ y_dual) >= -1e-4:
                fb = False
                obj_a = res_s.fun
        if fb:
            obj_a = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs").fun
        ta = (time.perf_counter() - ta0) * 1000.0

        full_t.append(tf)
        inf_t.append(ti)
        acc_t.append(ta)
        reds.append(100.0 * (1.0 - len(keep) / scale))
        opts.append(1.0 if abs(obj_a - res_f.fun) / max(abs(res_f.fun), 1e-8) <= 1e-4 else 0.0)

    print(f"{f'{scale}x{scale} ({regime})':<20} | {np.mean(reds):>9.1f}% | {100*np.mean(opts):>9.1f}% | {np.mean(full_t):>8.2f} ms | {np.mean(inf_t):>8.3f} ms | {np.mean(acc_t):>8.2f} ms | {np.mean(full_t)/np.mean(acc_t):>10.2f}x", flush=True)
print("=" * 112, flush=True)
