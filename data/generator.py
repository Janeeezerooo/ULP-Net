import numpy as np

def encode_lp_instance(A: np.ndarray, b: np.ndarray, c: np.ndarray, y: np.ndarray = None, sort_l2: bool = True):
    m, n = A.shape
    r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
    A_hat, b_hat = A / r_norm[:, None], b / r_norm
    c_hat = c / max(float(np.max(np.abs(c))), 1e-8)
    if sort_l2:
        r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
        A_hat, b_hat = A_hat[r_idx], b_hat[r_idx]
        c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
        A_hat, c_hat = A_hat[:, c_idx], c_hat[c_idx]
    else:
        c_idx = np.arange(n)
    mat = np.zeros((1, m + 1, n + 1), dtype=np.float32)
    mat[0, 0, 1:], mat[0, 1:, 0], mat[0, 1:, 1:] = c_hat, b_hat, A_hat
    return (mat, y[c_idx], c_idx) if y is not None else (mat, c_idx)

def generate_lp_instance(m: int = 100, n: int = 100, regime: str = "MixedOblique", sort_l2: bool = True, rng=None):
    rng = rng or np.random.default_rng()
    if regime == "Dense":
        A = (rng.random((m, n)) < 0.70) * rng.normal(0, 1, (m, n))
    elif regime == "HighSparsity":
        A = (rng.random((m, n)) < 0.08) * rng.normal(0, 1, (m, n))
    elif regime == "ExtremeScale":
        scale = 10.0 ** rng.uniform(-3, 4, size=(m, 1))
        A = rng.normal(0, 1, (m, n)) * scale
    else:  # MixedOblique + hard-negative decoy columns
        A = (rng.random((m, n)) < 0.30) * rng.normal(0, 1, (m, n))

    x_feas = rng.uniform(0.1, 2.0, size=n)
    b = A @ x_feas + rng.uniform(0.5, 3.0, size=m)
    c = rng.normal(0, 1, size=n)
    y = np.zeros(n, dtype=np.float32)

    # Plant unbounded recession ray variables
    ray_cols = rng.choice(n, size=rng.integers(1, max(2, min(5, n // 4))), replace=False)
    for idx in ray_cols:
        A[:, idx] = -np.abs(A[:, idx]) - 0.1
        c[idx] = -abs(c[idx]) - 1.0
        y[idx] = 1.0

    # Plant hard-negative decoy columns (negative objective c_j < 0, but bounded by >=1 positive constraint)
    rem_cols = np.setdiff1d(np.arange(n), ray_cols)
    if len(rem_cols) > 0:
        decoy_cols = rng.choice(rem_cols, size=min(len(rem_cols), rng.integers(1, 4)), replace=False)
        for idx in decoy_cols:
            c[idx] = -abs(c[idx]) - 0.5
            block_rows = rng.choice(m, size=rng.integers(1, max(2, m // 10)), replace=False)
            A[block_rows, idx] = np.abs(A[block_rows, idx]) + 0.5

    mat, y_sorted, _ = encode_lp_instance(A, b, c, y=y, sort_l2=sort_l2)
    return mat, y_sorted, (A, b, c)
