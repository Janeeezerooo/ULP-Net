import numpy as np

def encode_lp_instance(A: np.ndarray, b: np.ndarray, c: np.ndarray, y: np.ndarray = None):
    m, n = A.shape
    r_norm = np.maximum(np.maximum(np.max(np.abs(A), axis=1), np.abs(b)), 1e-8)
    A_hat, b_hat = A / r_norm[:, None], b / r_norm
    c_hat = c / max(float(np.max(np.abs(c))), 1e-8)
    r_idx = np.argsort(-(np.linalg.norm(A_hat, axis=1) + np.abs(b_hat)))
    A_hat, b_hat = A_hat[r_idx], b_hat[r_idx]
    c_idx = np.argsort(-(np.linalg.norm(A_hat, axis=0) + np.abs(c_hat)))
    A_hat, c_hat = A_hat[:, c_idx], c_hat[c_idx]
    mat = np.zeros((1, m + 1, n + 1), dtype=np.float32)
    mat[0, 0, 1:], mat[0, 1:, 0], mat[0, 1:, 1:] = c_hat, b_hat, A_hat
    return (mat, y[c_idx], c_idx) if y is not None else (mat, c_idx)
