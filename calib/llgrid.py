"""Per-record gamma-grid log-likelihood, precomputed: LL[i, g] = log q_i(S_T) + g*log(S_T/F) - log Z_i(g)"""
import numpy as np
from . import kernel as kn

GRID = np.arange(-5.0, 30.0001, 0.05)


def build(data):
    n = len(data)
    LQ = np.empty(n); LM = np.empty(n); LZ = np.empty((n, len(GRID)))
    for i, r in enumerate(data):
        Ks = r['Ks']; q = kn._norm(Ks, r['pdf']); F = np.trapz(Ks * q, Ks)
        x = np.log(Ks / F)
        for j0 in range(0, len(GRID), 175):
            g = GRID[j0:j0 + 175]
            LZ[i, j0:j0 + 175] = np.log(np.trapz(q[:, None] * np.exp(x[:, None] * g[None, :]), Ks, axis=0))
        LQ[i] = np.log(max(np.interp(r['ST'], Ks, q, left=0, right=0), 1e-8 * q.max()))
        LM[i] = np.log(r['ST'] / F)
    return LQ[:, None] + LM[:, None] * GRID[None, :] - LZ


GAMMA_C = np.arange(-2.0, 12.0001, 1.0)      # coarse gamma grid used for the kappa estimate
KAPPA_C = np.arange(0.80, 1.1001, 0.05)       # kappa grid


def build_kappa(data):
    """LK[i, k, g] = log p_{gamma_g, kappa_k}(S_T,i)  (same definition as kernel.log_density)"""
    LK = np.empty((len(data), len(KAPPA_C), len(GAMMA_C)))
    for i, r in enumerate(data):
        for a, kp in enumerate(KAPPA_C):
            for b, gm in enumerate(GAMMA_C):
                LK[i, a, b] = kn.log_density(r['Ks'], r['pdf'], r['ST'], float(gm), float(kp))
    return LK


def mle(LL, idx):
    s = LL[idx].sum(0)
    return GRID[s.argmax()]


def mle_se(LL, idx, default_se=5.0):
    """Maximum-likelihood gamma and a curvature-based standard error (second difference on the grid).
    Falls back to default_se when there is no curvature or the maximum sits on the boundary."""
    s = LL[idx].sum(0); k = int(s.argmax()); h = GRID[1] - GRID[0]
    if 0 < k < len(GRID) - 1:
        d2 = (s[k + 1] - 2 * s[k] + s[k - 1]) / h ** 2
        se = float(np.sqrt(-1 / d2)) if d2 < 0 else default_se
    else:
        se = default_se
    return float(GRID[k]), se
