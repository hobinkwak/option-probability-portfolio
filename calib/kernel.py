"""
Q (risk-neutral) -> P (physical) density transform and estimation.

Transform family p_{gamma,kappa}(K):
  1) kappa : scale of the log return relative to the forward F,  x = ln(K/F) -> kappa * x   (kappa<1: realized vol < implied vol)
  2) gamma : CRRA discount-factor tilt,  p ~ q * (K/F)^gamma                                (SDF m ~ S_T^-gamma)
  gamma=0, kappa=1 gives p = q.

The parameters are not an ad-hoc additive correction; they are estimated by maximum likelihood on past
(q_t, realized S_T) pairs. Market premiums (call prices) stay under Q; only the settlement scenario
distribution of S_T is moved to P.
"""
import numpy as np
from scipy import optimize as sco


def _norm(Ks, pdf):
    return pdf / np.trapz(pdf, Ks)


def transform_pdf(Ks, pdf, gamma=0.0, kappa=1.0):
    q = _norm(Ks, np.asarray(pdf, float))
    F = np.trapz(Ks * q, Ks)
    if kappa != 1.0:
        Kp = F * (Ks / F) ** (1.0 / kappa)                      # location to evaluate in the original density
        q = np.interp(Kp, Ks, q, left=0.0, right=0.0) * (Kp / Ks) / kappa   # Jacobian dK'/dK
    if gamma != 0.0:
        q = q * (Ks / F) ** gamma
    s = np.trapz(q, Ks)
    return q / s if s > 0 else q


def log_density(Ks, pdf, ST, gamma=0.0, kappa=1.0, floor_rel=1e-8):
    """log p at ST. Densities off the grid or equal to zero are floored (the count is reported separately)."""
    p = transform_pdf(Ks, pdf, gamma, kappa)
    d = np.interp(ST, Ks, p, left=0.0, right=0.0)
    return np.log(max(d, floor_rel * p.max()))


def total_loglik(data, gamma, kappa):
    return sum(log_density(r['Ks'], r['pdf'], r['ST'], gamma, kappa) for r in data)


def fit(data, free=('gamma', 'kappa'), bounds=((-5.0, 30.0), (0.5, 1.5))):
    """Maximum likelihood. Parameters not in `free` are fixed at gamma=0, kappa=1."""
    def unpack(th):
        g, k = 0.0, 1.0
        i = 0
        if 'gamma' in free:
            g = th[i]; i += 1
        if 'kappa' in free:
            k = th[i]
        return g, k

    b = [bounds[0] if n == 'gamma' else bounds[1] for n in ('gamma', 'kappa') if n in free]
    x0 = [0.0 if n == 'gamma' else 1.0 for n in ('gamma', 'kappa') if n in free]
    if len(b) == 1:
        r = sco.minimize_scalar(lambda t: -total_loglik(data, *unpack([t])), bounds=b[0], method='bounded',
                                options={'xatol': 1e-4})
        th = [r.x]
    else:
        r = sco.minimize(lambda t: -total_loglik(data, *unpack(t)), x0, bounds=b, method='L-BFGS-B')
        th = r.x
    g, k = unpack(th)
    return dict(gamma=g, kappa=k, loglik=-float(r.fun))


def pit(data, gamma=0.0, kappa=1.0):
    """Probability integral transform u_t = F_p(S_T). Uniform(0,1) if the calibration is right."""
    out = []
    for r in data:
        p = transform_pdf(r['Ks'], r['pdf'], gamma, kappa)
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(r['Ks']))))
        out.append(np.interp(r['ST'], r['Ks'], cdf / cdf[-1], left=0.0, right=1.0))
    return np.array(out)


def estimate_gamma(data, as_of, min_n=36, bounds=(-5.0, 30.0)):
    """
    Point-in-time gamma estimate: uses only pairs already settled by as_of (settle_day < as_of).
    Returns None if fewer than min_n pairs (the caller decides whether to fall back to gamma=0).
    """
    d = [r for r in data if r['settle_day'] < as_of]
    if len(d) < min_n:
        return None
    return fit(d, free=('gamma',), bounds=(bounds, (0.5, 1.5)))['gamma']
