"""Per-regime gamma shrunk toward the pooled value by random effects (DerSimonian-Laird). If the regimes do not differ significantly, it is close to pooling."""
import numpy as np

from . import llgrid
from .base import BaseERModel


class ShrunkRegimeGamma(BaseERModel):
    """
    Shrink the gamma of regime k from (per-regime MLE g_k, standard error se_k):
        g_k* = (g_k/se_k^2 + mu/tau^2) / (1/se_k^2 + 1/tau^2)
        mu = random-effects mean, tau^2 = DerSimonian-Laird heterogeneity estimate (0 means full pooling)
    Records with a negative label (unknown regime) are excluded from the per-regime estimates (they only contribute to
    the pooled estimate). A regime with fewer than min_regime_n observations uses the pooled gamma.
    """
    name = 'shrunk'

    def __init__(self, min_n=36, min_regime_n=15):
        super().__init__(min_n)
        self.min_regime_n = min_regime_n

    def _fit(self, ds, idx):
        if ds.regimes is None:
            raise ValueError('ERDataset needs regime labels')
        pooled = llgrid.mle(ds.LL, idx); lab = ds.regimes[idx]
        est = {}
        for k in np.unique(ds.regimes):
            if k < 0:
                continue
            ik = idx[lab == k]
            if len(ik) >= self.min_regime_n:
                g, se = llgrid.mle_se(ds.LL, ik); est[int(k)] = (g, se, len(ik))
        shr, tau2, mu = {}, np.nan, pooled
        if len(est) >= 2:
            gs = np.array([v[0] for v in est.values()]); se = np.array([v[1] for v in est.values()])
            w = 1 / se ** 2; g0 = (w * gs).sum() / w.sum(); Q = (w * (gs - g0) ** 2).sum()
            c = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - (len(gs) - 1)) / c)
            w2 = 1 / (se ** 2 + tau2); mu = (w2 * gs).sum() / w2.sum()
            for (k, (g, s_, _)) in est.items():
                shr[k] = mu if tau2 == 0 else (g / s_ ** 2 + mu / tau2) / (1 / s_ ** 2 + 1 / tau2)
        gamma = {int(k): shr.get(int(k), pooled) for k in np.unique(ds.regimes) if k >= 0}
        self.params_ = {'pooled': pooled, 'gamma': gamma, 'raw': {k: v[0] for k, v in est.items()}, 'se': {k: v[1] for k, v in est.items()},
                        'tau2': tau2, 'mu': mu, 'n_regime': {k: v[2] for k, v in est.items()}, 'n': len(idx)}

    def gamma(self, regime=None):
        self._check()
        if regime is None or int(regime) not in self.params_['gamma']:
            return self.params_['pooled']
        return self.params_['gamma'][int(regime)]
