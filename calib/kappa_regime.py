"""Pooled gamma + per-regime volatility scale kappa. E[R] keeps one pooled gamma; the regime (e.g. ivol bucket) only scales the P-density volatility."""
import numpy as np

from . import llgrid
from .base import BaseERModel


class KappaRegime(BaseERModel):
    """
    gamma = pooled MLE. kappa_k = the grid value that maximizes the log-likelihood on regime-k observations (gamma fixed).
    A regime with fewer than min_regime_n observations uses the pooled kappa (full-sample maximum likelihood).
    Requires ds.LK (llgrid.build_kappa).
    """
    name = 'kappa_regime'

    def __init__(self, min_n=36, min_regime_n=15):
        super().__init__(min_n)
        self.min_regime_n = min_regime_n

    def _fit(self, ds, idx):
        if ds.regimes is None or getattr(ds, 'LK', None) is None:
            raise ValueError('ERDataset needs regime labels and LK (llgrid.build_kappa)')
        gp = llgrid.mle(ds.LL, idx); g = int(np.abs(llgrid.GAMMA_C - gp).argmin())
        kp_pool = llgrid.KAPPA_C[ds.LK[idx, :, g].sum(0).argmax()]
        lab = ds.regimes[idx]; kap, ns = {}, {}
        for k in np.unique(ds.regimes):
            ik = idx[lab == k]; ns[int(k)] = len(ik)
            kap[int(k)] = float(llgrid.KAPPA_C[ds.LK[ik, :, g].sum(0).argmax()]) if len(ik) >= self.min_regime_n else float(kp_pool)
        self.params_ = {'pooled': gp, 'kappa': kap, 'kappa_pooled': float(kp_pool), 'n_regime': ns, 'n': len(idx)}

    def gamma(self, regime=None):
        self._check()
        return self.params_['pooled']

    def kappa(self, regime=None):
        self._check()
        if regime is None or int(regime) not in self.params_['kappa']:
            return self.params_['kappa_pooled']
        return self.params_['kappa'][int(regime)]
