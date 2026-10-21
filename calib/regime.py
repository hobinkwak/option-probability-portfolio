"""Per-regime gamma model. A regime with fewer than min_regime_n observations falls back to the pooled gamma."""
import numpy as np

from . import llgrid
from .base import BaseERModel


class RegimeGamma(BaseERModel):
    """Per-regime gamma. Falls back to the pooled gamma when a regime has fewer than min_regime_n observations or the label is unknown."""
    name = 'regime'

    def __init__(self, min_n=36, min_regime_n=15):
        super().__init__(min_n)
        self.min_regime_n = min_regime_n

    def _fit(self, ds, idx):
        if ds.regimes is None:
            raise ValueError('ERDataset needs regime labels')
        pooled = llgrid.mle(ds.LL, idx)
        lab = ds.regimes[idx]
        gam, ns = {}, {}
        for k in np.unique(ds.regimes):
            ik = idx[lab == k]
            ns[int(k)] = len(ik)
            gam[int(k)] = llgrid.mle(ds.LL, ik) if len(ik) >= self.min_regime_n else pooled
        self.params_ = {'pooled': pooled, 'gamma': gam, 'n_regime': ns, 'n': len(idx)}

    def gamma(self, regime=None):
        self._check()
        if regime is None or int(regime) not in self.params_['gamma']:
            return self.params_['pooled']
        return self.params_['gamma'][int(regime)]
