"""Pooled model: a single gamma (maximum likelihood) estimated without regime splitting."""
from . import llgrid
from .base import BaseERModel


class PooledGamma(BaseERModel):
    """A single gamma (maximum likelihood), no regime split."""
    name = 'pooled'

    def _fit(self, ds, idx):
        self.params_ = {'gamma': llgrid.mle(ds.LL, idx), 'n': len(idx)}

    def gamma(self, regime=None):
        self._check()
        return self.params_['gamma']
