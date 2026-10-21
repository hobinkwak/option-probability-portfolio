"""
Shared base for expected-return (E[R]) models. The Q->P transform parameter (gamma) is estimated from past
(q_t, realized S_T) pairs.
  transform: p ~ q * (K/F)^gamma  (power of the pricing kernel; a correction parameter tied to the density
             construction pipeline, not a risk-aversion coefficient)

  ERDataset    : records + per-record gamma-grid log-likelihood (LL) + regime labels. Shared across models
  BaseERModel  : parent class (fit(ds, as_of) / gamma(regime) / tilt / expected_excess)
Child classes live in separate files: q_measure.py (QMeasure), pooled.py (PooledGamma), regime.py (RegimeGamma)
"""
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import kernel as kn
from . import llgrid
from .pipeline import PIPELINE


class ERDataset:
    """(q_t, S_T) records + per-record gamma-grid log-likelihood LL (n x G) + regime labels. Shared/reused across models."""

    def __init__(self, records, regimes=None, LL=None):
        self.records = records
        self.n = len(records)
        self.LL = llgrid.build(records) if LL is None else LL
        self.regimes = None if regimes is None else np.asarray(regimes)
        if self.regimes is not None and len(self.regimes) != self.n:
            raise ValueError('regimes length differs from records')
        self.settle = pd.DatetimeIndex([r['settle_day'] for r in records])
        # Record of the density construction pipeline. Legacy files (no record) are assumed to use the fixed pipeline.
        pl = records[0].get('pipeline') if records else None
        self.pipeline_assumed = pl is None
        self.pipeline = dict(PIPELINE) if pl is None else pl

    @classmethod
    def from_pickle(cls, path, regimes=None, cache_ll=True):
        records = pickle.load(open(path, 'rb'))
        ll_path = path + '.LL.npy'
        LL = np.load(ll_path) if cache_ll and os.path.exists(ll_path) else None
        ds = cls(records, regimes, LL)
        if cache_ll and LL is None:
            np.save(ll_path, ds.LL)
        return ds

    def known_before(self, as_of):
        """Indices of records settled by as_of (decision at the close). The expiry-day SOQ is printed at that
        day's open, so settle_day <= as_of is included (future information is blocked)."""
        return np.where(self.settle <= pd.Timestamp(as_of))[0]


class BaseERModel(ABC):
    name = 'base'

    def __init__(self, min_n=36):
        self.min_n = min_n
        self.params_ = {}
        self.fitted_ = False

    def fit(self, ds, as_of=None, full_sample=False):
        """as_of is required (point-in-time). A full-sample fit (diagnostics only) must be requested explicitly with
        full_sample=True. In backtests and live optimization, pass as_of = the decision time (the close) every time."""
        if full_sample == (as_of is not None):
            raise ValueError('specify exactly one of as_of or full_sample=True (prevents accidental full-sample use)')
        idx = np.arange(ds.n) if full_sample else ds.known_before(as_of)
        if len(idx) < self.min_n:
            raise ValueError(f'{self.name}: {len(idx)} observations < min_n={self.min_n}')
        self._fit(ds, idx)
        self.pipeline_ = ds.pipeline
        self.fitted_ = True
        return self

    @abstractmethod
    def _fit(self, ds, idx):
        ...

    @abstractmethod
    def gamma(self, regime=None):
        ...

    def _check(self):
        if not self.fitted_:
            raise ValueError('call .fit() first')

    def tilt(self, Ks, pdf, regime=None):
        self._check()
        return kn.transform_pdf(Ks, pdf, self.gamma(regime))

    def expected_excess(self, Ks, pdf, regime=None):
        """E_P[S_T] / E_Q[S_T] - 1  (expected excess return over the forward, per period)"""
        q = kn._norm(Ks, np.asarray(pdf, float))
        p = self.tilt(Ks, pdf, regime)
        return float(np.trapz(Ks * p, Ks) / np.trapz(Ks * q, Ks) - 1)
