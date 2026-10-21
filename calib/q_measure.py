"""P = Q (expected return = carry). No estimation. Baseline."""
from .base import BaseERModel


class QMeasure(BaseERModel):
    """P = Q (expected return = carry). No estimation. Baseline."""
    name = 'Q'

    def __init__(self):
        super().__init__(min_n=0)

    def _fit(self, ds, idx):
        self.params_ = {'gamma': 0.0, 'n': len(idx)}

    def gamma(self, regime=None):
        return 0.0
