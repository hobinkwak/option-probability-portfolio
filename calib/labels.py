"""Regime labels from an external regime time series. The regime file (CSV: first column date, second column regime code 1~6)
is given by the environment variable OPP_MACRO_REGIME_PATH. The signal is assumed to be lagged already, so the value on the
given date is used as is (no extra shift)."""
import os

import numpy as np
import pandas as pd

RISK_OFF_CODES = (2, 4, 6)     # 1,3,5 = risk-on / 2,4,6 = risk-off


def macro_risk_off(dates, path=None):
    """1 if the regime at each date is risk-off, 0 if risk-on (uses the last value on or before the date)."""
    path = path or os.environ.get('OPP_MACRO_REGIME_PATH')
    if not path:
        raise RuntimeError('a regime file path is required: path argument or environment variable OPP_MACRO_REGIME_PATH')
    rg = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return np.array([int(rg.loc[:pd.Timestamp(d)].iloc[-1] in RISK_OFF_CODES) for d in dates])
