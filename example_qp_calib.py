"""
Examples of the new features: (1) exact QP solve, (2) delta-band candidates with a fixed lambda,
(3) point-in-time Q -> P calibration. Runs independently of main.py (option chain monitor + implied PDF + Monte Carlo
optimization):   python example_qp_calib.py
"""
import warnings

import numpy as np
import pandas as pd

from calib import PooledGamma, RegimeGamma, ShrunkRegimeGamma, ERDataset
from optim import BuyWriteOptimizer
from optim.candidates import select_candidates
from pdf import OptionImpliedPDF

warnings.simplefilter("ignore")

# ------------------------------------------------------------------ common: chain -> implied PDF
target_date, target_exp = pd.to_datetime('2026-04-17'), pd.to_datetime('2026-05-15')
ocs_all = pd.read_csv(f'sample_data/{target_date.strftime("%Y%m%d")}.csv', parse_dates=True, index_col=[0])
ocs_all['EXPIRE_DATE'] = pd.to_datetime(ocs_all['EXPIRE_DATE'])
ocs = ocs_all.loc[ocs_all.EXPIRE_DATE.isin([target_exp]) & (ocs_all.SYMBOL == 'SPX')].sort_values('STRIKE').drop_duplicates('STRIKE')
div_yld, rf = 1.38 * 0.01, 3.5 * 0.01

prob_model = OptionImpliedPDF(ocs, rf, div_yld)
prob_model.fit(method='sabr', K_interval=0.5, mny_bounds=(0.125, 0.125), flatten=True, check_arbitrage=True)
S, T = prob_model.S, prob_model.oc_raw['T'].unique()[0]
Ks, pdf = prob_model.Ks, prob_model.pdf.values

# ------------------------------------------------------------------ 1. exact QP solve (default)
# The objective is a quadratic form in the weights x, so it is solved as a convex QP. Unlike random search
# (method='brute'), the solution is exact and fast.
cand_strikes = np.arange(7150, 7500, 25)
opt = BuyWriteOptimizer(Ks, S, T, rf, random_seed=42)
opt.fit(pdfs=pdf, N=100_000)
opt.optimize(price=prob_model.price, mnys=cand_strikes / S, risk_aversion=np.arange(0.0, 1.1, 0.25), tcr=1, method='qp')
sols = opt.sols
sols.columns = cand_strikes
print('\n[1] QP optimal weights (rows: risk_aversion, columns: strike)')
print(sols.loc[:, sols.sum(axis=0) > 1e-4].round(3))

# ------------------------------------------------------------------ 2. delta-band candidates + fixed lambda
# Use only listed strikes that are ATM or above with delta >= 0.10, and set the risk aversion lambda directly.
# lambda=0: maximize expected P&L within the candidates, lambda=1: minimize variance.
# With tcr=None the cover ratio (sum of weights <= 1) is optimized as well.
ix = select_candidates(ocs['STRIKE'].values, ocs['C_DELTA'].values, S, min_moneyness=1.0, min_delta=0.10, max_n=30)
K_c, prem_c = ocs['STRIKE'].values[ix], ocs['C_MID'].values[ix]
band = (ocs['C_DELTA'].values[ix] >= 0.15) & (ocs['C_DELTA'].values[ix] <= 0.25)       # e.g. target delta 0.20 +- 0.05
res = opt.optimize_fixed_lambda(K_c[band], prem_c[band], lams=[0.0, 0.5], tcr=1.0)
print('\n[2] %d candidates with delta 0.15~0.25, fixed-lambda optimization' % band.sum())
for r in res:
    w = r['weights'][r['weights'] > 1e-4]
    print(f"  lambda={r['lam']}: strikes {dict(w.round(3))} / E[call P&L]={r['er_calls']:.4f}, vol reduction={r['achieved']:.3f}")
# Under Q the expected P&L of selling a call is about carry (roughly zero), so a negative value makes zero coverage
# (no call sold) optimal.
free = opt.optimize_fixed_lambda(K_c[band], prem_c[band], lams=[0.0], tcr=None)[0]
print(f"  free cover ratio (tcr=None): total coverage {free['weights'].sum():.2f}")

# ------------------------------------------------------------------ 3. Q -> P calibration (point-in-time gamma)
# Under the implied density Q alone, E[R] collapses to carry. gamma in p ~ q * (K/F)^gamma is estimated by maximum
# likelihood from past (q_t, realized S_T) pairs, and the scenarios are tilted toward P. The data below is synthetic;
# in practice accumulate (q_t, S_T) pairs every month.
rng = np.random.default_rng(0)
n_obs, sigma, drift = 72, 0.045, 0.012                 # monthly vol 4.5%, true expected excess return +1.2%
grid = np.linspace(70, 140, 700)
records = []
for i in range(n_obs):
    f = 100.0
    q = np.exp(-0.5 * ((np.log(grid / f) + 0.5 * sigma ** 2) / sigma) ** 2) / grid        # Q: lognormal density (normalized internally)
    st = f * np.exp(drift + sigma * rng.standard_normal() - 0.5 * sigma ** 2)             # realized value drawn from P (larger drift than Q)
    records.append(dict(Ks=grid, pdf=q, ST=st, settle_day=pd.Timestamp('2019-01-18') + pd.DateOffset(months=i)))
regimes = np.arange(n_obs) % 3                                                              # e.g. three regime labels (ivol terciles)
ds = ERDataset(records, regimes=regimes)
as_of = records[-1]['settle_day']                                                           # only pairs settled by this date are used (blocks future information)
print('\n[3] Q->P gamma estimate (as_of=%s, %d observations)' % (as_of.date(), len(ds.known_before(as_of))))
for name, M in (('PooledGamma ', PooledGamma), ('RegimeGamma ', RegimeGamma), ('ShrunkRegime', ShrunkRegimeGamma)):
    m = M().fit(ds, as_of=as_of)
    print(f'  {name}: gamma(regime 0..2) =', [round(m.gamma(k), 2) for k in range(3)], '/ expected excess return E_P[S]/E_Q[S]-1 = %.3f%%' % (m.expected_excess(grid, records[-1]['pdf'], regime=0) * 100))

# Connect the calibrated model to the optimizer: fit(er_model=..., regime=...) pulls gamma automatically and makes the scenarios P.
opt_p = BuyWriteOptimizer(Ks, S, T, rf, random_seed=42)
opt_p.fit(pdfs=pdf, N=100_000, er_model=PooledGamma().fit(ds, as_of=as_of))
r_q = opt.optimize_fixed_lambda(K_c[band], prem_c[band], lams=[0.0], tcr=1.0)[0]
r_p = opt_p.optimize_fixed_lambda(K_c[band], prem_c[band], lams=[0.0], tcr=1.0)[0]
print('  best call expected P&L among the candidates: Q %.4f -> P %.4f (the larger drift under P lowers the expected P&L of selling calls)' % (r_q['er_calls'], r_p['er_calls']))
