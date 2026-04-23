import warnings

import numpy as np
import pandas as pd
import scipy as sp

from monitor import OptionChainMonitor
from optim import BuyWriteOptimizer
from pdf import OptionImpliedPDF

warnings.simplefilter("ignore")

target_date = '2026-04-17'
target_exp = '2026-05-15'
target_date = pd.to_datetime(target_date)
target_exp = pd.to_datetime(target_exp)
ocs_all = pd.read_csv(f'sample_data/{target_date.strftime("%Y%m%d")}.csv',
                      parse_dates=True, index_col=[0])
ocs_all['EXPIRE_DATE'] = pd.to_datetime(ocs_all['EXPIRE_DATE'])
ocs = ocs_all.loc[ocs_all.EXPIRE_DATE.isin([target_exp])]
ocs = ocs.loc[ocs.SYMBOL == 'SPX']

div_yld = 1.38 * 0.01
rf = 3.5 * 0.01

################### Option Chain Monitor ###################

chain_model = OptionChainMonitor(ocs, rf, div_yld)
gex = chain_model.compute_gex()
vex = chain_model.compute_vex()
chain_model.compute_vanna_profile()
zero_vanna = chain_model.zero_vanna
largest_gex_k, smallest_gex_k = gex['NET'].idxmax(), gex.NET.idxmin()
gpf, zero = chain_model.compute_gamma_profile(boundary=None, n_levels=1000)
chain_model.visualize_gex(boundary=0.20)
chain_model.visualize_gamma_profile(boundary=0.20)

res = pd.DataFrame([[chain_model.S, largest_gex_k, (largest_gex_k / chain_model.S - 1) * 100,
                     smallest_gex_k, (smallest_gex_k / chain_model.S - 1) * 100,
                     gex.NET.sum() / 1e9,
                     vex.NET.sum() / 1e9,
                     zero if zero is not None else 0,
                     ((zero / chain_model.S - 1) * 100) if zero is not None else 0,
                     zero_vanna if zero_vanna is not None else 0,
                     ((zero_vanna / chain_model.S - 1) * 100) if zero_vanna is not None else 0,
                     ]],
                   columns=['S', 'BigGexK', 'BigGexK/S',
                            'SmallGexK', 'SmallGexK/S',
                            'NetGex', 'NetVex', 'Flip', 'Flip/S',
                            'VANNA Flip', 'VANNA Flip/S'
                            ]).round(2)
res.columns = ['S', 'HighGex', 'HighGex/S', 'LowGex', 'LowGex/S', 'NetGex(1B$)', 'NetVex(1B$)', 'Flip', 'Flip/S',
               'VANNA Flip', 'VANNA Flip/S']
print(res)

################### Implied PDF ###################

model_params = {
    'method': 'sabr',
    'filter_iv_sigma': None,
    'K_interval': 0.5,
    'mny_bounds': (0.125, 0.125),
    'filter_pdf_sigma': None,
    'maximum_q': 1.0,
    'flatten': True,  # 안 건드려도 됨
    'check_arbitrage': True
}

prob_model = OptionImpliedPDF(ocs, rf, div_yld)
prob_model.fit(**model_params)
pdf = prob_model.pdf.to_frame(target_date)
pdf.index = (prob_model.Ks / prob_model.S - 1) * 100
prob_model.visualize_diagnostics()

ev = sp.integrate.trapezoid(prob_model.pdf.values * prob_model.Ks, prob_model.Ks) / prob_model.normalizer
biggest_prob_k = prob_model.Ks[prob_model.pdf.argmax()]
print(pd.DataFrame([[prob_model.S, ev, (ev / prob_model.S - 1) * 100, biggest_prob_k,
                     (biggest_prob_k / prob_model.S - 1) * 100]],
                   columns=['S', 'EV', 'EV/S', 'HighProb', 'HighProb/S']).round(2))

################## Covered Call Optimization ###################
S, T = prob_model.S, prob_model.oc_raw['T'].unique()[0]
Ks, price = prob_model.Ks, prob_model.price
Ks_oc_raw = ocs['STRIKE'].values
price_oc_raw = ocs['C_MID'].values
mny_pools = Ks_oc_raw / S

target_mnys = np.arange(7150, 7500, 25) / S
target_idx = np.searchsorted(mny_pools, target_mnys, 'left')
Ks_selected = Ks_oc_raw[target_idx]
calls_selected = price_oc_raw[target_idx]

random_seed = 42
port_model = BuyWriteOptimizer(Ks, S, T, rf, random_seed)
port_fit_params = {
    'pdfs': prob_model.pdf.values,
    'N': int(1e5),
}
port_model.fit(**port_fit_params)

risk_aversions = np.arange(0.0, 1.1, 0.1)
port_optim_params = {
    "price": price,
    "mnys": Ks_selected / S,
    "risk_aversion": risk_aversions,
    "tcr": 1,
    "n_core": 20,
    'n_simul': None,
    'bm_exposure': None,
    'timing_penalty': 0.1,
}
port_model.optimize(**port_optim_params)
port_model.plot_heatmap()
sols = port_model.sols
sols.columns = Ks_selected
print(sols.loc[:, sols.sum(axis=0) > 0])

port_model.plot_return_distribution(risk_aversion=0.1, figsize=(12, 6))

sols_dfs = pd.DataFrame()
for ra in risk_aversions:
    sols_df = pd.DataFrame(
        [tup for tup in port_model.analyze_solution(risk_aversion=ra).items() if tup[0] != 'weights'],
        columns=['metric', ra])
    sols_df = sols_df.set_index('metric')
    sols_dfs = pd.concat([sols_dfs, sols_df], axis=1)
print(sols_dfs)
