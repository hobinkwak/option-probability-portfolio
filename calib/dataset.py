"""
Build (implied PDF q_t, realized settlement S_T) pairs for each monthly roll -> calib/cache/pdf_dataset{SUF}.pkl
(the fixed pipeline is in calib/pipeline.py, default SUF=_w45_96)
Run: python -m calib.dataset  (requires the environment variables OPP_CHAIN_DIR and OPP_DATA_ROOT)

- entry date d_k : third Friday of the month (or the last chain file on or before it)
- target expiry  : third Friday of the next month (adjusted for Saturday labeling and holidays)
- realized value : SOQ on the expiry day (the Friday if the expiry is labeled Saturday), SPX AM settlement
- rf, div        : external data loader preprocess.DataLoader (OPP_DATA_ROOT); the r/q columns inside the chain
                   have unit errors and are not used
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from calib.pipeline import PIPELINE, START, DEFAULT_SUF


def _require_env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f'environment variable {name} is required')
    return v


def _chain_dir():
    """Directory of daily chain CSVs: {OPP_CHAIN_DIR}/{yyyy}/{yyyymmdd}.csv"""
    return _require_env('OPP_CHAIN_DIR').replace(chr(92), '/').rstrip('/') + '/'


# Defaults are the fixed pipeline (SABR, +-45%, 1996~). The environment variables are overrides for sensitivity
# experiments; when overridden, SUF must also be set separately (so the fixed dataset is not overwritten).
SUF = os.environ.get('KME_SUF', DEFAULT_SUF)
WIDTH = float(os.environ.get('KME_WIDTH', PIPELINE['mny_bounds'][0]))
METHOD = os.environ.get('KME_METHOD', PIPELINE['method'])
_overridden = (WIDTH != PIPELINE['mny_bounds'][0]) or (METHOD != PIPELINE['method'])
if _overridden and SUF == DEFAULT_SUF:
    raise SystemExit('when KME_WIDTH/KME_METHOD is changed, KME_SUF must be set to a different value (protects the fixed dataset)')
CACHE = os.path.join(HERE, 'cache', f'pdf_dataset{SUF}.pkl')

MODEL_PARAMS = dict(PIPELINE, method=METHOD, mny_bounds=(WIDTH, WIDTH))


def third_friday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1)
    first_fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + pd.Timedelta(days=14)


def chain_dates(year):
    p = f'{_chain_dir()}{year}/'
    if not os.path.isdir(p):
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(pd.to_datetime([f[:8] for f in os.listdir(p) if f.endswith('.csv')])))


def build_schedule(start=None, end=None):
    start = start or os.environ.get('KME_START', START)
    end = pd.Timestamp(end) if end else pd.Timestamp.today()
    avail = {y: chain_dates(y) for y in range(pd.Timestamp(start).year, end.year + 1)}
    allds = pd.DatetimeIndex(sorted(np.concatenate([v.values for v in avail.values()])))
    months = pd.period_range(start, end, freq='M')
    rows = []
    for m_ in months:
        e0 = third_friday(m_.year, m_.month)
        e1 = third_friday((m_ + 1).year, (m_ + 1).month)
        if e1 > end or e0 < pd.Timestamp(start):
            continue
        cand = allds[allds <= e0]
        if len(cand) == 0:
            continue
        rows.append((cand[-1], e1))
    return rows


def parity_div(ocs, rf):
    """Back out d = r - ln(F/S)/T from the median near-ATM (0.97~1.03) put-call-parity forward"""
    o = ocs.copy()
    for c, b, a in (('C_MID', 'C_BID', 'C_ASK'), ('P_MID', 'P_BID', 'P_ASK')):
        if c not in o.columns or o[c].isna().all():
            o[c] = (o[b] + o[a]) / 2
    S = float(o['UNDERLYING_LAST'].iloc[0]); T = float(o['T'].iloc[0])
    near = o[(o.STRIKE / S > 0.97) & (o.STRIKE / S < 1.03) & (o.C_MID > 0) & (o.P_MID > 0)]
    if len(near) < 3:
        return None
    Fpar = np.median(near.STRIKE + (near.C_MID - near.P_MID) * np.exp(rf * T))
    return float(rf - np.log(Fpar / S) / T)


def fit_one(entry, target_exp, rf, div):
    warnings.simplefilter('ignore')
    from pdf import OptionImpliedPDF

    f = f'{_chain_dir()}{entry.year}/{entry.strftime("%Y%m%d")}.csv'
    oc = pd.read_csv(f, parse_dates=True, index_col=[0])
    oc['EXPIRE_DATE'] = pd.to_datetime(oc['EXPIRE_DATE'])
    oc = oc.loc[oc.SYMBOL == 'SPX']
    exps = oc.EXPIRE_DATE.unique()
    diffs = {pd.Timestamp(e): (pd.Timestamp(e) - target_exp).days for e in exps}
    cands = [e for e, dd in diffs.items() if -1 <= dd <= 2]
    if not cands:
        return None
    exp = sorted(cands, key=lambda e: (abs(diffs[e]), diffs[e] < 0))[0]
    ocs = oc.loc[oc.EXPIRE_DATE == exp]
    if len(ocs) < 20:
        return None
    if div is None or not np.isfinite(div):
        div = parity_div(ocs, rf)            # no dividend-yield source (before 2000): back out from the same-day chain's parity forward
        if div is None:
            return None
    pm = OptionImpliedPDF(ocs, rf, div)
    pm.fit(**MODEL_PARAMS)
    return dict(entry=entry, exp=exp, target_exp=target_exp, S=float(pm.S), T=float(ocs['T'].iloc[0]),
                rf=float(rf), div=float(div), Ks=pm.Ks.copy(), pdf=pm.pdf.values.copy(),
                pipeline=dict(MODEL_PARAMS))


def main(n_jobs=6):
    from joblib import Parallel, delayed
    root = _require_env('OPP_DATA_ROOT')      # root of the external project that provides preprocess.DataLoader
    os.chdir(root)
    sys.path.insert(0, root)
    from preprocess import DataLoader
    loader = DataLoader('SPX')
    soq = loader.get_settlement_price('soq')['PX_LAST']
    _, _, _, _, div, _ = loader.get_price()
    rf = (loader._read_value_sheet()['USGG3M INDEX'].dropna() / 100)      # from 1990 (the rf from get_price starts in 2000, so use this directly)

    sched = build_schedule()
    print(f'schedule: {len(sched)} entries', flush=True)
    jobs = [(d, e, float(rf.loc[:d].iloc[-1]),
             float(div.loc[:d].iloc[-1]) if d >= div.index[0] else None) for d, e in sched]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_safe_fit)(*j) for j in jobs)

    out, dropped = [], []
    for r_, j in zip(res, jobs):
        if r_ is None or isinstance(r_, str):
            dropped.append((j[0], j[1], r_))
            continue
        settle_day = r_['exp'] - pd.Timedelta(days=1) if r_['exp'].weekday() == 5 else r_['exp']
        s_ = soq.loc[:settle_day]
        if len(s_) == 0 or (settle_day - s_.index[-1]).days > 3:
            dropped.append((j[0], j[1], 'no soq'))
            continue
        r_['ST'] = float(s_.iloc[-1])
        r_['settle_day'] = s_.index[-1]
        out.append(r_)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(out, open(CACHE, 'wb'))
    print(f'saved {len(out)} / dropped {len(dropped)} -> {CACHE}')
    for d in dropped:
        print('dropped', d)


def _safe_fit(*a):
    try:
        return fit_one(*a)
    except Exception as e:
        return f'ERR {type(e).__name__}: {e}'


if __name__ == '__main__':
    main()
