import warnings
from collections.abc import Iterable

import random
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import optimize as sco

from optim.utils import init_params, mjd_pdf, sampling


class BuyWriteOptimizer:

    def __init__(self, log_ret: pd.Series, dt, Ks, S, T, r, random_seed=42):
        self.ret_series = log_ret
        self.ret = log_ret.values
        self.dt = dt
        self.Ks = Ks
        self.S = S
        self.T = T
        self.r = r

        self.STs = None
        self.fitted = False
        self.sols = None
        self.optimal_vals = None
        self.Ks_selected = None
        self.calls_selected = None

        self._call_payoff = None
        self._equity_ret = None

        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _fit_mjd_params(self, returns):
        init_guess = init_params(returns, self.dt, threshold=3)
        res = sco.minimize(mjd_pdf, init_guess,
                           args=(returns, self.dt, True),
                           bounds=[(-1, 1), (1e-4, 1), (0, 50), (-1, 1), (1e-4, 1)])
        return res.x

    def _fit_mjd_regime(self, regime_labels, current_regime,
                        blend_lambda=0.5, min_regime_obs=60):
        """
        국면별 수익률 시계열로 MJD 파라미터 추정

        Args:
            regime_labels: 수익률 시계열과 같은 길이의 array (0 or 1).
                           0 = 저변동, 1 = 고변동.
            current_regime: 현재 국면 (0 or 1)
            blend_lambda: 국면별 파라미터와 전체 파라미터의 혼합 비율.
                          1.0 = 국면 파라미터만 사용, 0.0 = 전체 파라미터만 사용.
            min_regime_obs: 국면별 최소 관측치. 이보다 적으면 blend_lambda 자동 축소.

        """

        common_dates = regime_labels.index.intersection(self.ret_series.index)
        regime_labels_ = regime_labels.loc[common_dates]
        ret_series = self.ret_series.loc[common_dates]
        regime_dates = regime_labels_[regime_labels_ == current_regime].index

        # 전체 기간 피팅
        params_full = self._fit_mjd_params(self.ret)
        # 현재 국면에 해당하는 수익률 추출
        ret_regime = self.ret_series[regime_dates].values

        n_regime = len(ret_regime)
        if n_regime < 20:
            # 관측치 너무 적으면 전체 기간 파라미터만 사용
            warnings.warn(f"Regime obs too few ({n_regime}), using full-sample params")
            return params_full

        # 국면별 피팅
        params_regime = self._fit_mjd_params(ret_regime)

        # blend_lambda 자동 조정: 관측치가 min_regime_obs보다 적으면 축소
        effective_lambda = blend_lambda * min(n_regime / min_regime_obs, 1.0)

        params_blended = effective_lambda * params_regime + (1 - effective_lambda) * params_full
        return params_blended

    def _sampling(self, pdfs, N, random_seed):
        cdfs = np.array(
            [sp.integrate.trapezoid(pdfs[:i], self.Ks[:i]) for i in np.arange(len(pdfs))])
        # dx = np.diff(self.Ks)
        # mid = 0.5 * (pdfs[:-1] + pdfs[1:])
        # cdfs = np.concatenate(([0.0], np.cumsum(mid * dx)))
        STs = sampling(cdfs, self.Ks, N=N, add_noise=True, random_seed=random_seed)
        return STs

    def fit(self, pdfs=None, use_parametric=False, N=10000,
            regime_labels=None, current_regime=None,
            blend_lambda=0.5, min_regime_obs=60):
        """
        PDF 기반 만기 주가 경로 샘플링

        prior (Q-measure):  implied PDF (옵션시장 가격에서 추출)
        likelihood (P-measure): MJD (과거 수익률 기반, 국면 조건부)
        posterior: prior × likelihood → 샘플링 대상

        Args:
            pdfs: implied PDF (None이면 parametric only)
            use_parametric: MJD parametric PDF 사용 여부
            N: 샘플 수
            regime_labels: 수익률 시계열과 같은 길이의 0/1 array (저변동/고변동)
            current_regime: 현재 국면 (0 or 1). regime_labels와 함께 사용.
            blend_lambda: 국면 파라미터 혼합 비율 (1.0=국면only, 0.0=전체only)
            min_regime_obs: 국면별 최소 관측치 (이하면 blend_lambda 자동 축소)
        """
        if pdfs is None:
            use_parametric = True
        else:
            pdfs = pdfs / sp.integrate.trapezoid(pdfs, self.Ks)

        if use_parametric:
            # 국면 정보가 있으면 조건부 피팅, 없으면 전체 피팅
            if regime_labels is not None and current_regime is not None:
                params = self._fit_mjd_regime(regime_labels, current_regime,
                                              blend_lambda, min_regime_obs)
            else:
                params = self._fit_mjd_params(self.ret)

            # MJD로 P-measure PDF 생성
            pdfs_parametric = mjd_pdf(params, log_ret=np.log(self.Ks / self.S),
                                      dt=self.T, nll=False)

            # posterior = P × Q (또는 P only)
            if pdfs is not None:
                posterior = pdfs_parametric * pdfs
                pdfs = posterior / sp.integrate.trapezoid(posterior, self.Ks)
            else:
                pdfs = pdfs_parametric

        self.STs = self._sampling(pdfs, N, self.random_seed)
        self.fitted = True

    def _compute_payoffs(self, calls, Ks):
        self._call_payoff = (
                calls * np.exp(self.r * self.T)
                - np.maximum(self.STs[:, np.newaxis] - Ks, 0)
        ).astype(np.float32)

        self._equity_ret = ((self.STs - self.S) / self.S).astype(np.float32)

    def _objective(self, x, risk_aversion):
        weighted = self._call_payoff @ x  # (N_paths,)
        ris = self._equity_ret + weighted / self.S
        er = ris.mean()
        var = ris.var(ddof=1)
        return -er * (1 - risk_aversion) + var * risk_aversion

    def _objective_robust(self, x, risk_aversion, epsilon=0.05):
        """
        Robust optimization: CVaR penalty로 PDF 추정 오차 고려.

        기본 objective + epsilon * CVaR(5%) penalty.
        tail 리스크에 대한 보수적 조정.
        """
        weighted = self._call_payoff @ x
        ris = self._equity_ret + weighted / self.S
        er = ris.mean()
        var = ris.var(ddof=1)

        # CVaR (Expected Shortfall) at 5%
        n_tail = max(int(len(ris) * 0.05), 1)
        cvar = -np.partition(ris, n_tail)[:n_tail].mean()  # partial sort (faster than full sort)

        base_obj = -er * (1 - risk_aversion) + var * risk_aversion
        return base_obj + epsilon * cvar

    def _objective_for_scipy(self, x, risk_aversion, robust, epsilon):
        """scipy.optimize용 wrapper"""
        if robust:
            return self._objective_robust(x, risk_aversion, epsilon)
        return self._objective(x, risk_aversion)

    # ================================================================ Optimize
    def optimize(self, price, mnys, risk_aversion=0.5, tcr=None,
                 n_simul=1000, n_core=4, robust=False, epsilon=0.05):
        """
        Args:
            price: fitted call price array (from PDF model)
            mnys: target moneyness array
            risk_aversion: scalar or array (0~1)
            tcr: total cover ratio (비중 합 상한). None이면 1.
            n_simul: brute-force 시뮬 수. None이면 COBYLA.
            n_core: 병렬 코어 수 (brute-force에만 사용)
            robust: robust optimization (CVaR penalty) 사용 여부
            epsilon: robust penalty 강도
        """
        if not self.fitted:
            raise ValueError(".fit() 먼저")

        if not isinstance(risk_aversion, Iterable):
            risk_aversion = [risk_aversion]
        risk_aversion = np.array(risk_aversion)
        risk_aversion = np.sort(risk_aversion)  # warm-start를 위해 정렬

        if not (np.all(risk_aversion >= 0) & np.all(risk_aversion <= 1)):
            raise ValueError("risk aversion은 0~1 값")

        mny_pools = self.Ks / self.S
        if (mnys.min() < mny_pools.min()) or (mnys.max() > mny_pools.max()):
            warnings.warn("입력한 타겟 mnys가 행사가 범위를 초과하여, 강제 조정 (clipping)")
            mnys = mnys[(mnys < mny_pools.max()) & (mnys > mny_pools.min())]
        target_idx = np.searchsorted(mny_pools, mnys, 'left')
        calls_selected = price[target_idx]
        Ks_selected = self.Ks[target_idx]

        self._compute_payoffs(calls_selected, Ks_selected)

        ndim = len(mnys)
        upper = tcr if tcr is not None else 1.0

        if n_simul is not None:
            from joblib import Parallel, delayed
            ps = self._init_ps(calls_selected, int(n_simul), tcr, self.random_seed)
            res = Parallel(n_jobs=n_core)(
                delayed(self._loop_optimize_fast)(ps, risk_aversion=ra)
                for ra in risk_aversion
            )
        else:
            # ---- COBYLA path with warm-start ----
            bounds = [(0, upper)] * ndim
            constraints = [{'type': 'ineq', 'fun': lambda x: upper - np.sum(x)}]

            res = []
            x_prev = np.zeros(ndim)

            for ra in risk_aversion:
                result = sco.minimize(
                    fun=self._objective_for_scipy,
                    x0=x_prev,
                    args=(ra, robust, epsilon),
                    bounds=bounds,
                    constraints=constraints,
                    method='COBYLA',
                    options={'maxiter': 10000, 'rhobeg': 0.1},
                    tol=1e-10,
                )
                x_prev = result.x.copy()  # warm-start for next λ
                res.append((result.x, result.fun))

        optimal_ps, optimal_vals = list(zip(*res))

        optimal_ps = pd.DataFrame(optimal_ps, columns=mnys, index=risk_aversion).round(4)
        optimal_vals = pd.Series(optimal_vals, index=risk_aversion).round(6)

        self.sols = optimal_ps
        self.optimal_vals = optimal_vals
        self.Ks_selected = Ks_selected
        self.calls_selected = calls_selected

        return optimal_ps

    @staticmethod
    def _init_ps(calls, n_simul, tcr, random_seed):
        np.random.seed(random_seed)
        ps = np.random.dirichlet(np.ones(len(calls)), size=n_simul).round(6).astype(np.float32)
        if tcr is not None:
            ps = np.hstack((np.ones((n_simul, 1)) * tcr, ps))
        else:
            ps = np.hstack((np.random.uniform(low=0.1, high=1, size=(n_simul, 1)), ps))
        ps = ps[:, 1:] * ps[:, :1]
        return ps.astype(np.float32)

    def _loop_optimize_fast(self, ps, risk_aversion):
        weighted = ps @ self._call_payoff.T  # (n_simul, N_paths)
        ris = self._equity_ret[np.newaxis, :] + weighted / self.S
        er = ris.mean(axis=1)
        var = ris.var(axis=1, ddof=1)
        objs = -er * (1 - risk_aversion) + var * risk_aversion
        best_idx = objs.argmin()
        return ps[best_idx], objs[best_idx]

    def analyze_solution(self, risk_aversion_idx=None):
        if self.sols is None:
            raise ValueError("optimize() 먼저")

        if risk_aversion_idx is None:
            risk_aversion_idx = len(self.sols) // 2

        x = self.sols.iloc[risk_aversion_idx].values
        weighted = self._call_payoff @ x
        ris = self._equity_ret + weighted / self.S

        return {
            'weights': self.sols.iloc[risk_aversion_idx],
            'expected_return': float(ris.mean()),
            'volatility': float(ris.std()),
            'sharpe': float(ris.mean() / ris.std()) if ris.std() > 0 else 0,
            'max_loss': float(ris.min()),
            'var_95': float(np.percentile(ris, 5)),
            'cvar_95': float(ris[ris <= np.percentile(ris, 5)].mean()),
            'skewness': float(sp.stats.skew(ris)),
            'kurtosis': float(sp.stats.kurtosis(ris)),
        }

    def plot_heatmap(self, annualize=True, figsize=(12, 7), cmap='YlOrRd',
                     fmt='.2%', vmin=None, vmax=None, title=None, save_path='plot/'):
        if self.sols is None:
            raise ValueError("optimize()를 먼저 실행하세요.")

        data = self.sols.copy()
        data = data.loc[:, data.sum(axis=0) > 0]

        # 라벨 정리
        col_labels = [f'{m:.2%}' for m in data.columns] if annualize else data.columns.tolist()
        row_labels = [f'{ra:.2f}' for ra in data.index]

        fig, ax = plt.subplots(figsize=figsize)

        # 히트맵 그리기
        norm = mcolors.Normalize(
            vmin=vmin if vmin is not None else data.values.min(),
            vmax=vmax if vmax is not None else data.values.max()
        )
        im = ax.imshow(data.values, cmap=cmap, norm=norm, aspect='auto')

        # 축 설정
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        ax.set_xlabel('Moneyness (K/S)', fontsize=11)
        ax.set_ylabel('Risk Aversion (λ)', fontsize=11)

        # 셀 텍스트
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                text = format(val, fmt.replace('.', '.').lstrip('.'))
                # fmt 파싱: 기본은 percentage
                if '%' in fmt:
                    text = f'{val:{fmt}}'
                else:
                    text = f'{val:{fmt}}'
                color = 'white' if norm(val) > 0.65 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=8, color=color)

        # 컬러바
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Weight', fontsize=10)

        # 제목
        if title is None:
            title = f'Covered Call Optimal Weights (S={self.S:,.0f}, T={self.T:.2f})'
        ax.set_title(title, fontsize=13, pad=12)

        # 총 비중 합 표시 (오른쪽에 row-sum annotation)
        for i, row_sum in enumerate(data.values.sum(axis=1)):
            ax.annotate(f'Σ={row_sum:.2%}',
                        xy=(data.shape[1], i), xycoords='data',
                        xytext=(8, 0), textcoords='offset points',
                        fontsize=8, va='center', color='#555555')

        fig.tight_layout()
        plt.savefig(f'{save_path}Covered Call Heatmap.png')
        plt.show()
