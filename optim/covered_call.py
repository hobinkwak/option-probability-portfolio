import random
import warnings
from collections.abc import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize as sco
from scipy.stats import kurtosis as calc_kurtosis
from scipy.stats import skew as calc_skew

from optim.utils import sampling


class BuyWriteOptimizer:

    def __init__(self, Ks, S, T, r, random_seed=42):
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
        self.posterior = None

        self._call_payoff = None
        self._equity_ret = None

        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _sampling(self, pdfs, N, random_seed):
        # cdfs = np.array(
        #     [sp.integrate.trapezoid(pdfs[:i], self.Ks[:i]) for i in np.arange(len(pdfs))])
        """
        아래 방식이 조금 더 정확함. 사다리꼴 적분 근사.
        """
        dx = np.diff(self.Ks)
        mid = 0.5 * (pdfs[:-1] + pdfs[1:])
        cdfs = np.concatenate(([0.0], np.cumsum(mid * dx)))
        STs = sampling(cdfs, self.Ks, N=N, add_noise=True, random_seed=random_seed)
        return STs

    def fit(self, pdfs, N=10000):
        """
        Implied PDF 기반 만기 주가 경로 샘플링.
        """
        pdfs = pdfs / sp.integrate.trapezoid(pdfs, self.Ks)
        self.STs = self._sampling(pdfs, N, self.random_seed)
        self.fitted = True

    def _compute_payoffs(self, calls, Ks):
        self._call_payoff = (
                calls * np.exp(self.r * self.T)
                - np.maximum(self.STs[:, np.newaxis] - Ks, 0)
        ).astype(np.float32)

        self._equity_ret = ((self.STs - self.S) / self.S).astype(np.float32)

    def _objective(self, x, risk_aversion, bm_exposure=1.0, timing_penalty=0):
        weighted = self._call_payoff @ x  # (N_paths,)
        ris = self._equity_ret + weighted / self.S
        if bm_exposure is not None:
            ris_bm = self._equity_ret * bm_exposure
            ris = ris - ris_bm
        er = ris.mean()
        var = ris.var(ddof=1)

        # Ref Paper: Covered Calls Uncovered
        timing_var=  0
        if timing_penalty > 0:
            delta_scenarios = (self.STs[:, np.newaxis] > self.Ks_selected).astype(np.float32) @ x
            delta_mean = delta_scenarios.mean()
            timing_var = ((delta_scenarios - delta_mean) ** 2 * self._equity_ret ** 2).mean()

        return -er * (1 - risk_aversion) + var * risk_aversion + timing_var * timing_penalty

    def _objective_for_scipy(self, x, risk_aversion, bm_exposure, timing_penalty):
        """scipy.optimize용 wrapper"""
        return self._objective(x, risk_aversion, bm_exposure, timing_penalty)

    # ================================================================ Optimize
    def optimize(self, price, mnys, risk_aversion=0.5, tcr=None,
                 n_simul=1000, n_core=4, bm_exposure=None, timing_penalty=0):
        """
        Args:
            price: fitted call price array (from PDF model)
            mnys: target moneyness array
            risk_aversion: scalar or array (0~1)
            tcr: total cover ratio (비중 합 상한). None이면 1.
            n_simul: brute-force 시뮬 수. None이면 COBYLA.
            n_core: 병렬 코어 수 (brute-force에만 사용)
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

        if n_simul is not None:
            from joblib import Parallel, delayed
            ps = self._init_ps(calls_selected, int(n_simul), tcr, self.random_seed)
            res = Parallel(n_jobs=n_core)(
                delayed(self._loop_optimize_fast)(ps, risk_aversion=ra)
                for ra in risk_aversion
            )
        else:
            bounds = [(0, 1)] * ndim
            if tcr is None:
                upper = 1
                constraints = [{'type': 'ineq', 'fun': lambda x: upper - np.sum(x)}]
            else:
                constraints = [{'type': 'eq', 'fun': lambda x: tcr - np.sum(x)}]

            res = []
            x_prev = np.zeros(ndim)
            cobyla_option = {'maxiter': 10000, 'rhobeg': 0.1, 'tol': 1e-10}
            slsqp_option = {'maxiter': 10000, 'ftol': 1e-12}

            for ra in risk_aversion:
                result = sco.minimize(
                    fun=self._objective_for_scipy,
                    x0=x_prev,
                    args=(ra, bm_exposure, timing_penalty),
                    bounds=bounds,
                    constraints=constraints,
                    method='SLSQP' if tcr is not None else 'COBYLA',
                    options=slsqp_option if tcr is not None else cobyla_option
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

    def analyze_solution(self, risk_aversion=None):
        if self.sols is None:
            raise ValueError("optimize() 먼저")

        if risk_aversion is None:
            risk_aversion = self.sols.index[len(self.sols) // 2]

        x = self.sols.loc[risk_aversion].values
        weighted = self._call_payoff @ x
        ris = self._equity_ret + weighted / self.S

        return {
            'weights': self.sols.loc[risk_aversion],
            'expected_return': float(ris.mean() * (1 / self.T)),
            'volatility': float(ris.std() * np.sqrt((1 / self.T))),
            'sharpe': (float(ris.mean() / ris.std()) * np.sqrt(1 / self.T)) if ris.std() > 0 else 0,
            'max_loss': float(ris.min()),
            'var_95': float(np.percentile(ris, 5)),
            'cvar_95': float(ris[ris <= np.percentile(ris, 5)].mean()),
            'skewness': float(calc_skew(ris)),
            'kurtosis': float(calc_kurtosis(ris)),
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

    def plot_return_distribution(self, risk_aversion=None, bins=100,
                                 figsize=(12, 6), save_path='plot/'):
        """
        주식 단독 보유 vs 커버드콜 포지션의 수익률 분포 비교.

        Args:
            bins: 히스토그램 bin 수
            figsize: 그림 크기
            save_path: 저장 경로
        """
        if self.sols is None:
            raise ValueError("optimize() 먼저")

        if risk_aversion is None:
            risk_aversion = self.sols.index[len(self.sols) // 2]

        x = self.sols.loc[risk_aversion].values
        weighted = self._call_payoff @ x
        cc_ret = self._equity_ret + weighted / self.S
        eq_ret = self._equity_ret

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- 왼쪽: 겹친 히스토그램 ----
        ax = axes[0]
        ax.hist(eq_ret, bins=bins, alpha=0.4, color='#3b82f6', label='Equity Only', density=True)
        ax.hist(cc_ret, bins=bins, alpha=0.5, color='#ef4444', label='Covered Call', density=True)

        ax.axvline(eq_ret.mean(), color='#3b82f6', linestyle='--', linewidth=1.2)
        ax.axvline(cc_ret.mean(), color='#ef4444', linestyle='--', linewidth=1.2)

        ax.set_xlabel('Return', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Return Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # ---- 오른쪽: 통계 비교 ----
        ax2 = axes[1]
        ax2.axis('off')

        stats = [
            ('', 'Equity', 'Covered Call'),
            ('E[R]', f'{eq_ret.mean() * (1 / self.T) :.2%}', f'{cc_ret.mean() * (1 / self.T):.2%}'),
            ('Std', f'{eq_ret.std() * np.sqrt((1 / self.T)) :.2%}', f'{cc_ret.std() * np.sqrt((1 / self.T)):.2%}'),
            ('Sharpe', f'{(eq_ret.mean() / eq_ret.std()) * np.sqrt((1 / self.T))  :.3f}' if eq_ret.std() > 0 else '-',
             f'{(cc_ret.mean() / cc_ret.std()) * np.sqrt((1 / self.T)):.3f}' if cc_ret.std() > 0 else '-'),
            ('VaR 5%', f'{np.percentile(eq_ret, 5):.4%}', f'{np.percentile(cc_ret, 5):.4%}'),
            ('CVaR 5%', f'{eq_ret[eq_ret <= np.percentile(eq_ret, 5)].mean():.4%}',
             f'{cc_ret[cc_ret <= np.percentile(cc_ret, 5)].mean():.4%}'),
            ('Max Loss', f'{eq_ret.min():.4%}', f'{cc_ret.min():.4%}'),
            ('Skew', f'{calc_skew(eq_ret):.3f}', f'{calc_skew(cc_ret):.3f}'),
            ('Kurtosis', f'{calc_kurtosis(eq_ret):.3f}', f'{calc_kurtosis(cc_ret):.3f}'),
        ]

        table = ax2.table(
            cellText=[row for row in stats],
            colWidths=[0.3, 0.35, 0.35],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # 헤더 스타일
        for j in range(3):
            table[0, j].set_facecolor('#334155')
            table[0, j].set_text_props(color='white', fontweight='bold')

        for i in range(1, len(stats)):
            table[i, 0].set_text_props(fontweight='bold')

        ax2.set_title(f'Risk Metrics (λ={risk_aversion:.2f})', fontsize=12)

        fig.suptitle(
            f'Covered Call Analysis  (S={self.S:,.0f}, T={self.T:.2f}, Σ={x.sum():.0%})',
            fontsize=13, y=0.98
        )
        fig.tight_layout()
        plt.savefig(f'{save_path}Return Distribution.png', bbox_inches='tight')
        plt.show()

        return fig, axes
