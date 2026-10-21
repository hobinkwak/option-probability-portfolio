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
from calib.kernel import transform_pdf
from calib.pipeline import same_pipeline


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
        self.gamma = 0.0       # Q->P CRRA tilt (0 means the scenario distribution = Q)
        self.er_model = None   # E[R] model passed via fit(er_model=...) (for reference)
        self.qp_status = None  # solver/gap/success per lambda when method='qp'

        self._call_payoff = None
        self._equity_ret = None
        self._itm_indicator = None

        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _sampling(self, pdfs, N, random_seed):
        # cdfs = np.array(
        #     [sp.integrate.trapezoid(pdfs[:i], self.Ks[:i]) for i in np.arange(len(pdfs))])
        """
        The approach below is slightly more accurate: trapezoid-rule approximation.
        """
        dx = np.diff(self.Ks)
        mid = 0.5 * (pdfs[:-1] + pdfs[1:])
        cdfs = np.concatenate(([0.0], np.cumsum(mid * dx)))
        STs = sampling(cdfs, self.Ks, N=N, add_noise=True, random_seed=random_seed)
        return STs

    def fit(self, pdfs, N=10000, gamma=0.0, er_model=None, regime=None):
        """
        Sample terminal prices from the implied PDF.

        gamma: Q->P transform (CRRA discount factor) parameter. p ~ q * (K/F)^gamma. 0 keeps the scenarios at Q.
            Market premiums (call prices, the price argument) stay under Q; only the settlement scenario distribution of S_T moves to P.
            Estimate it from past (q, realized value) pairs with calib.kernel.estimate_gamma() (point-in-time).
        er_model: a fitted calib.BaseERModel child (PooledGamma / RegimeGamma, ...). If given,
            it replaces the gamma argument with er_model.gamma(regime).
        regime: current regime label when er_model is per-regime (e.g. 0/1 from calib.labels.macro_risk_off).
        """
        self.er_model = er_model
        if er_model is not None:
            if not same_pipeline(getattr(er_model, 'pipeline_', None)):
                warnings.warn('the gamma of er_model was estimated under a pipeline different from the fixed density pipeline (calib/pipeline.py): re-estimate gamma')
            gamma = er_model.gamma(regime)
        kappa = float(er_model.kappa(regime)) if er_model is not None and hasattr(er_model, 'kappa') else 1.0   # volatility scale (1 if absent)
        pdfs = pdfs / sp.integrate.trapezoid(pdfs, self.Ks)
        self.gamma = float(gamma); self.kappa = kappa
        if self.gamma != 0.0 or kappa != 1.0:
            pdfs = transform_pdf(self.Ks, pdfs, gamma=self.gamma, kappa=kappa)
        self.STs = self._sampling(pdfs, N, self.random_seed)
        self.fitted = True

    def _compute_payoffs(self, calls, Ks):
        self._call_payoff = (
                calls * np.exp(self.r * self.T)
                - np.maximum(self.STs[:, np.newaxis] - Ks, 0)
        ).astype(np.float32)

        self._equity_ret = ((self.STs - self.S) / self.S).astype(np.float32)
        # for timing: per-scenario, per-strike ITM indicator
        self._itm_indicator = (self.STs[:, np.newaxis] > Ks).astype(np.float32)

    def _objective(self, x, risk_aversion, bm_exposure=1.0, timing_penalty=0):
        weighted = self._call_payoff @ x  # (N_paths,)
        ris = self._equity_ret + weighted / self.S
        if bm_exposure is not None:
            ris_bm = self._equity_ret * bm_exposure
            ris = ris - ris_bm
        er = ris.mean()
        var = ris.var(ddof=1)

        # Ref Paper: Covered Calls Uncovered (2015)
        timing_var = 0
        if timing_penalty > 0:
            delta_scenarios = self._itm_indicator @ x
            delta_mean = delta_scenarios.mean()
            timing_var = ((delta_scenarios - delta_mean) ** 2 * self._equity_ret ** 2).mean()

        return -er * (1 - risk_aversion) + var * risk_aversion + timing_var * timing_penalty

    def _objective_for_scipy(self, x, risk_aversion, bm_exposure, timing_penalty):
        """wrapper for scipy.optimize"""
        return self._objective(x, risk_aversion, bm_exposure, timing_penalty)

    # ================================================================ Quadratic form
    def _quadratic_terms(self, bm_exposure=None, timing_penalty=0):
        """
        Precomputation based on the objective being a quadratic form in x.

        ris = a + P x  (a: equity (minus benchmark) return, P: per-strike call payoff / S)
        E[R]   = mean(a) + mu'x
        Var[R] = x'Cx + 2c'x + var(a)
        timing = x'Mx,  M = mean(c_i c_i' * ret_i^2),  c_i = ITM indicator (centered)

        Returns:
            dict(C, c, mu, a_mean, a_var, M)  (M is None if timing_penalty <= 0)
        """
        P = self._call_payoff.astype(np.float64) / self.S
        a = self._equity_ret.astype(np.float64)
        if bm_exposure is not None:
            a = a * (1 - bm_exposure)

        n = len(a)
        mu = P.mean(axis=0)
        Pc = P - mu
        ac = a - a.mean()
        M = None
        if timing_penalty > 0:
            itm = self._itm_indicator.astype(np.float64)
            Ic = itm - itm.mean(axis=0)
            w = self._equity_ret.astype(np.float64) ** 2
            M = (Ic * w[:, np.newaxis]).T @ Ic / n

        return {
            'C': Pc.T @ Pc / (n - 1),
            'c': Pc.T @ ac / (n - 1),
            'mu': mu,
            'a_mean': a.mean(),
            'a_var': ac @ ac / (n - 1),
            'M': M,
        }

    @staticmethod
    def _qp_matrices(terms, risk_aversion, timing_penalty):
        """f(x) = x'Qx + q'x + const"""
        Q = risk_aversion * terms['C']
        if terms['M'] is not None:
            Q = Q + timing_penalty * terms['M']
        q = -(1 - risk_aversion) * terms['mu'] + 2 * risk_aversion * terms['c']
        const = -(1 - risk_aversion) * terms['a_mean'] + risk_aversion * terms['a_var']
        return Q, q, const

    _GAP_TOL = 1e-6   # tolerance of the duality (Frank-Wolfe) gap on the scale-normalized objective
    _FEAS_TOL = 1e-8

    @staticmethod
    def _project(x, tcr):
        """Post-correct to the box and sum constraints (removes solver numerical error)."""
        x = np.clip(x, 0, 1)
        total = x.sum()
        if tcr is None:
            if total > 1:
                x = x / total
        elif total > 0:
            x = np.clip(x * (tcr / total), 0, 1)
        else:
            x = np.full(len(x), tcr / len(x))
        return x

    @staticmethod
    def _fw_gap(g, x, tcr):
        """
        Optimality certificate of a convex problem: gap = g'x - min_{s in feasible} g's  (>= 0, 0 means optimal).
        It is exact because it is a linear program over the feasible set.
        """
        ndim = len(g)
        if tcr is None:
            lp = sco.linprog(g, A_ub=np.ones((1, ndim)), b_ub=[1.0], bounds=[(0, 1)] * ndim, method='highs')
        else:
            lp = sco.linprog(g, A_eq=np.ones((1, ndim)), b_eq=[tcr], bounds=[(0, 1)] * ndim, method='highs')
        if not lp.success:
            return np.inf
        return float(g @ x - lp.fun)

    def _solve_qp(self, terms, risk_aversion, timing_penalty, tcr, x0):
        """
        Convex QP: min x'Qx + q'x  s.t. 0<=x<=1, and sum(x)<=1 (tcr None) or sum(x)==tcr.
        Q is PSD (covariance + timing), so the KKT solution is globally optimal.

        Convergence procedure:
          1) SLSQP (warm start) -> post projection -> verify optimality with the Frank-Wolfe gap
          2) if gap > tolerance, retry with SLSQP from a uniform start, then trust-constr, and keep the best candidate
        Returns:
            x, objective value, status (dict: solver, gap, success)
        """
        Q, q, const = self._qp_matrices(terms, risk_aversion, timing_penalty)
        ndim = len(q)

        # scale normalization (returns are of order 1e-3 or smaller, which causes ftol numerical issues)
        scale = max(np.abs(Q).max(), np.abs(q).max(), 1e-12)
        Qs, qs = Q / scale, q / scale
        fun = lambda x: x @ Qs @ x + qs @ x
        jac = lambda x: 2 * Qs @ x + qs
        bounds = [(0, 1)] * ndim
        ones = np.ones(ndim)

        if tcr is None:
            cons = [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum(), 'jac': lambda x: -ones}]
            lin = sco.LinearConstraint(ones.reshape(1, -1), -np.inf, 1.0)
        else:
            cons = [{'type': 'eq', 'fun': lambda x: x.sum() - tcr, 'jac': lambda x: ones}]
            lin = sco.LinearConstraint(ones.reshape(1, -1), tcr, tcr)

        uniform = np.full(ndim, (1.0 if tcr is None else tcr) / ndim)

        def run(name, start):
            try:
                if name == 'SLSQP':
                    r = sco.minimize(fun, start, jac=jac, bounds=bounds, constraints=cons,
                                     method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-14})
                else:
                    r = sco.minimize(fun, start, jac=jac, hess=lambda x: 2 * Qs, bounds=sco.Bounds(0, 1),
                                     constraints=[lin], method='trust-constr',
                                     options={'maxiter': 2000, 'gtol': 1e-10, 'xtol': 1e-12})
                x = self._project(np.nan_to_num(r.x), tcr)
            except Exception as e:  # a failure inside the solver moves on to the next candidate
                warnings.warn(f"QP {name} exception (lambda={risk_aversion}): {e}")
                return None
            return x, fun(x), self._fw_gap(jac(x), x, tcr), name

        best = None
        for name, start in (('SLSQP', x0), ('SLSQP', uniform), ('trust-constr', uniform)):
            cand = run(name, start)
            if cand is None:
                continue
            if best is None or cand[1] < best[1] - 1e-15:
                best = cand
            if best[2] <= self._GAP_TOL:
                break

        if best is None:
            raise RuntimeError(f"QP solve failed (lambda={risk_aversion})")

        x, _, gap, name = best
        ok = bool(gap <= self._GAP_TOL)
        if not ok:
            warnings.warn(f"QP optimality not certified (lambda={risk_aversion}): gap={gap:.2e} ({name})")
        status = {'solver': name, 'gap': gap, 'success': ok}
        return x, float(x @ Q @ x + q @ x + const), status

    # ================================================================ Optimize
    def optimize(self, price, mnys, risk_aversion=0.5, tcr=None,
                 n_simul=None, n_core=4, bm_exposure=None, timing_penalty=0,
                 method=None):
        """
        Args:
            price: fitted call price array (from PDF model)
            mnys: target moneyness array
            risk_aversion: scalar or array (0~1)
            tcr: total cover ratio (upper bound on the sum of weights). 1 if None.
            n_simul: number of brute-force simulations (used only with method='brute')
            n_core: number of parallel cores (brute-force only)
            method: 'qp' (exact solution, default) | 'brute' (Dirichlet random search) | 'scipy' (old COBYLA/SLSQP, for comparison).
                    If None, 'brute' when n_simul is given, otherwise 'qp'.
        """
        if not self.fitted:
            raise ValueError("call .fit() first")

        if not isinstance(risk_aversion, Iterable):
            risk_aversion = [risk_aversion]
        risk_aversion = np.array(risk_aversion)
        risk_aversion = np.sort(risk_aversion)  # sort for warm start

        if not (np.all(risk_aversion >= 0) & np.all(risk_aversion <= 1)):
            raise ValueError("risk aversion must be between 0 and 1")

        mny_pools = self.Ks / self.S
        if (mnys.min() < mny_pools.min()) or (mnys.max() > mny_pools.max()):
            warnings.warn("the target mnys exceed the strike range; clipping")
            mnys = mnys[(mnys < mny_pools.max()) & (mnys > mny_pools.min())]
        target_idx = np.searchsorted(mny_pools, mnys, 'left')
        calls_selected = price[target_idx]
        Ks_selected = self.Ks[target_idx]

        self._compute_payoffs(calls_selected, Ks_selected)

        ndim = len(mnys)

        if method is None:
            method = 'brute' if n_simul is not None else 'qp'
        if method not in ('qp', 'brute', 'scipy'):
            raise ValueError("method must be 'qp' | 'brute' | 'scipy'")

        if method == 'qp':
            terms = self._quadratic_terms(bm_exposure, timing_penalty)
            res, statuses = [], []
            # start from uniform weights that satisfy the sum constraint, then warm-start across lambdas in order
            x_prev = np.full(ndim, (1.0 if tcr is None else tcr) / ndim)
            for ra in risk_aversion:
                x, val, st = self._solve_qp(terms, ra, timing_penalty, tcr, x_prev)
                x_prev = x
                res.append((x, val))
                statuses.append(st)
            self.qp_status = pd.DataFrame(statuses, index=risk_aversion)
        elif method == 'brute':
            from joblib import Parallel, delayed
            n_simul = 1000 if n_simul is None else int(n_simul)
            ps = self._init_ps(calls_selected, n_simul, tcr, self.random_seed)
            terms = self._quadratic_terms(bm_exposure, timing_penalty)
            res = Parallel(n_jobs=n_core)(
                delayed(self._loop_optimize_fast)(ps, ra, terms, timing_penalty)
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

    def optimize_vol_target(self, strikes, premiums, vol_reduction, tcr=1.0,
                            bm_exposure=None, timing_penalty=0, tol=1e-3, max_iter=40):
        """
        Fixed-risk-level mode: bisect the risk aversion lambda so that the volatility reduction versus equity equals vol_reduction,
        then solve the QP that maximizes expected return at that risk level (lambda is an output).
        Takes listed strikes and market premiums directly (no grid approximation).

        Args:
            strikes: array of listed strikes (candidates)
            premiums: current call price of each strike (market mid). Converted to a terminal value internally with e^{rT}
            vol_reduction: target volatility reduction (scalar or iterable, e.g. 0.1 = 10% lower than equity)
            tcr: total cover ratio (sum <= 1 if None)
        Returns:
            list[dict(target, weights(Series), lam, achieved, feasible, path, er_stock, er_calls, sd_stock, sd_port)]
            (in the order of vol_reduction. er_*: expected per-period return under the scenarios (model distribution), path: lambda search path)
            feasible=False: even lambda=1 (variance minimization) falls short of the target -> returns the lambda=1 solution
        """
        if not self.fitted:
            raise ValueError("call .fit() first")
        strikes = np.asarray(strikes, dtype=float)
        premiums = np.asarray(premiums, dtype=float)
        targets = [vol_reduction] if np.isscalar(vol_reduction) else list(vol_reduction)

        self._compute_payoffs(premiums, strikes)
        terms = self._quadratic_terms(bm_exposure, timing_penalty)
        ndim = len(strikes)
        a_var = terms['a_var']

        def reduction(x):
            var = x @ terms['C'] @ x + 2 * terms['c'] @ x + a_var
            return 1.0 - np.sqrt(max(var, 0.0) / a_var)

        x_start = np.full(ndim, (1.0 if tcr is None else tcr) / ndim)

        def solve(lam, x0):
            x, _, _ = self._solve_qp(terms, lam, timing_penalty, tcr, x0)
            return x, reduction(x)

        x_hi, red_hi = solve(1.0, x_start)          # variance minimization (maximum achievable reduction)
        x_lo, red_lo = solve(0.0, x_start)          # expected-return maximization (minimum reduction)
        out = []
        for tgt in targets:
            path = [(1.0, red_hi), (0.0, red_lo)]     # (lambda, achieved reduction) search path (diagnostics)
            if red_hi < tgt - tol:
                x, lam, red, ok = x_hi, 1.0, red_hi, False
            elif red_lo >= tgt - tol:
                x, lam, red, ok = x_lo, 0.0, red_lo, True
            else:
                lo, hi, x_prev = 0.0, 1.0, x_hi
                x, red, lam = x_hi, red_hi, 1.0
                for _ in range(max_iter):
                    lam = 0.5 * (lo + hi)
                    x, red = solve(lam, x_prev)
                    path.append((lam, red))
                    x_prev = x
                    if abs(red - tgt) <= tol:
                        break
                    if red < tgt:
                        lo = lam
                    else:
                        hi = lam
                ok = True
            var_x = x @ terms['C'] @ x + 2 * terms['c'] @ x + a_var
            out.append({'target': tgt, 'weights': pd.Series(np.round(x, 6), index=strikes), 'lam': lam,
                        'achieved': red, 'feasible': ok, 'path': path,
                        'er_stock': float(terms['a_mean']), 'er_calls': float(terms['mu'] @ x),
                        'sd_stock': float(np.sqrt(a_var)), 'sd_port': float(np.sqrt(max(var_x, 0.0)))})
        self.Ks_selected, self.calls_selected = strikes, premiums
        return out

    def optimize_fixed_lambda(self, strikes, premiums, lams, tcr=1.0, bm_exposure=None, timing_penalty=0):
        """
        Set the risk aversion lambda directly, without a risk target. Takes listed strikes (candidates) and market premiums directly.
        Use it when the risk level is set by restricting candidates beforehand (e.g. a delta band) (lambda=0: maximize expected return, 1: minimize variance).

        Returns:
            list[dict(lam, weights(Series), er_stock, er_calls, sd_stock, sd_port, achieved)]  (in the order of lams)
        """
        if not self.fitted:
            raise ValueError("call .fit() first")
        strikes = np.asarray(strikes, dtype=float); premiums = np.asarray(premiums, dtype=float)
        lams = [lams] if np.isscalar(lams) else list(lams)
        self._compute_payoffs(premiums, strikes)
        terms = self._quadratic_terms(bm_exposure, timing_penalty)
        ndim = len(strikes); a_var = terms['a_var']; x_prev = np.full(ndim, (1.0 if tcr is None else tcr) / ndim)
        out = []
        for lam in lams:
            if ndim == 1 and tcr is not None:
                x = np.array([float(tcr)])
            else:
                x, _, _ = self._solve_qp(terms, lam, timing_penalty, tcr, x_prev); x_prev = x
            var = x @ terms['C'] @ x + 2 * terms['c'] @ x + a_var
            out.append({'lam': lam, 'weights': pd.Series(np.round(x, 6), index=strikes), 'er_stock': float(terms['a_mean']), 'er_calls': float(terms['mu'] @ x),
                        'sd_stock': float(np.sqrt(a_var)), 'sd_port': float(np.sqrt(max(var, 0.0))), 'achieved': float(1 - np.sqrt(max(var, 0.0) / a_var))})
        self.Ks_selected, self.calls_selected = strikes, premiums
        return out

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

    def _loop_optimize_fast(self, ps, risk_aversion, terms, timing_penalty=0):
        # evaluate all candidates at once through the quadratic form (reflects bm_exposure and timing_penalty)
        Q, q, const = self._qp_matrices(terms, risk_aversion, timing_penalty)
        p64 = ps.astype(np.float64)
        objs = np.einsum('ij,jk,ik->i', p64, Q, p64) + p64 @ q + const
        best_idx = objs.argmin()
        return ps[best_idx], objs[best_idx]

    def analyze_solution(self, risk_aversion=None):
        if self.sols is None:
            raise ValueError("call optimize() first")

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
            raise ValueError("run optimize() first.")

        data = self.sols.copy()
        data = data.loc[:, data.sum(axis=0) > 0]

        # tidy labels
        col_labels = [f'{m:.2%}' for m in data.columns] if annualize else data.columns.tolist()
        row_labels = [f'{ra:.2f}' for ra in data.index]

        fig, ax = plt.subplots(figsize=figsize)

        # draw the heatmap
        norm = mcolors.Normalize(
            vmin=vmin if vmin is not None else data.values.min(),
            vmax=vmax if vmax is not None else data.values.max()
        )
        im = ax.imshow(data.values, cmap=cmap, norm=norm, aspect='auto')

        # axis setup
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        ax.set_xlabel('Moneyness (K/S)', fontsize=11)
        ax.set_ylabel('Risk Aversion (λ)', fontsize=11)

        # cell text
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                # parse fmt: percentage by default
                if '%' in fmt:
                    text = f'{val:{fmt}}'
                else:
                    text = f'{val:{fmt}}'
                color = 'white' if norm(val) > 0.65 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=8, color=color)

        # colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Weight', fontsize=10)

        # title
        if title is None:
            title = f'Covered Call Optimal Weights (S={self.S:,.0f}, T={self.T:.2f})'
        ax.set_title(title, fontsize=13, pad=12)

        # show the total weight (row-sum annotation on the right)
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
        Compare the return distribution of holding equity alone vs the covered-call position.

        Args:
            bins: number of histogram bins
            figsize: figure size
            save_path: save path
        """
        if self.sols is None:
            raise ValueError("call optimize() first")

        if risk_aversion is None:
            risk_aversion = self.sols.index[len(self.sols) // 2]

        x = self.sols.loc[risk_aversion].values
        weighted = self._call_payoff @ x
        cc_ret = self._equity_ret + weighted / self.S
        eq_ret = self._equity_ret

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- left: overlaid histograms ----
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

        # ---- right: statistics comparison ----
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

        # header style
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
