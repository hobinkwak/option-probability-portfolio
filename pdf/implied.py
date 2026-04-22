import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from curve import SVIFitter
from curve.utils import check_call_spread_arbitrage, check_butterfly_arbitrage, clean_arbitrage_prices
from gbs import GBSEquation

warnings.filterwarnings('ignore')


class OptionImpliedPDF:

    def __init__(self, data, rf, dividend):
        self.oc_raw = data
        self.S = data['UNDERLYING_LAST'].values[0]
        self.r = rf
        self.d = dividend
        self.T = data['T'].values[0]

        self.pdf = None
        self.Ks = None
        self.ivs = None
        self.price = None
        self.moments = None
        self.normalizer = 1.0

        self._preprocess()
        self.calc_implied_vol()

    def _preprocess(self):
        self.call = self.oc_raw[['STRIKE', 'C_IV', 'C_MID', 'C_BID', 'C_ASK', 'T']]
        self.put = self.oc_raw[['STRIKE', 'P_IV', 'P_MID', 'P_BID', 'P_ASK', 'T']]

        self.call = self.call.dropna(subset=['C_IV', 'C_MID'])
        self.put = self.put.dropna(subset=['P_IV', 'P_MID'])

    def calc_implied_vol(self):
        if 'C_IV' not in self.call.columns or self.call['C_IV'].isna().any():
            self.call['C_IV'] = pd.Series(
                GBSEquation.calc_implied_vol(self.call.C_MID, self.S, self.call.STRIKE,
                                             self.call['T'], self.r, self.d, True),
                index=self.call.index
            )

        if 'P_IV' not in self.put.columns or self.put['P_IV'].isna().any():
            self.put['P_IV'] = pd.Series(
                GBSEquation.calc_implied_vol(self.put.P_MID, self.S, self.put.STRIKE,
                                             self.put['T'], self.r, self.d, False),
                index=self.put.index
            )

    def fit(self, method='sabr', filter_iv_sigma=3, K_interval=0.5,
            mny_bounds=(0.15, 0.15), filter_pdf_sigma=3, maximum_q=0.99,
            flatten=False, check_arbitrage=True, regularization=0.01):
        """
        Args:
            method: 'svi', 'sabr', 'pchip'
            filter_iv_sigma: IV 스무딩 (None이면 스킵)
            K_interval: Strike 간격
            mny_bounds: (min, max) moneyness 범위
            filter_pdf_sigma: PDF 스무딩 (None이면 스킵)
            maximum_q: PDF clipping quantile
            flatten: Wing 영역 IV 평탄화
            check_arbitrage: 차익거래 체크
            regularization: SVI regularization
        """
        method_ = method.lower()

        # (1) Call/Put IV 병합
        Ks, ivs = self._concatenate_call_put()

        valid_mask = np.isfinite(ivs) & np.isfinite(Ks)
        if not np.any(valid_mask):
            raise ValueError("All IVs are invalid")

        if np.sum(~valid_mask) > 0:
            warnings.warn(f"Removed {np.sum(~valid_mask)} invalid IV points")
            Ks = Ks[valid_mask]
            ivs = ivs[valid_mask]

        # (2) IV 필터링
        if filter_iv_sigma is not None:
            ivs = gaussian_filter1d(ivs, sigma=filter_iv_sigma)

        # (3) Arbitrage check
        if check_arbitrage:
            call_prices = self._calc_call_price(Ks, ivs, self.T)
            valid_mask = np.isfinite(call_prices)
            if not np.all(valid_mask):
                warnings.warn(f"Removed {np.sum(~valid_mask)}")
                Ks = Ks[valid_mask]
                ivs = ivs[valid_mask]
                call_prices = call_prices[valid_mask]

            if len(Ks) > 2:
                cs_violations = check_call_spread_arbitrage(Ks, call_prices)
                bf_violations = check_butterfly_arbitrage(Ks, call_prices)

                if len(cs_violations) > 0 or len(bf_violations) > 0:
                    warnings.warn(f"Arbitrage: {len(cs_violations)} spreads, {len(bf_violations)} butterflies")
                    call_prices = clean_arbitrage_prices(Ks, call_prices)
                    ivs = np.array([
                        GBSEquation.calc_implied_vol(p, self.S, k, self.T, self.r, self.d, True)
                        for p, k in zip(call_prices, Ks)
                    ])
                    valid_mask = np.isfinite(ivs)
                    Ks = Ks[valid_mask]
                    ivs = ivs[valid_mask]

        # (4): Strike 확장
        min_mny, max_mny = mny_bounds
        min_K = self.S * (1 - min_mny)
        max_K = self.S * (1 + max_mny)

        Ks_new = np.arange(min_K, max_K + K_interval, K_interval)

        # (5) Volatility curve fitting
        if method_ == 'svi':
            F = self.S * np.exp(self.r * self.T)
            log_mny = np.log(Ks / F)
            var = ivs ** 2 * self.T

            fitter = SVIFitter(F, self.T, regularization=regularization)
            params, loss = fitter.fit(log_mny, var)

            log_mny_new = np.log(Ks_new / F)
            var_new = fitter.svi_slice(log_mny_new, params)
            ivs_new = np.sqrt(var_new / self.T)

        elif method_ == 'sabr':
            from pysabr import Hagan2002LognormalSABR
            from pysabr import hagan_2002_lognormal_sabr as sabr
            F = self.S * np.exp(self.r * self.T)
            sabr_model = Hagan2002LognormalSABR(f=F, t=self.T, beta=0.5)
            alpha, rho, volvol = sabr_model.fit(Ks, ivs * 100)  # % 변환

            ivs_new = np.array([
                sabr.lognormal_vol(k, F, self.T, alpha, 0.5, rho, volvol)
                for k in Ks_new
            ])

        elif method_ == 'pchip':
            interpolator = PchipInterpolator(Ks, ivs)
            ivs_new = interpolator(Ks_new)

        else:
            raise ValueError(f"Unknown method: {method}")

        # (6) Flatten wings
        if flatten:
            left_bound_idx = np.searchsorted(Ks_new, Ks.min())
            right_bound_idx = np.searchsorted(Ks_new, Ks.max())

            if left_bound_idx > 0:
                ivs_new[:left_bound_idx] = ivs_new[left_bound_idx]
            if right_bound_idx < len(Ks_new):
                ivs_new[right_bound_idx:] = ivs_new[right_bound_idx - 1]

        prices_new = self._calc_call_price(Ks_new, ivs_new, self.T)
        pdf = self._calc_pdf_finite_diff(prices_new, K_interval)

        # (6) PDF 후처리
        if filter_pdf_sigma is not None:
            pdf = gaussian_filter1d(pdf, sigma=filter_pdf_sigma)

        pdf = np.clip(pdf, 0, np.quantile(pdf, maximum_q))

        if len(pdf) > 2:
            pdf = pdf[1:-1]
            Ks_new = Ks_new[1:-1]
            ivs_new = ivs_new[1:-1]
            prices_new = prices_new[1:-1]

        self.normalizer = sp.integrate.trapezoid(pdf, Ks_new)

        if self.normalizer <= 0:
            warnings.warn(f"Invalid normalizer: {self.normalizer}")
            self.normalizer = 1.0

        try:
            call_prices_all = self._calc_call_price(Ks_new, ivs_new, self.T)
            put_prices_all = self._calc_put_price(Ks_new, ivs_new, self.T)
            self.moments = self.compute_model_free_moments(
                Ks_new, call_prices_all, put_prices_all, self.S, self.T, self.r
            )
        except:
            self.moments = None

        self.Ks = Ks_new
        self.ivs = ivs_new
        self.price = prices_new
        self.pdf = pd.Series(pdf, index=((Ks_new / self.S - 1) * 100).round(2))

        return self

    def _concatenate_call_put(self):
        call_above_atm = self.call[self.call.STRIKE > self.S]
        put_below_atm = self.put[self.put.STRIKE <= self.S]

        if len(call_above_atm) == 0 or len(put_below_atm) == 0:
            warnings.warn("No clear ATM split, using all data")
            all_strikes = np.concatenate([self.put.STRIKE.values, self.call.STRIKE.values])
            all_ivs = np.concatenate([self.put.P_IV.values, self.call.C_IV.values])

            sorted_idx = np.argsort(all_strikes)
            all_strikes = all_strikes[sorted_idx]
            all_ivs = all_ivs[sorted_idx]

            unique_mask = np.concatenate([[True], np.diff(all_strikes) > 1e-6])
            return all_strikes[unique_mask], all_ivs[unique_mask]

        call_K_st = call_above_atm.index[0]
        put_K_st = put_below_atm.index[-1]

        call_K_sti = np.where(self.call.index == call_K_st)[0][0]
        put_K_sti = np.where(self.put.index == put_K_st)[0][0]

        call_ivg_half = self.call['C_IV'].values[call_K_sti:]
        put_ivg_half = self.put['P_IV'].values[:put_K_sti + 1]

        ivs = np.hstack((put_ivg_half, call_ivg_half))
        Ks = np.hstack((self.put.STRIKE.values[:put_K_sti + 1],
                        self.call.STRIKE.values[call_K_sti:]))

        return Ks, ivs

    def _calc_call_price(self, Ks, ivs, T):
        """Call price 계산"""
        return GBSEquation(self.S, Ks, T, self.r, ivs, self.d).call_price()

    def _calc_put_price(self, Ks, ivs, T):
        """Put price 계산"""
        return GBSEquation(self.S, Ks, T, self.r, ivs, self.d).put_price()

    def _calc_pdf_finite_diff(self, prices, K_interval, steps_range=[1, 2]):
        """
        Finite difference PDF 계산
            - steps_range 중 1은 촘촘하지만 noisy하고 2는 덜 noisy. 둘 중 양수 pdf나오면 채택

        Args:
            K_interval:  ΔK
        """

        n = len(prices)
        pdf = np.zeros(n)
        const = np.exp(self.r * self.T)

        for i in range(1, n - 1):
            for s in steps_range:
                if i - s >= 0 and i + s < n:
                    d2C = (prices[i - int(s)] - 2 * prices[i] + prices[i + int(s)]) / ((K_interval * s) ** 2)
                    pdf_value = const * d2C

                    if pdf_value >= 0:  # 양수면 채택
                        pdf[i] = pdf_value
                        break

        pdf[np.isnan(pdf)] = 0
        return pdf

    @staticmethod
    def compute_model_free_moments(strikes, call_prices, put_prices, S, T, r):
        """
        Model-free implied moments (Bakshi, Kapadia & Madan 2003)
        # Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options
        - Log-return 기반 Variance, Skewness, Kurtosis
        - V, W, X contracts → standardized moments
        """
        F = S * np.exp(r * T)
        discount = np.exp(-r * T)

        # Sort strikes
        idx = np.argsort(strikes)
        K = strikes[idx]
        C = call_prices[idx]
        P = put_prices[idx]

        # OTM options 사용
        atm_idx = np.searchsorted(K, F)

        otm_calls = C[atm_idx:]
        otm_puts = P[:atm_idx]
        K_calls = K[atm_idx:]
        K_puts = K[:atm_idx]

        if len(K_calls) < 2 or len(K_puts) < 2:
            return {'mean': F, 'variance': np.nan, 'volatility': np.nan,
                    'skewness': np.nan, 'kurtosis': np.nan}

        # V contract: E[R^2] where R = ln(S_T/F)
        # V = ∫ (2(1 - ln(K/F))/K^2) * OTM_price dK
        def integrand_v(K_arr, prices):
            return 2 * (1 - np.log(K_arr / F)) / K_arr ** 2 * prices

        V = discount * (
                sp.integrate.trapezoid(integrand_v(K_puts, otm_puts), K_puts) +
                sp.integrate.trapezoid(integrand_v(K_calls, otm_calls), K_calls)
        )

        # W contract: E[R^3]
        # W = ∫ (6*ln(K/F) - 3*ln(K/F)^2) / K^2 * OTM_price dK
        def integrand_w(K_arr, prices):
            lnk = np.log(K_arr / F)
            return (6 * lnk - 3 * lnk ** 2) / K_arr ** 2 * prices

        W = discount * (
                sp.integrate.trapezoid(integrand_w(K_puts, otm_puts), K_puts) +
                sp.integrate.trapezoid(integrand_w(K_calls, otm_calls), K_calls)
        )

        # X contract: E[R^4]
        # X = ∫ (12*ln(K/F)^2 - 4*ln(K/F)^3) / K^2 * OTM_price dK
        def integrand_x(K_arr, prices):
            lnk = np.log(K_arr / F)
            return (12 * lnk ** 2 - 4 * lnk ** 3) / K_arr ** 2 * prices

        X = discount * (
                sp.integrate.trapezoid(integrand_x(K_puts, otm_puts), K_puts) +
                sp.integrate.trapezoid(integrand_x(K_calls, otm_calls), K_calls)
        )

        # Risk-neutral mean of log-return
        mu = np.exp(r * T) - 1 - V / 2 - W / 6 - X / 24

        # Standardized moments
        variance = V - mu ** 2
        skewness = (W - 3 * mu * V + 2 * mu ** 3) / (variance ** 1.5) if variance > 0 else np.nan
        kurtosis = (X - 4 * mu * W + 6 * mu ** 2 * V - 3 * mu ** 4) / (variance ** 2) - 3 if variance > 0 else np.nan

        return {
            'mean': mu,
            'variance': variance,
            'volatility': np.sqrt(max(variance, 0) / T),
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def visualize_diagnostics(self, save_path='plot/'):

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Vol Curve
        axes[0, 0].plot(self.Ks, self.ivs, 'b-', linewidth=2)
        axes[0, 0].axvline(self.S, color='r', linestyle='--', label='ATM')
        axes[0, 0].set_xlabel('Strike')
        axes[0, 0].set_ylabel('Implied Volatility')
        axes[0, 0].set_title('Volatility Smile')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # PDF
        axes[0, 1].plot(self.pdf.index, self.pdf.values, 'g-', linewidth=2)
        axes[0, 1].axvline(0, color='r', linestyle='--', label='ATM')
        axes[0, 1].set_xlabel('Moneyness (%)')
        axes[0, 1].set_ylabel('Probability Density')
        axes[0, 1].set_title('Risk-Neutral PDF')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Moments
        if self.moments:
            moment_names = ['Variance', 'Volatility', 'Skewness', 'Kurtosis']
            moment_vals = [self.moments['variance'], self.moments['volatility'],
                           self.moments['skewness'], self.moments['kurtosis']]

            axes[1, 0].bar(moment_names, moment_vals, color=['blue', 'green', 'orange', 'red'])
            axes[1, 0].set_title('Model-Free Implied Moments')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # CDF
        cdf = np.cumsum(self.pdf.values)
        cdf = cdf / cdf[-1]
        axes[1, 1].plot(self.pdf.index, cdf, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Moneyness (%)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Risk-Neutral CDF')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}Implied PDF Dashboard.png')
        plt.show()

    def visualize(self, save_path='plot/'):
        pos_pdfs = self.pdf[self.pdf.index > 0]
        neg_pdfs = self.pdf[self.pdf.index < 0]
        abs_neg_pdfs = neg_pdfs.copy()
        abs_neg_pdfs.index = abs(abs_neg_pdfs.index)
        abs_neg_pdfs = abs_neg_pdfs[::-1]

        plt.subplot(2, 1, 1)
        plt.plot(self.pdf.index, self.pdf, ".")
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Strike')
        plt.ylabel('pdf')
        plt.subplot(2, 1, 2)
        plt.plot(pos_pdfs.index, pos_pdfs, color='darkblue', label='pos')
        plt.plot(abs_neg_pdfs.index, abs_neg_pdfs, color='darkred', label='neg')
        plt.xlabel('Strike')
        plt.ylabel('pdf')
        plt.legend()
        plt.savefig(f'{save_path}implied pdf.png')
        plt.show()


if __name__ == '__main__':
    pass
