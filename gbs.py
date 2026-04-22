from collections.abc import Iterable
from copy import copy

import numpy as np
from scipy import optimize as sco
from scipy import stats


class GBSEquation:

    def __init__(self, S, K, T, rf, sigma, dividend=0):
        self.S = S
        self.K = np.array(K) if isinstance(K, Iterable) else K
        self.T = np.array(T) if isinstance(T, Iterable) else T
        self.rf = rf
        self.sigma = np.array(sigma) if isinstance(sigma, Iterable) else sigma
        self.dividend = dividend

        self.call = None
        self.put = None

        self.calc_d1()
        self.calc_d2()

    def calc_d1(self):
        """
        기초자산 가격에 대한 옵션 가격의 민감도 (=delta)
        """
        d1 = (np.log(self.S / self.K) + (self.rf - self.dividend + (self.sigma ** 2) / 2) * self.T) / (
                self.sigma * np.sqrt(self.T))
        self.d1 = d1

    def calc_d2(self):
        """
        만기에 옵션이 행사될 확률
        """
        if self.d1 is None:
            self.calc_d1()
        d2 = self.d1 - self.sigma * np.sqrt(self.T)
        self.d2 = d2

    def _moment(self, order=1, call=True):
        if order == 1:
            return self.expectation(call=call)
        elif order == 2:
            return
        elif order == 3:
            return
        else:
            ValueError("moment order는 1,2,3만 지원")

    def expectation(self, call=True):
        if call:
            if self.call is None:
                return self.call_price()
            else:
                return self.call
        else:
            if self.put is None:
                return self.put_price()
            else:
                return self.put

    def price(self, call=True):
        if call:
            return self.call_price()
        else:
            return self.put_price()

    def call_price(self):
        if not isinstance(self.T, Iterable):
            if self.T == 0:
                self.call = 0
                return self.call

        self.call = self.S * np.exp(-self.dividend * self.T) * stats.norm.cdf(self.d1) - self.K * np.exp(
            -self.rf * self.T) * stats.norm.cdf(self.d2)
        if np.mean(np.isnan(self.call)) > 0:
            raise ValueError("Call Price NaN Value")

        return self.call

    def put_price(self):
        if not isinstance(self.T, Iterable):
            if self.T == 0:
                self.put = 0
                return self.put

        self.put = self.K * np.exp(-self.rf * self.T) * stats.norm.cdf(-self.d2) - self.S * np.exp(
            -self.dividend * self.T) * stats.norm.cdf(-self.d1)
        if np.mean(np.isnan(self.put)) > 0:
            raise ValueError("Put Price NaN Value")
        return self.put

    def delta(self, call=True, long=True):
        pos_sign = 1 if long else -1
        if call:
            return stats.norm.cdf(self.d1) * pos_sign
        else:
            return (stats.norm.cdf(self.d1) - 1) * pos_sign

    def gamma(self, long=True):
        """
        Long position: Long Gamma / Short Theta
        기초자산 가격 변화에 따른 delta의 민감도
        ATM에서 감마가 가장 큼 (ATM에서 만기가 줄어듦에 따라 점점 커짐, 만기 직전에는 폭발함)
        """
        pos_sign = 1 if long else -1
        g = stats.norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return g * pos_sign

    def vega(self, long=True):
        pos_sign = 1 if long else -1
        v = self.S * np.sqrt(self.T) * stats.norm.pdf(self.d1)
        return v * pos_sign

    def rho(self, call=True):
        """
        ATM일 떄 베가가 가장 큼
        ATM일 때 변동성의 변화에 포지션이 매우 민감하게 반응한다는 뜻
        만기로 가까워질수록 값이 작아짐

        """
        sign = 1 if call else - 1
        return sign * self.K * self.T * np.exp(-self.rf * self.T) * stats.norm.cdf(sign * self.d2)

    def theta(self, call=True):
        """
        ATM에서 시간에 대한 민감도가 제일 큼
        long; negative
        short; positive
        """
        sign = -1 if call else 1
        theta = (-self.sigma * self.S * stats.norm.pdf(self.d1)) / (2 * np.sqrt(self.T)) - sign * (
                self.rf * self.K * np.exp(-self.rf * self.T) * stats.norm.cdf(sign * self.d2))
        return theta

    def vanna(self):
        vanna = np.exp(-self.dividend * self.T) * self.d1 * (self.d2 / self.sigma)
        return vanna

    def greeks(self, call=True, long=True):
        d = self.delta(call, long)
        g = self.gamma(long)
        v = self.vega(long)
        r = self.rho(call)
        t = self.theta(call)
        return d, g, v, r, t

    @staticmethod
    def calc_implied_vol(market_price, S, K, T, r, dv, call=True,
                         precision=1e-4, initial_guess=0.5, max_iter=500):
        """
        Newton-Raphson implied volatility solver.
        σ_new = σ_current - f(σ) / f'(σ)
        f(σ) = BS(σ) - market_price
        f'(σ) = vega
        """
        is_iterable = isinstance(market_price, Iterable)
        result = np.array([np.nan]) if not is_iterable else np.full(len(market_price), np.nan)
        ivol = np.array([initial_guess]) if not is_iterable else np.full(len(market_price), initial_guess)
        market_price_ = np.atleast_1d(np.array(market_price, dtype=float)).copy()

        active = np.arange(len(result))

        for _ in range(max_iter):
            if len(active) == 0:
                break

            model = GBSEquation(S, K if not is_iterable else np.asarray(K)[active],
                                T if not is_iterable else np.asarray(T)[active] if isinstance(T, Iterable) else T,
                                r, ivol[active], dv)
            P = model.price(call=call)
            diff = P - market_price_[active]

            converged = np.abs(diff) < precision
            result[active[converged]] = ivol[active[converged]]
            active = active[~converged]

            if len(active) == 0:
                break

            diff = diff[~converged]
            grad = model.vega()[~converged] if is_iterable or not converged.all() else model.vega()
            grad = np.where(np.abs(grad) < 1e-20, 1e-20, grad)

            ivol[active] -= diff / grad

        result[(result < 0) | (result > 100)] = np.nan
        return result[0] if not is_iterable else result

    @staticmethod
    def calc_implied_vol_mse(market_price, S, K, T, r, dv, call=True, initial_guess=0.5):
        is_iterable = isinstance(market_price, Iterable)
        x_init = np.array([initial_guess]) if not is_iterable else np.ones(len(market_price)) * initial_guess
        market_price_ = np.array([market_price]) if not is_iterable else copy(np.array(market_price))

        def objective_function(sigma, S, K, T, r, dv, call):
            model = GBSEquation(S, K, T, r, sigma, dv)
            p = model.price(call=call)
            return np.mean((market_price_ - p) ** 2)

        result = sco.minimize(objective_function, x0=x_init, bounds=[(1e-6, 5)] * len(x_init),
                              args=(S, K, T, r, dv, call))
        return result.x if result.success else None


if __name__ == '__main__':
    gbs = GBSEquation(100, 110, 0.1, 0.01, 0.25, 0.00)
    gbs.call_price()
