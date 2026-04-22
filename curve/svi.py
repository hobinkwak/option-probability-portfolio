import warnings

import numpy as np
from scipy import optimize as sco

warnings.filterwarnings('ignore')


class SVIFitter:

    def __init__(self, F, T, regularization=0.01):
        self.F = F
        self.T = T
        self.reg = regularization

    @staticmethod
    def svi_slice(k, params):
        if params.ndim > 1:
            a = params[:, 0]
            b = params[:, 1]
            rho = params[:, 2]
            m = params[:, 3]
            sigma = params[:, 4]
        else:
            a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def natural_constraints(self, params):
        """
        Natural SVI constraints (Gatheral & Jacquier 2014)
        """
        a, b, rho, m, sigma = params

        # Cons 1: No butterfly arbitrage
        c1 = (4 - b ** 2 * (1 + abs(rho)) ** 2)

        # Cons 2: Calendar spread arbitrage
        c2 = a + b * sigma * np.sqrt(1 - rho ** 2)

        # Cons 3: Symmetry
        c3 = (a - m * b * (rho + np.sign(m))) * (4 - a + m * b * (rho + np.sign(m))) - b ** 2 * (rho + np.sign(m)) ** 2

        return np.array([c1, c2, c3])

    def fit(self, log_mnys, variance, weights=None):
        """
        Args:
            log_mnys: log(K/F)
            variance: IV^2 * T
            weights: observation weights
        """
        if weights is None:
            weights = np.ones(len(log_mnys))

        # Init guess
        a_init = np.mean(variance)
        b_init = 0.3
        rho_init = -0.3
        m_init = np.median(log_mnys)
        sigma_init = 0.1

        x0 = [a_init, b_init, rho_init, m_init, sigma_init]

        # Bounds
        bounds = [
            (0, 2 * np.max(variance)),  # a
            (0, 2),  # b
            (-0.999, 0.999),  # rho
            (-1, 1),  # m
            (0.01, 1)  # sigma
        ]

        def objective(params):
            pred = self.svi_slice(log_mnys, params)
            mse = np.mean(weights * (variance - pred) ** 2)

            # L2
            reg_term = self.reg * np.sum(np.array(params) ** 2)

            return mse + reg_term

        def constraint_func(params):
            return self.natural_constraints(params)

        constraints = {'type': 'ineq', 'fun': constraint_func}

        result = sco.minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-10}
        )

        if not result.success:
            warnings.warn(f"SVI fitting failed: {result.message}")

        return result.x, result.fun
