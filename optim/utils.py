import math

import numpy as np
from scipy import stats


def sampling(cdfs, strike, N, add_noise=True, noise_std=0.005, random_seed=42):
    np.random.seed(random_seed)
    rand = np.random.uniform(0, cdfs.max(), int(N))
    sampled = np.searchsorted(cdfs, rand)
    sampled_K = strike[sampled]

    if add_noise:
        noise = np.random.normal(0, noise_std, len(sampled_K))
        sampled_K = sampled_K + noise
        sampled_K = np.clip(sampled_K, strike.min(), strike.max())
    sampled_K = sampled_K.astype(np.float32)
    return sampled_K


def simulate_merton(S0, mu, sigma, lambda_jump, mu_jump, sigma_jump, n_steps, dt, n_paths=100):
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)  # 브라운 운동
        dN = np.random.poisson(lambda_jump * dt, n_paths)  # 점프 발생 횟수
        J = np.random.normal(mu_jump, sigma_jump, n_paths) * dN  # 점프 크기

        S[t, :] = S[t - 1, :] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + J
        )
    return S


def init_params(log_ret, dt, threshold=2):
    # Calculate log returns
    n = len(log_ret)

    # Calculate statistics of log returns
    mu_hat = np.mean(log_ret)
    sigma_hat = np.std(log_ret)

    # Identify jumps
    jump_indices = np.where(np.abs(log_ret - mu_hat) > threshold * sigma_hat)[0]
    no_jump_indices = np.where(np.abs(log_ret - mu_hat) <= threshold * sigma_hat)[0]

    # Estimate jump intensity (lambda)
    lamb_hat = len(jump_indices) / (n * dt)

    # Estimate jump sizes
    if len(jump_indices) > 0:
        jump_sizes = log_ret[jump_indices]
        mu_j_hat = np.mean(jump_sizes)
        sigma_j_hat = np.std(jump_sizes)
    else:
        mu_j_hat = 0
        sigma_j_hat = 0

    # Adjusted drift estimation
    # Remove jumps to estimate diffusion component
    diffusion_returns = log_ret[no_jump_indices]
    mu_diffusion_hat = np.mean(diffusion_returns) / dt
    sigma_diffusion_hat = np.std(diffusion_returns) / np.sqrt(dt)

    # Adjust drift for jump component
    k_hat = np.exp(mu_j_hat + 0.5 * sigma_j_hat ** 2) - 1
    mu_hat_adj = mu_diffusion_hat + lamb_hat * k_hat

    mu_j_hat = -0.05 if len(jump_indices) == 0 else mu_j_hat
    sigma_j_hat = 0.05 if len(jump_indices) <= 1 else sigma_j_hat

    return mu_hat_adj, sigma_diffusion_hat, lamb_hat, mu_j_hat, sigma_j_hat


def mjd_pdf(params, log_ret, dt, nll=False):
    mu, sigma, lambda_j, mu_j, sigma_j = params
    sigma = np.clip(sigma, 1e-5, None)
    lambda_j = np.clip(lambda_j, 1e-5, None)
    sigma_j = np.clip(sigma_j, 1e-5, None)

    res = 0
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    N = int(stats.poisson.ppf(0.99, dt * lambda_j))
    for n in range(N + 1):
        pois_prob = np.exp(-lambda_j * dt) * (lambda_j * dt) ** n / math.factorial(n)
        # pois_prob = stats.poisson.pmf(n, lambda_j * dt)
        drift = (mu - 0.5 * sigma ** 2 - lambda_j * k) * dt + n * mu_j
        vol = np.sqrt(dt * sigma ** 2 + n * sigma_j ** 2)
        res += pois_prob * stats.norm.pdf(log_ret, drift, vol)

    if nll:
        res = -np.sum(np.log(res))
    return res
