# option probability portfolio

Extract risk-neutral probability distributions from option chains and optimize covered call strike allocation using Monte Carlo simulation with Merton Jump-Diffusion dynamics.

---

## Pipeline

```
Option Chain → IV Surface Fitting → Breeden-Litzenberger PDF → Monte Carlo Sampling → Covered Call Optimization
                (SABR / SVI)           ∂²C/∂K²                  (Implied × MJD)         Mean-Variance + CVaR
```

## Core Mathematics

### 1. Implied PDF (Breeden-Litzenberger)

$$q(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$$

Fit the IV surface via SABR or SVI, then recover the risk-neutral density through finite differences on a dense strike grid. Butterfly and call-spread arbitrage conditions are enforced before extraction.

### 2. Model-Free Implied Moments (Bakshi-Kapadia-Madan 2003)

$$V = 2\int \frac{1 - \ln(K/F)}{K^2} \cdot O(K)\,dK, \quad W = \int \frac{6\ln(K/F) - 3\ln^2(K/F)}{K^2} \cdot O(K)\,dK$$

Compute variance, skewness, and kurtosis directly from OTM option prices — no model assumptions required.

### 3. Covered Call Objective

$$\min_{\mathbf{x}} \; -(1-\lambda)\,\mathbb{E}[R] + \lambda\,\text{Var}[R] + \varepsilon \cdot \text{CVaR}_\alpha[R]$$

Terminal prices $S_T$ are sampled from the posterior of the implied PDF (Q-measure) and a Merton Jump-Diffusion model (P-measure):

$$p(S_T) \propto q(S_T) \times f_{\text{MJD}}(S_T \mid \mu, \sigma, \lambda, \mu_J, \sigma_J)$$

MJD parameters are estimated conditionally on the current volatility regime, then blended with full-sample estimates.

## Usage

```python
from pdf import OptionImpliedPDF
from optim import BuyWriteOptimizer

# 1. Extract implied PDF
model = OptionImpliedPDF(option_chain_df, rf=0.035, dividend=0.0138)
model.fit(method='sabr', mny_bounds=(0.15, 0.15), check_arbitrage=True)

# 2. Optimize covered call
optimizer = BuyWriteOptimizer(log_ret, dt=1/252, Ks=model.Ks, S=model.S, T=T, r=rf)
optimizer.fit(pdfs=model.pdf.values, use_parametric=True, N=100_000)
weights = optimizer.optimize(
    price=model.price,
    mnys=target_strikes / S,
    risk_aversion=np.arange(0.1, 1.0, 0.1),
    robust=True,
)
```

## Key Features

- **IV Surface Fitting**: SABR, SVI, and PCHIP interpolation with automatic arbitrage detection and correction
- **Regime-Aware MJD**: Volatility-regime-conditional parameter estimation blended with full-sample estimates
- **Robust Optimization**: CVaR penalty to guard against PDF estimation error in the tails
- **Solver Options**: Brute-force Monte Carlo (parallelized) or COBYLA with warm-start across risk aversion levels
- **Greeks Monitor**: GEX, VEX, gamma/vanna profiles, and flip point computation

## Output Examples

### Implied PDF Diagnostics
![Implied PDF Dashboard](plot/Implied PDF Dashboard.png)

### Net Gamma Exposure by Strike
![GEX](plot/GEX.png)

### Gamma Profile
![Gamma Profile](plot/Gamma Profile.png)

### Covered Call Optimal Weights
![Covered Call Heatmap](plot/Covered Call Heatmap.png)

## References

- Breeden & Litzenberger (1978) — *Prices of State-Contingent Claims Implicit in Option Prices*
- Bakshi, Kapadia & Madan (2003) — *Stock Return Characteristics, Skew Laws, and Differential Pricing*
- Hagan et al. (2002) — *Managing Smile Risk* (SABR)
- Gatheral & Jacquier (2014) — *Arbitrage-Free SVI Volatility Surfaces*
- Merton (1976) — *Option Pricing When Underlying Stock Returns Are Discontinuous*

## License

MIT
