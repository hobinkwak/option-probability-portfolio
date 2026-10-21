"""
Fixed density construction pipeline. gamma (the E[R] model) is a correction parameter tied to this setup,
so it must be re-estimated whenever the setup changes. Changes are only for sensitivity experiments.
"""
START = '1996-01-01'
PIPELINE = dict(
    method='sabr',              # IV smile fit: SABR (beta=0.5 fixed, pysabr); best likelihood vs SVI/PCHIP, fits every month
    K_interval=0.5,
    mny_bounds=(0.45, 0.45),    # PDF grid width +-45% (+-12.5% truncates the left tail and biases the mean)
    filter_iv_sigma=None,
    filter_pdf_sigma=None,
    maximum_q=1.0,
    flatten=True,
    check_arbitrage=True,
)
DEFAULT_SUF = '_w45_96'          # cache/pdf_dataset{DEFAULT_SUF}.pkl is the standard dataset of the fixed pipeline


def signature(p=None):
    p = PIPELINE if p is None else p
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in p.items()}


def same_pipeline(p):
    """Whether p (dict) uses the same settings as the fixed pipeline. Returns True if p is None (no information)."""
    if p is None:
        return True
    return signature({k: p.get(k) for k in PIPELINE}) == signature()
