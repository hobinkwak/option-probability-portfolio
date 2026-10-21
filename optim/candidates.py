"""Candidate strike selection (restricting the call-writing candidates)"""
import numpy as np


def select_candidates(strikes, deltas, spot, min_moneyness=1.0, min_delta=0.10, max_n=30):
    """
    Indices of listed strikes with K >= spot * min_moneyness (ATM or above) and delta >= min_delta (excludes far OTM).

    Args:
        strikes: listed strikes (ascending)
        deltas: call delta of each strike
        spot: underlying price on the entry date
        min_moneyness: lower bound of strike / spot (1.0 = ATM or above, excludes ITM)
        min_delta: delta floor (0.10 excludes far-OTM strikes with delta below 0.10)
        max_n: cap on the number of candidates. If exceeded, thin evenly in strike order (QP dimension)
    Returns:
        np.ndarray: indices into strikes. If no candidate qualifies, the single highest-delta strike at or above ATM
    """
    strikes = np.asarray(strikes, float); deltas = np.asarray(deltas, float)
    ix = np.where((strikes >= spot * min_moneyness) & (deltas >= min_delta))[0]
    if len(ix) == 0:
        cand = np.where(strikes >= spot * min_moneyness)[0]
        ix = cand[[int(np.argmax(deltas[cand]))]] if len(cand) else np.array([int(np.argmin(np.abs(strikes - spot)))])
    if len(ix) > max_n:
        ix = ix[np.unique(np.round(np.linspace(0, len(ix) - 1, max_n)).astype(int))]
    return ix
