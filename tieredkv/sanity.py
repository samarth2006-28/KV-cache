"""Small dependency-free statistics for paired reconstruction/attention data."""
import numpy as np


def _ranks(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i+1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i+j-1)/2
        i = j
    return ranks


def spearman(scores, errors):
    """Return rho, or None when constant/insufficient data make it undefined."""
    x = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(errors, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired scores/errors must be finite and aligned")
    if len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
