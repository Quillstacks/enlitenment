"""Auto-checker helpers for Chapter 7, Uncertainty Estimation."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3


def _close(a, b, tol=_TOL):
    """Scalar / array closeness with the chapter's default tolerance."""
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# 🎲 MC Dropout: predictive_stats(samples) -> (mean, var)
# ---------------------------------------------------------------------------

def check_predictive_stats(fn):
    """Sanity-check predictive_stats over a synthetic (T, n, K) tensor.

    The convention: axis 0 is T (the number of forward passes), axis 1 is n
    (samples), axis 2 is K (classes). Both outputs should have shape (n, K).
    """
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=(20, 50, 5)).astype(np.float64)
    samples /= samples.sum(axis=2, keepdims=True)
    expected_mean = samples.mean(axis=0)
    expected_var  = samples.var(axis=0)

    got = fn(samples)
    if got is None:
        print(f"  {_NONE} predictive_stats: not implemented yet "
              f"(expected mean and var of shape {expected_mean.shape})")
        return

    if not (isinstance(got, tuple) and len(got) == 2):
        print(f"  {_FAIL} predictive_stats: should return a tuple (mean, var), "
              f"got {type(got).__name__}")
        return

    mean, var = got
    mean = np.asarray(mean, dtype=float)
    var  = np.asarray(var,  dtype=float)

    if mean.shape != expected_mean.shape:
        print(f"  {_FAIL} predictive_stats: mean has shape {mean.shape} "
              f"(expected {expected_mean.shape})")
        if mean.shape == (samples.shape[0], samples.shape[2]):
            print(f"       Hint: looks like you reduced over axis 1 instead of axis 0. "
                  f"T (the pass count) is axis 0.")
        return
    if var.shape != expected_var.shape:
        print(f"  {_FAIL} predictive_stats: var has shape {var.shape} "
              f"(expected {expected_var.shape})")
        return

    if not _close(mean, expected_mean):
        if _close(mean, samples.mean(axis=1)):
            print(f"  {_FAIL} predictive_stats: mean averaged over the sample axis "
                  f"instead of the T axis")
            print(f"       Hint: T (axis 0) is the pass count; use "
                  f"np.mean(samples, axis=0).")
            return
        print(f"  {_FAIL} predictive_stats: mean values do not match "
              f"np.mean(samples, axis=0)")
        return

    if not _close(var, expected_var):
        if _close(var, samples.std(axis=0)):
            print(f"  {_FAIL} predictive_stats: returned standard deviation, "
                  f"not variance")
            print(f"       Hint: predictive variance is np.var(...), not np.std(...).")
            return
        if _close(var, samples.var(axis=0, ddof=1)):
            print(f"  {_FAIL} predictive_stats: used ddof=1 (sample variance)")
            print(f"       Hint: stick with the default ddof=0 — the population "
                  f"variance over the T passes.")
            return
        print(f"  {_FAIL} predictive_stats: variance values do not match "
              f"np.var(samples, axis=0)")
        return

    print(f"  {_OK} predictive_stats: mean and var look right")
