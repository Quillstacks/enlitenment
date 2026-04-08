"""Auto-checker helpers for Chapter 2 notebook."""

import numpy as np

_OK = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL = 0.01


def _close(a, b, tol=_TOL):
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def _close_rel(a, b, rtol=0.1):
    """Relative tolerance check — suitable when values span many orders of magnitude."""
    try:
        a, b = float(a), float(b)
        if b == 0:
            return abs(a) < 1e-30
        return abs(a - b) / abs(b) < rtol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Machine epsilon
# ---------------------------------------------------------------------------

def check_machine_epsilon(fn, dtype):
    got = fn(dtype)

    # Compute expected by halving
    eps = dtype(1)
    while dtype(1) + dtype(eps) > dtype(1):
        last = eps
        eps = dtype(eps / 2)
    expected = float(last)

    label = dtype.__name__

    if got is None:
        print(f"  {_NONE} Machine epsilon ({label}): not implemented yet "
              f"(expected {expected:.6e})")
        return

    if _close_rel(got, expected, rtol=0.1):
        print(f"  {_OK} Machine epsilon ({label}) = {float(got):.6e}")
        return

    # Mistake: off-by-one — returned the eps that was NOT distinguishable
    off_by_one = float(dtype(expected) / dtype(2))
    if _close_rel(got, off_by_one, rtol=0.1):
        print(f"  {_FAIL} Machine epsilon ({label}) = {float(got):.6e}  "
              f"(expected {expected:.6e})")
        print(f"       Hint: you returned the epsilon that is no longer "
              f"distinguishable from zero.")
        print(f"       You need the last epsilon where dtype(1) + eps > dtype(1).")
        return

    # Mistake: returned np.finfo(dtype).eps directly
    finfo_eps = float(np.finfo(dtype).eps)
    if _close_rel(got, finfo_eps, rtol=0.01) and not _close_rel(got, expected, rtol=0.1):
        print(f"  {_FAIL} Machine epsilon ({label}) = {float(got):.6e}  "
              f"(expected {expected:.6e})")
        print(f"       Hint: that looks like np.finfo(dtype).eps. Implement the "
              f"halving search yourself.")
        return

    print(f"  {_FAIL} Machine epsilon ({label}) = {float(got):.6e}  "
          f"(expected {expected:.6e})")
    print(f"       Hint: start with eps = 1, keep halving while "
          f"dtype(1) + dtype(eps) > dtype(1),")
    print(f"       and return the last eps that was still distinguishable.")


# ---------------------------------------------------------------------------
# Cancellation error
# ---------------------------------------------------------------------------

def check_cancellation(fn, a, b, eps):
    got = fn(a, b, eps)

    naive = (a + eps) - b
    exact = eps
    expected = (naive, exact)

    if got is None:
        print(f"  {_NONE} Cancellation error: not implemented yet "
              f"(expected ({naive:.6e}, {exact:.6e}))")
        return

    # Check it is a tuple/list of length 2
    try:
        g0, g1 = got
    except (TypeError, ValueError):
        print(f"  {_FAIL} Cancellation error: expected a tuple of two values, "
              f"got {type(got).__name__}")
        print(f"       Hint: return (naive_result, true_result).")
        return

    ok_naive = _close_rel(g0, naive, rtol=0.01) or _close(g0, naive, tol=1e-30)
    ok_exact = _close_rel(g1, exact, rtol=0.01) or _close(g1, exact, tol=1e-30)

    if ok_naive and ok_exact:
        print(f"  {_OK} Cancellation: naive = {float(g0):.6e}, "
              f"true = {float(g1):.6e}")
        return

    # Mistake: returned single value instead of tuple
    # (already caught above)

    # Mistake: swapped order
    ok_swap0 = _close_rel(g0, exact, rtol=0.01) or _close(g0, exact, tol=1e-30)
    ok_swap1 = _close_rel(g1, naive, rtol=0.01) or _close(g1, naive, tol=1e-30)
    if ok_swap0 and ok_swap1:
        print(f"  {_FAIL} Cancellation: got ({float(g0):.6e}, {float(g1):.6e})  "
              f"(expected ({naive:.6e}, {exact:.6e}))")
        print(f"       Hint: the order is (naive_result, true_result), not the "
              f"other way around.")
        return

    print(f"  {_FAIL} Cancellation: got ({float(g0):.6e}, {float(g1):.6e})  "
          f"(expected ({naive:.6e}, {exact:.6e}))")
    print(f"       Hint: naive_result = (a + eps) - b, true_result = eps "
          f"(the mathematically exact answer).")


# ---------------------------------------------------------------------------
# Safe subtract
# ---------------------------------------------------------------------------

def check_safe_subtract(fn, a, eps, b):
    got = fn(a, eps, b)

    # When a == b, the safe rearrangement gives exactly eps
    expected = eps + (a - b)

    if got is None:
        print(f"  {_NONE} Safe subtract: not implemented yet "
              f"(expected {expected:.6e})")
        return

    if _close_rel(got, expected, rtol=0.01) or _close(got, expected, tol=1e-30):
        print(f"  {_OK} Safe subtract = {float(got):.6e}")
        return

    # Mistake: still using naive formula (a + eps) - b
    naive = (a + eps) - b
    if (_close_rel(got, naive, rtol=0.01) or _close(got, naive, tol=1e-30)) and \
       not (_close_rel(naive, expected, rtol=0.01)):
        print(f"  {_FAIL} Safe subtract = {float(got):.6e}  "
              f"(expected {expected:.6e})")
        print(f"       Hint: you are still computing (a + eps) - b, which "
              f"suffers from cancellation.")
        print(f"       Rearrange the terms: eps + (a - b) avoids adding a "
              f"tiny eps to a large a.")
        return

    # Mistake: wrong sign
    if _close_rel(got, -expected, rtol=0.01):
        print(f"  {_FAIL} Safe subtract = {float(got):.6e}  "
              f"(expected {expected:.6e})")
        print(f"       Hint: the sign is flipped. Check the order of "
              f"subtraction: eps + (a - b), not eps + (b - a).")
        return

    print(f"  {_FAIL} Safe subtract = {float(got):.6e}  "
          f"(expected {expected:.6e})")
    print(f"       Hint: rearrange (a + eps) - b as eps + (a - b). "
          f"When a == b the second term is exactly 0.")
