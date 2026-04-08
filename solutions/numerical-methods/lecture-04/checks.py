"""Auto-checker helpers for Chapter 4 — Root Finding."""

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


# ---------------------------------------------------------------------------
# Brute-force root finding
# ---------------------------------------------------------------------------

def check_brute_root(fn, f, a, b, n):
    """Check brute_force_root(f, a, b, n) -> x with smallest |f(x)|."""
    got = fn(f, a, b, n)

    xs = np.linspace(a, b, n)
    vals = np.abs(f(xs))
    best_idx = np.argmin(vals)
    expected = xs[best_idx]

    if got is None:
        print(f"  {_NONE} Brute-force root: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} Brute-force root = {got:.4f}")
        return

    # Mistake 1: returned f(x) instead of x
    if _close(got, f(expected)):
        print(f"  {_FAIL} Brute-force root = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you returned f(x) instead of x. The function should")
        print(f"       return the x value where |f(x)| is smallest.")
        return

    # Mistake 2: returned the index instead of the x value
    if _close(got, best_idx):
        print(f"  {_FAIL} Brute-force root = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you returned the index, not the x value. Use the")
        print(f"       index to look up the corresponding x from your grid.")
        return

    # Mistake 3: forgot abs — found min of f(x) rather than min of |f(x)|
    raw_vals = f(xs)
    raw_min_idx = np.argmin(raw_vals)
    raw_min_x = xs[raw_min_idx]
    if _close(got, raw_min_x) and not _close(raw_min_x, expected):
        print(f"  {_FAIL} Brute-force root = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you minimized f(x) instead of |f(x)|. A root is where")
        print(f"       f(x) is closest to zero, not where it is most negative.")
        return

    print(f"  {_FAIL} Brute-force root = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: create a grid of n points from a to b with np.linspace(),")
    print(f"       evaluate |f(x)| at each point, and return the x with the smallest value.")


# ---------------------------------------------------------------------------
# Newton's method
# ---------------------------------------------------------------------------

def check_newton(fn, f, f_prime, x0, true_root, tol=1e-8, max_iter=100):
    """Check newton_method(f, f_prime, x0, tol, max_iter) -> (root, history)."""
    result = fn(f, f_prime, x0, tol, max_iter)

    # Compute expected
    x = x0
    history = [x]
    for _ in range(max_iter):
        fx = f(x)
        dfx = f_prime(x)
        if dfx == 0:
            break
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    expected_root = history[-1]

    if result is None:
        print(f"  {_NONE} Newton's method: not implemented yet (expected root {expected_root:.6f})")
        return

    # Unpack
    try:
        got_root, got_history = result
    except (TypeError, ValueError):
        print(f"  {_FAIL} Newton's method: expected a tuple (root, history), got {type(result).__name__}")
        print(f"       Hint: return both the final root and the list of iterates.")
        return

    if got_root is None:
        print(f"  {_NONE} Newton's method: not implemented yet (expected root {expected_root:.6f})")
        return

    if _close(got_root, expected_root, tol=1e-6):
        print(f"  {_OK} Newton root = {got_root:.8f}  ({len(got_history)-1} iterations)")
        return

    # Mistake 1: converged to a different root (valid but not the target)
    if abs(f(got_root)) < 1e-6 and not _close(got_root, true_root, tol=0.1):
        print(f"  {_FAIL} Newton root = {got_root:.8f}  (expected near {true_root:.4f})")
        print(f"       Hint: the method converged to a different root of f(x).")
        print(f"       Try a starting point closer to the target root.")
        return

    # Mistake 2: forgot to update x — history is all x0
    if got_history is not None and len(got_history) > 1:
        if all(_close(h, x0, tol=1e-10) for h in got_history):
            print(f"  {_FAIL} Newton root = {got_root:.8f}  (expected {expected_root:.6f})")
            print(f"       Hint: all history values equal x0. Make sure you update x")
            print(f"       with x_new = x - f(x)/f'(x) inside your loop.")
            return

    # Mistake 3: applied f(x)*f'(x) instead of f(x)/f'(x)
    x_test = x0
    x_mult = x_test - f(x_test) * f_prime(x_test)
    if got_history is not None and len(got_history) > 1 and _close(got_history[1], x_mult):
        print(f"  {_FAIL} Newton root = {got_root:.8f}  (expected {expected_root:.6f})")
        print(f"       Hint: you multiplied f(x) by f'(x) instead of dividing.")
        print(f"       The update rule is x_new = x - f(x) / f'(x).")
        return

    print(f"  {_FAIL} Newton root = {got_root:.8f}  (expected {expected_root:.6f})")
    print(f"       Hint: the update rule is x_new = x - f(x) / f'(x). Check your")
    print(f"       loop and convergence condition.")


# ---------------------------------------------------------------------------
# Secant method
# ---------------------------------------------------------------------------

def check_secant(fn, f, x0, x1, true_root, tol=1e-8, max_iter=100):
    """Check secant_method(f, x0, x1, tol, max_iter) -> (root, history)."""
    result = fn(f, x0, x1, tol, max_iter)

    # Compute expected
    xa, xb = x0, x1
    history = [xa, xb]
    for _ in range(max_iter):
        fa = f(xa)
        fb = f(xb)
        if fb - fa == 0:
            break
        x_new = xb - fb * (xb - xa) / (fb - fa)
        history.append(x_new)
        if abs(x_new - xb) < tol:
            break
        xa, xb = xb, x_new
    expected_root = history[-1]

    if result is None:
        print(f"  {_NONE} Secant method: not implemented yet (expected root {expected_root:.6f})")
        return

    # Unpack
    try:
        got_root, got_history = result
    except (TypeError, ValueError):
        print(f"  {_FAIL} Secant method: expected a tuple (root, history), got {type(result).__name__}")
        print(f"       Hint: return both the final root and the list of iterates.")
        return

    if got_root is None:
        print(f"  {_NONE} Secant method: not implemented yet (expected root {expected_root:.6f})")
        return

    if _close(got_root, expected_root, tol=1e-6):
        print(f"  {_OK} Secant root = {got_root:.8f}  ({len(got_history)-2} iterations)")
        return

    # Mistake 1: converged to a different root
    if abs(f(got_root)) < 1e-6 and not _close(got_root, true_root, tol=0.1):
        print(f"  {_FAIL} Secant root = {got_root:.8f}  (expected near {true_root:.4f})")
        print(f"       Hint: the method converged to a different root of f(x).")
        print(f"       Try starting points closer to the target root.")
        return

    # Mistake 2: used Newton's formula (divided by f'(x) instead of finite difference)
    # If the second history entry matches Newton's update, they likely used the derivative
    if got_history is not None and len(got_history) > 2:
        # Check if the student accidentally implemented Newton instead of secant
        # Newton step from x1: x1 - f(x1)/f'(x1) where f'(x) = 3x^2 - 1
        # We can't assume the derivative, so we check the denominator pattern
        pass

    # Mistake 3: wrong denominator — used (x1 - x0) instead of (f(x1) - f(x0))
    fa0, fb0 = f(x0), f(x1)
    wrong_denom = x1 - fb0 * (x1 - x0) / (x1 - x0) if (x1 - x0) != 0 else None
    if wrong_denom is not None and got_history is not None and len(got_history) > 2:
        if _close(got_history[2], wrong_denom):
            print(f"  {_FAIL} Secant root = {got_root:.8f}  (expected {expected_root:.6f})")
            print(f"       Hint: the denominator should be f(x1) - f(x0), not x1 - x0.")
            print(f"       The secant method approximates the derivative using function values.")
            return

    # Mistake 4: swapped numerator/denominator in the fraction
    if (fb0 - fa0) != 0:
        swapped = x1 - fb0 * (fb0 - fa0) / (x1 - x0)
        if got_history is not None and len(got_history) > 2 and _close(got_history[2], swapped):
            print(f"  {_FAIL} Secant root = {got_root:.8f}  (expected {expected_root:.6f})")
            print(f"       Hint: the fraction is (x1 - x0) / (f(x1) - f(x0)), not the")
            print(f"       other way around. Check the secant update formula.")
            return

    print(f"  {_FAIL} Secant root = {got_root:.8f}  (expected {expected_root:.6f})")
    print(f"       Hint: the update is x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0)).")
    print(f"       Then shift: x0, x1 = x1, x_new.")
