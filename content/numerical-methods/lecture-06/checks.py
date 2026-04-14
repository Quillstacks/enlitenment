"""Auto-checker helpers for Chapter 6 notebook (Integration & SGD)."""

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
# Composite Trapezoid Rule
# ---------------------------------------------------------------------------

def check_trapezoid(fn, f, a, b, n):
    got = fn(f, a, b, n)

    # Reference implementation
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    expected = f(x[0]) + f(x[-1])
    for i in range(1, n):
        expected += 2 * f(x[i])
    expected *= h / 2

    if got is None:
        print(f"  {_NONE} Trapezoid: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} Trapezoid({n}) = {got:.6f}")
        return

    # Mistake: forgot the 1/2 factor on endpoints (all weights = 2)
    no_half_endpoints = 2 * f(x[0]) + 2 * f(x[-1])
    for i in range(1, n):
        no_half_endpoints += 2 * f(x[i])
    no_half_endpoints *= h / 2
    if _close(got, no_half_endpoints):
        print(f"  {_FAIL} Trapezoid({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: endpoints should have weight 1, not 2. The pattern")
        print(f"       is 1-2-2-...-2-1, then multiply by h/2.")
        return

    # Mistake: wrong h (used n+1 instead of n subintervals)
    h_wrong = (b - a) / (n + 1)
    wrong_h_val = f(x[0]) + f(x[-1])
    for i in range(1, n):
        wrong_h_val += 2 * f(x[i])
    wrong_h_val *= h_wrong / 2
    if _close(got, wrong_h_val):
        print(f"  {_FAIL} Trapezoid({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: h = (b - a) / n, not (b - a) / (n + 1).")
        print(f"       n is the number of subintervals, n + 1 is the number of points.")
        return

    # Mistake: off-by-one (summing interior points including an endpoint)
    off_by_one = f(x[0]) + f(x[-1])
    for i in range(1, n + 1):
        off_by_one += 2 * f(x[min(i, n)])
    off_by_one *= h / 2
    if _close(got, off_by_one):
        print(f"  {_FAIL} Trapezoid({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: the interior sum runs from i=1 to i=n-1.")
        print(f"       Index n is the right endpoint, not an interior point.")
        return

    print(f"  {_FAIL} Trapezoid({n}) = {got:.6f}  (expected {expected:.6f})")
    print(f"       Hint: weights are 1-2-2-...-2-1, then multiply the sum by h/2.")


# ---------------------------------------------------------------------------
# Composite Simpson's Rule
# ---------------------------------------------------------------------------

def check_simpson(fn, f, a, b, n):
    got = fn(f, a, b, n)

    # Reference implementation
    if n % 2 != 0:
        print(f"  {_FAIL} Simpson requires even n, got {n}.")
        return

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    expected = f(x[0]) + f(x[-1])
    for i in range(1, n):
        if i % 2 == 1:
            expected += 4 * f(x[i])
        else:
            expected += 2 * f(x[i])
    expected *= h / 3

    if got is None:
        print(f"  {_NONE} Simpson: not implemented yet (expected {expected:.6f})")
        return

    if _close(got, expected):
        print(f"  {_OK} Simpson({n}) = {got:.6f}")
        return

    # Mistake: all weights are 4 (forgot the alternating 2s)
    all_fours = f(x[0]) + f(x[-1])
    for i in range(1, n):
        all_fours += 4 * f(x[i])
    all_fours *= h / 3
    if _close(got, all_fours):
        print(f"  {_FAIL} Simpson({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: not all interior weights are 4. The pattern is")
        print(f"       1, 4, 2, 4, 2, ..., 4, 1. Even-indexed interior points get 2.")
        return

    # Mistake: wrong denominator (used h/2 instead of h/3)
    wrong_denom = f(x[0]) + f(x[-1])
    for i in range(1, n):
        if i % 2 == 1:
            wrong_denom += 4 * f(x[i])
        else:
            wrong_denom += 2 * f(x[i])
    wrong_denom *= h / 2
    if _close(got, wrong_denom):
        print(f"  {_FAIL} Simpson({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: the outer multiplier is h/3, not h/2.")
        print(f"       You may be mixing up Simpson and Trapezoid formulas.")
        return

    # Mistake: used odd n (didn't check or enforce even n)
    if n % 2 != 0:
        print(f"  {_FAIL} Simpson({n}) = {got:.6f}  (expected {expected:.6f})")
        print(f"       Hint: Simpson's rule requires an even number of subintervals.")
        return

    print(f"  {_FAIL} Simpson({n}) = {got:.6f}  (expected {expected:.6f})")
    print(f"       Hint: weights are 1, 4, 2, 4, 2, ..., 4, 1, then multiply by h/3.")


# ---------------------------------------------------------------------------
# Monte Carlo Integration
# ---------------------------------------------------------------------------

def check_monte_carlo(fn, f, a, b, n_samples):
    # Seed before student call
    np.random.seed(42)
    got = fn(f, a, b, n_samples)

    # Seed before reference call
    np.random.seed(42)
    xs = np.random.uniform(a, b, n_samples)
    expected = (b - a) * np.mean(f(xs))

    tol = 0.1  # relaxed tolerance for stochastic methods

    if got is None:
        print(f"  {_NONE} Monte Carlo: not implemented yet (expected ~{expected:.4f})")
        return

    if _close(got, expected, tol=tol):
        print(f"  {_OK} Monte Carlo({n_samples}) = {got:.4f}")
        return

    # Mistake: forgot (b - a) factor
    np.random.seed(42)
    xs2 = np.random.uniform(a, b, n_samples)
    no_volume = np.mean(f(xs2))
    if _close(got, no_volume, tol=tol):
        print(f"  {_FAIL} Monte Carlo = {got:.4f}  (expected ~{expected:.4f})")
        print(f"       Hint: the Monte Carlo estimator is (b - a) * mean(f(x_i)),")
        print(f"       not just mean(f(x_i)). You forgot to multiply by the interval length.")
        return

    # Mistake: used sum instead of mean (off by factor n_samples)
    np.random.seed(42)
    xs3 = np.random.uniform(a, b, n_samples)
    sum_not_mean = (b - a) * np.sum(f(xs3))
    if _close(got, sum_not_mean, tol=tol * n_samples):
        print(f"  {_FAIL} Monte Carlo = {got:.4f}  (expected ~{expected:.4f})")
        print(f"       Hint: use np.mean(), not np.sum(). The estimator averages")
        print(f"       over all samples, then multiplies by (b - a).")
        return

    print(f"  {_FAIL} Monte Carlo = {got:.4f}  (expected ~{expected:.4f})")
    print(f"       Hint: draw n_samples uniform points in [a, b], evaluate f,")
    print(f"       then return (b - a) * np.mean(f(samples)).")


# ---------------------------------------------------------------------------
# Hit-or-Miss Monte Carlo
# ---------------------------------------------------------------------------

def check_hit_or_miss(fn, f, a, b, f_max, n_samples):
    # Seed before student call
    np.random.seed(42)
    got = fn(f, a, b, f_max, n_samples)

    # Seed before reference call
    np.random.seed(42)
    x = np.random.uniform(a, b, n_samples)
    y = np.random.uniform(0, f_max, n_samples)
    hits = np.sum(y <= f(x))
    expected = (b - a) * f_max * hits / n_samples

    tol = 0.15  # relaxed tolerance for stochastic methods

    if got is None:
        print(f"  {_NONE} Hit-or-miss: not implemented yet (expected ~{expected:.4f})")
        return

    if _close(got, expected, tol=tol):
        print(f"  {_OK} Hit-or-miss({n_samples}) = {got:.4f}")
        return

    # Mistake: forgot to scale by box area
    np.random.seed(42)
    x2 = np.random.uniform(a, b, n_samples)
    y2 = np.random.uniform(0, f_max, n_samples)
    hits2 = np.sum(y2 <= f(x2))
    no_scale = hits2 / n_samples
    if _close(got, no_scale, tol=tol):
        print(f"  {_FAIL} Hit-or-miss = {got:.4f}  (expected ~{expected:.4f})")
        print(f"       Hint: you computed the fraction of hits, but forgot to")
        print(f"       multiply by the box area (b - a) * f_max.")
        return

    # Mistake: only scaled by (b - a), forgot f_max
    np.random.seed(42)
    x3 = np.random.uniform(a, b, n_samples)
    y3 = np.random.uniform(0, f_max, n_samples)
    hits3 = np.sum(y3 <= f(x3))
    no_fmax = (b - a) * hits3 / n_samples
    if _close(got, no_fmax, tol=tol):
        print(f"  {_FAIL} Hit-or-miss = {got:.4f}  (expected ~{expected:.4f})")
        print(f"       Hint: the bounding box has height f_max, not 1.")
        print(f"       Multiply by (b - a) * f_max, not just (b - a).")
        return

    print(f"  {_FAIL} Hit-or-miss = {got:.4f}  (expected ~{expected:.4f})")
    print(f"       Hint: sample x in [{a}, {b}] and y in [0, f_max],")
    print(f"       count how many satisfy y <= f(x),")
    print(f"       then return (b - a) * f_max * count / n_samples.")


# ---------------------------------------------------------------------------
# SGD Step
# ---------------------------------------------------------------------------

def check_sgd_step(fn, theta, grad, lr):
    got = fn(theta, grad, lr)
    expected = theta - lr * grad

    if got is None:
        print(f"  {_NONE} SGD step: not implemented yet (expected {expected})")
        return

    got_arr = np.asarray(got, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)

    if np.allclose(got_arr, expected_arr, atol=_TOL):
        print(f"  {_OK} SGD step: theta_new = {got_arr}")
        return

    # Mistake: gradient ascent (added instead of subtracted)
    ascent = theta + lr * grad
    if np.allclose(got_arr, np.asarray(ascent, dtype=float), atol=_TOL):
        print(f"  {_FAIL} SGD step = {got_arr}  (expected {expected_arr})")
        print(f"       Hint: you are adding the gradient (ascent). SGD subtracts")
        print(f"       the gradient to move downhill: theta_new = theta - lr * grad.")
        return

    # Mistake: forgot the learning rate
    no_lr = theta - grad
    if np.allclose(got_arr, np.asarray(no_lr, dtype=float), atol=_TOL):
        print(f"  {_FAIL} SGD step = {got_arr}  (expected {expected_arr})")
        print(f"       Hint: you subtracted the raw gradient without scaling by")
        print(f"       the learning rate. The update is theta - lr * grad.")
        return

    print(f"  {_FAIL} SGD step = {got_arr}  (expected {expected_arr})")
    print(f"       Hint: SGD update rule is theta_new = theta - lr * grad.")
