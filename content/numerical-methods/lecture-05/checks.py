"""Auto-checker helpers for Chapter 5 notebook (Global Optimization)."""

import numpy as np

_OK = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL = 0.05


def _close(a, b, tol=_TOL):
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Shekel landscape helper — compute the known global minimum
# ---------------------------------------------------------------------------

def _find_global(f, bounds, n_grid=100000):
    """Brute-force scan to locate the global minimum of f on bounds."""
    xs = np.linspace(bounds[0], bounds[1], n_grid)
    ys = np.array([f(xi) for xi in xs])
    idx = int(np.argmin(ys))
    return xs[idx], ys[idx]


# ---------------------------------------------------------------------------
# Random Restart
# ---------------------------------------------------------------------------

def check_random_restart(fn, optimizer_fn, f, f_prime,
                         n_restarts, bounds, known_global_x, known_global_f):
    """
    Validate the student's random_restart implementation.

    Parameters
    ----------
    fn : callable
        Student's random_restart(optimizer_fn, f, f_prime,
                                 n_restarts, bounds) -> (best_x, best_f, all_results)
    optimizer_fn : callable
        The local_optimize reference provided in the notebook.
    f, f_prime : callables
        Shekel landscape and its derivative.
    n_restarts : int
    bounds : tuple
    known_global_x : float
        Approximate x-location of the global minimum.
    known_global_f : float
        Approximate function value at the global minimum.
    """
    np.random.seed(42)
    got = fn(optimizer_fn, f, f_prime, n_restarts, bounds)

    # --- not implemented ---
    if got is None:
        print(f"  {_NONE} random_restart: not implemented yet")
        return

    # --- unpack ---
    if not isinstance(got, tuple):
        print(f"  {_FAIL} random_restart: expected a tuple (best_x, best_f, all_results), "
              f"got {type(got).__name__}")
        return

    if len(got) == 1:
        print(f"  {_FAIL} random_restart: returned only one value. "
              f"Expected (best_x, best_f, all_results).")
        return

    if len(got) == 2:
        print(f"  {_FAIL} random_restart: returned two values. "
              f"Expected three: (best_x, best_f, all_results).")
        return

    if len(got) != 3:
        print(f"  {_FAIL} random_restart: expected 3 return values, got {len(got)}")
        return

    best_x, best_f, all_results = got

    # --- recompute expected values with same seed ---
    np.random.seed(42)
    exp_results = []
    exp_best_x = None
    exp_best_f = float('inf')
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[0], bounds[1])
        x_star, history, n_evals = optimizer_fn(f, f_prime, x0)
        f_star = f(x_star)
        exp_results.append({'x0': x0, 'x_final': x_star, 'f_final': f_star})
        if f_star < exp_best_f:
            exp_best_f = f_star
            exp_best_x = x_star

    all_ok = True

    # --- check best_x ---
    if _close(best_x, exp_best_x):
        print(f"  {_OK} best_x = {best_x:.6f}")
    else:
        all_ok = False
        # diagnose: did they return the worst?
        worst_f = max(r['f_final'] for r in exp_results)
        worst_x = [r['x_final'] for r in exp_results if _close(r['f_final'], worst_f)]
        if worst_x and _close(best_x, worst_x[0]):
            print(f"  {_FAIL} best_x = {best_x:.6f}  (expected {exp_best_x:.6f})")
            print(f"       Hint: you returned the worst result, not the best. "
                  f"Check your comparison operator.")
        else:
            print(f"  {_FAIL} best_x = {best_x:.6f}  (expected {exp_best_x:.6f})")

    # --- check best_f ---
    if _close(best_f, exp_best_f):
        print(f"  {_OK} best_f = {best_f:.6f}")
    else:
        all_ok = False
        if _close(best_f, -exp_best_f):
            print(f"  {_FAIL} best_f = {best_f:.6f}  (expected {exp_best_f:.6f})")
            print(f"       Hint: sign is flipped. The Shekel function returns negative values.")
        else:
            print(f"  {_FAIL} best_f = {best_f:.6f}  (expected {exp_best_f:.6f})")

    # --- check all_results ---
    if all_results is None:
        all_ok = False
        print(f"  {_FAIL} all_results is None. Return a list of per-run results.")
    elif not hasattr(all_results, '__len__'):
        all_ok = False
        print(f"  {_FAIL} all_results should be a list, got {type(all_results).__name__}")
    elif len(all_results) != n_restarts:
        all_ok = False
        print(f"  {_FAIL} all_results has {len(all_results)} entries "
              f"(expected {n_restarts})")
        print(f"       Hint: append one result per restart.")
    else:
        print(f"  {_OK} all_results has {len(all_results)} entries")

    if all_ok:
        print(f"  {_OK} random_restart looks correct!")


# ---------------------------------------------------------------------------
# Basin Hopping
# ---------------------------------------------------------------------------

def check_basin_hopping(fn, optimizer_fn, f, f_prime,
                        x0, n_jumps, jump_range, bounds,
                        known_global_x, known_global_f):
    """
    Validate the student's basin_hopping implementation.

    Parameters
    ----------
    fn : callable
        Student's basin_hopping(optimizer_fn, f, f_prime,
                                x0, n_jumps, jump_range, bounds)
              -> (best_x, best_f, jump_history)
    optimizer_fn : callable
        The local_optimize reference provided in the notebook.
    f, f_prime : callables
    x0 : float
    n_jumps : int
    jump_range : float
    bounds : tuple
    known_global_x, known_global_f : float
    """
    np.random.seed(42)
    got = fn(optimizer_fn, f, f_prime, x0, n_jumps, jump_range, bounds)

    # --- not implemented ---
    if got is None:
        print(f"  {_NONE} basin_hopping: not implemented yet")
        return

    # --- unpack ---
    if not isinstance(got, tuple):
        print(f"  {_FAIL} basin_hopping: expected a tuple (best_x, best_f, jump_history), "
              f"got {type(got).__name__}")
        return

    if len(got) == 2:
        print(f"  {_FAIL} basin_hopping: returned two values. "
              f"Expected three: (best_x, best_f, jump_history).")
        return

    if len(got) != 3:
        print(f"  {_FAIL} basin_hopping: expected 3 return values, got {len(got)}")
        return

    best_x, best_f, jump_history = got

    # --- recompute expected values with same seed ---
    np.random.seed(42)
    exp_best_x = x0
    exp_best_f = f(x0)
    exp_jump_history = [x0]

    current_x = x0
    for jump_num in range(n_jumps):
        x_local, hist, n_evals = optimizer_fn(f, f_prime, current_x)
        f_local = f(x_local)
        if f_local < exp_best_f:
            exp_best_f = f_local
            exp_best_x = x_local
        if jump_num < n_jumps - 1:
            delta = np.random.uniform(-jump_range, jump_range)
            x_next = float(np.clip(exp_best_x + delta, bounds[0], bounds[1]))
            current_x = x_next
            exp_jump_history.append(x_next)
        else:
            current_x = exp_best_x

    all_ok = True

    # --- check best_x ---
    if _close(best_x, exp_best_x):
        print(f"  {_OK} best_x = {best_x:.6f}")
    else:
        all_ok = False
        # diagnose: did they return the last, not the best?
        if _close(best_x, current_x) and not _close(current_x, exp_best_x):
            print(f"  {_FAIL} best_x = {best_x:.6f}  (expected {exp_best_x:.6f})")
            print(f"       Hint: you returned the last converged point, not the best one. "
                  f"Track the overall best separately.")
        else:
            print(f"  {_FAIL} best_x = {best_x:.6f}  (expected {exp_best_x:.6f})")

    # --- check best_f ---
    if _close(best_f, exp_best_f):
        print(f"  {_OK} best_f = {best_f:.6f}")
    else:
        all_ok = False
        print(f"  {_FAIL} best_f = {best_f:.6f}  (expected {exp_best_f:.6f})")

    # --- check jump_history ---
    if jump_history is None:
        all_ok = False
        print(f"  {_FAIL} jump_history is None. Return a list of jump positions.")
    elif not hasattr(jump_history, '__len__'):
        all_ok = False
        print(f"  {_FAIL} jump_history should be a list, got {type(jump_history).__name__}")
    else:
        exp_len = len(exp_jump_history)
        if len(jump_history) != exp_len:
            all_ok = False
            print(f"  {_FAIL} jump_history has {len(jump_history)} entries "
                  f"(expected {exp_len})")
            if len(jump_history) == n_jumps:
                print(f"       Hint: the history should include the initial x0 "
                      f"plus each jump destination ({exp_len} total).")
            elif len(jump_history) == n_jumps + 1:
                print(f"       Hint: there are {n_jumps} jumps but only {n_jumps - 1} "
                      f"random perturbations (no jump after the last convergence).")
        else:
            print(f"  {_OK} jump_history has {len(jump_history)} entries")

    if all_ok:
        print(f"  {_OK} basin_hopping looks correct!")
