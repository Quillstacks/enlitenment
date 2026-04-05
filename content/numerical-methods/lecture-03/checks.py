"""Auto-checker helpers for Chapter 3 notebook."""

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


def _close_arr(a, b, tol=_TOL):
    try:
        return np.allclose(a, b, atol=tol)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Brute-Force Solver
# ---------------------------------------------------------------------------

def check_brute_force(fn, X, Y, w_range, b_range, h):
    got = fn(X, Y, w_range, b_range, h)

    # Compute expected internally
    A = np.array([[X[0], 1], [X[1], 1]])
    best_w, best_b, best_err = None, None, float('inf')
    for w in np.arange(w_range[0], w_range[1] + h, h):
        for b in np.arange(b_range[0], b_range[1] + h, h):
            preds = A @ np.array([w, b])
            err = np.mean(np.abs(Y - preds))
            if err < best_err:
                best_err = err
                best_w = w
                best_b = b
    expected = (best_w, best_b, best_err)

    # Not implemented
    if got is None:
        print(f"  {_NONE} Brute-force: not implemented yet "
              f"(expected w={expected[0]:.4f}, b={expected[1]:.4f}, "
              f"error={expected[2]:.4f})")
        return

    # Unpack student result
    try:
        g_w, g_b, g_err = got
    except (TypeError, ValueError):
        print(f"  {_FAIL} Brute-force: expected a tuple (w_best, b_best, min_error), "
              f"got {type(got).__name__}")
        return

    # Correct
    if _close(g_w, best_w) and _close(g_b, best_b) and _close(g_err, best_err):
        print(f"  {_OK} Brute-force: w={g_w:.4f}, b={g_b:.4f}, MAE={g_err:.4f}")
        return

    # --- Diagnose common mistakes ---

    # Mistake 1: used max instead of min
    worst_err = -float('inf')
    worst_w, worst_b = None, None
    for w in np.arange(w_range[0], w_range[1] + h, h):
        for b in np.arange(b_range[0], b_range[1] + h, h):
            preds = A @ np.array([w, b])
            err = np.mean(np.abs(Y - preds))
            if err > worst_err:
                worst_err = err
                worst_w = w
                worst_b = b
    if _close(g_w, worst_w) and _close(g_b, worst_b):
        print(f"  {_FAIL} Brute-force: w={g_w:.4f}, b={g_b:.4f}, error={g_err:.4f}  "
              f"(expected w={best_w:.4f}, b={best_b:.4f}, error={best_err:.4f})")
        print(f"       Hint: you are maximizing the error instead of minimizing it.")
        return

    # Mistake 2: used MSE instead of MAE
    best_w_mse, best_b_mse, best_mse = None, None, float('inf')
    for w in np.arange(w_range[0], w_range[1] + h, h):
        for b in np.arange(b_range[0], b_range[1] + h, h):
            preds = A @ np.array([w, b])
            err = np.mean((Y - preds) ** 2)
            if err < best_mse:
                best_mse = err
                best_w_mse = w
                best_b_mse = b
    if _close(g_w, best_w_mse) and _close(g_b, best_b_mse):
        print(f"  {_FAIL} Brute-force: w={g_w:.4f}, b={g_b:.4f}, error={g_err:.4f}  "
              f"(expected w={best_w:.4f}, b={best_b:.4f}, error={best_err:.4f})")
        print(f"       Hint: you are using MSE (squared error). The task asks for MAE "
              f"(mean absolute error).")
        return

    # Mistake 3: swapped w and b roles
    if _close(g_w, best_b) and _close(g_b, best_w):
        print(f"  {_FAIL} Brute-force: w={g_w:.4f}, b={g_b:.4f}, error={g_err:.4f}  "
              f"(expected w={best_w:.4f}, b={best_b:.4f}, error={best_err:.4f})")
        print(f"       Hint: w and b appear to be swapped. Check the order of your "
              f"return values.")
        return

    # Generic fallback
    print(f"  {_FAIL} Brute-force: w={g_w:.4f}, b={g_b:.4f}, error={g_err:.4f}  "
          f"(expected w={best_w:.4f}, b={best_b:.4f}, error={best_err:.4f})")
    print(f"       Hint: loop over w and b in their ranges with step h. For each "
          f"pair, compute predictions = X*w + b, then MAE = mean(abs(Y - preds)). "
          f"Track the minimum.")


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def check_conditioning(fn, X, Y, perturbations):
    A = np.array([[X[0], 1], [X[1], 1]])
    exact = np.linalg.solve(A, Y)

    expected = []
    for p in perturbations:
        Y_pert = Y.copy()
        Y_pert[0] += p
        sol_pert = np.linalg.solve(A, Y_pert)
        expected.append(np.linalg.norm(sol_pert - exact))
    expected = np.array(expected)

    got = fn(X, Y, perturbations)

    # Not implemented
    if got is None:
        print(f"  {_NONE} Conditioning: not implemented yet "
              f"(expected {expected.round(4)})")
        return

    got = np.asarray(got, dtype=float)

    # Correct
    if _close_arr(got, expected):
        print(f"  {_OK} Conditioning: norm changes = {got.round(4)}")
        return

    # --- Diagnose common mistakes ---

    # Mistake 1: perturbed X instead of Y
    wrong_perturb = []
    for p in perturbations:
        X_pert = X.copy()
        X_pert[0] += p
        A_pert = np.array([[X_pert[0], 1], [X_pert[1], 1]])
        try:
            sol_pert = np.linalg.solve(A_pert, Y)
            wrong_perturb.append(np.linalg.norm(sol_pert - exact))
        except np.linalg.LinAlgError:
            wrong_perturb.append(float('inf'))
    if _close_arr(got, np.array(wrong_perturb)):
        print(f"  {_FAIL} Conditioning: {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: you are perturbing X instead of Y. The task asks to "
              f"perturb Y[0] and solve with the original coefficient matrix.")
        return

    # Mistake 2: forgot norm, returned raw solution differences
    raw_diffs = []
    for p in perturbations:
        Y_pert = Y.copy()
        Y_pert[0] += p
        sol_pert = np.linalg.solve(A, Y_pert)
        raw_diffs.append(sol_pert - exact)
    raw_diffs = np.array(raw_diffs)
    if got.shape == raw_diffs.shape and _close_arr(got, raw_diffs):
        print(f"  {_FAIL} Conditioning: shape mismatch or raw differences returned  "
              f"(expected {expected.round(4)})")
        print(f"       Hint: return np.linalg.norm() of the solution change, "
              f"not the raw difference vector.")
        return

    # Mistake 3: returned absolute solutions not changes
    abs_sols = []
    for p in perturbations:
        Y_pert = Y.copy()
        Y_pert[0] += p
        sol_pert = np.linalg.solve(A, Y_pert)
        abs_sols.append(np.linalg.norm(sol_pert))
    if _close_arr(got, np.array(abs_sols)):
        print(f"  {_FAIL} Conditioning: {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: return the norm of the change (perturbed - exact), "
              f"not the norm of the perturbed solution itself.")
        return

    # Generic fallback
    print(f"  {_FAIL} Conditioning: {got.round(4)}  (expected {expected.round(4)})")
    print(f"       Hint: for each perturbation p, add p to Y[0], solve with "
          f"np.linalg.solve, and return np.linalg.norm(sol_perturbed - sol_exact).")


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def check_stability(fn, X, w, b, perturbations):
    ref = w * X + b

    expected = []
    for p in perturbations:
        pred_w = (w + p) * X + b
        pred_b = w * X + (b + p)
        change_w = np.max(np.abs(pred_w - ref))
        change_b = np.max(np.abs(pred_b - ref))
        expected.append(max(change_w, change_b))
    expected = np.array(expected)

    got = fn(X, w, b, perturbations)

    # Not implemented
    if got is None:
        print(f"  {_NONE} Stability: not implemented yet "
              f"(expected {expected.round(4)})")
        return

    got = np.asarray(got, dtype=float)

    # Correct
    if _close_arr(got, expected):
        print(f"  {_OK} Stability: max output changes = {got.round(4)}")
        return

    # --- Diagnose common mistakes ---

    # Mistake 1: perturbed both w and b simultaneously
    wrong_both = []
    for p in perturbations:
        pred_both = (w + p) * X + (b + p)
        wrong_both.append(np.max(np.abs(pred_both - ref)))
    if _close_arr(got, np.array(wrong_both)):
        print(f"  {_FAIL} Stability: {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: you are perturbing w and b at the same time. Perturb "
              f"each one separately and take the worst case.")
        return

    # Mistake 2: wrong reference (used perturbed as reference)
    wrong_ref = []
    for p in perturbations:
        pred_w = (w + p) * X + b
        pred_b = w * X + (b + p)
        change_w = np.max(np.abs(pred_w - pred_b))
        wrong_ref.append(change_w)
    if _close_arr(got, np.array(wrong_ref)):
        print(f"  {_FAIL} Stability: {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: compare each perturbed prediction to the unperturbed "
              f"reference (w*X + b), not to each other.")
        return

    # Mistake 3: returned sum instead of max
    wrong_sum = []
    for p in perturbations:
        pred_w = (w + p) * X + b
        pred_b = w * X + (b + p)
        change_w = np.sum(np.abs(pred_w - ref))
        change_b = np.sum(np.abs(pred_b - ref))
        wrong_sum.append(max(change_w, change_b))
    if _close_arr(got, np.array(wrong_sum)):
        print(f"  {_FAIL} Stability: {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: use np.max() for the largest absolute change, "
              f"not np.sum().")
        return

    # Generic fallback
    print(f"  {_FAIL} Stability: {got.round(4)}  (expected {expected.round(4)})")
    print(f"       Hint: for each perturbation p, compute predictions with "
          f"(w+p, b) and (w, b+p) separately. Compare each to the unperturbed "
          f"reference and return the max absolute change.")


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def check_residuals(fn, X, Y, w, b):
    expected = Y - (w * X + b)

    got = fn(X, Y, w, b)

    # Not implemented
    if got is None:
        print(f"  {_NONE} Residuals: not implemented yet "
              f"(expected {expected.round(4)})")
        return

    got = np.asarray(got, dtype=float)

    # Correct
    if _close_arr(got, expected):
        print(f"  {_OK} Residuals = {got.round(4)}")
        return

    # --- Diagnose common mistakes ---

    # Mistake 1: wrong sign (prediction - Y instead of Y - prediction)
    wrong_sign = (w * X + b) - Y
    if _close_arr(got, wrong_sign):
        print(f"  {_FAIL} Residuals = {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: the sign is flipped. Residuals are Y - predictions, "
              f"not predictions - Y.")
        return

    # Mistake 2: forgot to compute predictions, returned Y - X
    wrong_no_pred = Y - X
    if _close_arr(got, wrong_no_pred):
        print(f"  {_FAIL} Residuals = {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: you are computing Y - X. You need to compute "
              f"predictions first: preds = w*X + b, then Y - preds.")
        return

    # Mistake 3: returned predictions instead of residuals
    preds = w * X + b
    if _close_arr(got, preds):
        print(f"  {_FAIL} Residuals = {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: those are the predictions, not the residuals. "
              f"Residuals = Y - predictions.")
        return

    # Mistake 4: returned absolute residuals
    abs_resid = np.abs(expected)
    if _close_arr(got, abs_resid):
        print(f"  {_FAIL} Residuals = {got.round(4)}  (expected {expected.round(4)})")
        print(f"       Hint: return the signed residuals (Y - preds), "
              f"not the absolute values.")
        return

    # Generic fallback
    print(f"  {_FAIL} Residuals = {got.round(4)}  (expected {expected.round(4)})")
    print(f"       Hint: compute predictions = w*X + b, then return Y - predictions.")
