"""Auto-checker helpers for Chapter 1 notebook."""

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
# Predict
# ---------------------------------------------------------------------------

def check_predict(fn, x, w, b):
    got = fn(x, w, b)
    expected = w * x + b

    if got is None:
        print(f"  {_NONE} Predict: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} predict({x}, {w}, {b}) = {got:.4f}")
        return

    no_bias = w * x
    if _close(got, no_bias):
        print(f"  {_FAIL} predict = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you forgot to add the bias term b.")
    elif _close(got, b * x + w):
        print(f"  {_FAIL} predict = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: it looks like you swapped w and b. The model is w*x + b.")
    else:
        print(f"  {_FAIL} predict = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: the linear model is y = w*x + b.")


# ---------------------------------------------------------------------------
# Mean Absolute Error
# ---------------------------------------------------------------------------

def check_mae(fn, y_true, y_pred):
    got = fn(y_true, y_pred)
    expected = np.mean(np.abs(y_true - y_pred))

    if got is None:
        print(f"  {_NONE} MAE: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} MAE = {got:.4f}")
        return

    no_abs = np.mean(y_true - y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    sum_not_mean = np.sum(np.abs(y_true - y_pred))

    if _close(got, no_abs):
        print(f"  {_FAIL} MAE = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you forgot np.abs(). Negative errors are cancelling")
        print(f"       positive ones.")
    elif _close(got, mse):
        print(f"  {_FAIL} MAE = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: that is the Mean Squared Error. Use np.abs() instead")
        print(f"       of squaring the differences.")
    elif _close(got, sum_not_mean):
        print(f"  {_FAIL} MAE = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you computed the sum of absolute errors, not the mean.")
        print(f"       Divide by the number of samples (use np.mean instead of np.sum).")
    else:
        print(f"  {_FAIL} MAE = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: MAE = mean of |y_true - y_pred|. Use np.mean(np.abs(...)).")


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def check_grid_search(fn, X, Y, w_range, b_range, h):
    got = fn(X, Y, w_range, b_range, h)

    # Run the reference grid search
    best_w_exp = None
    best_b_exp = None
    best_loss_exp = float('inf')
    w_values = np.arange(w_range[0], w_range[1] + h, h)
    b_values = np.arange(b_range[0], b_range[1] + h, h)
    loss_grid_exp = np.zeros((len(w_values), len(b_values)))

    for i, w in enumerate(w_values):
        for j, b in enumerate(b_values):
            preds = w * X + b
            loss = np.mean(np.abs(Y - preds))
            loss_grid_exp[i, j] = loss
            if loss < best_loss_exp:
                best_loss_exp = loss
                best_w_exp = w
                best_b_exp = b

    if got is None:
        print(f"  {_NONE} Grid search: not implemented yet")
        print(f"       (expected best_w={best_w_exp:.2f}, best_b={best_b_exp:.2f}, "
              f"best_loss={best_loss_exp:.4f})")
        return

    got_w, got_b, got_loss, got_grid = got

    # Check all three values
    w_ok = _close(got_w, best_w_exp)
    b_ok = _close(got_b, best_b_exp)
    loss_ok = _close(got_loss, best_loss_exp)

    if w_ok and b_ok and loss_ok:
        print(f"  {_OK} Grid search: best_w={got_w:.2f}, best_b={got_b:.2f}, "
              f"best_loss={got_loss:.4f}")
        return

    # Diagnose: returned max instead of min
    worst_loss = np.max(loss_grid_exp)
    if _close(got_loss, worst_loss):
        print(f"  {_FAIL} Grid search: best_loss={got_loss:.4f}  "
              f"(expected {best_loss_exp:.4f})")
        print(f"       Hint: you found the maximum loss, not the minimum.")
        print(f"       Initialize best_loss to infinity and keep the combination")
        print(f"       with the smallest loss.")
        return

    # Diagnose: swapped w and b
    if _close(got_w, best_b_exp) and _close(got_b, best_w_exp):
        print(f"  {_FAIL} Grid search: best_w={got_w:.2f}, best_b={got_b:.2f}  "
              f"(expected w={best_w_exp:.2f}, b={best_b_exp:.2f})")
        print(f"       Hint: it looks like w and b are swapped. The outer loop")
        print(f"       should iterate over w, the inner loop over b.")
        return

    # Generic failure
    print(f"  {_FAIL} Grid search: best_w={got_w:.2f}, best_b={got_b:.2f}, "
          f"best_loss={got_loss:.4f}")
    print(f"       (expected w={best_w_exp:.2f}, b={best_b_exp:.2f}, "
          f"loss={best_loss_exp:.4f})")
    print(f"       Hint: loop over w and b values in the given ranges with step h,")
    print(f"       compute MAE for each combination, and track the minimum.")


# ---------------------------------------------------------------------------
# Computational Cost
# ---------------------------------------------------------------------------

def check_operations(fn, n_w, n_b, n_data):
    got = fn(n_w, n_b, n_data)
    expected = n_w * n_b * n_data * 4

    if got is None:
        print(f"  {_NONE} Operations: not implemented yet (expected {expected})")
        return

    if _close(got, expected):
        print(f"  {_OK} Operations = {got}")
        return

    no_data = n_w * n_b * 4
    if _close(got, no_data):
        print(f"  {_FAIL} Operations = {got}  (expected {expected})")
        print(f"       Hint: you forgot the inner data loop. For each (w, b)")
        print(f"       combination, the model evaluates every data point.")
        return

    one_param = n_w * n_data * 4
    if _close(got, one_param) or _close(got, n_b * n_data * 4):
        print(f"  {_FAIL} Operations = {got}  (expected {expected})")
        print(f"       Hint: you only counted one parameter loop. Grid search")
        print(f"       iterates over all combinations of w AND b.")
        return

    no_ops = n_w * n_b * n_data
    if _close(got, no_ops):
        print(f"  {_FAIL} Operations = {got}  (expected {expected})")
        print(f"       Hint: each data point requires 4 operations per (w, b)")
        print(f"       combination: one multiply, one add for the prediction,")
        print(f"       then one subtract and one abs for the error.")
        return

    print(f"  {_FAIL} Operations = {got}  (expected {expected})")
    print(f"       Hint: total = n_w * n_b * n_data * 4. There are two nested")
    print(f"       parameter loops, one data loop, and 4 operations per data point.")
