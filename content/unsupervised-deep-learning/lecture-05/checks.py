"""Auto-checker helpers for Chapter 5 - Dimensionality Reduction."""

import numpy as np

_OK   = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL  = 0.01


# ---------------------------------------------------------------------------
# Variance-based feature mask
# ---------------------------------------------------------------------------

def check_variance_mask(fn, X, threshold):
    """Check that fn returns a boolean mask of features with variance > threshold."""
    got = fn(X, threshold)

    variances = np.var(X, axis=0)
    expected = variances > threshold
    n_kept = int(np.sum(expected))

    if got is None:
        print(f"  {_NONE} variance_mask: not implemented yet (expected {n_kept} of {X.shape[1]} features kept)")
        return

    got_arr = np.asarray(got).ravel()

    # Mistake: returned indices (integers in [0, d)) instead of a mask
    if got_arr.dtype != bool and got_arr.dtype.kind in "iu":
        uniq = np.unique(got_arr)
        if uniq.min() >= 0 and uniq.max() < X.shape[1] and len(got_arr) < X.shape[1]:
            print(f"  {_FAIL} variance_mask: returned indices, not a boolean mask")
            print(f"       Hint: return an array of True / False, one entry per feature.")
            return

    try:
        got_bool = got_arr.astype(bool)
    except Exception:
        print(f"  {_FAIL} variance_mask: cannot cast result to boolean")
        return

    if got_bool.shape == expected.shape and np.array_equal(got_bool, expected):
        print(f"  {_OK} variance_mask: {int(np.sum(got_bool))} of {X.shape[1]} features kept")
        return

    # Mistake: used std instead of var (comparable when threshold is small)
    expected_std = np.std(X, axis=0) > threshold
    if got_bool.shape == expected_std.shape and np.array_equal(got_bool, expected_std):
        print(f"  {_FAIL} variance_mask: {int(np.sum(got_bool))} kept  (expected {n_kept})")
        print(f"       Hint: use np.var(X, axis=0), not np.std. The threshold is compared to variance.")
        return

    # Mistake: used >= instead of strict >
    expected_ge = variances >= threshold
    if got_bool.shape == expected_ge.shape and np.array_equal(got_bool, expected_ge):
        print(f"  {_FAIL} variance_mask: {int(np.sum(got_bool))} kept  (expected {n_kept})")
        print(f"       Hint: the exercise asks for strictly greater than threshold (use >, not >=).")
        return

    # Mistake: wrong axis (variance per row instead of per column)
    if X.shape[0] != X.shape[1]:
        expected_axis1 = np.var(X, axis=1) > threshold
        if got_bool.shape == expected_axis1.shape and np.array_equal(got_bool, expected_axis1):
            print(f"  {_FAIL} variance_mask: computed variance per sample, not per feature")
            print(f"       Hint: use axis=0 so variance is computed down each column.")
            return

    # Mistake: inverted mask
    if got_bool.shape == expected.shape and np.array_equal(got_bool, ~expected):
        print(f"  {_FAIL} variance_mask: mask is inverted")
        print(f"       Hint: True means KEEP the feature (variance > threshold).")
        return

    print(f"  {_FAIL} variance_mask: {int(np.sum(got_bool))} kept  (expected {n_kept})")
    print(f"       Hint: return np.var(X, axis=0) > threshold  (shape: ({X.shape[1]},)).")


# ---------------------------------------------------------------------------
# Per-feature centering
# ---------------------------------------------------------------------------

def check_center_data(fn, X):
    """Check that fn subtracts the per-feature mean."""
    got = fn(X)
    expected = X - X.mean(axis=0)

    if got is None:
        print(f"  {_NONE} center_data: not implemented yet (expected array of shape {X.shape})")
        return

    got = np.asarray(got, dtype=float)

    if got.shape == expected.shape and np.allclose(got, expected, atol=_TOL):
        max_abs_mean = float(np.abs(got.mean(axis=0)).max())
        print(f"  {_OK} center_data: shape {got.shape}, max |column mean| = {max_abs_mean:.2e}")
        return

    # Mistake: subtracted the scalar mean of all entries
    expected_scalar = X - X.mean()
    if got.shape == expected_scalar.shape and np.allclose(got, expected_scalar, atol=_TOL):
        print(f"  {_FAIL} center_data: subtracted the scalar mean of all values")
        print(f"       Hint: use X.mean(axis=0) to get one mean per feature, not X.mean().")
        return

    # Mistake: also divided by std (z-scored)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    expected_z = (X - X.mean(axis=0)) / std_safe
    if got.shape == expected_z.shape and np.allclose(got, expected_z, atol=_TOL):
        print(f"  {_FAIL} center_data: also divided by std (this is z-scoring, not centering)")
        print(f"       Hint: centering only subtracts the mean. Leave the scale alone here.")
        return

    # Mistake: returned X unchanged
    if got.shape == X.shape and np.allclose(got, X, atol=_TOL):
        print(f"  {_FAIL} center_data: returned X unchanged")
        print(f"       Hint: subtract the per-column mean before returning.")
        return

    # Mistake: subtracted rowwise mean
    expected_row = X - X.mean(axis=1, keepdims=True)
    if got.shape == expected_row.shape and np.allclose(got, expected_row, atol=_TOL):
        print(f"  {_FAIL} center_data: subtracted the per-sample mean, not the per-feature mean")
        print(f"       Hint: use axis=0 in X.mean, not axis=1.")
        return

    print(f"  {_FAIL} center_data: values do not match")
    print(f"       Hint: return X - X.mean(axis=0).")


# ---------------------------------------------------------------------------
# Reference PCA (used by several checkers)
# ---------------------------------------------------------------------------

def _reference_pca(X_c, k):
    """Return top-k eigenvalues and eigenvectors of covariance matrix.

    Vectors are returned as rows: shape (k, d).
    """
    n = X_c.shape[0]
    cov = X_c.T @ X_c / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order][:k], eigvecs[:, order][:, :k].T


def _vecs_match(A, B, tol=_TOL):
    """Check that two (k, d) arrays match row-by-row up to a sign flip."""
    if A.shape != B.shape:
        return False
    for i in range(A.shape[0]):
        a, b = A[i], B[i]
        if not (np.allclose(a, b, atol=tol) or np.allclose(a, -b, atol=tol)):
            return False
    return True


# ---------------------------------------------------------------------------
# PCA components
# ---------------------------------------------------------------------------

def check_pca_components(fn, X_c, k):
    """Check that fn returns the top-k eigenvectors of the covariance matrix."""
    got = fn(X_c, k)

    eigvals, exp_vecs = _reference_pca(X_c, k)
    d = X_c.shape[1]

    if got is None:
        print(f"  {_NONE} pca_components: not implemented yet (expected shape ({k}, {d}))")
        return

    got = np.asarray(got, dtype=float)

    if got.ndim == 1:
        got = got.reshape(1, -1)

    # Correct shape (k, d)
    if got.shape == (k, d) and _vecs_match(got, exp_vecs):
        frac = eigvals.sum() / np.trace(X_c.T @ X_c / (X_c.shape[0] - 1))
        print(f"  {_OK} pca_components: top {k} components capture {100 * frac:.1f}% of the variance")
        return

    # Mistake: shape transposed (d, k)
    if got.shape == (d, k) and _vecs_match(got.T, exp_vecs):
        print(f"  {_FAIL} pca_components: shape {got.shape} is transposed (expected ({k}, {d}))")
        print(f"       Hint: return components as rows — one row per component, each of length d.")
        return

    # Mistake: returned smallest eigenvectors instead of largest
    _, all_vecs = _reference_pca(X_c, d)
    smallest_k = all_vecs[-k:][::-1]
    if got.shape == (k, d) and _vecs_match(got, smallest_k):
        print(f"  {_FAIL} pca_components: returned the k SMALLEST eigenvectors")
        print(f"       Hint: sort eigenvalues in DESCENDING order before slicing the first k.")
        return

    # Mistake: returned the eigenvalues instead of vectors
    if got.ndim == 2 and got.shape[0] == 1 and got.shape[1] == k:
        print(f"  {_FAIL} pca_components: returned only {k} values (looks like eigenvalues)")
        print(f"       Hint: return eigenvectors (each of length d), not eigenvalues.")
        return
    if got.ndim == 1 and len(got) == k:
        print(f"  {_FAIL} pca_components: returned {len(got)} values (looks like eigenvalues)")
        print(f"       Hint: return the eigenvectors, not the eigenvalues.")
        return

    # Mistake: components not unit length
    norms = np.linalg.norm(got, axis=1)
    if got.shape == (k, d) and not np.allclose(norms, 1.0, atol=0.05):
        print(f"  {_FAIL} pca_components: rows have non-unit norms (min={norms.min():.3f}, max={norms.max():.3f})")
        print(f"       Hint: use np.linalg.eigh or np.linalg.svd — their eigenvectors are already unit length.")
        return

    print(f"  {_FAIL} pca_components: shape {got.shape}  (expected ({k}, {d}))")
    print(f"       Hint: compute covariance, eigendecompose with np.linalg.eigh,")
    print(f"       sort eigenvalues descending, return top-k eigenvectors as rows.")

