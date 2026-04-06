"""Auto-checker helpers for Chapter 1 notebook."""

import numpy as np
import hashlib

_OK = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL = 0.01


def _h(val):
    """Hash a scalar rounded to 4 decimal places."""
    return hashlib.sha256(f"{val:.4f}".encode()).hexdigest()[:16]


def _h_arr(arr):
    """Hash a full array rounded to 4 decimal places."""
    s = ",".join(f"{v:.4f}" for v in arr.flat)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _close(got_hash, expected_hash):
    return got_hash == expected_hash


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------

_EUCL_OK      = "78128f7a7ec02f52"
_EUCL_NO_SQRT = "a311d2cd6ba1f730"
_EUCL_IS_MANH = "502147371e08ea79"

def check_euclidean(fn, x, y):
    got = fn(x, y)

    if got is None:
        print(f"  {_NONE} Euclidean: not implemented yet")
        return

    gh = _h(got)

    if _close(gh, _EUCL_OK):
        print(f"  {_OK} Euclidean d = {got:.4f}")
    elif _close(gh, _EUCL_NO_SQRT):
        print(f"  {_FAIL} Euclidean d = {got:.4f}")
        print(f"       Hint: that is the squared distance. Don't forget np.sqrt().")
    elif _close(gh, _EUCL_IS_MANH):
        print(f"  {_FAIL} Euclidean d = {got:.4f}")
        print(f"       Hint: that is the Manhattan distance. You need to square the")
        print(f"       differences, not take the absolute value.")
    elif got < 0:
        print(f"  {_FAIL} Euclidean d = {got:.4f}")
        print(f"       Hint: the sign is flipped. Distance should be non-negative.")
    else:
        print(f"  {_FAIL} Euclidean d = {got:.4f}")
        print(f"       Hint: revisit the formula. The steps are: subtract, square, sum, root.")


# ---------------------------------------------------------------------------
# Manhattan
# ---------------------------------------------------------------------------

_MANH_OK      = "502147371e08ea79"
_MANH_IS_EUCL = "78128f7a7ec02f52"
_MANH_NO_ABS  = "3bb373b4104d7065"
_MANH_SQ_SUM  = "a311d2cd6ba1f730"

def check_manhattan(fn, x, y):
    got = fn(x, y)

    if got is None:
        print(f"  {_NONE} Manhattan: not implemented yet")
        return

    gh = _h(got)

    if _close(gh, _MANH_OK):
        print(f"  {_OK} Manhattan d = {got:.4f}")
    elif _close(gh, _MANH_IS_EUCL):
        print(f"  {_FAIL} Manhattan d = {got:.4f}")
        print(f"       Hint: that is the Euclidean distance. For Manhattan, use absolute")
        print(f"       values instead of squaring, and no square root at the end.")
    elif _close(gh, _MANH_NO_ABS):
        print(f"  {_FAIL} Manhattan d = {got:.4f}")
        print(f"       Hint: differences can be negative. Wrap them in np.abs().")
    elif _close(gh, _MANH_SQ_SUM):
        print(f"  {_FAIL} Manhattan d = {got:.4f}")
        print(f"       Hint: you are squaring the differences. Manhattan uses absolute")
        print(f"       values, not squares.")
    else:
        print(f"  {_FAIL} Manhattan d = {got:.4f}")
        print(f"       Hint: revisit the formula. The steps are: subtract, abs, sum.")


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------

_COS_SIM_OK  = "7b6e0c57dde99985"
_COS_DIST_OK = "d63738433fb18db3"
_COS_RAW_DOT = "dbb0d3b71bf6b3df"

def check_cosine_sim(fn, x, y):
    got = fn(x, y)

    if got is None:
        print(f"  {_NONE} Cosine similarity: not implemented yet")
        return

    gh = _h(got)

    if _close(gh, _COS_SIM_OK):
        print(f"  {_OK} Cosine similarity = {got:.4f}")
    elif _close(gh, _COS_RAW_DOT):
        print(f"  {_FAIL} Cosine similarity = {got:.4f}")
        print(f"       Hint: that is the raw dot product. You need to divide by the")
        print(f"       product of the two vector norms to normalize it.")
    elif _close(gh, _COS_DIST_OK):
        print(f"  {_FAIL} Cosine similarity = {got:.4f}")
        print(f"       Hint: that is the cosine distance (1 - similarity).")
        print(f"       This function should return the similarity, not the distance.")
    elif got is not None and (float(got) > 1.0 or float(got) < -1.0):
        print(f"  {_FAIL} Cosine similarity = {got:.4f}")
        print(f"       Hint: cosine similarity should be between -1 and +1.")
        print(f"       Check that you are dividing by both norms.")
    else:
        print(f"  {_FAIL} Cosine similarity = {got:.4f}")
        print(f"       Hint: numerator = dot product, denominator = norm(x) * norm(y).")


def check_cosine_dist(fn, x, y):
    got = fn(x, y)

    if got is None:
        print(f"  {_NONE} Cosine distance: not implemented yet")
        return

    gh = _h(got)

    if _close(gh, _COS_DIST_OK):
        print(f"  {_OK} Cosine distance = {got:.4f}")
    elif _close(gh, _COS_SIM_OK):
        print(f"  {_FAIL} Cosine distance = {got:.4f}")
        print(f"       Hint: that is the similarity, not the distance.")
        print(f"       Cosine distance = 1 - similarity.")
    else:
        print(f"  {_FAIL} Cosine distance = {got:.4f}")
        print(f"       Hint: cosine distance = 1 - cosine_similarity(x, y).")


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------

_MAHAL_OK      = "3241fdea44ec7db6"
_MAHAL_SQ      = "16fb98d60d5c6ad6"
_MAHAL_IS_EUCL = "78128f7a7ec02f52"
_MAHAL_NO_INV  = "341595f8b8ab64df"

def check_mahalanobis(fn, x, y, cov):
    got = fn(x, y, cov)

    if got is None:
        print(f"  {_NONE} Mahalanobis: not implemented yet")
        return

    gh = _h(got)

    if _close(gh, _MAHAL_OK):
        print(f"  {_OK} Mahalanobis d = {got:.4f}")
    elif _close(gh, _MAHAL_SQ):
        print(f"  {_FAIL} Mahalanobis d = {got:.4f}")
        print(f"       Hint: that is the squared Mahalanobis distance.")
        print(f"       Don't forget to take the square root at the end.")
    elif _close(gh, _MAHAL_IS_EUCL):
        print(f"  {_FAIL} Mahalanobis d = {got:.4f}")
        print(f"       Hint: that is the Euclidean distance. You need to transform the")
        print(f"       difference vector through the inverse covariance matrix.")
    elif _close(gh, _MAHAL_NO_INV):
        print(f"  {_FAIL} Mahalanobis d = {got:.4f}")
        print(f"       Hint: you are using the covariance matrix directly. You need its")
        print(f"       inverse: np.linalg.inv(cov).")
    else:
        print(f"  {_FAIL} Mahalanobis d = {got:.4f}")
        print(f"       Hint: the steps are: delta = x - y, invert cov, then")
        print(f"       sqrt(delta @ cov_inv @ delta).")


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------

_ZSCORE_OK   = "61e8267d11c4bf43"
_Z_CENTERED  = "121cf491a037621b"
_Z_SCALED    = "be2f6d352bd60a72"
_Z_BY_VAR    = "03d618a3f743c9cb"

def check_zscore(result, data):
    if result is None:
        print(f"  {_NONE} Z-score: not implemented yet")
        return

    rh = _h_arr(result)

    if _close(rh, _ZSCORE_OK):
        print(f"  {_OK} Z-score normalization correct (mean\u22480, std\u22481)")
        return

    if _close(rh, _Z_CENTERED):
        print(f"  {_FAIL} You subtracted the mean but forgot to divide by the")
        print(f"       standard deviation.")
        return

    if _close(rh, _Z_SCALED):
        print(f"  {_FAIL} You divided by the standard deviation but forgot to")
        print(f"       subtract the mean first.")
        return

    if _close(rh, _Z_BY_VAR):
        print(f"  {_FAIL} You divided by the variance instead of the standard deviation.")
        print(f"       Use np.std(), not np.var().")
        return

    means = result.mean(axis=0)
    stds = result.std(axis=0)

    if not np.allclose(means, 0, atol=_TOL):
        print(f"  {_FAIL} Column means are {means.round(4)}, expected ~0.")
        print(f"       Hint: subtract np.mean(data, axis=0) before dividing.")
    elif not np.allclose(stds, 1, atol=_TOL):
        print(f"  {_FAIL} Column stds are {stds.round(4)}, expected ~1.")
        print(f"       Hint: divide by np.std(data, axis=0).")
    else:
        print(f"  {_FAIL} Values don't match expected output.")
        print(f"       Hint: z = (data - mean) / std, applied per column (axis=0).")


# ---------------------------------------------------------------------------
# Min-max normalization
# ---------------------------------------------------------------------------

_MINMAX_OK     = "df466ec24dcf3caf"
_MM_NO_SHIFT   = "e9ae04ec2af4e507"
_MM_BY_MAX     = "9905202b41cc1a80"
_MM_BY_MAXONLY = "dc3c5d0485c11241"

def check_minmax(result, data):
    if result is None:
        print(f"  {_NONE} Min-max: not implemented yet")
        return

    rh = _h_arr(result)

    if _close(rh, _MINMAX_OK):
        print(f"  {_OK} Min-max scaling correct (min\u22480, max\u22481)")
        return

    if _close(rh, _MM_NO_SHIFT):
        print(f"  {_FAIL} You divided by the range but forgot to subtract the minimum.")
        return

    if _close(rh, _MM_BY_MAX):
        print(f"  {_FAIL} You divided by the max instead of the range (max - min).")
        return

    if _close(rh, _MM_BY_MAXONLY):
        print(f"  {_FAIL} You divided by max without subtracting min.")
        print(f"       The formula is (x - min) / (max - min).")
        return

    mins = result.min(axis=0)
    maxs = result.max(axis=0)

    if not np.allclose(mins, 0, atol=_TOL):
        print(f"  {_FAIL} Column mins are {mins.round(4)}, expected ~0.")
        print(f"       Hint: subtract np.min(data, axis=0) in the numerator.")
    elif not np.allclose(maxs, 1, atol=_TOL):
        print(f"  {_FAIL} Column maxs are {maxs.round(4)}, expected ~1.")
        print(f"       Hint: divide by (max - min), not just max.")
    else:
        print(f"  {_FAIL} Values don't match expected output.")
        print(f"       Hint: (data - min) / (max - min), per column (axis=0).")


# ---------------------------------------------------------------------------
# Distance comparison: raw vs. normalized
# ---------------------------------------------------------------------------

def check_distance_comparison(fn_euclid, fn_manhattan, fn_cosine, fn_mahal,
                              x_raw, y_raw, cov_raw, features_z, idx_x=0, idx_y=11):
    """Validate and print a side-by-side table of raw vs z-scored distances."""

    if features_z is None:
        print(f"  {_NONE} Implement zscore_normalize() first to compare distances.")
        return

    x_z = features_z[idx_x]
    y_z = features_z[idx_y]
    cov_z = np.cov(features_z, rowvar=False)

    metrics = [
        ("Euclidean",   lambda xv, yv, _: fn_euclid(xv, yv)),
        ("Manhattan",   lambda xv, yv, _: fn_manhattan(xv, yv)),
        ("Cosine",      lambda xv, yv, _: fn_cosine(xv, yv)),
        ("Mahalanobis", lambda xv, yv, c: fn_mahal(xv, yv, c)),
    ]

    dash = "\u2500"
    print(f"  {'Metric':<14} {'Raw':>10} {'Z-Score':>10}")
    print(f"  {dash*14} {dash*10} {dash*10}")

    all_ok = True
    for name, fn in metrics:
        got_raw = fn(x_raw, y_raw, cov_raw)
        got_z = fn(x_z, y_z, cov_z)

        if got_raw is None or got_z is None:
            emdash = "\u2014"
            print(f"  {_NONE} {name:<14} {emdash:>10} {emdash:>10}  (not implemented)")
            all_ok = False
            continue

        # For the comparison table we validate each value via its
        # individual checker (already called earlier in the notebook).
        # Here we just display the values.
        print(f"  {_OK} {name:<12} {got_raw:10.4f} {got_z:10.4f}")

    if all_ok:
        print()
        print(f"  {_OK} All distances computed.")


# ---------------------------------------------------------------------------
# Pairwise distance matrix (safe builder)
# ---------------------------------------------------------------------------

def pairwise_distances(fn, features):
    """Build an n x n distance matrix using fn. Returns None if fn is not implemented."""
    test = fn(features[0], features[1])
    if test is None:
        print(f"  {_NONE} Implement the distance function first.")
        return None
    n = len(features)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = fn(features[i], features[j])
    return D
