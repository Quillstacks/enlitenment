"""Auto-checker helpers for Chapter 4 — Density-Based Clustering."""

import numpy as np
from scipy.spatial.distance import cdist

_OK   = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL  = 0.01


def _close(a, b, tol=_TOL):
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Epsilon-neighborhoods
# ---------------------------------------------------------------------------

def check_epsilon_neighbors(fn, point_idx, X, eps):
    """Check that fn returns sorted indices within eps of X[point_idx]."""
    got = fn(point_idx, X, eps)

    dists = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
    expected = np.sort(np.where(dists <= eps)[0])
    n_exp = len(expected)

    if got is None:
        print(f"  {_NONE} epsilon_neighbors: not implemented yet (expected {n_exp} neighbors)")
        return

    got = np.asarray(got).ravel()

    if np.array_equal(np.sort(got), expected):
        print(f"  {_OK} epsilon_neighbors: {len(got)} neighbors (including self)")
        return

    # Mistake: excluded self
    expected_no_self = expected[expected != point_idx]
    if np.array_equal(np.sort(got), expected_no_self):
        print(f"  {_FAIL} epsilon_neighbors: {len(got)} neighbors  (expected {n_exp})")
        print(f"       Hint: the epsilon-neighborhood includes the point itself (distance 0 <= eps).")
        return

    # Mistake: used strict < instead of <=
    expected_strict = np.sort(np.where(dists < eps)[0])
    if np.array_equal(np.sort(got), expected_strict):
        print(f"  {_FAIL} epsilon_neighbors: {len(got)} neighbors  (expected {n_exp})")
        print(f"       Hint: use <= (not <) when comparing distances to eps.")
        return

    # Mistake: used squared distance instead of Euclidean
    sq_dists = np.sum((X - X[point_idx]) ** 2, axis=1)
    expected_sq = np.sort(np.where(sq_dists <= eps)[0])
    if np.array_equal(np.sort(got), expected_sq):
        print(f"  {_FAIL} epsilon_neighbors: {len(got)} neighbors  (expected {n_exp})")
        print(f"       Hint: compare Euclidean distances (not squared distances) to eps.")
        print(f"       Apply np.sqrt before comparing, or compare squared distances to eps**2.")
        return

    # Mistake: returned distances instead of indices
    if len(got) > 0 and not np.issubdtype(got.dtype, np.integer):
        print(f"  {_FAIL} epsilon_neighbors returned float values")
        print(f"       Hint: return the indices (integer positions) of neighbors, not their distances.")
        return

    print(f"  {_FAIL} epsilon_neighbors: {len(got)} neighbors  (expected {n_exp})")
    print(f"       Hint: compute distances from X[point_idx] to all rows of X,")
    print(f"       then return np.sort(np.where(dists <= eps)[0]).")


# ---------------------------------------------------------------------------
# Core mask
# ---------------------------------------------------------------------------

def check_core_mask(fn, X, eps, min_samples):
    """Check that fn returns boolean core-point mask."""
    got = fn(X, eps, min_samples)

    D = cdist(X, X, "euclidean")
    counts = np.sum(D <= eps, axis=1)
    expected = counts >= min_samples
    n_core = int(np.sum(expected))

    if got is None:
        print(f"  {_NONE} core_mask: not implemented yet (expected {n_core} core points out of {len(X)})")
        return

    got = np.asarray(got).ravel()
    got_bool = got.astype(bool) if got.dtype != bool else got

    if np.array_equal(got_bool, expected):
        print(f"  {_OK} core_mask: {int(np.sum(got_bool))} core points out of {len(X)}")
        return

    # Mistake: off by one — used > instead of >=
    expected_strict = counts > min_samples
    if np.array_equal(got_bool, expected_strict):
        print(f"  {_FAIL} core_mask: {int(np.sum(got_bool))} core points  (expected {n_core})")
        print(f"       Hint: a point is core if it has >= min_samples neighbors, not strictly >.")
        return

    # Mistake: didn't count self as neighbor
    counts_no_self = counts - 1
    expected_no_self = counts_no_self >= min_samples
    if np.array_equal(got_bool, expected_no_self):
        print(f"  {_FAIL} core_mask: {int(np.sum(got_bool))} core points  (expected {n_core})")
        print(f"       Hint: each point is its own neighbor (distance 0 <= eps).")
        print(f"       The neighbor count includes the point itself.")
        return

    # Mistake: used strict < for distance comparison
    counts_strict = np.sum(D < eps, axis=1)
    expected_strict_d = counts_strict >= min_samples
    if np.array_equal(got_bool, expected_strict_d):
        print(f"  {_FAIL} core_mask: {int(np.sum(got_bool))} core points  (expected {n_core})")
        print(f"       Hint: use <= (not <) when counting neighbors within eps.")
        return

    # Mistake: returned counts instead of boolean
    if got.dtype in (np.int32, np.int64, np.float64) and np.allclose(got, counts):
        print(f"  {_FAIL} core_mask returned neighbor counts, not a boolean mask")
        print(f"       Hint: return a boolean array (True / False), not the raw counts.")
        return

    print(f"  {_FAIL} core_mask: {int(np.sum(got_bool))} core points  (expected {n_core})")
    print(f"       Hint: for each point, count how many points (including itself) lie within eps.")
    print(f"       A point is core if that count >= min_samples.")


# ---------------------------------------------------------------------------
# k-distance plot
# ---------------------------------------------------------------------------

def check_k_distances(fn, X, k):
    """Check that fn returns sorted k-th NN distances (descending)."""
    got = fn(X, k)

    D = cdist(X, X, "euclidean")
    np.fill_diagonal(D, np.inf)
    sorted_dists = np.sort(D, axis=1)
    kth = sorted_dists[:, k - 1]
    expected = np.sort(kth)[::-1]

    if got is None:
        print(f"  {_NONE} k_distances: not implemented yet (expected array of length {len(X)})")
        return

    got = np.asarray(got, dtype=float).ravel()

    if len(got) == len(expected) and np.allclose(got, expected, atol=_TOL):
        print(f"  {_OK} k_distances: range [{got[-1]:.4f}, {got[0]:.4f}]")
        return

    # Mistake: sorted ascending instead of descending
    expected_asc = np.sort(kth)
    if len(got) == len(expected_asc) and np.allclose(got, expected_asc, atol=_TOL):
        print(f"  {_FAIL} k_distances: sorted ascending  (expected descending)")
        print(f"       Hint: sort descending so the k-distance plot reads left-to-right")
        print(f"       from largest to smallest.")
        return

    # Mistake: used k index directly (0-indexed — got k-th instead of (k-1)-th)
    if k < sorted_dists.shape[1]:
        kth_off = sorted_dists[:, k]
        expected_off = np.sort(kth_off)[::-1]
        if len(got) == len(expected_off) and np.allclose(got, expected_off, atol=_TOL):
            print(f"  {_FAIL} k_distances: off-by-one in neighbor index")
            print(f"       Hint: the k-th nearest neighbor is at index k-1 after sorting")
            print(f"       (Python is 0-indexed, but k is 1-indexed: 1st NN, 2nd NN, ...).")
            return

    # Mistake: included self-distance (didn't exclude diagonal)
    D_with_self = cdist(X, X, "euclidean")
    sorted_with_self = np.sort(D_with_self, axis=1)
    kth_self = sorted_with_self[:, k]
    expected_self = np.sort(kth_self)[::-1]
    if len(got) == len(expected_self) and np.allclose(got, expected_self, atol=_TOL):
        print(f"  {_FAIL} k_distances: included self-distance (0.0)")
        print(f"       Hint: exclude self-distances before finding the k-th nearest neighbor.")
        print(f"       Set the diagonal to np.inf or skip index 0 after sorting each row.")
        return

    # Mistake: not sorted at all
    if len(got) == len(expected):
        got_sorted_desc = np.sort(got)[::-1]
        if np.allclose(got_sorted_desc, expected, atol=_TOL):
            print(f"  {_FAIL} k_distances: values correct but not sorted")
            print(f"       Hint: sort the k-th nearest-neighbor distances in descending order.")
            return

    print(f"  {_FAIL} k_distances: shape or values don't match (expected array of length {len(X)})")
    print(f"       Hint: compute pairwise distances, exclude self, sort each row,")
    print(f"       take column k-1, then sort the result descending.")


# ---------------------------------------------------------------------------
# Reference DBSCAN (used internally by optional checkers)
# ---------------------------------------------------------------------------

def _reference_dbscan(X, eps, min_samples):
    """Internal reference DBSCAN implementation."""
    n = len(X)
    D = cdist(X, X, "euclidean")
    counts = np.sum(D <= eps, axis=1)
    is_core = counts >= min_samples
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        queue = [i]
        visited[i] = True
        labels[i] = cid
        head = 0
        while head < len(queue):
            q = queue[head]
            head += 1
            nbrs = np.where(D[q] <= eps)[0]
            for nb in nbrs:
                if labels[nb] == -1:
                    labels[nb] = cid
                if not visited[nb] and is_core[nb]:
                    visited[nb] = True
                    queue.append(nb)
        cid += 1
    return labels


def _labels_equivalent(a, b):
    """Check if two label arrays are equivalent up to relabeling (-1 = noise stays fixed)."""
    if len(a) != len(b):
        return False
    if not np.array_equal(a == -1, b == -1):
        return False
    non_noise = a != -1
    if not np.any(non_noise):
        return True
    a_nn, b_nn = a[non_noise], b[non_noise]
    fwd, rev = {}, {}
    for ai, bi in zip(a_nn, b_nn):
        if ai in fwd:
            if fwd[ai] != bi:
                return False
        else:
            fwd[ai] = bi
        if bi in rev:
            if rev[bi] != ai:
                return False
        else:
            rev[bi] = ai
    return True


# ---------------------------------------------------------------------------
# Optional — DBSCAN from scratch
# ---------------------------------------------------------------------------

def check_dbscan(fn, X, eps, min_samples):
    """Check that fn returns DBSCAN labels matching the reference implementation."""
    got = fn(X, eps, min_samples)

    if got is None:
        print(f"  {_NONE} dbscan: not implemented yet")
        return

    got = np.asarray(got).ravel()
    ref = _reference_dbscan(X, eps, min_samples)

    n_clusters_ref = len(set(ref) - {-1})
    n_noise_ref = int(np.sum(ref == -1))
    n_clusters_got = len(set(got) - {-1})
    n_noise_got = int(np.sum(got == -1))

    if _labels_equivalent(got, ref):
        print(f"  {_OK} dbscan: {n_clusters_got} clusters, {n_noise_got} noise points")
        return

    # Mistake: all points labeled as noise
    if n_clusters_got == 0:
        print(f"  {_FAIL} dbscan: all points labeled as noise  (expected {n_clusters_ref} clusters)")
        print(f"       Hint: core points and their density-reachable neighbors should form clusters.")
        return

    # Mistake: no noise detected
    if n_noise_got == 0 and n_noise_ref > 0:
        print(f"  {_FAIL} dbscan: 0 noise points  (expected {n_noise_ref})")
        print(f"       Hint: points that are neither core nor density-reachable from a core point are noise (-1).")
        return

    # Mistake: wrong number of clusters
    if n_clusters_got != n_clusters_ref:
        print(f"  {_FAIL} dbscan: {n_clusters_got} clusters  (expected {n_clusters_ref})")
        print(f"       Hint: check your cluster expansion. Each core point's eps-neighbors")
        print(f"       that are also core should propagate the same cluster label.")
        return

    # Correct counts but wrong assignments
    print(f"  {_FAIL} dbscan: correct cluster/noise counts but assignments differ")
    print(f"       Hint: border points should be assigned to the cluster of the first")
    print(f"       core point that reaches them during expansion.")


# ---------------------------------------------------------------------------
# Optional — Cluster count (for sensitivity sweep)
# ---------------------------------------------------------------------------

def check_count_clusters(fn, X, eps, min_samples):
    """Check that fn returns the number of DBSCAN clusters (excluding noise)."""
    got = fn(X, eps, min_samples)

    ref = _reference_dbscan(X, eps, min_samples)
    expected = len(set(ref) - {-1})

    if got is None:
        print(f"  {_NONE} count_clusters: not implemented yet (expected {expected})")
        return

    got = int(got)

    if got == expected:
        print(f"  {_OK} count_clusters = {got}  (eps={eps}, min_samples={min_samples})")
        return

    # Mistake: counted noise as a cluster
    expected_with_noise = len(set(ref))
    if got == expected_with_noise and -1 in ref:
        print(f"  {_FAIL} count_clusters = {got}  (expected {expected})")
        print(f"       Hint: noise (label -1) is not a cluster. Exclude it from the count.")
        return

    print(f"  {_FAIL} count_clusters = {got}  (expected {expected})")
    print(f"       Hint: run DBSCAN with the given eps and min_samples,")
    print(f"       then count unique labels excluding -1.")


# ---------------------------------------------------------------------------
# Optional — Mutual reachability distance (HDBSCAN)
# ---------------------------------------------------------------------------

def check_mutual_reachability(fn, D, k):
    """Check that fn returns the mutual reachability distance matrix."""
    got = fn(D, k)

    n = D.shape[0]
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)
    core_dist = np.sort(D_work, axis=1)[:, k - 1]
    expected = np.maximum(D, core_dist[:, None])
    expected = np.maximum(expected, core_dist[None, :])
    np.fill_diagonal(expected, 0)

    if got is None:
        print(f"  {_NONE} mutual_reachability: not implemented yet (expected {n}x{n} matrix)")
        return

    got = np.asarray(got, dtype=float)

    if got.shape == expected.shape and np.allclose(got, expected, atol=_TOL):
        print(f"  {_OK} mutual_reachability: {n}x{n} matrix, max = {got.max():.4f}")
        return

    # Mistake: returned original distances unchanged
    if np.allclose(got, D, atol=_TOL):
        print(f"  {_FAIL} mutual_reachability: returned the original distance matrix unchanged")
        print(f"       Hint: inflate each distance to at least max(core_dist(a), core_dist(b), D(a,b)).")
        return

    # Mistake: only applied one core distance (row broadcast but not column)
    partial = np.maximum(D, core_dist[:, None])
    np.fill_diagonal(partial, 0)
    if np.allclose(got, partial, atol=_TOL):
        print(f"  {_FAIL} mutual_reachability: only applied core_dist of point a, not point b")
        print(f"       Hint: take the maximum of D(a,b), core_dist(a), AND core_dist(b).")
        return

    # Mistake: non-zero diagonal
    if got.shape == expected.shape and np.allclose(got[np.triu_indices(n, 1)],
                                                    expected[np.triu_indices(n, 1)], atol=_TOL):
        if not np.allclose(np.diag(got), 0, atol=_TOL):
            print(f"  {_FAIL} mutual_reachability: diagonal should be zero")
            print(f"       Hint: set np.fill_diagonal(mr, 0) before returning.")
            return

    # Mistake: used minimum instead of maximum
    mr_min = np.minimum(D, core_dist[:, None])
    mr_min = np.minimum(mr_min, core_dist[None, :])
    np.fill_diagonal(mr_min, 0)
    if np.allclose(got, mr_min, atol=_TOL):
        print(f"  {_FAIL} mutual_reachability: used minimum instead of maximum")
        print(f"       Hint: mutual reachability uses np.maximum, not np.minimum.")
        return

    print(f"  {_FAIL} mutual_reachability: values don't match expected")
    print(f"       Hint: core_dist(i) = distance to the k-th nearest neighbor of point i.")
    print(f"       mutual_reachability(a,b) = max(core_dist(a), core_dist(b), D(a,b)).")
    print(f"       Zero the diagonal before returning.")
