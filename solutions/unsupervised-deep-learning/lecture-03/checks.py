"""Auto-checker helpers for Chapter 3 — Hierarchical Clustering."""

import numpy as np

_OK   = "\u2705"
_FAIL = "\u274c"
_NONE = "\u2b1c"
_TOL  = 0.01


def _close(a, b, tol=_TOL):
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def _submatrix(D, a, b):
    """Extract D[a, :][:, b] safely for any index arrays."""
    return D[np.ix_(a, b)]


# ---------------------------------------------------------------------------
# Single linkage
# ---------------------------------------------------------------------------

def check_single_linkage(fn, cluster_a, cluster_b, dist_matrix):
    """Check that fn returns the minimum pairwise distance between two clusters."""
    got = fn(cluster_a, cluster_b, dist_matrix)

    expected = float(np.min(_submatrix(dist_matrix, cluster_a, cluster_b)))

    if got is None:
        print(f"  {_NONE} single_linkage: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} single_linkage = {got:.4f}")
        return

    # Mistake: used max (complete linkage)
    complete = float(np.max(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, complete):
        print(f"  {_FAIL} single_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: single linkage uses the minimum pairwise distance, not the maximum.")
        print(f"       Replace np.max with np.min.")
        return

    # Mistake: used mean (average linkage)
    average = float(np.mean(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, average, tol=0.05):
        print(f"  {_FAIL} single_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: single linkage uses the minimum, not the mean pairwise distance.")
        return

    # Mistake: only looked at the diagonal of the submatrix
    diag_len = min(len(cluster_a), len(cluster_b))
    if diag_len > 0:
        diag_min = float(np.min(
            [dist_matrix[cluster_a[i], cluster_b[i]] for i in range(diag_len)]
        ))
        if _close(got, diag_min, tol=0.05):
            print(f"  {_FAIL} single_linkage = {got:.4f}  (expected {expected:.4f})")
            print(f"       Hint: consider all pairs (a, b) with a in cluster A and b in cluster B,")
            print(f"       not just pairs at the same position. Use dist_matrix[np.ix_(cluster_a, cluster_b)].")
            return

    print(f"  {_FAIL} single_linkage = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: extract the submatrix dist_matrix[np.ix_(cluster_a, cluster_b)],")
    print(f"       which holds every (a, b) pairwise distance. Return its minimum.")


# ---------------------------------------------------------------------------
# Complete linkage
# ---------------------------------------------------------------------------

def check_complete_linkage(fn, cluster_a, cluster_b, dist_matrix):
    """Check that fn returns the maximum pairwise distance between two clusters."""
    got = fn(cluster_a, cluster_b, dist_matrix)

    expected = float(np.max(_submatrix(dist_matrix, cluster_a, cluster_b)))

    if got is None:
        print(f"  {_NONE} complete_linkage: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected):
        print(f"  {_OK} complete_linkage = {got:.4f}")
        return

    # Mistake: used min (single linkage)
    single = float(np.min(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, single):
        print(f"  {_FAIL} complete_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: complete linkage uses the maximum pairwise distance, not the minimum.")
        print(f"       Replace np.min with np.max.")
        return

    # Mistake: used mean
    average = float(np.mean(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, average, tol=0.05):
        print(f"  {_FAIL} complete_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: complete linkage uses the maximum, not the mean pairwise distance.")
        return

    # Mistake: diagonal only
    diag_len = min(len(cluster_a), len(cluster_b))
    if diag_len > 0:
        diag_max = float(np.max(
            [dist_matrix[cluster_a[i], cluster_b[i]] for i in range(diag_len)]
        ))
        if _close(got, diag_max, tol=0.05):
            print(f"  {_FAIL} complete_linkage = {got:.4f}  (expected {expected:.4f})")
            print(f"       Hint: check every pair (a, b), not just pairs at the same index.")
            print(f"       Use dist_matrix[np.ix_(cluster_a, cluster_b)] to get the full submatrix.")
            return

    print(f"  {_FAIL} complete_linkage = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: extract dist_matrix[np.ix_(cluster_a, cluster_b)] and return its maximum.")


# ---------------------------------------------------------------------------
# Average linkage
# ---------------------------------------------------------------------------

def check_average_linkage(fn, cluster_a, cluster_b, dist_matrix):
    """Check that fn returns the mean pairwise distance between two clusters."""
    got = fn(cluster_a, cluster_b, dist_matrix)

    expected = float(np.mean(_submatrix(dist_matrix, cluster_a, cluster_b)))

    if got is None:
        print(f"  {_NONE} average_linkage: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected, tol=0.02):
        print(f"  {_OK} average_linkage = {got:.4f}")
        return

    # Mistake: returned min (single linkage)
    single = float(np.min(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, single):
        print(f"  {_FAIL} average_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: average linkage uses the mean, not the minimum pairwise distance.")
        return

    # Mistake: returned max (complete linkage)
    complete = float(np.max(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, complete):
        print(f"  {_FAIL} average_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: average linkage uses the mean, not the maximum pairwise distance.")
        return

    # Mistake: averaged within one cluster (mean of row or column, not the cross-submatrix)
    mean_row = float(np.mean(dist_matrix[cluster_a[0], cluster_b]))
    if _close(got, mean_row, tol=0.05):
        print(f"  {_FAIL} average_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: average over all pairs (a, b) - one from each cluster.")
        print(f"       Use dist_matrix[np.ix_(cluster_a, cluster_b)].mean().")
        return

    # Mistake: summed instead of averaged (off by n_a * n_b)
    total = float(np.sum(_submatrix(dist_matrix, cluster_a, cluster_b)))
    if _close(got, total, tol=max(0.1, abs(total) * 0.01)):
        print(f"  {_FAIL} average_linkage = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: divide the total by the number of pairs, not just sum them.")
        print(f"       .mean() on the submatrix handles this automatically.")
        return

    print(f"  {_FAIL} average_linkage = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: return dist_matrix[np.ix_(cluster_a, cluster_b)].mean() -")
    print(f"       the mean distance over every (a, b) cross-pair.")


# ---------------------------------------------------------------------------
# Optional — Ward distance
# ---------------------------------------------------------------------------

def check_ward_distance(fn, cluster_a_pts, cluster_b_pts):
    """Check that fn returns the Ward merge cost (increase in total WCSS)."""
    got = fn(cluster_a_pts, cluster_b_pts)

    n_a = len(cluster_a_pts)
    n_b = len(cluster_b_pts)
    merged = np.vstack([cluster_a_pts, cluster_b_pts])

    mu_a = cluster_a_pts.mean(axis=0)
    mu_b = cluster_b_pts.mean(axis=0)
    mu_m = merged.mean(axis=0)

    wcss_a = float(np.sum((cluster_a_pts - mu_a) ** 2))
    wcss_b = float(np.sum((cluster_b_pts - mu_b) ** 2))
    wcss_m = float(np.sum((merged - mu_m) ** 2))
    expected = wcss_m - wcss_a - wcss_b

    if got is None:
        print(f"  {_NONE} ward_distance: not implemented yet (expected {expected:.4f})")
        return

    if _close(got, expected, tol=max(0.05, abs(expected) * 0.005)):
        print(f"  {_OK} ward_distance = {got:.4f}")
        return

    # Mistake: returned the merged WCSS without subtracting individual clusters
    if _close(got, wcss_m, tol=max(0.05, abs(wcss_m) * 0.005)):
        print(f"  {_FAIL} ward_distance = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: Ward distance is the INCREASE in WCSS, not the merged WCSS itself.")
        print(f"       Subtract wcss_a and wcss_b from the merged WCSS.")
        return

    # Mistake: used Euclidean distance between centroids
    centroid_dist = float(np.sqrt(np.sum((mu_a - mu_b) ** 2)))
    if _close(got, centroid_dist, tol=0.05):
        print(f"  {_FAIL} ward_distance = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: Ward distance is not the distance between centroids.")
        print(f"       Compute WCSS for the merged cluster and subtract the two individual WCSSs.")
        return

    # Mistake: used the Ward formula with size factor (n_a * n_b / (n_a + n_b)) * ||mu_a - mu_b||^2
    ward_formula = float(n_a * n_b / (n_a + n_b) * np.sum((mu_a - mu_b) ** 2))
    if _close(got, ward_formula, tol=max(0.05, abs(ward_formula) * 0.01)):
        print(f"  {_OK} ward_distance = {got:.4f}  (equivalent Ward formula - correct!)")
        return

    # Mistake: squared centroid distance without size weighting
    sq_centroid = float(np.sum((mu_a - mu_b) ** 2))
    if _close(got, sq_centroid, tol=0.05):
        print(f"  {_FAIL} ward_distance = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: the centroid distance alone ignores cluster sizes.")
        print(f"       Compute merged WCSS minus individual WCSSs, or use the size-weighted formula.")
        return

    print(f"  {_FAIL} ward_distance = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: stack both clusters, compute WCSS of the merged group,")
    print(f"       then subtract the individual WCSSs. WCSS of a set S:")
    print(f"       np.sum((S - S.mean(axis=0)) ** 2)")


# ---------------------------------------------------------------------------
# Optional — Cophenetic correlation
# ---------------------------------------------------------------------------

def check_cophenetic_correlation(fn, Z, dist_matrix):
    """Check that fn returns the cophenetic correlation coefficient."""
    got = fn(Z, dist_matrix)

    if got is None:
        print(f"  {_NONE} cophenetic_correlation: not implemented yet")
        return

    n = dist_matrix.shape[0]

    # Build cophenetic matrix from Z
    # Z is scipy linkage: rows are (idx1, idx2, distance, count)
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)
    expected, _ = cophenet(Z, condensed)
    expected = float(expected)

    if _close(got, expected, tol=0.02):
        print(f"  {_OK} cophenetic_correlation = {got:.4f}")
        return

    if _close(got, 1.0 - expected, tol=0.02):
        print(f"  {_FAIL} cophenetic_correlation = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: check the sign convention. High cophenetic correlation means")
        print(f"       the dendrogram preserves pairwise distances well (close to 1.0).")
        return

    print(f"  {_FAIL} cophenetic_correlation = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: build the cophenetic distance matrix from Z (the merge heights),")
    print(f"       then compute Pearson correlation between the original distances and")
    print(f"       the cophenetic distances. scipy.cluster.hierarchy.cophenet can verify.")
