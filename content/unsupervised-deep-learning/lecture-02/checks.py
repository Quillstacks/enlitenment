"""Auto-checker helpers for Chapter 2 — Centroid Clustering."""

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


# ---------------------------------------------------------------------------
# Assignment step
# ---------------------------------------------------------------------------

def check_assign_clusters(fn, data, centroids):
    """Check that fn assigns each point to its nearest centroid."""
    got = fn(data, centroids)

    if got is None:
        print(f"  {_NONE} assign_clusters: not implemented yet")
        return

    got = np.asarray(got)
    k = len(centroids)

    # Expected: argmin of squared distance to each centroid
    dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (n, k)
    expected = np.argmin(dists, axis=1)

    if got.shape != expected.shape:
        print(f"  {_FAIL} assign_clusters: got shape {got.shape}, expected {expected.shape}")
        print(f"       Hint: return a 1-D integer array of length {len(data)},")
        print(f"       one cluster index per point.")
        return

    if np.array_equal(got, expected):
        n_clusters = len(np.unique(got))
        print(f"  {_OK} assign_clusters: {len(data)} points \u2192 {n_clusters}/{k} clusters populated")
        return

    n_wrong = int(np.sum(got != expected))

    if n_wrong == len(data):
        print(f"  {_FAIL} assign_clusters: all {n_wrong} assignments are wrong")
        print(f"       Hint: for each point, compute its distance to every centroid,")
        print(f"       then pick the centroid with the smallest distance.")
        print(f"       np.argmin() on the distance array gives the nearest index.")
    elif n_wrong > len(data) // 2:
        print(f"  {_FAIL} assign_clusters: {n_wrong}/{len(data)} assignments are wrong")
        print(f"       Hint: check you are taking the minimum, not the maximum.")
    else:
        print(f"  {_FAIL} assign_clusters: {n_wrong}/{len(data)} assignments differ from expected")
        print(f"       Hint: verify the distance formula. Squared Euclidean distance")
        print(f"       between x and c is np.sum((x - c) ** 2).")


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

def check_update_centroids(fn, data, assignments, k):
    """Check that fn returns the mean of each cluster."""
    got = fn(data, assignments, k)

    if got is None:
        print(f"  {_NONE} update_centroids: not implemented yet")
        return

    got = np.asarray(got, dtype=float)
    d = data.shape[1]

    if got.shape != (k, d):
        print(f"  {_FAIL} update_centroids: got shape {got.shape}, expected ({k}, {d})")
        print(f"       Hint: return a 2-D array with one row per cluster.")
        return

    expected = np.array([data[assignments == i].mean(axis=0) for i in range(k)])

    if np.allclose(got, expected, atol=_TOL):
        print(f"  {_OK} update_centroids: {k} centroids placed at cluster means")
        return

    # Diagnosis: used median instead of mean
    expected_med = np.array([np.median(data[assignments == i], axis=0) for i in range(k)])
    if np.allclose(got, expected_med, atol=_TOL):
        print(f"  {_FAIL} update_centroids: looks like you used the median, not the mean")
        print(f"       Hint: K-Means centroids are arithmetic means. Use .mean(axis=0).")
        return

    # Diagnosis: global mean applied to all clusters
    global_mean = data.mean(axis=0)
    if all(np.allclose(got[i], global_mean, atol=_TOL) for i in range(k)):
        print(f"  {_FAIL} update_centroids: all centroids are identical (the global mean)")
        print(f"       Hint: compute the mean separately per cluster.")
        print(f"       Try: data[assignments == i].mean(axis=0) for each i.")
        return

    print(f"  {_FAIL} update_centroids: centroids don\u2019t match cluster means")
    print(f"       Expected centroid 0: {expected[0].round(3)}")
    print(f"       Got      centroid 0: {got[0].round(3)}")
    print(f"       Hint: for cluster i, select data[assignments == i] and call .mean(axis=0).")


# ---------------------------------------------------------------------------
# Inertia (WCSS)
# ---------------------------------------------------------------------------

def check_inertia(fn, data, assignments, centroids):
    """Check that fn returns the within-cluster sum of squares."""
    got = fn(data, assignments, centroids)

    if got is None:
        print(f"  {_NONE} compute_inertia: not implemented yet")
        return

    k = len(centroids)
    expected = float(sum(
        np.sum((data[assignments == i] - centroids[i]) ** 2)
        for i in range(k)
    ))

    if _close(got, expected, tol=max(0.1, abs(expected) * 0.001)):
        print(f"  {_OK} compute_inertia = {got:.4f}")
        return

    # Diagnosis: used distance instead of squared distance
    expected_dist = float(sum(
        np.sum(np.sqrt(np.sum((data[assignments == i] - centroids[i]) ** 2, axis=1)))
        for i in range(k)
    ))
    if _close(got, expected_dist, tol=max(0.5, abs(expected_dist) * 0.01)):
        print(f"  {_FAIL} compute_inertia = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: inertia uses squared distances, not raw distances.")
        print(f"       Remove np.sqrt() \u2014 use (x \u2212 centroid) ** 2 directly.")
        return

    # Diagnosis: squared norms without subtracting centroid
    expected_no_sub = float(np.sum(data ** 2))
    if _close(got, expected_no_sub, tol=max(0.5, abs(expected_no_sub) * 0.01)):
        print(f"  {_FAIL} compute_inertia = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: subtract the centroid before squaring.")
        print(f"       The formula is \u2225x\u1d62 \u2212 \u03bc\u2096\u2225\u00b2, not \u2225x\u1d62\u2225\u00b2.")
        return

    print(f"  {_FAIL} compute_inertia = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: J = \u03a3\u2096 \u03a3\u1d62\u2208C\u2096 \u2225x\u1d62 \u2212 \u03bc\u2096\u2225\u00b2")
    print(f"       Loop over clusters, select assigned points, sum squared distances.")


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------

def check_silhouette(fn, data, assignments):
    """Check that fn returns the correct mean silhouette score."""
    got = fn(data, assignments)

    if got is None:
        print(f"  {_NONE} silhouette_score_manual: not implemented yet")
        return

    k = int(np.max(assignments)) + 1
    n = len(data)
    s_vals = np.zeros(n)

    for i in range(n):
        c_i = int(assignments[i])

        same_mask = assignments == c_i
        same_mask = same_mask.copy()
        same_mask[i] = False  # exclude self

        if same_mask.sum() == 0:
            s_vals[i] = 0.0
            continue

        # a(i): mean distance to own cluster, excluding self
        diffs_a = data[same_mask] - data[i]
        a_i = float(np.mean(np.sqrt(np.sum(diffs_a ** 2, axis=1))))

        # b(i): mean distance to nearest other cluster
        b_i = np.inf
        for c in range(k):
            if c == c_i:
                continue
            other_mask = assignments == c
            if other_mask.sum() == 0:
                continue
            diffs_b = data[other_mask] - data[i]
            mean_dist = float(np.mean(np.sqrt(np.sum(diffs_b ** 2, axis=1))))
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        s_vals[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    expected = float(np.mean(s_vals))

    if _close(got, expected, tol=0.02):
        print(f"  {_OK} silhouette_score_manual = {got:.4f}")
        return

    if _close(got, -expected, tol=0.02):
        print(f"  {_FAIL} silhouette = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: check the sign. The formula is (b \u2212 a) / max(a, b),")
        print(f"       not (a \u2212 b) / max(a, b).")
        return

    if got is not None and abs(float(got)) > 1.0 + 1e-6:
        print(f"  {_FAIL} silhouette = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: silhouette values must lie in [\u22121, +1].")
        print(f"       Make sure you divide by max(a(i), b(i)), not by a(i) + b(i).")
        return

    print(f"  {_FAIL} silhouette = {got:.4f}  (expected {expected:.4f})")
    print(f"       Hint: for each point i:")
    print(f"         a(i) = mean dist to other members of its own cluster")
    print(f"         b(i) = mean dist to all members of the nearest other cluster")
    print(f"         s(i) = (b(i) \u2212 a(i)) / max(a(i), b(i))")
    print(f"       Return the mean of s(i) over all points.")


# ---------------------------------------------------------------------------
# Optional — K-Means++ initialization
# ---------------------------------------------------------------------------

def check_kmeans_pp_init(fn, data, k):
    """Check K-Means++ init: output shape and all centroids are data points."""
    got = fn(data, k)

    if got is None:
        print(f"  {_NONE} kmeans_pp_init: not implemented yet")
        return

    got = np.asarray(got, dtype=float)
    d = data.shape[1]

    if got.shape != (k, d):
        print(f"  {_FAIL} kmeans_pp_init: got shape {got.shape}, expected ({k}, {d})")
        print(f"       Hint: return a (k, d) array of selected data points.")
        return

    # Each centroid must be an actual data point
    not_found = 0
    for c in got:
        if not np.any(np.all(np.abs(data - c) < 1e-6, axis=1)):
            not_found += 1

    if not_found > 0:
        print(f"  {_FAIL} kmeans_pp_init: {not_found}/{k} centroids are not actual data points")
        print(f"       Hint: select centroids by sampling indices from data,")
        print(f"       not by interpolating between points.")
        return

    # Centroids must be distinct
    _, counts = np.unique(got, axis=0, return_counts=True)
    if np.any(counts > 1):
        print(f"  {_FAIL} kmeans_pp_init: duplicate centroids found")
        print(f"       Hint: sample without replacement.")
        return

    print(f"  {_OK} kmeans_pp_init: {k} distinct data points selected as initial centroids")


# ---------------------------------------------------------------------------
# Optional — K-Medoids assignment
# ---------------------------------------------------------------------------

def check_kmedoids_assign(fn, data, medoid_indices):
    """Check that fn assigns each point to its nearest medoid."""
    got = fn(data, medoid_indices)

    if got is None:
        print(f"  {_NONE} kmedoids_assign: not implemented yet")
        return

    got = np.asarray(got)
    medoids = data[medoid_indices]
    k = len(medoid_indices)

    dists = np.sum((data[:, None, :] - medoids[None, :, :]) ** 2, axis=2)
    expected = np.argmin(dists, axis=1)

    if got.shape != expected.shape:
        print(f"  {_FAIL} kmedoids_assign: got shape {got.shape}, expected {expected.shape}")
        return

    if np.array_equal(got, expected):
        print(f"  {_OK} kmedoids_assign: {len(data)} points assigned to {k} medoids")
        return

    n_wrong = int(np.sum(got != expected))
    print(f"  {_FAIL} kmedoids_assign: {n_wrong}/{len(data)} assignments differ from expected")
    print(f"       Hint: use data[medoid_indices] to get medoid coordinates,")
    print(f"       then assign each point to the nearest one by Euclidean distance.")


# ---------------------------------------------------------------------------
# Optional — K-Medoids update (find medoid per cluster)
# ---------------------------------------------------------------------------

def check_kmedoids_update(fn, data, assignments, k):
    """Check that fn returns the index of the true medoid per cluster."""
    got = fn(data, assignments, k)

    if got is None:
        print(f"  {_NONE} kmedoids_update: not implemented yet")
        return

    got = np.asarray(got, dtype=int)

    if got.shape != (k,):
        print(f"  {_FAIL} kmedoids_update: got shape {got.shape}, expected ({k},)")
        print(f"       Hint: return a 1-D array of k global data indices.")
        return

    expected = np.zeros(k, dtype=int)
    for cluster in range(k):
        mask = assignments == cluster
        indices = np.where(mask)[0]
        if len(indices) == 0:
            expected[cluster] = -1
            continue
        cluster_data = data[indices]
        # pairwise squared distances within cluster
        diffs = cluster_data[:, None, :] - cluster_data[None, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        local_idx = np.argmin(sq_dists.sum(axis=1))
        expected[cluster] = indices[local_idx]

    if np.array_equal(got, expected):
        print(f"  {_OK} kmedoids_update: {k} medoid indices correctly identified")
        return

    wrong = int(np.sum(got != expected))
    print(f"  {_FAIL} kmedoids_update: {wrong}/{k} medoid indices are wrong")
    print(f"       Hint: the medoid of a cluster is the member with the smallest")
    print(f"       total (or mean) distance to all other cluster members.")
    print(f"       Compute pairwise distances within the cluster and argmin the row sums.")


# ---------------------------------------------------------------------------
# Optional — Mini-Batch update step
# ---------------------------------------------------------------------------

def check_minibatch_step(fn, data, centroids, batch_size, seed=0):
    """Check one mini-batch centroid update."""
    rng = np.random.default_rng(seed)
    batch_idx = rng.choice(len(data), batch_size, replace=False)
    batch = data[batch_idx]

    got = fn(batch, centroids)

    if got is None:
        print(f"  {_NONE} minibatch_step: not implemented yet")
        return

    got = np.asarray(got, dtype=float)

    if got.shape != centroids.shape:
        print(f"  {_FAIL} minibatch_step: got shape {got.shape}, expected {centroids.shape}")
        print(f"       Hint: return updated centroids with the same shape as the input.")
        return

    # Expected: assign batch to nearest centroid, compute means
    k = len(centroids)
    dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    batch_asgn = np.argmin(dists, axis=1)
    expected = centroids.copy()
    for i in range(k):
        mask = batch_asgn == i
        if mask.sum() > 0:
            expected[i] = batch[mask].mean(axis=0)

    if np.allclose(got, expected, atol=0.5):
        print(f"  {_OK} minibatch_step: centroids updated from batch of {batch_size}")
        return

    if np.allclose(got, centroids, atol=_TOL):
        print(f"  {_FAIL} minibatch_step: centroids did not change")
        print(f"       Hint: compute new means from the batch, don\u2019t return the originals.")
        return

    print(f"  {_FAIL} minibatch_step: centroids don\u2019t match expected mini-batch means")
    print(f"       Hint: assign batch points to their nearest centroid,")
    print(f"       then recompute each centroid as the mean of its assigned batch points.")
