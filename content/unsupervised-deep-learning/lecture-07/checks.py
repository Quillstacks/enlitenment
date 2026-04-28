"""Auto-checker helpers for Chapter 7, Uncertainty Estimation."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3


def _close(a, b, tol=_TOL):
    """Scalar / array closeness with the chapter's default tolerance."""
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# 🎲 MC Dropout: predictive_stats(samples) -> (mean, var)
# ---------------------------------------------------------------------------

def check_predictive_stats(fn):
    """Sanity-check predictive_stats over a synthetic (T, n, K) tensor.

    The convention: axis 0 is T (the number of forward passes), axis 1 is n
    (samples), axis 2 is K (classes). Both outputs should have shape (n, K).
    """
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=(20, 50, 5)).astype(np.float64)
    samples /= samples.sum(axis=2, keepdims=True)
    expected_mean = samples.mean(axis=0)
    expected_var  = samples.var(axis=0)

    got = fn(samples)
    if got is None:
        print(f"  {_NONE} predictive_stats: not implemented yet "
              f"(expected mean and var of shape {expected_mean.shape})")
        return

    if not (isinstance(got, tuple) and len(got) == 2):
        print(f"  {_FAIL} predictive_stats: should return a tuple (mean, var), "
              f"got {type(got).__name__}")
        return

    mean, var = got
    mean = np.asarray(mean, dtype=float)
    var  = np.asarray(var,  dtype=float)

    if mean.shape != expected_mean.shape:
        print(f"  {_FAIL} predictive_stats: mean has shape {mean.shape} "
              f"(expected {expected_mean.shape})")
        if mean.shape == (samples.shape[0], samples.shape[2]):
            print(f"       Hint: looks like you reduced over axis 1 instead of axis 0. "
                  f"T (the pass count) is axis 0.")
        return
    if var.shape != expected_var.shape:
        print(f"  {_FAIL} predictive_stats: var has shape {var.shape} "
              f"(expected {expected_var.shape})")
        return

    if not _close(mean, expected_mean):
        if _close(mean, samples.mean(axis=1)):
            print(f"  {_FAIL} predictive_stats: mean averaged over the sample axis "
                  f"instead of the T axis")
            print(f"       Hint: T (axis 0) is the pass count; use "
                  f"np.mean(samples, axis=0).")
            return
        print(f"  {_FAIL} predictive_stats: mean values do not match "
              f"np.mean(samples, axis=0)")
        return

    if not _close(var, expected_var):
        if _close(var, samples.std(axis=0)):
            print(f"  {_FAIL} predictive_stats: returned standard deviation, "
                  f"not variance")
            print(f"       Hint: predictive variance is np.var(...), not np.std(...).")
            return
        if _close(var, samples.var(axis=0, ddof=1)):
            print(f"  {_FAIL} predictive_stats: used ddof=1 (sample variance)")
            print(f"       Hint: stick with the default ddof=0 — the population "
                  f"variance over the T passes.")
            return
        print(f"  {_FAIL} predictive_stats: variance values do not match "
              f"np.var(samples, axis=0)")
        return

    print(f"  {_OK} predictive_stats: mean and var look right")


# ---------------------------------------------------------------------------
# 🪞 Find the Odds: auroc_from_scores(scores, is_odd) -> float
# ---------------------------------------------------------------------------

def _reference_auroc(scores, is_odd):
    """Rank-sum AUROC, used by check_auroc_from_scores as ground truth."""
    scores = np.asarray(scores, dtype=float)
    is_odd = np.asarray(is_odd, dtype=bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(is_odd.sum())
    n_neg = len(scores) - n_pos
    return float((ranks[is_odd].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def check_auroc_from_scores(fn):
    """Sanity-check auroc_from_scores on a synthetic two-class score pile."""
    rng = np.random.default_rng(0)
    neg = rng.normal(0.0, 1.0, size=60)
    pos = rng.normal(1.5, 1.0, size=40)
    scores = np.concatenate([neg, pos])
    is_odd = np.concatenate([np.zeros(60, bool), np.ones(40, bool)])
    perm = rng.permutation(len(scores))
    scores, is_odd = scores[perm], is_odd[perm]

    expected = _reference_auroc(scores, is_odd)
    got = fn(scores, is_odd)

    if got is None:
        print(f"  {_NONE} auroc_from_scores: not implemented yet "
              f"(expected {expected:.4f})")
        return

    got_f = float(np.asarray(got))

    if _close(got_f, expected):
        print(f"  {_OK} auroc_from_scores = {got_f:.4f}")
        return

    if _close(got_f, 1.0 - expected):
        print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
        print(f"       Hint: looks like 1 - AUROC. Higher scores should mean more "
              f"anomalous; check the sort direction or the label polarity.")
        return

    order = np.argsort(scores)
    ranks0 = np.empty(len(scores), dtype=float)
    ranks0[order] = np.arange(0, len(scores))
    n_pos = int(is_odd.sum()); n_neg = len(scores) - n_pos
    off_by_one = float((ranks0[is_odd].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    if _close(got_f, off_by_one):
        print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
        print(f"       Hint: ranks should run 1..n, not 0..n-1. Use "
              f"np.arange(1, len(scores) + 1).")
        return

    ranks1 = np.empty(len(scores), dtype=float)
    ranks1[order] = np.arange(1, len(scores) + 1)
    no_offset = float(ranks1[is_odd].sum() / (n_pos * n_neg))
    if _close(got_f, no_offset):
        print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
        print(f"       Hint: subtract n_pos * (n_pos + 1) / 2 from the positive "
              f"rank sum before dividing.")
        return

    wrong_denom = float((ranks1[is_odd].sum() - n_pos * (n_pos + 1) / 2) / len(scores))
    if _close(got_f, wrong_denom):
        print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
        print(f"       Hint: the denominator is n_pos * n_neg, not the total count.")
        return

    if _close(got_f, float(ranks1[is_odd].sum())):
        print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
        print(f"       Hint: you returned the rank sum. Subtract the offset and "
              f"divide by n_pos * n_neg to get a number in [0, 1].")
        return

    print(f"  {_FAIL} auroc_from_scores = {got_f:.4f}  (expected {expected:.4f})")
    print(f"       Hint: AUROC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) "
          f"with ranks running 1..n on ascending-sorted scores.")


# ---------------------------------------------------------------------------
# 🪞 Find the Odds: mahalanobis_score(Z, mean, cov_inv) -> (n,) of d^2
# ---------------------------------------------------------------------------

def check_mahalanobis_score(fn):
    """Sanity-check mahalanobis_score on a small anisotropic Gaussian."""
    rng = np.random.default_rng(1)
    n, k = 40, 4
    cov = np.diag([4.0, 1.0, 0.25, 0.0625])
    mean = np.array([0.5, -0.3, 0.1, 0.0])
    Z = rng.multivariate_normal(mean, cov, size=n)
    cov_inv = np.linalg.inv(cov)

    diff = Z - mean
    expected = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)

    got = fn(Z, mean, cov_inv)
    if got is None:
        print(f"  {_NONE} mahalanobis_score: not implemented yet "
              f"(expected shape ({n},), first value {expected[0]:.4f})")
        return

    got = np.asarray(got, dtype=float)

    if got.shape != expected.shape:
        print(f"  {_FAIL} mahalanobis_score: shape {got.shape} (expected {expected.shape})")
        if got.shape == (n, n):
            print(f"       Hint: you returned the full pairwise quadratic form. "
                  f"Take the diagonal, or use np.einsum('ij,jk,ik->i', ...) to "
                  f"keep only the per-row term.")
        elif got.shape == (k,) or got.shape == ():
            print(f"       Hint: reduce per row, not across all rows. The output "
                  f"should have one number per input.")
        return

    if _close(got, expected):
        print(f"  {_OK} mahalanobis_score: per-row distances look right")
        return

    euclid_sq = (diff ** 2).sum(axis=1)
    if _close(got, euclid_sq):
        print(f"  {_FAIL} mahalanobis_score: distances do not match (first "
              f"got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: this is squared Euclidean distance; you forgot to "
              f"weight by cov_inv. The form is diff @ cov_inv @ diff.T per row.")
        return

    cov_form = np.einsum('ij,jk,ik->i', diff, cov, diff)
    if _close(got, cov_form):
        print(f"  {_FAIL} mahalanobis_score: distances match diff @ cov @ diff.T "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: Mahalanobis uses the *inverse* covariance, not the "
              f"covariance itself.")
        return

    if _close(got, np.sqrt(np.clip(expected, 0, None))):
        print(f"  {_FAIL} mahalanobis_score: returned distance, not squared distance "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: stop before the square root. The chapter uses the "
              f"squared form because it is the negative log-density up to a constant.")
        return

    no_center = np.einsum('ij,jk,ik->i', Z, cov_inv, Z)
    if _close(got, no_center):
        print(f"  {_FAIL} mahalanobis_score: distances match Z @ cov_inv @ Z.T "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: subtract train_mean from each row first, then apply "
              f"cov_inv.")
        return

    print(f"  {_FAIL} mahalanobis_score: distances do not match "
          f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
    print(f"       Hint: per row, compute (z - mean) @ cov_inv @ (z - mean). "
          f"np.einsum('ij,jk,ik->i', diff, cov_inv, diff) does it in one line.")


# ---------------------------------------------------------------------------
# 🎁 Misc class: misc_class_decision(probs_misc, threshold) -> bool array
# ---------------------------------------------------------------------------

def check_misc_class_decision(fn):
    """Sanity-check misc_class_decision over a synthetic probability vector."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.0, 1.0, size=200).astype(np.float64)
    threshold = 0.5
    expected = probs > threshold

    got = fn(probs, threshold)
    if got is None:
        print(f"  {_NONE} misc_class_decision: not implemented yet "
              f"(expected boolean array of shape {expected.shape}, "
              f"{int(expected.sum())} True)")
        return

    got_arr = np.asarray(got)

    if got_arr.shape != expected.shape:
        print(f"  {_FAIL} misc_class_decision: shape {got_arr.shape} "
              f"(expected {expected.shape})")
        return

    if got_arr.dtype != bool:
        if np.issubdtype(got_arr.dtype, np.floating) and _close(got_arr, probs):
            print(f"  {_FAIL} misc_class_decision: returned the probabilities "
                  f"unchanged")
            print(f"       Hint: compare probs_misc against threshold and "
                  f"return the boolean result.")
            return
        if np.issubdtype(got_arr.dtype, np.integer) and _close(got_arr, expected.astype(int)):
            print(f"  {_FAIL} misc_class_decision: returned 0/1 ints, expected "
                  f"a boolean array")
            print(f"       Hint: a comparison like `probs > t` already returns "
                  f"booleans; do not cast to int.")
            return
        print(f"  {_FAIL} misc_class_decision: dtype {got_arr.dtype} "
              f"(expected bool)")
        return

    if np.array_equal(got_arr, expected):
        t2 = 0.8
        expected_2 = probs > t2
        got_2 = np.asarray(fn(probs, t2))
        if not np.array_equal(got_2, expected_2):
            if np.array_equal(got_2, expected):
                print(f"  {_FAIL} misc_class_decision: threshold argument is ignored, "
                      f"the function appears to use a hard-coded value")
                print(f"       Hint: use the `threshold` parameter in your "
                      f"comparison, not a literal like 0.5.")
                return
            print(f"  {_FAIL} misc_class_decision: passes at threshold=0.5 but "
                  f"fails at threshold=0.8")
            return
        print(f"  {_OK} misc_class_decision: {int(got_arr.sum())}/{len(probs)} "
              f"flagged at t=0.5, threshold argument wired up")
        return

    if np.array_equal(got_arr, ~expected):
        print(f"  {_FAIL} misc_class_decision: inverted, flagging inputs *below* "
              f"the threshold")
        print(f"       Hint: misc means high prob_misc; use `>` (or `>=`), "
              f"not `<`.")
        return

    n_diff = int((got_arr != expected).sum())
    print(f"  {_FAIL} misc_class_decision: {n_diff}/{len(probs)} entries differ "
          f"from `probs_misc > threshold`")
    print(f"       Hint: return a boolean array, True where prob_misc exceeds "
          f"the threshold.")


# ---------------------------------------------------------------------------
# ⚡ Free energy: free_energy(logits, T=1.0) -> (n,)
# ---------------------------------------------------------------------------

def check_free_energy(fn):
    """Sanity-check free_energy on a synthetic (n, K) logit array.

    The reference: E(x) = -T * logsumexp(logits / T, axis=-1).
    Lower energy = more confident (in-distribution-like).
    """
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 2.0, size=(40, 5)).astype(np.float64)

    def _ref(zz, T):
        m = zz.max(axis=-1, keepdims=True)
        return -T * (m.squeeze(-1) + T * np.log(np.sum(np.exp((zz - m) / T), axis=-1)))

    expected_T1 = _ref(z, 1.0)
    expected_T2 = _ref(z, 2.0)

    got = fn(z)
    if got is None:
        print(f"  {_NONE} free_energy: not implemented yet "
              f"(expected an array of shape {expected_T1.shape})")
        return

    got = np.asarray(got, dtype=float)
    if got.shape != expected_T1.shape:
        print(f"  {_FAIL} free_energy: returned shape {got.shape} "
              f"(expected {expected_T1.shape})")
        print(f"       Hint: reduce over the class axis (axis=-1), keep the sample axis.")
        return

    if _close(got, -expected_T1):
        print(f"  {_FAIL} free_energy: sign is flipped")
        print(f"       Hint: E(x) = -T * logsumexp(z / T). The minus sign is what "
              f"makes LOWER = in-distribution.")
        return

    if _close(got, z.max(axis=-1)):
        print(f"  {_FAIL} free_energy: looks like max(logits), not -logsumexp(logits)")
        print(f"       Hint: replace np.max with np.log(np.sum(np.exp(...))).")
        return
    if _close(got, -z.max(axis=-1)):
        print(f"  {_FAIL} free_energy: returned -max(logits), not -logsumexp(logits)")
        print(f"       Hint: max is the T -> 0 limit. Use logsumexp at the requested T.")
        return

    sm = np.exp(z - z.max(axis=-1, keepdims=True))
    sm = sm / sm.sum(axis=-1, keepdims=True)
    if _close(got, -np.log(sm.max(axis=-1))):
        print(f"  {_FAIL} free_energy: looks like -log(max softmax)")
        print(f"       Hint: do logsumexp directly on the logits. Softmax normalises "
              f"away the absolute mass that energy is meant to read.")
        return

    if _close(got, np.log(np.sum(np.exp(z), axis=-1))):
        print(f"  {_FAIL} free_energy: returned +logsumexp(z), missing the leading -T")
        print(f"       Hint: multiply by -T at the end.")
        return

    if not _close(got, expected_T1):
        print(f"  {_FAIL} free_energy at T=1.0 disagrees with the reference")
        print(f"       Hint: -T * logsumexp(logits / T, axis=-1). Use the max-shift "
              f"trick for stability.")
        return

    got_T2 = np.asarray(fn(z, T=2.0), dtype=float)
    if _close(got_T2, expected_T1):
        print(f"  {_FAIL} free_energy: output is the same at T=1.0 and T=2.0")
        print(f"       Hint: T scales BOTH the division inside logsumexp AND the "
              f"factor outside. Do not hardcode T=1.")
        return
    if _close(got_T2, 2.0 * expected_T1):
        print(f"  {_FAIL} free_energy: T multiplied the outside but the inside "
              f"still uses logits/1.0")
        print(f"       Hint: divide the logits by T before logsumexp.")
        return
    if _close(got_T2, expected_T1 / 2.0):
        print(f"  {_FAIL} free_energy: T divided the inside but the outside still "
              f"multiplies by 1.0")
        print(f"       Hint: the leading factor is -T, not -1.")
        return
    if not _close(got_T2, expected_T2):
        print(f"  {_FAIL} free_energy at T=2.0 disagrees with the reference")
        print(f"       Hint: -T * logsumexp(logits / T). T appears in two places.")
        return

    print(f"  {_OK} free_energy: matches -T * logsumexp(logits / T) at T=1 and T=2")


# ---------------------------------------------------------------------------
# 🧭 LOF in Latent Space: knn_distance_score(Q, fit_Z, k) -> (n_q,)
# ---------------------------------------------------------------------------

def check_knn_distance_score(fn):
    """Sanity-check knn_distance_score on a small synthetic latent cloud."""
    rng = np.random.default_rng(2)
    d = 4
    fit_Z = rng.normal(0.0, 1.0, size=(60, d))
    Q     = rng.normal(0.0, 1.0, size=(15, d))
    k     = 5

    D = np.linalg.norm(Q[:, None, :] - fit_Z[None, :, :], axis=-1)
    D_sorted = np.sort(D, axis=-1)
    expected = D_sorted[:, :k].mean(axis=-1)

    got = fn(Q, fit_Z, k)
    if got is None:
        print(f"  {_NONE} knn_distance_score: not implemented yet "
              f"(expected shape ({Q.shape[0]},), first value {expected[0]:.4f})")
        return

    got = np.asarray(got, dtype=float)

    if got.shape != expected.shape:
        print(f"  {_FAIL} knn_distance_score: shape {got.shape} "
              f"(expected {expected.shape})")
        if got.shape == (Q.shape[0], fit_Z.shape[0]):
            print(f"       Hint: you returned the full pairwise distance matrix. "
                  f"Sort each row and average the first k values.")
        elif got.shape == (fit_Z.shape[0],):
            print(f"       Hint: you reduced over the wrong axis. The output has "
                  f"one number per query row in Q, not per training row in fit_Z.")
        return

    if _close(got, expected):
        print(f"  {_OK} knn_distance_score at k={k}: per-query mean distances look right")
        return

    max_k = D_sorted[:, :k].max(axis=-1)
    if _close(got, max_k):
        print(f"  {_FAIL} knn_distance_score: distances do not match "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: looks like you took the max of the k nearest, not the "
              f"mean. Use .mean(axis=-1) on the first k sorted distances.")
        return

    min_k = D_sorted[:, 0]
    if _close(got, min_k):
        print(f"  {_FAIL} knn_distance_score: returned the single nearest distance "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: average over the first k sorted distances, not just "
              f"the first one.")
        return

    desc = np.sort(D, axis=-1)[:, ::-1][:, :k].mean(axis=-1)
    if _close(got, desc):
        print(f"  {_FAIL} knn_distance_score: distances match the k *farthest* "
              f"neighbours (first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: sort ascending and take the first k.")
        return

    sq = (D_sorted[:, :k] ** 2).mean(axis=-1)
    if _close(got, sq):
        print(f"  {_FAIL} knn_distance_score: returned mean *squared* distance "
              f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: take the square root before averaging.")
        return

    sum_k = D_sorted[:, :k].sum(axis=-1)
    if _close(got, sum_k):
        print(f"  {_FAIL} knn_distance_score: returned the *sum* of the k nearest "
              f"distances (first got {got[0]:.4f}, expected {expected[0]:.4f})")
        print(f"       Hint: divide by k, or use .mean(axis=-1).")
        return

    got_k2 = np.asarray(fn(Q, fit_Z, 2), dtype=float)
    if _close(got, got_k2):
        print(f"  {_FAIL} knn_distance_score: output is the same at k={k} and k=2")
        print(f"       Hint: the k argument is being ignored.")
        return

    print(f"  {_FAIL} knn_distance_score: per-query mean distances do not match "
          f"(first got {got[0]:.4f}, expected {expected[0]:.4f})")
    print(f"       Hint: for each row of Q, sort distances to fit_Z ascending and "
          f"average the first k.")


# ---------------------------------------------------------------------------
# 🚦 Decision routing: route_decisions(scores, accept_t, reject_t) -> (n,) str
# ---------------------------------------------------------------------------

def check_route_decisions(fn):
    """Sanity-check route_decisions on a synthetic score vector."""
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.0, 1.0, size=200).astype(np.float64)
    accept_t, reject_t = 0.3, 0.7

    expected = np.full(scores.shape, 'defer', dtype='<U6')
    expected[scores < accept_t] = 'accept'
    expected[scores > reject_t] = 'reject'
    n_a = int((expected == 'accept').sum())
    n_d = int((expected == 'defer').sum())
    n_r = int((expected == 'reject').sum())

    got = fn(scores, accept_t, reject_t)
    if got is None:
        print(f"  {_NONE} route_decisions: not implemented yet "
              f"(expected {n_a} accept, {n_d} defer, {n_r} reject)")
        return

    got_arr = np.asarray(got)

    if got_arr.shape != expected.shape:
        print(f"  {_FAIL} route_decisions: shape {got_arr.shape} "
              f"(expected {expected.shape})")
        return

    if np.issubdtype(got_arr.dtype, np.integer):
        print(f"  {_FAIL} route_decisions: returned integer codes, expected "
              f"strings 'accept' / 'defer' / 'reject'")
        print(f"       Hint: build a string array with np.full(n, 'defer', "
              f"dtype='<U6') and assign labels by mask.")
        return

    if np.issubdtype(got_arr.dtype, np.bool_):
        print(f"  {_FAIL} route_decisions: returned booleans, expected three-way "
              f"string labels")
        return

    labels = set(np.unique(got_arr).tolist())
    valid = {'accept', 'defer', 'reject'}
    unknown = labels - valid
    if unknown:
        print(f"  {_FAIL} route_decisions: unknown labels {sorted(unknown)} "
              f"(expected subset of {{'accept', 'defer', 'reject'}}, lower-case)")
        return

    if np.array_equal(got_arr, expected):
        # Sweep one threshold to confirm both arguments are wired up.
        got_wide = np.asarray(fn(scores, 0.1, 0.9))
        if np.array_equal(got_wide, got_arr):
            print(f"  {_FAIL} route_decisions: same output at (0.3, 0.7) and "
                  f"(0.1, 0.9), threshold arguments look ignored")
            return
        print(f"  {_OK} route_decisions: {n_a}/{n_d}/{n_r} accept/defer/reject "
              f"at (0.3, 0.7), thresholds wired up")
        return

    swapped = np.full(scores.shape, 'defer', dtype='<U6')
    swapped[scores > reject_t] = 'accept'
    swapped[scores < accept_t] = 'reject'
    if np.array_equal(got_arr, swapped):
        print(f"  {_FAIL} route_decisions: accept and reject are swapped")
        print(f"       Hint: lower score = more in-distribution = accept. "
              f"Higher score = more OOD = reject.")
        return

    no_accept_t = np.full(scores.shape, 'accept', dtype='<U6')
    no_accept_t[scores > reject_t] = 'reject'
    if np.array_equal(got_arr, no_accept_t):
        print(f"  {_FAIL} route_decisions: defer bin is empty, accept_t looks "
              f"unused")
        return

    no_reject_t = np.full(scores.shape, 'defer', dtype='<U6')
    no_reject_t[scores < accept_t] = 'accept'
    if np.array_equal(got_arr, no_reject_t):
        print(f"  {_FAIL} route_decisions: reject bin is empty, reject_t looks "
              f"unused")
        return

    inclusive = np.full(scores.shape, 'defer', dtype='<U6')
    inclusive[scores <= accept_t] = 'accept'
    inclusive[scores >= reject_t] = 'reject'
    if np.array_equal(got_arr, inclusive):
        print(f"  {_FAIL} route_decisions: boundary inputs at exactly accept_t / "
              f"reject_t are placed in accept / reject, expected defer")
        print(f"       Hint: the contract uses strict inequalities, `<` and `>`.")
        return

    n_diff = int((got_arr != expected).sum())
    print(f"  {_FAIL} route_decisions: {n_diff}/{len(scores)} entries differ "
          f"from the expected three-bin assignment")
    print(f"       Hint: 'accept' if score < accept_t, 'reject' if score > "
          f"reject_t, otherwise 'defer'.")


# ---------------------------------------------------------------------------
# 🛡️ Confidence head: accuracy_at_confidence(conf, correct, threshold) -> float
# ---------------------------------------------------------------------------

def check_accuracy_at_confidence(fn):
    """Sanity-check accuracy_at_confidence on a synthetic confidence/correct pair."""
    rng = np.random.default_rng(2)
    n = 400
    confidence = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    p_correct = np.clip(0.4 + 0.55 * confidence, 0.0, 1.0)
    correct = rng.uniform(0.0, 1.0, size=n) < p_correct

    threshold = 0.7
    mask     = confidence > threshold
    n_pass   = int(mask.sum())
    expected = float(correct[mask].mean()) if n_pass > 0 else float('nan')

    got = fn(confidence, correct, threshold)
    if got is None:
        print(f"  {_NONE} accuracy_at_confidence: not implemented yet "
              f"(expected {expected:.4f} on {n_pass}/{n} accepted)")
        return

    got_arr = np.asarray(got)

    if got_arr.ndim != 0:
        if got_arr.shape == confidence.shape and got_arr.dtype == bool:
            print(f"  {_FAIL} accuracy_at_confidence: returned the boolean mask, "
                  f"not a scalar accuracy")
            return
        if got_arr.shape == confidence.shape:
            print(f"  {_FAIL} accuracy_at_confidence: returned an array of shape "
                  f"{got_arr.shape} (expected a scalar)")
            return
        print(f"  {_FAIL} accuracy_at_confidence: returned shape {got_arr.shape} "
              f"(expected a scalar)")
        return

    got_f = float(got_arr)

    inv_mask = confidence < threshold
    if inv_mask.sum() > 0:
        wrong_dir = float(correct[inv_mask].mean())
        if _close(got_f, wrong_dir):
            print(f"  {_FAIL} accuracy_at_confidence = {got_f:.4f}  "
                  f"(expected {expected:.4f})")
            print(f"       Hint: you accepted the LOW-confidence subset. "
                  f"Use `confidence > threshold`, not `<`.")
            return

    if _close(got_f, float(n_pass)):
        print(f"  {_FAIL} accuracy_at_confidence = {got_f:.4f}  "
              f"(expected {expected:.4f})")
        print(f"       Hint: you returned the count of accepted entries.")
        return

    n_correct_accepted = int(correct[mask].sum()) if n_pass > 0 else 0
    if _close(got_f, float(n_correct_accepted)):
        print(f"  {_FAIL} accuracy_at_confidence = {got_f:.4f}  "
              f"(expected {expected:.4f})")
        print(f"       Hint: you returned the count of correct accepted "
              f"predictions. Divide by the number accepted to get an accuracy.")
        return

    total_acc = float(correct.mean())
    if _close(got_f, total_acc) and not _close(expected, total_acc):
        print(f"  {_FAIL} accuracy_at_confidence = {got_f:.4f}  "
              f"(expected {expected:.4f})")
        print(f"       Hint: looks like the total accuracy. The threshold "
              f"argument is being ignored.")
        return

    got_extreme = fn(confidence, correct, 1.5)
    if got_extreme is not None:
        ge = np.asarray(got_extreme)
        if ge.ndim == 0:
            ge_f = float(ge)
            if not np.isnan(ge_f):
                if _close(ge_f, 0.0):
                    print(f"  {_FAIL} accuracy_at_confidence: returned 0.0 when "
                          f"no entries pass the threshold (expected NaN)")
                    print(f"       Hint: guard with `if mask.sum() == 0: "
                          f"return float('nan')`.")
                    return

    if _close(got_f, expected):
        print(f"  {_OK} accuracy_at_confidence = {got_f:.4f} on "
              f"{n_pass}/{n} accepted")
        return

    print(f"  {_FAIL} accuracy_at_confidence = {got_f:.4f}  "
          f"(expected {expected:.4f})")
    print(f"       Hint: mask = confidence > threshold; "
          f"return correct[mask].mean(), or NaN if mask is empty.")


# ---------------------------------------------------------------------------
# 🌡️ Calibration: calibration_bins(conf, correct, n_bins) -> (conf, acc, count)
# ---------------------------------------------------------------------------

def _reference_calibration_bins(confidence, correct, n_bins):
    confidence = np.asarray(confidence, dtype=float)
    correct    = np.asarray(correct,    dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges, right=False) - 1, 0, n_bins - 1)
    bin_conf  = np.full(n_bins, np.nan)
    bin_acc   = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_conf[b] = confidence[mask].mean()
            bin_acc[b]  = correct[mask].mean()
    return bin_conf, bin_acc, bin_count


def check_calibration_bins(fn):
    """Sanity-check calibration_bins on a synthetic confidence-and-correctness pile."""
    rng = np.random.default_rng(0)
    n_bulk = 400
    conf_bulk = rng.uniform(0.95, 1.00, size=n_bulk)
    corr_bulk = rng.uniform(size=n_bulk) < 0.55
    n_mid = 80
    conf_mid = rng.uniform(0.30, 0.90, size=n_mid)
    corr_mid = rng.uniform(size=n_mid) < 0.20
    conf_extra = np.array([0.5, 0.5, 1.0, 0.0])
    corr_extra = np.array([True, False, True, False])
    confidence = np.concatenate([conf_bulk, conf_mid, conf_extra])
    correct    = np.concatenate([corr_bulk, corr_mid, corr_extra]).astype(bool)

    n_bins   = 10
    expected = _reference_calibration_bins(confidence, correct, n_bins)
    exp_conf, exp_acc, exp_count = expected

    got = fn(confidence, correct, n_bins)
    if got is None:
        print(f"  {_NONE} calibration_bins: not implemented yet "
              f"(expected three arrays of shape ({n_bins},))")
        return

    if not (isinstance(got, tuple) and len(got) == 3):
        print(f"  {_FAIL} calibration_bins: should return a tuple "
              f"(bin_conf, bin_acc, bin_count), got {type(got).__name__}")
        return

    got_conf, got_acc, got_count = (np.asarray(x) for x in got)

    if got_conf.shape != (n_bins,) or got_acc.shape != (n_bins,) or got_count.shape != (n_bins,):
        print(f"  {_FAIL} calibration_bins: shapes "
              f"{got_conf.shape}, {got_acc.shape}, {got_count.shape} "
              f"(expected three arrays of shape ({n_bins},))")
        if got_conf.shape == (n_bins + 1,):
            print(f"       Hint: you returned bin *edges*, not per-bin means.")
        return

    got_alt = fn(confidence, correct, 5)
    if got_alt is not None and isinstance(got_alt, tuple) and len(got_alt) == 3:
        if np.asarray(got_alt[0]).shape == (n_bins,):
            print(f"  {_FAIL} calibration_bins: passed n_bins=5 but returned "
                  f"shape ({n_bins},)")
            print(f"       Hint: build edges from `np.linspace(0, 1, n_bins + 1)`.")
            return

    empty = exp_count == 0
    if empty.any():
        if not np.array_equal(got_count[empty], np.zeros(int(empty.sum()), dtype=got_count.dtype)):
            print(f"  {_FAIL} calibration_bins: empty bins should report count = 0")
            return
        if not (np.isnan(got_conf[empty]).all() and np.isnan(got_acc[empty]).all()):
            if (got_conf[empty] == 0).all() and (got_acc[empty] == 0).all():
                print(f"  {_FAIL} calibration_bins: empty bins return 0 for conf and acc")
                print(f"       Hint: empty bins should be NaN, not 0.")
                return
            print(f"  {_FAIL} calibration_bins: empty bins should be NaN for conf and acc")
            return

    if int(got_count.sum()) != len(confidence):
        if int(got_count.sum()) == len(confidence) - int((confidence == 1.0).sum()):
            print(f"  {_FAIL} calibration_bins: confidence == 1.0 was dropped "
                  f"(counts sum to {int(got_count.sum())}, expected {len(confidence)})")
            print(f"       Hint: clip the digitize result into [0, n_bins - 1].")
            return
        print(f"  {_FAIL} calibration_bins: counts sum to {int(got_count.sum())} "
              f"(expected {len(confidence)})")
        return

    if not np.array_equal(got_count, exp_count):
        print(f"  {_FAIL} calibration_bins: bin counts do not match the reference")
        return

    nonempty = exp_count > 0
    if not _close(got_conf[nonempty], exp_conf[nonempty]):
        print(f"  {_FAIL} calibration_bins: per-bin mean confidence does not match")
        print(f"       Hint: average the *confidence* values inside each bin.")
        return
    if not _close(got_acc[nonempty], exp_acc[nonempty]):
        print(f"  {_FAIL} calibration_bins: per-bin mean accuracy does not match")
        print(f"       Hint: average the boolean correctness inside each bin.")
        return

    print(f"  {_OK} calibration_bins: bin conf, acc, and count all match the reference")
