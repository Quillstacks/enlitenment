"""Auto-checker helpers for Chapter 8, Transfer Learning."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3


def _close(a, b, tol=_TOL):
    """Scalar / array closeness with the chapter's default tolerance."""
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# 🧱  encode
# ---------------------------------------------------------------------------

def check_encode(fn, X, mlp):
    """Validate the encoder forward pass: shape, ReLU, L2 norm."""
    got = fn(X, mlp)

    if got is None:
        print(f"  {_NONE} encode: not implemented yet (expected shape ({X.shape[0]}, 16))")
        return

    got = np.asarray(got)
    expected_shape = (X.shape[0], mlp.coefs_[1].shape[1])

    if got.shape != expected_shape:
        print(f"  {_FAIL} encode shape {got.shape}  (expected {expected_shape})")
        print(f"       Hint: stop after the second weight matrix; do not apply the classifier head.")
        return

    norms = np.linalg.norm(got, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        # Diagnose: did they skip normalisation?
        h1 = np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])
        z_unnorm = np.maximum(0, h1 @ mlp.coefs_[1] + mlp.intercepts_[1])
        if _close(np.linalg.norm(got, axis=1), np.linalg.norm(z_unnorm, axis=1), tol=1e-3):
            print(f"  {_FAIL} encode: rows are not L2-normalised")
            print(f"       Hint: divide each row by its L2 norm so cosine similarity becomes a dot product later.")
            return
        print(f"  {_FAIL} encode: row norms range [{norms.min():.3f}, {norms.max():.3f}], expected ~1.0")
        return

    # Compare to reference forward pass.
    h1 = np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])
    z  = np.maximum(0, h1 @ mlp.coefs_[1] + mlp.intercepts_[1])
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    if _close(got, z_norm, tol=1e-2):
        print(f"  {_OK} encode: shape {got.shape}, rows on unit sphere, matches reference forward pass")
        return

    # Diagnose common mistakes.
    h1_no_relu = X @ mlp.coefs_[0] + mlp.intercepts_[0]
    z_no_relu  = (h1_no_relu @ mlp.coefs_[1] + mlp.intercepts_[1])
    z_no_relu  = z_no_relu / (np.linalg.norm(z_no_relu, axis=1, keepdims=True) + 1e-12)
    if _close(got, z_no_relu, tol=1e-2):
        print(f"  {_FAIL} encode: forgot the ReLU between the two layers")
        print(f"       Hint: apply np.maximum(0, ...) after EACH linear step, including the first.")
        return

    # Maybe they ran the head too.
    if len(mlp.coefs_) >= 3:
        z3 = np.maximum(0, h1 @ mlp.coefs_[1] + mlp.intercepts_[1])
        z3 = z3 @ mlp.coefs_[2] + mlp.intercepts_[2]
        z3 = z3 / (np.linalg.norm(z3, axis=1, keepdims=True) + 1e-12)
        if got.shape == z3.shape and _close(got, z3, tol=1e-2):
            print(f"  {_FAIL} encode: ran one layer too many, that is the classification head")
            print(f"       Hint: stop after coefs_[1]/intercepts_[1]; do not apply coefs_[2].")
            return

    print(f"  {_FAIL} encode: numerical mismatch with reference forward pass")
    print(f"       Hint: forward pass should be relu(relu(X @ W0 + b0) @ W1 + b1), then L2-normalise rows.")


# ---------------------------------------------------------------------------
# 🪜  linear_probe
# ---------------------------------------------------------------------------

def check_linear_probe(fn, emb_train, y_train, emb_test, y_test):
    from sklearn.linear_model import LogisticRegression
    got = fn(emb_train, y_train, emb_test, y_test)

    expected = float(LogisticRegression(max_iter=2000)
                     .fit(emb_train, y_train)
                     .score(emb_test, y_test))

    if got is None:
        print(f"  {_NONE} linear_probe: not implemented yet (expected ~{expected:.3f})")
        return

    if not isinstance(got, (int, float, np.floating)):
        print(f"  {_FAIL} linear_probe returned {type(got).__name__}, expected a float")
        print(f"       Hint: return the .score(...) value, which is a Python float.")
        return

    if abs(float(got) - expected) < 0.05:
        print(f"  {_OK} linear_probe accuracy = {float(got):.3f}  (reference {expected:.3f})")
        return

    if 0.0 <= got <= 1.0:
        print(f"  {_FAIL} linear_probe = {float(got):.3f}  (expected ~{expected:.3f})")
        print(f"       Hint: did you fit on emb_train then score on emb_test? Mixing them up gives a different number.")
    else:
        print(f"  {_FAIL} linear_probe = {got}, not in [0, 1]")
        print(f"       Hint: return the test accuracy as a float, not the model object.")


# ---------------------------------------------------------------------------
# 🎯  attribute_table  (validates student-filled rows)
# ---------------------------------------------------------------------------

def check_attribute_table(attrs_target, attr_names):
    n = len(attr_names)
    if not isinstance(attrs_target, dict):
        print(f"  {_FAIL} attrs_target is not a dict")
        return

    missing = [k for k in (0, 1) if k not in attrs_target]
    if missing:
        print(f"  {_FAIL} attrs_target missing keys {missing}; expected entries for 0 and 1")
        return

    not_filled = [k for k in (0, 1) if attrs_target[k] is None]
    if not_filled:
        print(f"  {_NONE} attrs_target rows for {not_filled} not filled in yet "
              f"(expected lists of {n} zeros/ones)")
        return

    issues = []
    for k in (0, 1):
        row = attrs_target[k]
        try:
            row = list(row)
        except TypeError:
            issues.append(f"row for {k} is not iterable")
            continue
        if len(row) != n:
            issues.append(f"row for {k} has length {len(row)}, expected {n}")
            continue
        if not all(v in (0, 1) for v in row):
            issues.append(f"row for {k} has non-binary values: {row}")

    if issues:
        print(f"  {_FAIL} attrs_target shape problems:")
        for s in issues:
            print(f"       . {s}")
        return

    # Soft sanity check: the two rows should differ. A 0 and a 1 are visually
    # opposite; if the rows are identical the bridge has nothing to work with.
    r0, r1 = list(attrs_target[0]), list(attrs_target[1])
    if r0 == r1:
        print(f"  {_FAIL} attrs_target: rows for 0 and 1 are identical; they should differ on at least one attribute")
        print(f"       Hint: a 0 has a closed loop and no straight strokes; a 1 has a vertical stroke and no loop.")
        return

    # Stronger soft check: warn if the row contradicts a strong visual prior.
    name_to_idx = {n_: i for i, n_ in enumerate(attr_names)}
    warn = []
    if 'has_loop' in name_to_idx:
        if r0[name_to_idx['has_loop']] != 1:
            warn.append("0 usually has a closed loop (`has_loop` = 1)")
        if r1[name_to_idx['has_loop']] != 0:
            warn.append("1 usually does not have a closed loop (`has_loop` = 0)")
    if 'vertical_stroke' in name_to_idx:
        if r1[name_to_idx['vertical_stroke']] != 1:
            warn.append("1 is dominated by a vertical stroke (`vertical_stroke` = 1)")
    if warn:
        print(f"  {_OK} attrs_target accepted, but check these:")
        for w in warn:
            print(f"       . {w}")
        return

    print(f"  {_OK} attrs_target rows look reasonable for 0 and 1")


# ---------------------------------------------------------------------------
# 🎯  class_centroids
# ---------------------------------------------------------------------------

def check_class_centroids(fn, emb, labels):
    got = fn(emb, labels)

    if got is None:
        print(f"  {_NONE} class_centroids: not implemented yet (expected dict over {len(set(labels))} classes)")
        return

    if not isinstance(got, dict):
        print(f"  {_FAIL} class_centroids returned {type(got).__name__}, expected a dict")
        print(f"       Hint: build {{label: centroid_vector}} so cosine_classify can index it later.")
        return

    expected_labels = set(int(l) for l in set(labels))
    got_labels = set(int(k) for k in got.keys())
    if got_labels != expected_labels:
        print(f"  {_FAIL} class_centroids keys {sorted(got_labels)}, expected {sorted(expected_labels)}")
        return

    issues = []
    for k, v in got.items():
        v = np.asarray(v)
        if v.shape != (emb.shape[1],):
            issues.append(f"label {k}: shape {v.shape}, expected ({emb.shape[1]},)")
            continue
        n = float(np.linalg.norm(v))
        if abs(n - 1.0) > 1e-2:
            # Diagnose: did they forget to L2-normalise?
            mask = np.asarray(labels) == k
            mean_unnorm = emb[mask].mean(axis=0)
            if _close(v, mean_unnorm, tol=1e-2):
                issues.append(f"label {k}: mean of embeddings, but not L2-normalised (norm = {n:.3f})")
            else:
                issues.append(f"label {k}: norm = {n:.3f}, expected ~1.0")

    if issues:
        print(f"  {_FAIL} class_centroids:")
        for s in issues:
            print(f"       . {s}")
        return

    # Reference compute and compare.
    ref = {}
    for k in expected_labels:
        mask = np.asarray(labels) == k
        c = emb[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        ref[int(k)] = c

    bad = [k for k in expected_labels if not _close(np.asarray(got[k]), ref[k], tol=1e-2)]
    if bad:
        print(f"  {_FAIL} class_centroids: numerical mismatch on labels {bad}")
        print(f"       Hint: average embeddings belonging to each class, THEN L2-normalise the average.")
        return

    print(f"  {_OK} class_centroids: {len(got)} classes, all rows on the unit sphere")


# ---------------------------------------------------------------------------
# 🎯  attribute_bridge
# ---------------------------------------------------------------------------

def check_attribute_bridge(fn, attrs_source, centroids_source, attrs_target):
    got = fn(attrs_source, centroids_source, attrs_target)

    if got is None:
        print(f"  {_NONE} attribute_bridge: not implemented yet "
              f"(expected dict over {sorted(attrs_target.keys())})")
        return

    if not isinstance(got, dict):
        print(f"  {_FAIL} attribute_bridge returned {type(got).__name__}, expected a dict")
        return

    expected_keys = set(attrs_target.keys())
    if set(got.keys()) != expected_keys:
        print(f"  {_FAIL} attribute_bridge keys {sorted(got.keys())}, "
              f"expected {sorted(expected_keys)}")
        return

    # Reference: A @ W = C least-squares.
    src_keys = sorted(attrs_source.keys())
    A = np.array([attrs_source[k] for k in src_keys], dtype=np.float64)
    C = np.array([np.asarray(centroids_source[k]) for k in src_keys], dtype=np.float64)
    W, *_ = np.linalg.lstsq(A, C, rcond=None)

    bad = []
    for k, v in got.items():
        v = np.asarray(v)
        if v.shape != (C.shape[1],):
            bad.append(f"label {k}: shape {v.shape}, expected ({C.shape[1]},)")
            continue
        n = float(np.linalg.norm(v))
        if abs(n - 1.0) > 1e-2:
            bad.append(f"label {k}: norm = {n:.3f}, expected ~1.0")
            continue
        ref = np.asarray(attrs_target[k], dtype=np.float64) @ W
        ref = ref / (np.linalg.norm(ref) + 1e-12)
        if not _close(v, ref, tol=5e-2):
            bad.append(f"label {k}: numerical mismatch with reference predicted centroid")

    if bad:
        print(f"  {_FAIL} attribute_bridge:")
        for s in bad:
            print(f"       . {s}")
        print(f"       Hint: stack source rows in matching order, solve with np.linalg.lstsq, "
              f"apply to target rows, then L2-normalise the result.")
        return

    print(f"  {_OK} attribute_bridge: {len(got)} target classes, predicted centroids match reference")


# ---------------------------------------------------------------------------
# 🎯  cosine_classify
# ---------------------------------------------------------------------------

def check_cosine_classify(fn, query_emb, class_emb_dict, expected):
    got = fn(query_emb, class_emb_dict)

    if got is None:
        print(f"  {_NONE} cosine_classify: not implemented yet (expected {expected})")
        return

    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)

    if got_arr.shape != expected_arr.shape:
        print(f"  {_FAIL} cosine_classify shape {got_arr.shape}, expected {expected_arr.shape}")
        return

    if np.array_equal(got_arr, expected_arr):
        print(f"  {_OK} cosine_classify: predictions {list(got_arr)}")
        return

    # Diagnose: argmin instead of argmax (returned the FURTHEST class).
    labels = sorted(class_emb_dict.keys())
    M = np.stack([np.asarray(class_emb_dict[l]) for l in labels])
    sims = query_emb @ M.T
    argmin_preds = np.array([labels[i] for i in np.argmin(sims, axis=1)])
    if np.array_equal(got_arr, argmin_preds):
        print(f"  {_FAIL} cosine_classify: returned the LEAST similar class")
        print(f"       Hint: use np.argmax, not np.argmin; high cosine similarity means closer.")
        return

    print(f"  {_FAIL} cosine_classify: got {list(got_arr)}, expected {list(expected_arr)}")
    print(f"       Hint: similarity = query_emb @ class_matrix.T, then argmax along axis 1, then label lookup.")


# ---------------------------------------------------------------------------
# 🔗  contrastive_loss
# ---------------------------------------------------------------------------

def check_contrastive_loss(fn, paired_embs, tau=0.1):
    """Sanity-check the diagonal contrastive loss.

    Two cases:
      (a) identical paired embeddings (perfect alignment): loss should be small.
      (b) shuffled "text" side (broken alignment): loss should be much larger.
    """
    got_aligned = fn(paired_embs, paired_embs, tau)

    if got_aligned is None:
        print(f"  {_NONE} contrastive_loss: not implemented yet "
              f"(expected ~0 for aligned pairs, ~log(N) for shuffled)")
        return

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(paired_embs))
    while np.array_equal(perm, np.arange(len(paired_embs))):
        perm = rng.permutation(len(paired_embs))
    got_shuffled = fn(paired_embs, paired_embs[perm], tau)

    if got_shuffled is None:
        print(f"  {_NONE} contrastive_loss returned None on shuffled pairs (expected a positive scalar)")
        return

    # Reference compute.
    def _ref(img, txt, t):
        S = (img @ txt.T) / t
        m_r = S.max(axis=1, keepdims=True)
        lse_r = np.log(np.exp(S - m_r).sum(axis=1)) + m_r.ravel()
        m_c = S.max(axis=0, keepdims=True)
        lse_c = np.log(np.exp(S - m_c).sum(axis=0)) + m_c.ravel()
        diag = np.diag(S)
        return float(0.5 * ((lse_r - diag).mean() + (lse_c - diag).mean()))

    ref_aligned  = _ref(paired_embs, paired_embs, tau)
    ref_shuffled = _ref(paired_embs, paired_embs[perm], tau)

    if not isinstance(got_aligned, (int, float, np.floating)):
        print(f"  {_FAIL} contrastive_loss returned {type(got_aligned).__name__}, expected a float")
        return

    if abs(float(got_aligned) - ref_aligned) > 1e-2:
        # Likely missed the symmetry: only computed image→text, not also text→image.
        S = (paired_embs @ paired_embs.T) / tau
        m_r = S.max(axis=1, keepdims=True)
        lse_r = np.log(np.exp(S - m_r).sum(axis=1)) + m_r.ravel()
        diag = np.diag(S)
        one_dir = float((lse_r - diag).mean())
        if abs(float(got_aligned) - one_dir) < 1e-2:
            print(f"  {_FAIL} contrastive_loss = {float(got_aligned):.4f}, "
                  f"matches a one-directional cross-entropy (expected the symmetric average)")
            print(f"       Hint: compute cross-entropy in BOTH directions (rows and columns) and average them.")
            return
        print(f"  {_FAIL} contrastive_loss aligned = {float(got_aligned):.4f}, expected ~{ref_aligned:.4f}")
        print(f"       Hint: use the symmetric InfoNCE: average of row-wise and column-wise cross-entropy.")
        return

    if got_shuffled <= got_aligned + 0.1:
        print(f"  {_FAIL} contrastive_loss did not increase when pairs were broken "
              f"(aligned {got_aligned:.3f}, shuffled {got_shuffled:.3f})")
        print(f"       Hint: the diagonal is the correct match; when you shuffle the text side, "
              f"the loss should rise sharply.")
        return

    print(f"  {_OK} contrastive_loss: aligned = {float(got_aligned):.4f}  "
          f"shuffled = {float(got_shuffled):.4f}  (reference {ref_aligned:.4f}, {ref_shuffled:.4f})")


# ---------------------------------------------------------------------------
# 🧩  topk_retrieve
# ---------------------------------------------------------------------------

def check_topk_retrieve(fn, query_emb, all_embs, k, expected):
    got = fn(query_emb, all_embs, k)

    if got is None:
        print(f"  {_NONE} topk_retrieve: not implemented yet (expected {list(expected)})")
        return

    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)

    if got_arr.shape != expected_arr.shape:
        print(f"  {_FAIL} topk_retrieve shape {got_arr.shape}, expected {expected_arr.shape}")
        return

    if np.array_equal(got_arr, expected_arr):
        print(f"  {_OK} topk_retrieve: indices {list(got_arr)}")
        return

    # Diagnose: ascending instead of descending.
    sims = all_embs @ query_emb
    asc = np.argsort(sims)[:k]
    if np.array_equal(got_arr, asc):
        print(f"  {_FAIL} topk_retrieve: returned the LEAST similar k")
        print(f"       Hint: descending order, np.argsort(...)[::-1].")
        return

    print(f"  {_FAIL} topk_retrieve: got {list(got_arr)}, expected {list(expected_arr)}")
    print(f"       Hint: similarity = all_embs @ query_emb, descending argsort, take the first k.")
