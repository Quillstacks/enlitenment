"""Auto-checker helpers for lecture-10, Weak Supervision."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3

ABSTAIN = -1


def _close(a, b, tol=_TOL):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# 🛠️  lf_coverage(L)
# ---------------------------------------------------------------------------
def check_lf_coverage(fn):
    rng = np.random.default_rng(0)
    L = rng.choice([-1, 0, 1], size=(40, 4), p=[0.3, 0.35, 0.35]).astype(int)
    got = fn(L)

    if got is None:
        print(f"  {_NONE} lf_coverage: not implemented yet (expected length-4 array)")
        return

    got = np.asarray(got)
    expected = (L != ABSTAIN).mean(axis=0)

    if got.shape != (4,):
        print(f"  {_FAIL} lf_coverage shape {got.shape}, expected (4,)")
        return

    if _close(got, expected):
        print(f"  {_OK} lf_coverage: {[round(float(x), 2) for x in got]}")
        return

    # Common mistake: counted total entries instead of per-LF
    if _close(np.full(4, expected.mean()), got):
        print(f"  {_FAIL} lf_coverage: returned the GLOBAL coverage, expected per-LF")
        print(f"       Hint: the result has one entry per LF — average over the rows (axis=0).")
        return

    # Inverted (counted abstains)
    inv = (L == ABSTAIN).mean(axis=0)
    if _close(got, inv):
        print(f"  {_FAIL} lf_coverage: returned the ABSTAIN fraction (= 1 − coverage)")
        print(f"       Hint: coverage is the fraction where the LF did NOT abstain.")
        return

    print(f"  {_FAIL} lf_coverage: numerical mismatch (got {got}, expected {expected})")
    print(f"       Hint: coverage[i] = mean(L[:, i] != ABSTAIN).")


# ---------------------------------------------------------------------------
# 🛠️  lf_empirical_accuracy(L_dev, y_dev)
# ---------------------------------------------------------------------------
def check_lf_empirical_accuracy(fn):
    rng = np.random.default_rng(1)
    n, n_lfs = 60, 3
    y = rng.integers(0, 2, size=n)
    # LF 0 is correct 80% of non-abstains; LF 1 is 50% (random); LF 2 abstains a lot
    L = np.full((n, n_lfs), ABSTAIN, dtype=int)
    for i in range(n):
        for j in range(n_lfs):
            if j == 0 and rng.random() < 0.7:
                L[i, j] = y[i] if rng.random() < 0.8 else 1 - y[i]
            elif j == 1 and rng.random() < 0.7:
                L[i, j] = rng.integers(0, 2)
            elif j == 2 and rng.random() < 0.3:
                L[i, j] = y[i]

    got = fn(L, y)
    if got is None:
        print(f"  {_NONE} lf_empirical_accuracy: not implemented yet")
        return

    got = np.asarray(got, dtype=float)
    expected = np.empty(n_lfs)
    for j in range(n_lfs):
        m = L[:, j] != ABSTAIN
        expected[j] = (L[m, j] == y[m]).mean() if m.any() else float('nan')

    if got.shape != (n_lfs,):
        print(f"  {_FAIL} lf_empirical_accuracy shape {got.shape}, expected {(n_lfs,)}")
        return

    if _close(got, expected, tol=1e-6):
        print(f"  {_OK} lf_empirical_accuracy: {[round(float(x), 2) for x in got]}")
        return

    # Mistake: included abstains as 'wrong'
    naive = np.empty(n_lfs)
    for j in range(n_lfs):
        naive[j] = (L[:, j] == y).mean()
    if _close(got, naive, tol=1e-6):
        print(f"  {_FAIL} lf_empirical_accuracy: counted abstains as wrong")
        print(f"       Hint: condition on non-abstain — mean(L[mask, i] == y[mask]) "
              f"where mask = L[:, i] != ABSTAIN.")
        return

    print(f"  {_FAIL} lf_empirical_accuracy: got {got}, expected {expected}")


# ---------------------------------------------------------------------------
# 🧮  majority_vote(L)
# ---------------------------------------------------------------------------
def check_majority_vote(fn):
    L = np.array([
        [1, 1, 0, -1],     # 2 vs 1, abstain ignored -> 1
        [0, 0, 1, 0],      # 3 vs 1 -> 0
        [-1, -1, 1, -1],   # only 1 vote -> 1
        [-1, -1, -1, -1],  # all abstain -> ABSTAIN
        [1, 0, 1, 0],      # tie 2-2 -> ABSTAIN or either (we accept ABSTAIN, 0, or 1)
    ], dtype=int)

    got = fn(L)
    if got is None:
        print(f"  {_NONE} majority_vote: not implemented yet (expected length-{len(L)})")
        return

    got = np.asarray(got)
    if got.shape != (len(L),):
        print(f"  {_FAIL} majority_vote shape {got.shape}, expected {(len(L),)}")
        return

    expected_strict = [1, 0, 1, ABSTAIN]
    for i in range(4):
        if got[i] != expected_strict[i]:
            print(f"  {_FAIL} majority_vote[{i}] = {got[i]}, expected {expected_strict[i]}")
            print(f"       Hint: for each row count votes for 0 and 1 separately, "
                  f"return argmax; ABSTAIN if both counts are 0.")
            return

    # Tie row: any of {ABSTAIN, 0, 1} acceptable (you just need to make ONE choice)
    if got[4] not in (ABSTAIN, 0, 1):
        print(f"  {_FAIL} majority_vote[tie] returned {got[4]}, expected one of -1/0/1")
        return

    print(f"  {_OK} majority_vote: {got.tolist()}")


# ---------------------------------------------------------------------------
# 🧮  dawid_skene_em(L, n_iters)
# ---------------------------------------------------------------------------
def check_dawid_skene_em(fn):
    """Synthetic ground truth: 3 LFs with accuracies 0.85, 0.75, 0.60.
    Run EM for a handful of steps; the recovered accuracies should be roughly right
    and the soft posteriors at confident examples should be near 0 or 1."""
    rng = np.random.default_rng(2)
    n = 200
    n_lfs = 3
    accs = np.array([0.85, 0.75, 0.60])
    cov  = np.array([0.7, 0.5, 0.6])     # how often each LF fires
    y = rng.integers(0, 2, size=n)

    L = np.full((n, n_lfs), ABSTAIN, dtype=int)
    for i in range(n):
        for j in range(n_lfs):
            if rng.random() < cov[j]:
                if rng.random() < accs[j]:
                    L[i, j] = y[i]
                else:
                    L[i, j] = 1 - y[i]

    got = fn(L, 30)
    if got is None:
        print(f"  {_NONE} dawid_skene_em: not implemented yet")
        return

    if not (isinstance(got, tuple) and len(got) == 2):
        print(f"  {_FAIL} dawid_skene_em: expected a tuple (alphas, soft_posteriors)")
        return

    alphas, soft = got
    alphas = np.asarray(alphas, dtype=float)
    soft   = np.asarray(soft,   dtype=float)

    if alphas.shape != (n_lfs,):
        print(f"  {_FAIL} alphas shape {alphas.shape}, expected ({n_lfs},)")
        return
    if soft.shape != (n,):
        print(f"  {_FAIL} soft posteriors shape {soft.shape}, expected ({n},)")
        return

    if (soft.min() < -1e-6) or (soft.max() > 1 + 1e-6):
        print(f"  {_FAIL} soft posteriors out of [0, 1]: range "
              f"[{soft.min():.3f}, {soft.max():.3f}]")
        return

    # Accuracies should be roughly in the right order: alpha[0] >= alpha[2]
    if not (alphas[0] >= alphas[2] - 0.05):
        print(f"  {_FAIL} dawid_skene_em: recovered accuracies {alphas} do not "
              f"reflect the LF quality ordering (LF 0 is the most accurate by far)")
        print(f"       Hint: at the M-step, each LF's accuracy is the agreement "
              f"between its non-abstain votes and the current soft posterior.")
        return

    # Soft posteriors should track ground truth
    pred = (soft > 0.5).astype(int)
    soft_acc = float((pred == y).mean())
    if soft_acc < 0.78:
        print(f"  {_FAIL} dawid_skene_em: soft posteriors do not track ground truth "
              f"(argmax accuracy = {soft_acc:.2f}, expected > 0.78)")
        return

    print(f"  {_OK} dawid_skene_em: recovered accuracies {[round(float(a), 2) for a in alphas]} "
          f"(true {[round(float(a), 2) for a in accs]}); argmax acc {soft_acc:.2f}")


# ---------------------------------------------------------------------------
# 🎓  soft_cross_entropy(soft_labels, model_probs)
# ---------------------------------------------------------------------------
def check_soft_cross_entropy(fn):
    rng = np.random.default_rng(3)
    n = 50
    soft  = rng.uniform(0.1, 0.9, size=n)
    probs = rng.uniform(0.1, 0.9, size=n)

    got = fn(soft, probs)
    if got is None:
        print(f"  {_NONE} soft_cross_entropy: not implemented yet")
        return

    expected = -np.mean(soft * np.log(probs) + (1 - soft) * np.log(1 - probs))

    if abs(float(got) - expected) < 1e-5:
        print(f"  {_OK} soft_cross_entropy = {float(got):.4f}  (reference {expected:.4f})")
        return

    # Forgot the (1 - soft) * log(1 - probs) term
    one_sided = -np.mean(soft * np.log(probs))
    if abs(float(got) - one_sided) < 1e-5:
        print(f"  {_FAIL} soft_cross_entropy: only used the y=1 branch")
        print(f"       Hint: binary CE has TWO terms — soft*log(p) AND (1-soft)*log(1-p).")
        return

    # Used hard labels (argmax)
    hard = (soft > 0.5).astype(float)
    hard_ce = -np.mean(hard * np.log(probs) + (1 - hard) * np.log(1 - probs))
    if abs(float(got) - hard_ce) < 1e-5:
        print(f"  {_FAIL} soft_cross_entropy: thresholded the soft labels first")
        print(f"       Hint: the WHOLE point is to keep the soft target as a probability — "
              f"don't argmax it.")
        return

    # Forgot to negate
    if abs(float(got) + expected) < 1e-5:
        print(f"  {_FAIL} soft_cross_entropy missing the negative sign")
        return

    print(f"  {_FAIL} soft_cross_entropy = {float(got):.4f}, expected {expected:.4f}")


# ---------------------------------------------------------------------------
# 🔍  pick_most_uncertain(model_probs)
# ---------------------------------------------------------------------------
def check_pick_most_uncertain(fn):
    probs = np.array([0.05, 0.95, 0.5, 0.51, 0.999, 0.49])

    got = fn(probs)
    if got is None:
        print(f"  {_NONE} pick_most_uncertain: not implemented yet")
        return

    if not isinstance(got, (int, np.integer)):
        print(f"  {_FAIL} pick_most_uncertain returned {type(got).__name__}, expected int")
        return

    # The peak entropy is at probs == 0.5; index 2 is the closest.
    expected = 2

    if int(got) == expected:
        print(f"  {_OK} pick_most_uncertain: index {int(got)}  (probs[idx] = {probs[got]:.3f})")
        return

    # Inverted (picked the most CONFIDENT)
    inverted = int(np.argmax(np.abs(probs - 0.5)))
    if int(got) == inverted:
        print(f"  {_FAIL} pick_most_uncertain: returned the most CONFIDENT index")
        print(f"       Hint: most uncertain = closest to 0.5; use argmin(|p - 0.5|) "
              f"or argmax of binary entropy.")
        return

    print(f"  {_FAIL} pick_most_uncertain returned {int(got)}, expected {expected}")
