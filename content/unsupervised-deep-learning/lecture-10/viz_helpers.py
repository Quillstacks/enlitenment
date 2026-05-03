"""Visualisation and dataset helpers for lecture-10 (Weak Supervision).

Sklearn-only, pyodide-compatible. The notebook uses ``digits`` restricted to
classes 0 and 1 — same task as chapter 8's K-shot transfer setup, but here
the supervision comes from labelling functions instead of a clean labelled
training set.
"""

from __future__ import annotations

import sys
sys.path.insert(0, '../..')
from plot_style import *  # noqa: F401,F403

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ABSTAIN = -1


# ---------------------------------------------------------------------------
# Dataset: binary digits (0 vs 1)
# ---------------------------------------------------------------------------

def load_binary_digits(seed: int = 0,
                       class_a: int = 4, class_b: int = 9):
    """Return (X, y, X_test, y_test, X_dev, y_dev).

    Picks the ``class_a`` / ``class_b`` subset of sklearn's digits and remaps
    those two labels to ``{0, 1}`` (``class_a`` -> 0, ``class_b`` -> 1) so the
    rest of the chapter machinery is binary. Default classes are 4 and 9 — a
    visually confusable pair on which weak supervision actually has work to do.

    Returns
    -------
    X, y           : training pool (~280 samples), labels in {0, 1}
    X_test, y_test : balanced test set (60 samples)
    X_dev, y_dev   : balanced labelled dev set (20 samples; for per-LF accuracy)
    """
    digits = load_digits()
    mask = (digits.target == class_a) | (digits.target == class_b)
    X_all = digits.data[mask] / 16.0
    y_all = (digits.target[mask] == class_b).astype(int)   # remap to {0, 1}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # Balanced test set: 30 of each class
    idx_test = []
    for c in (0, 1):
        cl = np.where(y_all == c)[0][:30]
        idx_test.extend(cl)
    idx_test = np.array(idx_test)
    pool_idx = np.array([i for i in range(len(X_all)) if i not in set(idx_test)])

    X_test, y_test = X_all[idx_test], y_all[idx_test]

    # Balanced dev set: 10 of each from the pool
    idx_dev = []
    for c in (0, 1):
        cl = pool_idx[y_all[pool_idx] == c][:10]
        idx_dev.extend(cl)
    idx_dev = np.array(idx_dev)
    train_idx = np.array([i for i in pool_idx if i not in set(idx_dev)])

    X_dev, y_dev = X_all[idx_dev], y_all[idx_dev]
    X, y = X_all[train_idx], y_all[train_idx]

    return X, y, X_test, y_test, X_dev, y_dev


# ---------------------------------------------------------------------------
# The five labelling functions
# ---------------------------------------------------------------------------

def lf_top_loop(X: np.ndarray) -> np.ndarray:
    """Mean intensity in the top-centre of an 8x8 image (rows 0-1, cols 2-5).

    A 9 has a closed loop at the top, so the top-centre region is filled.
    A 4 has an open top, so its top-centre region is dimmer.
    Empirical means on this dataset: 4 ~ 0.38, 9 ~ 0.62.
    Threshold: > 0.55 -> 9, < 0.32 -> 4, else abstain.
    """
    images = X.reshape(-1, 8, 8)
    top_centre = images[:, 0:2, 2:6].mean(axis=(1, 2))
    out = np.full(len(X), ABSTAIN, dtype=int)
    out[top_centre > 0.55] = 1   # label 1 = digit 9
    out[top_centre < 0.32] = 0   # label 0 = digit 4
    return out


def lf_left_loop_edge(X: np.ndarray) -> np.ndarray:
    """Mean intensity in the upper-left region (rows 0-2, cols 1-2).

    A 9's loop has a left edge that sits in cols 1-2; a 4's left vertical
    is more central (cols 2-3) and leaves cols 1-2 dimmer.  Empirical means
    on this dataset: 4 ~ 0.16, 9 ~ 0.38.
    Threshold: > 0.40 -> 9, < 0.10 -> 4, else abstain.
    """
    images = X.reshape(-1, 8, 8)
    left = images[:, 0:3, 1:3].mean(axis=(1, 2))
    out = np.full(len(X), ABSTAIN, dtype=int)
    out[left > 0.40] = 1   # label 1 = digit 9
    out[left < 0.10] = 0   # label 0 = digit 4
    return out


def lf_total_intensity(X: np.ndarray) -> np.ndarray:
    """Total ink. A weak auxiliary signal — 4s and 9s have nearly identical
    total ink (~19.5 vs ~19.7), so this LF is essentially random.  Kept for
    pedagogy: the label model should down-weight it once the agreement
    structure with the strong LFs reveals its near-zero accuracy.
    """
    total = X.sum(axis=1)
    out = np.full(len(X), ABSTAIN, dtype=int)
    out[total < 17.0] = 0   # very low ink -> 4 (very weak guess)
    out[total > 22.0] = 1   # very high ink -> 9 (very weak guess)
    return out


def lf_noisy_crowd(X: np.ndarray, y_true: np.ndarray, flip_rate: float = 0.20,
                   abstain_rate: float = 0.30, seed: int = 0) -> np.ndarray:
    """Simulated crowd annotator. Knows the true label most of the time but
    flips with probability ``flip_rate`` and abstains on a random subset."""
    rng = np.random.default_rng(seed)
    out = y_true.copy()
    flip = rng.random(len(y_true)) < flip_rate
    out[flip] = 1 - out[flip]
    abstain = rng.random(len(y_true)) < abstain_rate
    out[abstain] = ABSTAIN
    return out


def lf_pretrained(X: np.ndarray, X_pre: np.ndarray, y_pre: np.ndarray,
                  conf_threshold: float = 0.85) -> np.ndarray:
    """A small ``LogisticRegression`` pretrained on a 10-sample subset.

    Confidently overconfident on out-of-distribution shapes, abstains when
    its top-class probability is below ``conf_threshold``.
    """
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_pre, y_pre)
    probs = clf.predict_proba(X)
    out = np.full(len(X), ABSTAIN, dtype=int)
    confident = probs.max(axis=1) > conf_threshold
    out[confident] = probs.argmax(axis=1)[confident]
    return out


def _logistic_predict_proba(X, w, b):
    """Numerically-stable sigmoid for the binary logistic head."""
    z = X @ w + b
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_on_soft(X: np.ndarray, soft_pos: np.ndarray, soft_ce_fn,
                            n_iters: int = 50, l2: float = 1e-3, seed: int = 0):
    """Fit a logistic regression on soft labels using the student's
    ``soft_cross_entropy`` as the loss. Returns (w, b).

    ``soft_pos`` is the soft posterior P(y = 1 | input) provided by the label
    model — a real number in [0, 1] per example.  L-BFGS-B does the heavy
    lifting; the student function defines the objective.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    n = len(X)

    def pack(w, b):
        return np.concatenate([w, [b]])

    def unpack(theta):
        return theta[:-1], float(theta[-1])

    def loss(theta):
        w, b = unpack(theta)
        probs = _logistic_predict_proba(X, w, b)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        ce = float(soft_ce_fn(soft_pos, probs))
        if ce is None:
            return float('inf')
        return ce + 0.5 * l2 * float(w @ w)

    theta0 = pack(rng.normal(0, 0.01, size=d), 0.0)
    # Probe to fail fast when the student hasn't implemented the loss yet.
    probe = soft_ce_fn(soft_pos, _logistic_predict_proba(X, theta0[:-1], theta0[-1]))
    if probe is None:
        return None, None

    result = minimize(loss, theta0, method="L-BFGS-B",
                      options={"maxiter": n_iters, "disp": False})
    return unpack(result.x)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _logistic_predict_proba(X, w, b)


def run_active_learning(X_pool: np.ndarray, y_pool: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         soft_pos_init: np.ndarray, soft_ce_fn, picker_fn,
                         n_queries: int = 12, seed: int = 0):
    """Active-learning loop: alternate between training on the current soft labels
    and querying the oracle for the most-uncertain unqueried pool example.

    Returns (queried_indices, accuracies_per_step).
    """
    soft = soft_pos_init.copy()
    queried: list[int] = []
    accs: list[float] = []

    for step in range(n_queries + 1):
        w, b = train_logistic_on_soft(X_pool, soft, soft_ce_fn, n_iters=80, seed=seed)
        if w is None:
            return queried, accs
        accs.append(float(((predict_proba(X_test, w, b) > 0.5).astype(int) == y_test).mean()))

        if step == n_queries:
            break
        train_probs = predict_proba(X_pool, w, b)
        unqueried = np.array([i for i in range(len(X_pool)) if i not in set(queried)])
        local_idx = picker_fn(train_probs[unqueried])
        if local_idx is None:
            return queried, accs
        global_idx = int(unqueried[int(local_idx)])
        queried.append(global_idx)
        # The oracle gives the true label; we hard-set the soft label there.
        soft[global_idx] = float(y_pool[global_idx])

    return queried, accs


def run_random_baseline(X_pool: np.ndarray, y_pool: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         soft_pos_init: np.ndarray, soft_ce_fn,
                         n_queries: int = 12, seed: int = 0):
    """Same loop as run_active_learning but picks a random unqueried example."""
    rng = np.random.default_rng(seed)
    soft = soft_pos_init.copy()
    queried: list[int] = []
    accs: list[float] = []
    available = list(range(len(X_pool)))

    for step in range(n_queries + 1):
        w, b = train_logistic_on_soft(X_pool, soft, soft_ce_fn, n_iters=80, seed=seed)
        if w is None:
            return queried, accs
        accs.append(float(((predict_proba(X_test, w, b) > 0.5).astype(int) == y_test).mean()))

        if step == n_queries:
            break
        idx = int(rng.choice(available))
        available.remove(idx)
        queried.append(idx)
        soft[idx] = float(y_pool[idx])

    return queried, accs


def make_judge(X_pre: np.ndarray, y_pre: np.ndarray):
    """Stronger pretrained classifier — used as the LLM-as-judge stand-in.

    A small MLP trained on a larger held-out subset; same interface as the
    other LFs (returns 0 / 1 / ABSTAIN per sample) but higher coverage and
    higher accuracy than ``lf_pretrained`` on average.
    """
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                        random_state=0, learning_rate_init=1e-3)
    clf.fit(X_pre, y_pre)

    def judge_lf(X, conf_threshold: float = 0.70):
        probs = clf.predict_proba(X)
        out = np.full(len(X), ABSTAIN, dtype=int)
        confident = probs.max(axis=1) > conf_threshold
        out[confident] = probs.argmax(axis=1)[confident]
        return out

    return judge_lf


# ---------------------------------------------------------------------------
# Build the (n_samples, n_lfs) label matrix
# ---------------------------------------------------------------------------

def apply_lfs(X: np.ndarray, lfs) -> np.ndarray:
    """Apply each lf to X and stack into an (n_samples, n_lfs) integer matrix."""
    return np.stack([lf(X) for lf in lfs], axis=1)


# ---------------------------------------------------------------------------
# Plotting (the two plots the notebook actually uses)
# ---------------------------------------------------------------------------

def plot_active_learning_curve(budgets, accs_uncertain, accs_random,
                                title='Active learning vs random sampling'):
    """Accuracy as a function of oracle budget."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    accs_uncertain = np.asarray(accs_uncertain)
    accs_random    = np.asarray(accs_random)
    if accs_uncertain.ndim == 2:
        ax.plot(budgets, accs_uncertain.mean(axis=0), color=_ACCENT, marker='o',
                linewidth=1.4, label='uncertainty sampling')
        ax.fill_between(budgets,
                        accs_uncertain.mean(axis=0) - accs_uncertain.std(axis=0),
                        accs_uncertain.mean(axis=0) + accs_uncertain.std(axis=0),
                        color=_ACCENT, alpha=0.15)
        ax.plot(budgets, accs_random.mean(axis=0), color=_TERRA, marker='s',
                linewidth=1.4, label='random sampling')
        ax.fill_between(budgets,
                        accs_random.mean(axis=0) - accs_random.std(axis=0),
                        accs_random.mean(axis=0) + accs_random.std(axis=0),
                        color=_TERRA, alpha=0.15)
    else:
        ax.plot(budgets, accs_uncertain, color=_ACCENT, marker='o',
                linewidth=1.4, label='uncertainty sampling')
        ax.plot(budgets, accs_random, color=_TERRA, marker='s',
                linewidth=1.4, label='random sampling')
    ax.set_xlabel('oracle queries (cumulative)')
    ax.set_ylabel('test accuracy')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# A helper that visualises a single LF firing on actual digits
# ---------------------------------------------------------------------------


def show_lf_examples(X, y, lf_outputs, lf_name, n_each=4):
    """Show one row each of: predicted, abstained.  The point is the
    precision-coverage trade-off in pictures: where the LF fires vs where
    it stays silent.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, n_each, figsize=(1.4 * n_each, 3.0))

    def pick(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0: return []
        return rng.choice(idx, size=min(n_each, len(idx)), replace=False)

    rows = [(pick(lf_outputs != ABSTAIN), 'fires',     _ACCENT),
            (pick(lf_outputs == ABSTAIN), 'abstains', _TEXT  )]

    for r, (idx, label, colour) in enumerate(rows):
        for c in range(n_each):
            ax = axes[r, c]
            if c < len(idx):
                ax.imshow(X[idx[c]].reshape(8, 8), cmap='gray_r')
                if r == 0:
                    pred = int(lf_outputs[idx[c]])
                    ax.set_title(f'pred {pred}, true {int(y[idx[c]])}',
                                 fontsize=8, color=_GOLDEN)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if c == 0:
                ax.text(-0.5, 0.5, label, transform=ax.transAxes, fontsize=9,
                        color=colour, ha='right', va='center')
    fig.suptitle(f'{lf_name}', fontsize=10, color=_GOLDEN, y=1.02)
    plt.tight_layout()
    plt.show()
