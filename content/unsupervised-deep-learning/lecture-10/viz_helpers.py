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


# ---------------------------------------------------------------------------
# 🧮  per-example MV vs EM comparison on test images
# ---------------------------------------------------------------------------

def show_label_model_predictions(X_test, y_test, mv_test, soft_test,
                                 n_each=4, seed=0):
    """Side-by-side per-example predictions of majority vote and EM.

    Top row: examples majority vote refused to label (no LF agreed strongly
    enough), where EM is the only signal. Bottom row: examples majority
    vote did vote on. Title colour: teal if EM's argmax matches the
    ground truth, terra if it doesn't.
    """
    rng = np.random.default_rng(seed)
    abstain_idx = np.where(mv_test == ABSTAIN)[0]
    voted_idx   = np.where(mv_test != ABSTAIN)[0]

    rows = [
        ('MV abstains', abstain_idx, _TEXT),
        ('MV votes',    voted_idx,   _GOLDEN),
    ]

    _, axes = plt.subplots(2, n_each, figsize=(1.6 * n_each, 3.6))
    for r, (row_label, pool, label_colour) in enumerate(rows):
        n = min(n_each, len(pool))
        pick = (rng.choice(pool, size=n, replace=False)
                if n else np.array([], dtype=int))
        for c in range(n_each):
            ax = axes[r, c]
            if c < n:
                i = int(pick[c])
                ax.imshow(X_test[i].reshape(8, 8), cmap='gray_r')
                true = int(y_test[i])
                mv = int(mv_test[i])
                dse = float(soft_test[i])
                dse_pred = int(dse > 0.5)
                mv_str = 'abstain' if mv == ABSTAIN else str(mv)
                col = _ACCENT if dse_pred == true else _TERRA
                ax.set_title(f'true {true}\nMV  {mv_str}\nEM  {dse:.2f}',
                             fontsize=8, color=col)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.text(-0.5, 0.5, row_label, transform=ax.transAxes,
                        fontsize=9, color=label_colour, ha='right', va='center')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🛠️  task hardness: why 4 vs 9 is the confusable pair
# ---------------------------------------------------------------------------

def show_task_hardness(X, y, n_each=6, seed=0):
    """Show real digits side by side so the student SEES the confusion.

    Top row: examples labelled 0 (digit 4); bottom row: examples labelled 1 (digit 9).
    The right-most column shows each class's mean image — the average 4 and the
    average 9 differ mostly at the top loop, which is exactly the structure
    ``lf_top_loop`` exploits.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, n_each + 1,
                             figsize=(1.2 * (n_each + 1), 2.8))

    classes = [(0, 'class 0  (digit 4)', _TERRA),
               (1, 'class 1  (digit 9)', _ACCENT)]

    for r, (c, label, colour) in enumerate(classes):
        idx = rng.choice(np.where(y == c)[0],
                         size=min(n_each, int((y == c).sum())),
                         replace=False)
        for k in range(n_each):
            ax = axes[r, k]
            if k < len(idx):
                ax.imshow(X[idx[k]].reshape(8, 8), cmap='gray_r')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if k == 0:
                ax.text(-0.45, 0.5, label, transform=ax.transAxes,
                        fontsize=9, color=colour, ha='right', va='center')

        # Mean image in the rightmost column
        ax = axes[r, n_each]
        mean_img = X[y == c].mean(axis=0).reshape(8, 8)
        ax.imshow(mean_img, cmap='gray_r')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0:
            ax.set_title('class mean', fontsize=8, color=_GOLDEN)

    fig.suptitle('the labelling problem: 4s and 9s share an open/closed top loop',
                 fontsize=10, color=_GOLDEN, y=1.04)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🛠️  the label matrix L itself, as a heatmap
# ---------------------------------------------------------------------------

def plot_label_matrix(L, lf_names, max_rows=80, sort_rows=True,
                      title='Label matrix L  (rows = pool examples, columns = LFs)'):
    """Render the (n_samples, n_lfs) matrix as a categorical heatmap.

    Three colours: red = vote 0, teal = vote 1, dim = ABSTAIN.  Sorting rows
    by total non-abstain votes makes the coverage structure visible at a glance:
    the top of the figure has rows where every LF fired, the bottom has rows
    almost no LF covered.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    L = np.asarray(L)
    if sort_rows:
        order = np.argsort(-(L != ABSTAIN).sum(axis=1))
        L = L[order]
    if max_rows is not None and L.shape[0] > max_rows:
        # Even sampling so the row order (high-coverage at top, low at bottom) is preserved
        idx = np.linspace(0, L.shape[0] - 1, max_rows).astype(int)
        L = L[idx]

    # Map {-1, 0, 1} -> {0, 1, 2} for ListedColormap
    M = (L + 1).astype(int)   # ABSTAIN(-1)->0, vote 0->1, vote 1->2
    cmap = ListedColormap(['#1d2c33', _TERRA, _ACCENT])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    n_lfs = L.shape[1]
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n_lfs, 5.0))
    ax.imshow(M, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(n_lfs))
    ax.set_xticklabels(lf_names, fontsize=8, rotation=20, ha='right')
    ax.set_yticks([])
    ax.set_xlabel('labelling function')
    ax.set_ylabel(f'pool example  ({L.shape[0]} shown, sorted by # votes)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_ACCENT,  label='vote = 1'),
        Patch(facecolor=_TERRA,   label='vote = 0'),
        Patch(facecolor='#1d2c33', edgecolor=_BORDER, label='ABSTAIN'),
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=_TEXT,
              loc='upper right', bbox_to_anchor=(1.02, 1.10), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🛠️  precision-coverage scatter
# ---------------------------------------------------------------------------

def plot_coverage_accuracy(coverages, accuracies, lf_names,
                           title='Coverage vs accuracy  (each dot = one LF)'):
    """The precision-coverage trade-off, one dot per LF."""
    coverages = np.asarray(coverages, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.5, color=_TEXT, linewidth=0.6, alpha=0.4, linestyle='--')
    ax.text(0.02, 0.51, 'random  (50%)', fontsize=8, color=_TEXT, alpha=0.8)
    ax.axhline(1.0, color=_TEXT, linewidth=0.6, alpha=0.3, linestyle=':')
    ax.text(0.02, 1.01, 'perfect  (100%)', fontsize=8, color=_TEXT, alpha=0.8)

    # Color each LF differently for visual identity
    palette = [_ACCENT, _GOLDEN, _TERRA, _ORANGE, _SAGE, _STEEL, _LAVENDER]
    for i, (c, a, name) in enumerate(zip(coverages, accuracies, lf_names)):
        if np.isnan(a):
            continue
        col = palette[i % len(palette)]
        ax.scatter([c], [a], s=110, color=col, edgecolors='none', zorder=3)
        ax.annotate(name, (c, a), xytext=(8, 0), textcoords='offset points',
                    fontsize=9, color=col, va='center')

    ax.set_xlim(0, 1.05); ax.set_ylim(0.35, 1.08)
    ax.set_xlabel('coverage  (fraction of inputs the LF voted on)')
    ax.set_ylabel('accuracy  (fraction correct when it voted)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🧮  EM label model (black box)
# ---------------------------------------------------------------------------

def _em_label_model_trace(L, n_iters=30, init_alpha=0.7):
    """One-coin agreement-based label model fitted by EM.

    Returns (history, soft_pos):
      history  : (n_iters + 1, n_lfs) array of per-LF alpha at each iteration
      soft_pos : (n_samples,) final P(y = 1 | L) for every row.
    """
    L = np.asarray(L)
    n, n_lfs = L.shape
    alphas = np.full(n_lfs, init_alpha, dtype=np.float64)
    history = [alphas.copy()]
    soft_pos = np.full(n, 0.5)
    soft_history = [soft_pos.copy()]
    for _ in range(n_iters):
        log_odds = np.zeros(n)
        for i in range(n_lfs):
            a = float(np.clip(alphas[i], 1e-3, 1 - 1e-3))
            vote = L[:, i]
            log_odds += np.where(vote == 1, np.log(a) - np.log(1 - a), 0.0)
            log_odds += np.where(vote == 0, np.log(1 - a) - np.log(a), 0.0)
        soft_pos = 1.0 / (1.0 + np.exp(-log_odds))
        for i in range(n_lfs):
            mask = L[:, i] != ABSTAIN
            if not mask.any():
                continue
            agree = np.where(L[mask, i] == 1, soft_pos[mask], 1.0 - soft_pos[mask])
            alphas[i] = float(np.clip(agree.mean(), 0.5 + 1e-3, 1 - 1e-3))
        history.append(alphas.copy())
        soft_history.append(soft_pos.copy())
    return np.array(history), soft_pos, np.array(soft_history)


def em_label_model(L, n_iters: int = 30):
    """Black-box agreement-based label model.

    Takes the label matrix ``L`` of shape ``(n_samples, n_lfs)`` and returns:
      alphas   : (n_lfs,) each LF's inferred accuracy
      soft_pos : (n_samples,) P(y = 1 | L) for every example

    The implementation is a one-coin model fitted by expectation-maximisation,
    initialised at alpha = 0.7 to break the symmetric saddle at 0.5. Students
    consume this as a black box; the convergence and posterior plots in the
    notebook show what it produces.
    """
    history, soft_pos, _ = _em_label_model_trace(L, n_iters=n_iters)
    return history[-1], soft_pos


def plot_em_label_model_trace(L, lf_names, n_iters=30, dev_accuracies=None,
                              title='EM label model: per-LF accuracy estimate over iterations'):
    """Plot how each LF's recovered accuracy moves over EM iterations.

    With no labelled training data, EM still converges to roughly the right
    per-LF accuracies, driven only by the agreement structure of L.  If
    ``dev_accuracies`` is provided (the empirical accuracies on the dev set),
    they are drawn as horizontal references — the line that EM is approaching.
    """
    history, _, _ = _em_label_model_trace(L, n_iters=n_iters)
    n_lfs = history.shape[1]
    palette = [_ACCENT, _GOLDEN, _TERRA, _ORANGE, _SAGE, _STEEL, _LAVENDER]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(history.shape[0])
    for i in range(n_lfs):
        col = palette[i % len(palette)]
        ax.plot(xs, history[:, i], color=col, marker='o', markersize=3,
                linewidth=1.4, label=f'{lf_names[i]}', markeredgecolor='none')
        if dev_accuracies is not None and not np.isnan(dev_accuracies[i]):
            ax.axhline(dev_accuracies[i], color=col, linewidth=0.6,
                       linestyle='--', alpha=0.5)
    ax.axhline(0.5, color=_TEXT, linewidth=0.4, alpha=0.4)
    ax.set_xlabel('EM iteration')
    ax.set_ylabel('inferred accuracy  (alpha)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right', fontsize=9)
    if dev_accuracies is not None:
        ax.text(0.02, 0.96, 'dashed line = empirical accuracy on dev set',
                transform=ax.transAxes, fontsize=8, color=_TEXT, alpha=0.7,
                va='top')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🧮  soft posterior histogram
# ---------------------------------------------------------------------------

def plot_soft_posteriors(soft_pos, y_true,
                         title='Soft posterior P(y=1 | L) per pool example'):
    """Histogram of the label model's per-example soft posterior, split by truth.

    A healthy label model concentrates probability near 0 for true-0 examples
    and near 1 for true-1 examples, with a thin band of genuine uncertainty
    in between.  Examples in the middle band are exactly the ones an active
    learner would query first.
    """
    soft_pos = np.asarray(soft_pos)
    y_true   = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(soft_pos[y_true == 0], bins=bins, color=_TERRA,
            alpha=0.65, edgecolor='none', label='true class 0')
    ax.hist(soft_pos[y_true == 1], bins=bins, color=_ACCENT,
            alpha=0.65, edgecolor='none', label='true class 1')
    ax.axvline(0.5, color=_TEXT, linewidth=0.6, alpha=0.5, linestyle='--')
    ax.text(0.51, ax.get_ylim()[1] * 0.92, 'argmax threshold',
            fontsize=8, color=_TEXT, alpha=0.7)
    ax.set_xlabel('soft posterior P(y=1 | L)')
    ax.set_ylabel('count')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='upper center')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_em_soft_pos_evolution(L, y_true, n_iters: int = 30,
                               snapshots=(0, 2, 5, 30),
                               title='Soft posterior P(y=1 | L) sharpening over EM iterations'):
    """Show how the per-example soft posterior moves from flat (iter 0) toward
    bimodal (iter ``n_iters``).

    One histogram panel per snapshot iteration, split by held-back ground truth.
    At iteration 0 every example sits at 0.5 (the flat prior); after a few
    iterations the two classes pull apart. Pedagogical companion to the
    per-LF accuracy trace, showing the same EM fit from the *example* side
    rather than the *LF* side. Each panel scales its own y-axis so the
    iteration-0 spike at 0.5 does not dwarf the spread-out later panels.
    """
    _, _, soft_history = _em_label_model_trace(L, n_iters=n_iters)
    y_true = np.asarray(y_true)
    snapshots = [int(s) for s in snapshots if 0 <= int(s) <= n_iters]

    n_panels = len(snapshots)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.0),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)
    for ax, k in zip(axes, snapshots):
        soft = soft_history[k]
        ax.hist(soft[y_true == 0], bins=bins, color=_TERRA,
                alpha=0.65, edgecolor='none', label='true class 0')
        ax.hist(soft[y_true == 1], bins=bins, color=_ACCENT,
                alpha=0.65, edgecolor='none', label='true class 1')
        ax.axvline(0.5, color=_TEXT, linewidth=0.4, alpha=0.4, linestyle='--')
        ax.set_title(f'iter {k}', fontsize=10, color=_TEXT, loc='left')
        ax.set_xlabel('P(y=1 | L)')
        tufte_axis(ax)

        # Iter 0 is one tall bar at 0.5 — call that out so the visual
        # is unmistakable.
        if k == 0:
            n_total = len(soft)
            ax.text(0.5, 0.92,
                    f'all {n_total} examples\nstart at 0.5',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=8, color=_TEXT, alpha=0.85)

    axes[0].set_ylabel('count')
    axes[-1].legend(frameon=False, labelcolor=_TEXT, loc='upper center',
                    fontsize=8)
    fig.suptitle(title, fontsize=10, color=_GOLDEN, x=0.02, ha='left')
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


# ---------------------------------------------------------------------------
# 🎓  verifier-generator gap as a bar chart
# ---------------------------------------------------------------------------

def plot_verifier_generator_gap(L, X, y, predict_fn, lf_total=None,
                                title='End-model accuracy vs number of LFs that fired'):
    """Bar chart of end-model accuracy vs # LFs that fired on each example.

    The verifier-generator gap is the chapter punchline: even on examples no
    LF fired on, the trained end model can still classify them correctly
    because it operates on the input features rather than the LF outputs.
    """
    L = np.asarray(L)
    n_votes = (L != ABSTAIN).sum(axis=1)
    if lf_total is None:
        lf_total = L.shape[1]

    preds = (predict_fn(X) > 0.5).astype(int)
    correct = (preds == y).astype(int)

    buckets = list(range(lf_total + 1))
    accs   = []
    counts = []
    for k in buckets:
        m = n_votes == k
        counts.append(int(m.sum()))
        accs.append(float(correct[m].mean()) if m.any() else 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    palette = [_TERRA, _ORANGE, _GOLDEN, _ACCENT, _SAGE]
    bars = ax.bar(buckets, accs,
                  color=[palette[min(k, len(palette) - 1)] for k in buckets],
                  edgecolor='none')
    for k, (b, c) in enumerate(zip(bars, counts)):
        if c == 0:
            ax.text(b.get_x() + b.get_width() / 2, 0.02, 'no examples',
                    ha='center', fontsize=7, color=_TEXT, alpha=0.6)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f'{c}', ha='center', fontsize=8, color=_TEXT)
    ax.axhline(0.5, color=_TEXT, linewidth=0.4, alpha=0.4)

    ax.set_xticks(buckets)
    ax.set_xlabel('# LFs that voted on the example')
    ax.set_ylabel('end-model test accuracy')
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ⚖️  judge effect: how the soft posterior shifts when the judge is added
# ---------------------------------------------------------------------------

def plot_judge_effect(soft_before, soft_after, y_true,
                      title='Adding the judge: per-example soft posterior shift'):
    """Per-example scatter of soft posterior before vs after adding the judge.

    Points off the diagonal moved.  Up-and-right means the judge confirmed
    class 1; down-and-left means it confirmed class 0; movement against the
    true-label colour is where the judge OVERRULED the rest of the LFs.
    """
    soft_before = np.asarray(soft_before)
    soft_after  = np.asarray(soft_after)
    y_true      = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color=_TEXT, linewidth=0.6, alpha=0.5,
            linestyle='--', label='unchanged')
    ax.scatter(soft_before[y_true == 0], soft_after[y_true == 0],
               s=22, color=_TERRA, alpha=0.7, edgecolors='none',
               label='true class 0')
    ax.scatter(soft_before[y_true == 1], soft_after[y_true == 1],
               s=22, color=_ACCENT, alpha=0.7, edgecolors='none',
               label='true class 1')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('soft posterior  (3 LFs)')
    ax.set_ylabel('soft posterior  (3 LFs + judge)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='upper left')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 🔍  active-learning picks: show the digits the picker actually chose
# ---------------------------------------------------------------------------

def show_active_learning_picks(X, queried_indices, y, soft_pos=None,
                               n_show=8, title=None):
    """Render the first n_show digits the active learner asked the oracle for.

    With uncertainty sampling, these *should* be the visually-confusable
    examples the LFs disagreed on most.  If ``soft_pos`` is provided, each
    picked example is annotated with its label-model posterior at pick time.
    """
    if title is None:
        title = (f'first {min(n_show, len(queried_indices))} examples picked by uncertainty sampling'
                 '   (most uncertain first)')
    n = min(n_show, len(queried_indices))
    if n == 0:
        print('  no queries made — picker returned None?')
        return

    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.7 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for k, idx in enumerate(queried_indices[:n]):
        ax = axes[k // cols, k % cols]
        ax.imshow(X[idx].reshape(8, 8), cmap='gray_r')
        true_lab = int(y[idx])
        if soft_pos is not None:
            ax.set_title(f'#{k+1}  true {true_lab}\np={soft_pos[idx]:.2f}',
                         fontsize=8, color=_GOLDEN if true_lab == 1 else _TERRA)
        else:
            ax.set_title(f'#{k+1}  true {true_lab}',
                         fontsize=8, color=_GOLDEN if true_lab == 1 else _TERRA)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis('off')

    fig.suptitle(title, fontsize=10, color=_GOLDEN, y=1.02)
    plt.tight_layout()
    plt.show()
