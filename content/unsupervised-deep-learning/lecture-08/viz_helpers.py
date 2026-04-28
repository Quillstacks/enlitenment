"""Visualization and optimization helpers for lecture-08 (Transfer Learning).

These helpers hide the longer plotting and optimisation plumbing so the
notebook cells can stay short and conceptual. They expect the same Tufte
dark-theme palette imported from ``plot_style``.
"""

import sys
sys.path.insert(0, '../..')
from plot_style import *  # noqa: F401,F403

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# K-shot sweep (cell 15)
# ---------------------------------------------------------------------------

def kshot_sweep(encode_fn, linear_probe_fn, backbone,
                X_tgt_pool, y_tgt_pool, X_tgt_test, y_tgt_test,
                Ks=(1, 2, 5, 10, 20, 50), n_repeats=5):
    """Compare frozen-backbone+probe vs train-from-scratch over a sweep of K.

    Returns two arrays of shape (len(Ks), n_repeats): probe and scratch
    accuracies. Plotting is left to ``plot_kshot_sweep``.
    """
    Ks = list(Ks)
    acc_probe = np.zeros((len(Ks), n_repeats))
    acc_scratch = np.zeros((len(Ks), n_repeats))
    emb_test = encode_fn(X_tgt_test, backbone)

    for ki, K in enumerate(Ks):
        for r in range(n_repeats):
            rng = np.random.default_rng(100 + r)
            idx = []
            for c in (0, 1):
                ci = np.where(y_tgt_pool == c)[0]
                idx.extend(rng.choice(ci, size=K, replace=False))
            idx = np.array(idx)
            Xs, ys = X_tgt_pool[idx], y_tgt_pool[idx]

            acc_probe[ki, r] = linear_probe_fn(
                encode_fn(Xs, backbone), ys, emb_test, y_tgt_test
            )
            scratch = MLPClassifier(
                hidden_layer_sizes=(32, 16), activation='relu',
                solver='adam', learning_rate_init=1e-3,
                max_iter=300, random_state=r,
            ).fit(Xs, ys)
            acc_scratch[ki, r] = scratch.score(X_tgt_test, y_tgt_test)
    return acc_probe, acc_scratch


def plot_kshot_sweep(Ks, acc_probe, acc_scratch):
    """Errorbar comparison of frozen probe vs from-scratch across K."""
    Ks = list(Ks)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(Ks, acc_probe.mean(axis=1), yerr=acc_probe.std(axis=1),
                color=_ACCENT, marker='o', linewidth=1.4, capsize=3,
                label='frozen backbone + probe')
    ax.errorbar(Ks, acc_scratch.mean(axis=1), yerr=acc_scratch.std(axis=1),
                color=_TERRA, marker='s', linewidth=1.4, capsize=3,
                label='train MLP from scratch on raw pixels')
    ax.set_xscale('log')
    ax.set_xlabel('shots per class (K)')
    ax.set_ylabel('test accuracy on 0 vs 1')
    ax.set_title('K-shot transfer: frozen backbone vs from-scratch',
                 fontsize=10, color=_GOLDEN)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, labelcolor=_TEXT)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Mini-CLIP attribute encoder training (cell 32)
# ---------------------------------------------------------------------------

def train_clip_attribute_encoder(contrastive_loss_fn, attrs_source,
                                 emb_src, y_src, attr_names,
                                 n_pairs=300, tau=0.1, maxiter=60, seed=0):
    """Fit the linear attribute encoder W by L-BFGS-B on the contrastive loss.

    Returns ``(W_attr, w0, idx_pairs)`` so a downstream diagnostic cell can
    re-use the same initial weights and pair indices.
    """
    rng = np.random.default_rng(seed)
    A_train = np.array([attrs_source[int(y)] for y in y_src], dtype=np.float64)

    n_total = len(emb_src)
    n_pairs = min(n_pairs, n_total)
    idx_pairs = rng.choice(n_total, size=n_pairs, replace=False)
    img_pairs = emb_src[idx_pairs]
    attr_pairs = A_train[idx_pairs]

    n_attrs, d_emb = len(attr_names), emb_src.shape[1]

    def wrapped_loss(w_flat):
        W = w_flat.reshape(n_attrs, d_emb)
        text = attr_pairs @ W
        text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)
        return float(contrastive_loss_fn(img_pairs, text, tau=tau))

    w0 = rng.normal(size=n_attrs * d_emb) * 0.1

    # Probe that the user's loss returns a number, not None.
    probe = contrastive_loss_fn(
        img_pairs, attr_pairs @ w0.reshape(n_attrs, d_emb), tau=tau,
    )
    if probe is None:
        print('⬜ Implement contrastive_loss above first.')
        return None, w0, idx_pairs

    print(f'Pairs: img embs {emb_src.shape},  attr vectors {A_train.shape}')
    print(f'Initial loss : {wrapped_loss(w0):.4f}')
    result = minimize(wrapped_loss, w0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'disp': False})
    W_attr = result.x.reshape(n_attrs, d_emb)
    print(f'Final loss   : {wrapped_loss(result.x):.4f}')
    print(f'Iterations   : {result.nit}')
    return W_attr, w0, idx_pairs


def plot_similarity_before_after(emb_src, A_train, w0, W_attr, y_src):
    """Image-vs-attribute similarity matrices on a one-per-class diagnostic batch."""
    classes = sorted(set(y_src.tolist() if hasattr(y_src, 'tolist') else y_src))
    diag_idx = np.array([np.where(y_src == c)[0][0] for c in classes])
    img_diag = emb_src[diag_idx]
    attr_diag = A_train[diag_idx]

    n_attrs, d_emb = W_attr.shape

    def sim_with(W):
        text = attr_diag @ W
        text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)
        return img_diag @ text.T

    S_before = sim_with(w0.reshape(n_attrs, d_emb))
    S_after = sim_with(W_attr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, S, title in [(axes[0], S_before, 'Before training'),
                         (axes[1], S_after, 'After training')]:
        im = ax.imshow(S, cmap='magma', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('attribute (class)')
        ax.set_ylabel('image (class)')
        ax.set_title(title, fontsize=10, color=_GOLDEN)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Compose-the-pieces query (cell 39)
# ---------------------------------------------------------------------------

def compose_query_visualize(W_attr, encode_fn, topk_retrieve_fn,
                            X_all, y_all, attr_names, query_attr,
                            backbone, k_retrieve=60, n_clusters=3, n_show=12):
    """Run an attribute query through the trained bridge, retrieve, cluster, show.

    ``encode_fn`` is the user's ``encode(X, mlp)``; ``backbone`` supplies the
    second argument so the helper can embed every digit in ``X_all``.
    """
    print('Query: ' + '  '.join(f'{n}={int(v)}'
                                 for n, v in zip(attr_names, query_attr)))

    query_emb = query_attr @ W_attr
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    emb_all = encode_fn(X_all, backbone)
    top = topk_retrieve_fn(query_emb, emb_all, k=k_retrieve)
    if top is None:
        print('⬜ Implement topk_retrieve above first.')
        return

    retrieved_labels = y_all[top]
    hist = Counter(retrieved_labels.tolist())
    print(f'Retrieved label histogram (top {k_retrieve}): '
          f'{dict(sorted(hist.items()))}')

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(emb_all[top])
    cluster_sizes = Counter(km.labels_.tolist())
    print(f'Cluster sizes : {dict(cluster_sizes)}')
    for c in sorted(cluster_sizes):
        mask = km.labels_ == c
        members = Counter(retrieved_labels[mask].tolist())
        print(f'  cluster {c}: {dict(sorted(members.items()))}')

    n_cols = 6
    n_rows = max(1, n_show // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.75 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes_flat, top[:n_show]):
        ax.imshow(X_all[idx].reshape(8, 8), cmap='magma')
        ax.set_title(str(int(y_all[idx])), fontsize=9, color=_GOLDEN)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    label_str = ', '.join(f'{n}={int(v)}'
                          for n, v in zip(attr_names, query_attr) if int(v) == 1)
    plt.suptitle(f'Top retrievals for "{label_str}"', y=1.02, color=_GOLDEN)
    plt.tight_layout()
    plt.show()
