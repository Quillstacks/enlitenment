"""Visualisation helpers for lecture 07 (Uncertainty Estimation).

Long matplotlib boilerplate lives here so the notebook cells stay short
and focused on the concept. All helpers honour the dark-theme Tufte style
defined in the shared `plot_style` module at the repo root.
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from plot_style import _ACCENT, _TERRA, _GOLDEN, _TEXT, tufte_axis


# ---------------------------------------------------------------------------
# Generic primitives
# ---------------------------------------------------------------------------

def hist_line(ax, x, bins, color, label, linewidth=1.8, alpha_fill=0.18, density=False):
    """Histogram drawn as a line over bin centers, with a light fill."""
    counts, edges = np.histogram(x, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, color=color, linewidth=linewidth, label=label)
    ax.fill_between(centers, 0, counts, color=color, alpha=alpha_fill, linewidth=0)


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def reshape_passes(df, prefix, n_passes, classes):
    """Reshape MC-Dropout pass columns into a (T, n, K) tensor."""
    out = np.empty((n_passes, len(df), len(classes)))
    for t in range(n_passes):
        for k, c in enumerate(classes):
            out[t, :, k] = df[f'{prefix}{t:02d}_c{c}'].values
    return out


# ---------------------------------------------------------------------------
# Section: The Confidence Trap
# ---------------------------------------------------------------------------

def plot_confidence_trap(images, df, indices=None):
    """Two-row image grid: input on top, model's confident-but-wrong call below."""
    if indices is None:
        odd = df[df['parity'] == 'odd'].sort_values('top1_softmax', ascending=False).head(10)
    else:
        odd = df[df['idx'].isin(indices)].sort_values('idx')

    pix_cols = [f'pixel_{j}' for j in range(784)]
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.6))
    for col, (_, row) in enumerate(odd.iterrows()):
        img = row[pix_cols].values.astype(float).reshape(28, 28)
        axes[0, col].imshow(img, cmap='magma')
        axes[0, col].set_title(f"true: {int(row['label'])}", fontsize=9, color=_GOLDEN)
        axes[1, col].imshow(img, cmap='magma', alpha=0.25)
        axes[1, col].text(14, 14, str(int(row['top1_label'])), ha='center', va='center',
                          fontsize=22, color=_ACCENT, fontweight='bold')
        axes[1, col].set_title(f"p = {row['top1_softmax']:.2f}", fontsize=9, color=_ACCENT)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    axes[0, 0].set_ylabel('input\n(unknown)', color=_TEXT)
    axes[1, 0].set_ylabel('model says\n(known label)', color=_TEXT)
    fig.suptitle('All inputs are odd digits, none seen during training', fontsize=10, color=_GOLDEN, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section: Calibration
# ---------------------------------------------------------------------------

def plot_reliability_panels(panels, n_bins=10, suptitle=None):
    """Side-by-side reliability diagrams. Each panel: dict with keys
    'title', 'bin_conf', 'bin_acc', 'bin_count'. ECE is computed per panel
    and shown in its title.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    edges   = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = 1.0 / n_bins

    for ax, panel in zip(axes, panels):
        bin_conf  = np.asarray(panel['bin_conf'],  dtype=float)
        bin_acc   = np.asarray(panel['bin_acc'],   dtype=float)
        bin_count = np.asarray(panel['bin_count'], dtype=float)
        ne        = bin_count > 0
        total     = bin_count.sum()
        gap       = np.where(ne, np.abs(bin_acc - bin_conf), 0.0)
        ece       = float(np.nansum(gap * bin_count) / total) if total > 0 else 0.0

        ax.plot([0, 1], [0, 1], color=_TEXT, linewidth=1.0, alpha=0.6,
                linestyle='--', label='perfect (acc = conf)')
        ax.bar(centers[ne], bin_acc[ne], width=width * 0.9,
               color=_ACCENT, alpha=0.65, edgecolor=_ACCENT, linewidth=1.2,
               label='accuracy in bin', align='center', zorder=2)
        bottom = np.minimum(bin_acc, bin_conf)
        ax.bar(centers[ne], gap[ne], width=width * 0.9,
               bottom=bottom[ne],
               color=_TERRA, alpha=0.45, hatch='///',
               edgecolor=_TERRA, linewidth=0,
               label='gap', align='center', zorder=3)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('confidence (top-1 softmax)')
        ax.set_title(f"{panel['title']}   ECE = {ece:.3f}",
                     fontsize=10, color=_GOLDEN)
        tufte_axis(ax)
    axes[0].set_ylabel('accuracy')
    axes[0].legend(frameon=False, labelcolor=_TEXT, loc='upper left', fontsize=8)
    if suptitle is not None:
        fig.suptitle(suptitle, color=_GOLDEN, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def apply_histogram_binning(confidence, calibration_bin_acc, n_bins=10):
    """Histogram-binning calibration: map each raw confidence to the accuracy
    its bin achieved on the calibration set.

    `calibration_bin_acc` is the `bin_acc` array returned by `calibration_bins`
    on the calibration data. Empty bins (NaN) fall back to the raw confidence.
    """
    confidence = np.asarray(confidence, dtype=float)
    cal_acc    = np.asarray(calibration_bin_acc, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx   = np.clip(np.digitize(confidence, edges, right=False) - 1, 0, n_bins - 1)
    out   = cal_acc[idx]
    return np.where(np.isnan(out), confidence, out)


def print_ece_breakdown(bin_conf, bin_acc, bin_count, label=''):
    """Pretty-print the per-bin contributions to ECE for one group."""
    bin_conf  = np.asarray(bin_conf,  dtype=float)
    bin_acc   = np.asarray(bin_acc,   dtype=float)
    bin_count = np.asarray(bin_count, dtype=float)

    total = bin_count.sum()
    gap   = np.abs(bin_acc - bin_conf)
    ece   = float(np.nansum(gap * bin_count) / total) if total > 0 else 0.0

    header = f'Per-bin breakdown {label}'.strip()
    print(header)
    print('-' * len(header))
    print(f'  ECE = {ece:.3f}    (sum of weight * |gap| over non-empty bins)')
    print()
    print('  bin edge       n     conf    acc    |gap|    weight * |gap|')
    edges = np.linspace(0, 1, len(bin_count) + 1)
    for i in range(len(bin_count)):
        if bin_count[i] == 0:
            continue
        w = bin_count[i] / total
        print(f'  [{edges[i]:.2f}, {edges[i+1]:.2f}]  {int(bin_count[i]):4d}   '
              f'{bin_conf[i]:.3f}  {bin_acc[i]:.3f}  {gap[i]:.3f}    '
              f'{w * gap[i]:.3f}')


# ---------------------------------------------------------------------------
# Section: MC Dropout
# ---------------------------------------------------------------------------

def plot_mc_variance(mean_probs, var_total, is_odd, p_drop=0.3):
    """Side-by-side scatters: mean top-1 confidence vs predictive variance.

    Left panel shows knowns only (evens, teal). Right panel shows unknowns only
    (odds, terra). Shared x and y axes so the two are directly comparable.
    """
    top1_mean = np.asarray(mean_probs).max(axis=1)
    var_total = np.asarray(var_total)
    x_min = max(0.18, float(top1_mean.min()) - 0.02)
    y_max = float(var_total.max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)

    for ax, mask, color, title in [
        (axes[0], ~is_odd, _ACCENT, f'evens, known (n={(~is_odd).sum()})'),
        (axes[1],  is_odd, _TERRA, f'odds, unknown (n={ is_odd.sum()})'),
    ]:
        ax.scatter(top1_mean[mask], var_total[mask],
                   s=14, color=color, alpha=0.45, linewidths=0)
        ax.set_xlabel('mean top-1 confidence (averaged over T passes)')
        ax.set_title(title, fontsize=10, color=color)
        tufte_axis(ax)

    axes[0].set_ylabel('predictive variance (sum across classes)')
    axes[0].set_xlim(x_min, 1.02)
    axes[0].set_ylim(-0.01, y_max)
    fig.suptitle(f'MC Dropout: confidence vs variance, p = {p_drop}',
                 color=_GOLDEN, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_high_variance_examples(mc_df, var_total, mean_probs, images, classes):
    """5+5 grid: lowest-variance evens on top, highest-variance odds below."""
    pix_cols = [f'pixel_{j}' for j in range(784)]
    mc_pix = mc_df.merge(images.drop(columns=['label', 'parity']), on='idx')
    mc_pix = mc_pix.assign(pvar=var_total, top1=mean_probs.argmax(axis=1))

    ev_low  = mc_pix[mc_pix['parity'] == 'even'].nsmallest(5, 'pvar')
    od_high = mc_pix[mc_pix['parity'] == 'odd' ].nlargest(5,  'pvar')

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.0))
    for row, group in enumerate([ev_low, od_high]):
        for col, (_, r) in enumerate(group.iterrows()):
            img = r[pix_cols].values.astype(float).reshape(28, 28)
            ax = axes[row, col]
            ax.imshow(img, cmap='magma')
            color = _ACCENT if r['parity'] == 'even' else _TERRA
            pred  = classes[int(r['top1'])]
            ax.set_title(f"true {int(r['label'])}  pred {pred}\nvar = {r['pvar']:.2f}",
                         fontsize=9, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    axes[0, 0].set_ylabel('lowest-variance\nevens (known)', color=_ACCENT, fontsize=9)
    axes[1, 0].set_ylabel('highest-variance\nodds (unknown)', color=_TERRA, fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section: Confidence Head
# ---------------------------------------------------------------------------

def plot_confidence_examples(ch_df, images):
    """3-row grid: certain evens, uncertain odds, saturated odd failures."""
    pix_cols = [f'pixel_{j}' for j in range(784)]
    ch_pix = ch_df.merge(images.drop(columns=['label', 'parity']), on='idx')

    ev_high = ch_pix[ch_pix['parity'] == 'even'].nlargest(5, 'confidence')
    od_low  = ch_pix[ch_pix['parity'] == 'odd' ].nsmallest(5, 'confidence')
    od_sat  = ch_pix[(ch_pix['parity'] == 'odd') & (ch_pix['confidence'] > 0.99)].head(5)

    fig, axes = plt.subplots(3, 5, figsize=(11, 7.0))
    row_titles = ['evens (known), head says certain',
                  'odds (unknown), head says uncertain',
                  'odds (unknown), head saturated (the failure mode)']
    for row, group in enumerate([ev_high, od_low, od_sat]):
        for col, (_, r) in enumerate(group.iterrows()):
            img = r[pix_cols].values.astype(float).reshape(28, 28)
            ax = axes[row, col]
            ax.imshow(img, cmap='magma')
            color = _ACCENT if r['parity'] == 'even' else _TERRA
            ax.set_title(f"true {int(r['label'])}  pred {int(r['top1_label'])}\nc = {r['confidence']:.3f}",
                         fontsize=8, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
        axes[row, 0].set_ylabel(row_titles[row], color=_TEXT, fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section: Find the Odds (VAE)
# ---------------------------------------------------------------------------

def plot_latent_pca(lat_df):
    """2D PCA scatter of VAE latent means, evens (known) vs odds (unknown)."""
    mu_cols = [f'mu_{k}' for k in range(16)]
    Z       = lat_df[mu_cols].values
    is_odd  = (lat_df['parity'] == 'odd').values

    Z2 = PCA(n_components=2, random_state=0).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z2[~is_odd, 0], Z2[~is_odd, 1],
               s=10, color=_ACCENT, alpha=0.45, linewidths=0,
               label=f'evens, known (n={(~is_odd).sum()})')
    ax.scatter(Z2[ is_odd, 0], Z2[ is_odd, 1],
               s=10, color=_TERRA,  alpha=0.45, linewidths=0,
               label=f'odds, unknown (n={ is_odd.sum()})')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('VAE latent means, projected to 2D by PCA', fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='upper right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_confusion_known_unknown(true_labels, pred_labels, even_classes=(0, 2, 4, 6, 8)):
    """Confusion matrix with all 10 true classes as rows and the 5 known
    prediction classes as columns. Row labels are colored teal for knowns
    (evens) and terra for unknowns (odds), so it is immediate which rows are
    which.
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    rows = list(range(10))
    cols = list(even_classes)
    M = np.zeros((len(rows), len(cols)), dtype=int)
    for i, t in enumerate(rows):
        for j, p in enumerate(cols):
            M[i, j] = int(((true_labels == t) & (pred_labels == p)).sum())

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    im = ax.imshow(M, cmap='magma', aspect='equal')

    M_max = M.max() if M.max() > 0 else 1
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if v == 0:
                continue
            tcolor = '#222222' if v > 0.55 * M_max else _TEXT
            ax.text(j, i, str(v), ha='center', va='center', color=tcolor, fontsize=9)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r}  (known)' if r % 2 == 0 else f'{r}  (unknown)' for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(_ACCENT if r % 2 == 0 else _TERRA)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols], color=_ACCENT)

    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label  (only the 5 known classes are reachable)')
    ax.set_title('Where do the predictions land?', fontsize=10, color=_GOLDEN)

    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()


def plot_recon_grid(ex_df):
    """Original-vs-reconstruction pairs: best evens on top, worst odds below."""
    orig_cols  = [f'orig_{j}'  for j in range(784)]
    recon_cols = [f'recon_{j}' for j in range(784)]

    ev_ex = ex_df[ex_df['parity'] == 'even'].sort_values('recon_error').head(5)
    od_ex = ex_df[ex_df['parity'] == 'odd' ].sort_values('recon_error', ascending=False).head(5)

    fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
    for col, (_, row) in enumerate(ev_ex.iterrows()):
        o = row[orig_cols ].values.astype(float).reshape(28, 28)
        r = row[recon_cols].values.astype(float).reshape(28, 28)
        axes[0, 2*col    ].imshow(o, cmap='magma')
        axes[0, 2*col + 1].imshow(r, cmap='magma')
        axes[0, 2*col    ].set_title(f"{int(row['label'])}", fontsize=9, color=_ACCENT)
        axes[0, 2*col + 1].set_title(f"e={row['recon_error']:.3f}", fontsize=9, color=_TEXT)
    for col, (_, row) in enumerate(od_ex.iterrows()):
        o = row[orig_cols ].values.astype(float).reshape(28, 28)
        r = row[recon_cols].values.astype(float).reshape(28, 28)
        axes[1, 2*col    ].imshow(o, cmap='magma')
        axes[1, 2*col + 1].imshow(r, cmap='magma')
        axes[1, 2*col    ].set_title(f"{int(row['label'])}", fontsize=9, color=_TERRA)
        axes[1, 2*col + 1].set_title(f"e={row['recon_error']:.3f}", fontsize=9, color=_TEXT)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    axes[0, 0].set_ylabel('evens\n(known)',  color=_ACCENT)
    axes[1, 0].set_ylabel('odds\n(unknown)', color=_TERRA)
    plt.tight_layout()
    plt.show()
