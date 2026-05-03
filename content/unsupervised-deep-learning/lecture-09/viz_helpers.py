"""Visualisation and training helpers for lecture-09 (Self-Supervised Learning).

The notebook runs in JupyterLite/pyodide, so everything here is numpy-only.
Students implement attention / masking / losses as standalone forward
functions; this module wires those into a tiny transformer that trains in
the browser via scipy.optimize, plus the plotting plumbing the notebook
calls.

A larger reference variant trained offline by ``generate_precomputed.py``
can be loaded via ``load_precomputed_weights``; the saved weight names
match the names this module uses for its tiny model, so loading drops in
transparently for sampling and inference.
"""

from __future__ import annotations

import sys
sys.path.insert(0, '../..')
from plot_style import *  # noqa: F401,F403

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

MASK_ID = 0  # slot 0 is reserved for [MASK] in the saved vocab


def load_corpus(filename: str = "die_raeuber.txt") -> str:
    return (HERE / filename).read_text(encoding="utf-8")


def build_char_vocab(text: str) -> tuple[dict, dict]:
    """Char-level vocab. Index 0 is reserved for [MASK]."""
    chars = sorted(set(text))
    stoi = {"\x00": 0}
    for i, c in enumerate(chars, start=1):
        stoi[c] = i
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def decode_ids(ids, itos: dict) -> str:
    """Decode an iterable of ids; renders [MASK] as ``_``."""
    out = []
    for i in ids:
        i = int(i)
        if i == 0:
            out.append("_")
        else:
            out.append(itos.get(i, "?"))
    return "".join(out)


def visible_vocab(stoi: dict, n: int = 40) -> str:
    """Pretty-print the first n vocab entries (excluding the [MASK] slot)."""
    items = [(i, c) for c, i in stoi.items() if i != 0]
    items.sort()
    rows = []
    for i, c in items[:n]:
        rows.append(f"  {i:3d}  {repr(c)}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Tiny numpy transformer
# ---------------------------------------------------------------------------
#
# The architecture is small enough to train in pyodide via scipy.optimize:
#   - 1 transformer block (attention + small FFN)
#   - d_model = 16, n_heads = 1, context = 16
#   - vocab depends on the corpus (~85 for Die Raeuber)
#   - ~5 k params total
#
# Layer names match the saved weights in ``generate_precomputed.py`` so the
# 'next scale up' weights drop in for sampling without architectural changes.

DEFAULT_D_MODEL = 8
DEFAULT_CONTEXT = 16
DEFAULT_D_FF    = 16


def init_params(vocab_size: int, d_model: int = DEFAULT_D_MODEL,
                context: int = DEFAULT_CONTEXT, d_ff: int = DEFAULT_D_FF,
                seed: int = 0) -> dict[str, np.ndarray]:
    """Random init that matches the saved-weights naming convention."""
    rng = np.random.default_rng(seed)
    s = 0.05
    return {
        "tok_emb"   : rng.normal(0, s, (vocab_size, d_model)).astype(np.float32),
        "pos_emb"   : rng.normal(0, s, (context, d_model)).astype(np.float32),
        "blk0_ln1_w": np.ones(d_model, dtype=np.float32),
        "blk0_ln1_b": np.zeros(d_model, dtype=np.float32),
        "blk0_qkv_w": rng.normal(0, s, (3 * d_model, d_model)).astype(np.float32),
        "blk0_proj_w": rng.normal(0, s, (d_model, d_model)).astype(np.float32),
        "blk0_ln2_w": np.ones(d_model, dtype=np.float32),
        "blk0_ln2_b": np.zeros(d_model, dtype=np.float32),
        "blk0_ffn1_w": rng.normal(0, s, (d_ff, d_model)).astype(np.float32),
        "blk0_ffn1_b": np.zeros(d_ff, dtype=np.float32),
        "blk0_ffn2_w": rng.normal(0, s, (d_model, d_ff)).astype(np.float32),
        "blk0_ffn2_b": np.zeros(d_model, dtype=np.float32),
        "ln_f_w"    : np.ones(d_model, dtype=np.float32),
        "ln_f_b"    : np.zeros(d_model, dtype=np.float32),
        "head_w"    : rng.normal(0, s, (vocab_size, d_model)).astype(np.float32),
    }


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def forward(params: dict, ids: np.ndarray, attention_fn, attn_mask=None):
    """Tiny-LM forward pass.

    Calls ``attention_fn(Q, K, V, attn_mask)`` for the attention step so the
    student's implementation is exercised here.  Returns ``(logits, hidden)``
    where ``hidden`` is the post-LN hidden state used by JEPA.
    """
    B, T = ids.shape
    pos = np.arange(T)
    h = params["tok_emb"][ids] + params["pos_emb"][pos][None, :, :]   # (B, T, D)

    # Block 0
    hn = _layernorm(h, params["blk0_ln1_w"], params["blk0_ln1_b"])
    qkv = hn @ params["blk0_qkv_w"].T                                 # (B, T, 3D)
    D = h.shape[-1]
    Q, K, V = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
    attn_out = attention_fn(Q, K, V, attn_mask)                       # (B, T, D)
    h = h + attn_out @ params["blk0_proj_w"].T

    hn = _layernorm(h, params["blk0_ln2_w"], params["blk0_ln2_b"])
    ff = _gelu(hn @ params["blk0_ffn1_w"].T + params["blk0_ffn1_b"])
    h = h + ff @ params["blk0_ffn2_w"].T + params["blk0_ffn2_b"]

    h = _layernorm(h, params["ln_f_w"], params["ln_f_b"])
    logits = h @ params["head_w"].T
    return logits, h


# ---------------------------------------------------------------------------
# Param flatten / unflatten for scipy.optimize
# ---------------------------------------------------------------------------

def flatten(params: dict) -> tuple[np.ndarray, list]:
    """Flatten params to a single 1D float64 array. Returns (flat, layout)."""
    layout = []
    flats = []
    for k, v in params.items():
        layout.append((k, v.shape, v.size))
        flats.append(v.astype(np.float64).ravel())
    return np.concatenate(flats), layout


def unflatten(flat: np.ndarray, layout: list) -> dict:
    out = {}
    i = 0
    for k, shape, size in layout:
        out[k] = flat[i:i + size].reshape(shape).astype(np.float32)
        i += size
    return out


# ---------------------------------------------------------------------------
# Training loop (scipy L-BFGS-B with finite-diff gradients)
# ---------------------------------------------------------------------------

def _stable_log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _ce_at_positions(logits, targets, positions_mask):
    """Average cross-entropy at the True positions of positions_mask."""
    log_probs = _stable_log_softmax(logits)
    n = int(positions_mask.sum())
    if n == 0:
        return 0.0
    flat_logp = log_probs[positions_mask]                              # (n, V)
    flat_tgt  = targets[positions_mask]                                # (n,)
    return -float(flat_logp[np.arange(n), flat_tgt].mean())


def train_lm(params: dict, ids: np.ndarray, attention_fn, mask_strategy_fn,
             use_causal_mask: bool, n_steps: int = 30, batch_size: int = 64,
             seed: int = 0, verbose: bool = True) -> tuple[dict, list]:
    """Train the tiny LM via scipy L-BFGS-B on a *fixed* batch.

    L-BFGS-B is a quasi-Newton optimiser; it expects a deterministic loss
    surface and will terminate early if successive evaluations of the same
    point disagree. We therefore sample a single batch up front and treat
    that batch as the training objective — small enough to fit, large enough
    that the patterns learned generalise to held-out windows.

    Parameters
    ----------
    params              initial weight dict
    ids                 corpus encoded as 1D array of int ids
    attention_fn        student's attention(Q, K, V, attn_mask)
    mask_strategy_fn    function(batch, rng) -> (corrupted_batch, mask_positions)
                        OR ``None`` to use plain causal next-token training
    use_causal_mask     True for causal LMs; False for masked LMs
    n_steps             max L-BFGS-B iterations
    batch_size          number of windows in the fixed training batch
    """
    flat, layout = flatten(params)
    rng = np.random.default_rng(seed)
    context = params["pos_emb"].shape[0]
    vocab_size = params["tok_emb"].shape[0]

    if use_causal_mask:
        mask = np.triu(np.full((context, context), -1e9), k=1).astype(np.float32)
    else:
        mask = None

    # --- Build the fixed batch once, up front ---
    if mask_strategy_fn is None:
        starts = rng.integers(0, len(ids) - context - 1, size=batch_size)
        windows = np.stack([ids[s:s + context + 1] for s in starts])
        x_batch = windows[:, :-1]
        y_batch = windows[:, 1:]
    else:
        starts = rng.integers(0, len(ids) - context, size=batch_size)
        clean_batch = np.stack([ids[s:s + context] for s in starts])
        corrupted_batch, mask_pos = mask_strategy_fn(clean_batch, rng)

    losses = []

    def loss_fn(flat_params):
        p = unflatten(flat_params, layout)
        if mask_strategy_fn is None:
            logits, _ = forward(p, x_batch, attention_fn, mask)
            log_probs = _stable_log_softmax(logits)                    # (B, T, V)
            B, T, _   = logits.shape
            loss = -float(log_probs.reshape(B * T, -1)[np.arange(B * T), y_batch.ravel()].mean())
        else:
            logits, _ = forward(p, corrupted_batch, attention_fn, mask)
            loss = _ce_at_positions(logits, clean_batch, mask_pos)
        losses.append(loss)
        return loss

    if verbose:
        init_loss = loss_fn(flat)
        print(f"  initial loss: {init_loss:.4f}  (uniform baseline ~{np.log(vocab_size):.4f})")

    result = minimize(loss_fn, flat, method="L-BFGS-B",
                      options={"maxiter": n_steps, "disp": False})
    final_params = unflatten(result.x, layout)
    if verbose:
        print(f"  final loss:   {losses[-1]:.4f}  ({result.nit} iterations, "
              f"{len(losses)} loss evals)")
    return final_params, losses


# ---------------------------------------------------------------------------
# Sampling (uses student's attention_fn during forward)
# ---------------------------------------------------------------------------

def sample(params: dict, prompt_ids: np.ndarray, attention_fn,
           max_new_tokens: int = 80, temperature: float = 0.8,
           seed: int = 0) -> np.ndarray:
    """Greedy / temperature autoregressive sampling from a causal LM."""
    rng = np.random.default_rng(seed)
    context = params["pos_emb"].shape[0]
    causal  = np.triu(np.full((context, context), -1e9), k=1).astype(np.float32)
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        seq = np.array(out[-context:], dtype=np.int64)[None, :]
        T = seq.shape[1]
        logits, _ = forward(params, seq, attention_fn, causal[:T, :T])
        last = logits[0, -1] / max(1e-6, temperature)
        last = last - last.max()
        probs = np.exp(last); probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Loading precomputed (offline-trained) weights
# ---------------------------------------------------------------------------

def load_precomputed_weights(filename: str) -> dict[str, np.ndarray] | None:
    """Load a .npz of precomputed weights. Returns None if missing."""
    path = HERE / filename
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "__meta__"}


def precomputed_meta(filename: str) -> dict | None:
    path = HERE / filename
    if not path.exists():
        return None
    z = np.load(path)
    if "__meta__" not in z.files:
        return None
    d_model, n_heads, context, n_layers = z["__meta__"]
    return {"d_model": int(d_model), "n_heads": int(n_heads),
            "context": int(context), "n_layers": int(n_layers)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(losses, title: str, baseline: float | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = np.arange(1, len(losses) + 1)
    ax.plot(xs, losses, color=_ACCENT, linewidth=1.4)
    if baseline is not None:
        ax.axhline(baseline, color=_TERRA, linewidth=0.8, linestyle="--",
                   label=f"uniform baseline ({baseline:.2f})")
        ax.legend(frameon=False, labelcolor=_TEXT)
    ax.set_xlabel("L-BFGS-B iteration")
    ax.set_ylabel("training loss")
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_attention_masks(context: int = 8):
    """Side-by-side: bidirectional (None) vs causal (upper-triangle blocked)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    # Bidirectional: every position attends to every other -> all-ones.
    bd = np.ones((context, context))
    # Causal: position i can attend to j <= i -> lower triangle (incl diagonal).
    cs = np.tril(np.ones((context, context)))
    for ax, M, title in [(axes[0], bd, "bidirectional (encoder)"),
                          (axes[1], cs, "causal (decoder)")]:
        ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(context)); ax.set_yticks(range(context))
        ax.set_xticklabels([f"{i}" for i in range(context)], fontsize=7)
        ax.set_yticklabels([f"{i}" for i in range(context)], fontsize=7)
        ax.set_xlabel("key position (j)")
        ax.set_ylabel("query position (i)")
        ax.set_title(title, fontsize=10, color=_GOLDEN)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()


def show_fill_in_examples(params: dict, ids: np.ndarray, itos: dict,
                          attention_fn, mask_fn, n_examples: int = 4,
                          seed: int = 0):
    """Run the masked LM on a handful of windows and show the predictions."""
    rng = np.random.default_rng(seed)
    context = params["pos_emb"].shape[0]
    starts = rng.integers(0, len(ids) - context, size=n_examples)

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.95 * n_examples))
    ax.axis("off")
    ax.set_title("masked-LM fill-ins  (— shown where mask was, * marks wrong predictions)",
                 fontsize=10, color=_GOLDEN, loc="left")
    y0 = 1.0
    for k, s in enumerate(starts):
        window = ids[s:s + context][None, :].copy()
        corrupted, mask_pos = mask_fn(window, rng)
        logits, _ = forward(params, corrupted, attention_fn, attn_mask=None)
        preds = logits.argmax(axis=-1)
        # Build a string with the corrupted view + predictions at masked positions.
        vis  = decode_ids(corrupted[0], itos)
        true = decode_ids(window[0], itos)
        recon_chars = []
        for j in range(context):
            if mask_pos[0, j]:
                p = int(preds[0, j])
                t = int(window[0, j])
                ch = itos.get(p, "?")
                if ch == "\n": ch = "⏎"
                if p != t:
                    recon_chars.append(f"\033[91m{ch}*\033[0m")
                else:
                    recon_chars.append(ch)
            else:
                ch = itos.get(int(window[0, j]), "?")
                recon_chars.append(ch if ch != "\n" else "⏎")
        # Render as plain text rows in matplotlib (escape codes won't render; show ` *` only)
        recon = ""
        for j in range(context):
            ch = itos.get(int(preds[0, j] if mask_pos[0, j] else window[0, j]), "?")
            if ch == "\n": ch = " "
            recon += ch
            if mask_pos[0, j] and int(preds[0, j]) != int(window[0, j]):
                recon += "*"
        # Replace newlines in vis/true for display
        vis_d = vis.replace("\n", "⏎")
        true_d = true.replace("\n", "⏎")
        y = y0 - (k + 1) / (n_examples + 1)
        ax.text(0.0,  y + 0.06, f"masked : {vis_d}",  family="monospace",
                fontsize=9, transform=ax.transAxes, color=_TEXT)
        ax.text(0.0,  y - 0.00, f"recon  : {recon}",  family="monospace",
                fontsize=9, transform=ax.transAxes, color=_GOLDEN)
        ax.text(0.0,  y - 0.06, f"true   : {true_d}", family="monospace",
                fontsize=9, transform=ax.transAxes, color=_ACCENT)
    plt.tight_layout()
    plt.show()


def show_span_vs_single(ids: np.ndarray, itos: dict, single_mask_fn, span_mask_fn,
                        context: int, seed: int = 1):
    """One window, two masking strategies, side-by-side text."""
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, len(ids) - context))
    window = ids[s:s + context][None, :].copy()
    sing_corrupted, _ = single_mask_fn(window, rng)
    span_corrupted, _ = span_mask_fn(window, rng)
    fig, ax = plt.subplots(figsize=(11, 1.6))
    ax.axis("off")
    true_d = decode_ids(window[0], itos).replace("\n", "⏎")
    sing_d = decode_ids(sing_corrupted[0], itos).replace("\n", "⏎")
    span_d = decode_ids(span_corrupted[0], itos).replace("\n", "⏎")
    ax.text(0.0, 0.80, f"original : {true_d}", family="monospace",
            fontsize=9, transform=ax.transAxes, color=_TEXT)
    ax.text(0.0, 0.45, f"single   : {sing_d}", family="monospace",
            fontsize=9, transform=ax.transAxes, color=_GOLDEN)
    ax.text(0.0, 0.10, f"span     : {span_d}", family="monospace",
            fontsize=9, transform=ax.transAxes, color=_TERRA)
    ax.set_title("Single-token vs span masking on the same window",
                 fontsize=10, color=_GOLDEN, loc="left")
    plt.tight_layout()
    plt.show()


def plot_jepa_variance(student_h_history: list, title: str = "embedding variance"):
    """Plot per-step embedding variance across the batch — the canonical
    collapse diagnostic for non-contrastive methods."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(1, len(student_h_history) + 1), student_h_history,
            color=_ACCENT, linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean embedding variance")
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()
