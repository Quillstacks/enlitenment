"""Visualisation and training helpers for lecture-09 (Self-Supervised Learning).

The notebook runs in JupyterLite/pyodide, so everything here is numpy-only.
Students implement attention / masking / losses as standalone functions; this
module wires those into a tiny transformer that trains in the browser via
an analytic backward pass and a manual Adam optimiser, plus the plotting
and demo plumbing the notebook calls.

The corpus is the synthetic banana-sentences dataset (~50 sentences, ~30
unique words).  At this size every section's tiny model genuinely learns the
patterns ("yellow banana peel", "ate a ripe banana", ...) and the demos
visibly succeed — fill-ins return real words, samples emit plausible
continuations, JEPA's variance trace shows collapse-vs-healthy clearly.
"""

from __future__ import annotations

import sys
sys.path.insert(0, '../..')
from plot_style import *  # noqa: F401,F403

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tokeniser (word-level)
# ---------------------------------------------------------------------------

MASK_ID = 0  # slot 0 is reserved for [MASK]


def load_banana_corpus(filename: str = "banana_corpus.txt") -> str:
    """Read the banana-sentences corpus.  Returns one whitespace-joined string."""
    raw = (HERE / filename).read_text(encoding="utf-8")
    # Collapse newlines into single spaces so the encoded id stream is
    # contiguous; sentence boundaries become regular spaces.
    return " ".join(line.strip() for line in raw.strip().splitlines() if line.strip())


def build_word_vocab(text: str) -> tuple[dict, dict]:
    """Word-level vocab.  Index 0 is reserved for [MASK]; real words start at 1."""
    words = sorted(set(text.split()))
    stoi = {"[MASK]": 0}
    for i, w in enumerate(words, start=1):
        stoi[w] = i
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def encode_words(text: str, stoi: dict) -> np.ndarray:
    """Encode a whitespace-tokenised string into a 1-D array of ids."""
    return np.array([stoi[w] for w in text.split()], dtype=np.int64)


def decode_ids(ids, itos: dict) -> str:
    """Decode an iterable of ids; renders [MASK] as an underscore."""
    out = []
    for i in ids:
        i = int(i)
        if i == MASK_ID:
            out.append("_")
        else:
            out.append(itos.get(i, "?"))
    return " ".join(out)


def visible_vocab(stoi: dict, n: int = 40) -> str:
    """Pretty-print the first n vocab entries (excluding the [MASK] slot)."""
    items = [(i, w) for w, i in stoi.items() if i != MASK_ID]
    items.sort()
    rows = [f"  {i:3d}  {w!r}" for i, w in items[:n]]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Tiny numpy transformer
# ---------------------------------------------------------------------------
#
# One attention block (single head, no head reshaping), small feed-forward
# layer, learned position embeddings.  Fits in a few thousand parameters so
# Adam with analytic gradients trains the tiny language model on the banana
# corpus to a low loss in a few seconds in pyodide.

DEFAULT_D_MODEL = 16
DEFAULT_CONTEXT = 12
DEFAULT_D_FF    = 32


def init_params(vocab_size: int, d_model: int = DEFAULT_D_MODEL,
                context: int = DEFAULT_CONTEXT, d_ff: int = DEFAULT_D_FF,
                seed: int = 0) -> dict[str, np.ndarray]:
    """Random init for the tiny one-block transformer."""
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


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def forward(params: dict, ids: np.ndarray, attention_fn, attn_mask=None):
    """Tiny-LM forward pass (used by demos that don't need a backward).

    Calls ``attention_fn(Q, K, V, attn_mask)`` for the attention step so the
    student's implementation is exercised here.  Returns ``(logits, hidden)``
    where ``hidden`` is the post-LN hidden state used by JEPA.
    """
    logits, hidden, _ = _forward_cache(params, ids, attention_fn, attn_mask)
    return logits, hidden


def _forward_cache(params: dict, ids: np.ndarray, attention_fn, attn_mask=None):
    """Forward pass returning ``(logits, hidden, cache)`` for analytic backward.

    Attention is computed twice: once internally to populate the gradient
    cache, and again via ``attention_fn`` (the student's implementation)
    whose output flows through the residual stream.  When the student's
    ``attention`` is correct the two computations are identical, and the
    extra work is negligible.  When it is not, the loss reflects the bug
    even though the gradient cache uses the canonical computation —
    ``check_attention`` is the dedicated detector for that case.
    """
    B, T = ids.shape
    D = params["tok_emb"].shape[1]

    pos = np.arange(T)
    tok = params["tok_emb"][ids]                                       # (B, T, D)
    pos_e = params["pos_emb"][pos][None, :, :]                         # (1, T, D)
    h0 = tok + pos_e                                                   # (B, T, D)

    # ---- Block 0: pre-LN + attention ----
    ln1_mean = h0.mean(axis=-1, keepdims=True)
    ln1_var  = h0.var(axis=-1, keepdims=True)
    ln1_inv  = 1.0 / np.sqrt(ln1_var + 1e-5)
    ln1_norm = (h0 - ln1_mean) * ln1_inv
    ln1_out  = ln1_norm * params["blk0_ln1_w"] + params["blk0_ln1_b"]  # (B, T, D)

    qkv = ln1_out @ params["blk0_qkv_w"].T                             # (B, T, 3D)
    Q, K, V = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]

    # Canonical scaled-dot-product attention; weights are cached for the
    # gradient path through softmax in ``_backward_from_hidden``.
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(D)                 # (B, T, T)
    if attn_mask is not None:
        scores = scores + attn_mask
    sm_max = scores.max(axis=-1, keepdims=True)
    sm_exp = np.exp(scores - sm_max)
    attn_weights = sm_exp / sm_exp.sum(axis=-1, keepdims=True)         # (B, T, T)
    attn_out = attn_weights @ V                                        # (B, T, D)

    # The student's attention exercises their code on the live forward path.
    if attention_fn is not None:
        attn_out = attention_fn(Q, K, V, attn_mask)

    proj_out = attn_out @ params["blk0_proj_w"].T                      # (B, T, D)
    h1 = h0 + proj_out

    # ---- Block 0: pre-LN + FFN ----
    ln2_mean = h1.mean(axis=-1, keepdims=True)
    ln2_var  = h1.var(axis=-1, keepdims=True)
    ln2_inv  = 1.0 / np.sqrt(ln2_var + 1e-5)
    ln2_norm = (h1 - ln2_mean) * ln2_inv
    ln2_out  = ln2_norm * params["blk0_ln2_w"] + params["blk0_ln2_b"]

    ffn1_pre  = ln2_out @ params["blk0_ffn1_w"].T + params["blk0_ffn1_b"]  # (B, T, d_ff)
    ffn1_post = _gelu(ffn1_pre)
    ffn2      = ffn1_post @ params["blk0_ffn2_w"].T + params["blk0_ffn2_b"]
    h2 = h1 + ffn2

    # ---- Final LN + head ----
    lnf_mean = h2.mean(axis=-1, keepdims=True)
    lnf_var  = h2.var(axis=-1, keepdims=True)
    lnf_inv  = 1.0 / np.sqrt(lnf_var + 1e-5)
    lnf_norm = (h2 - lnf_mean) * lnf_inv
    hidden   = lnf_norm * params["ln_f_w"] + params["ln_f_b"]
    logits   = hidden @ params["head_w"].T

    cache = {
        "ids": ids, "B": B, "T": T, "D": D,
        "h0": h0,
        "ln1_norm": ln1_norm, "ln1_inv": ln1_inv,
        "ln1_out": ln1_out,
        "Q": Q, "K": K, "V": V, "attn_weights": attn_weights, "attn_out": attn_out,
        "proj_out": proj_out, "h1": h1,
        "ln2_norm": ln2_norm, "ln2_inv": ln2_inv, "ln2_out": ln2_out,
        "ffn1_pre": ffn1_pre, "ffn1_post": ffn1_post, "ffn2": ffn2, "h2": h2,
        "lnf_norm": lnf_norm, "lnf_inv": lnf_inv, "hidden": hidden,
    }
    return logits, hidden, cache


def _gelu_grad(x):
    """Derivative of the tanh-approximation GELU used in ``_gelu``."""
    s = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    th = np.tanh(s)
    ds_dx = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + th) + 0.5 * x * (1 - th ** 2) * ds_dx


def _layernorm_backward(d_out, x, norm, inv_std, gamma):
    """LayerNorm backward over the last axis."""
    N = x.shape[-1]
    d_norm = d_out * gamma
    d_gamma = (d_out * norm).reshape(-1, N).sum(axis=0)
    d_beta  = d_out.reshape(-1, N).sum(axis=0)
    # dx through normalisation: standard formula
    sum_dnorm = d_norm.sum(axis=-1, keepdims=True)
    sum_dnorm_norm = (d_norm * norm).sum(axis=-1, keepdims=True)
    d_x = (1.0 / N) * inv_std * (N * d_norm - sum_dnorm - norm * sum_dnorm_norm)
    return d_x, d_gamma, d_beta


def _backward_from_hidden(d_hidden: np.ndarray, params: dict, cache: dict) -> dict:
    """Analytic backward starting from dL/d_hidden (output of final LN).

    Returns gradients keyed by the same names as ``params``.  Does NOT
    include ``head_w`` (which is downstream of ``hidden``); callers that
    start from dL/dlogits should add ``d_head_w`` themselves.

    The causal/bidirectional mask is implicit in the cached ``attn_weights``
    (softmax of ``-inf`` is zero), so the gradient through softmax already
    yields zero contribution at masked-out cells.  No mask argument needed.
    """
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    h2 = cache["h2"]; D = cache["D"]

    # ln_f
    d_h2, d_lnf_w, d_lnf_b = _layernorm_backward(
        d_hidden, h2, cache["lnf_norm"], cache["lnf_inv"], params["ln_f_w"])
    grads["ln_f_w"] += d_lnf_w
    grads["ln_f_b"] += d_lnf_b

    # h2 = h1 + ffn2 (residual)
    d_h1   = d_h2.copy()
    d_ffn2 = d_h2.copy()

    # ffn2 = ffn1_post @ ffn2_w.T + ffn2_b
    grads["blk0_ffn2_w"] += np.einsum("bti,btj->ij", d_ffn2, cache["ffn1_post"])
    grads["blk0_ffn2_b"] += d_ffn2.reshape(-1, D).sum(axis=0)
    d_ffn1_post = d_ffn2 @ params["blk0_ffn2_w"]

    # ffn1_post = gelu(ffn1_pre)
    d_ffn1_pre = d_ffn1_post * _gelu_grad(cache["ffn1_pre"])

    # ffn1_pre = ln2_out @ ffn1_w.T + ffn1_b
    d_ff = params["blk0_ffn1_w"].shape[0]
    grads["blk0_ffn1_w"] += np.einsum("bti,btj->ij", d_ffn1_pre, cache["ln2_out"])
    grads["blk0_ffn1_b"] += d_ffn1_pre.reshape(-1, d_ff).sum(axis=0)
    d_ln2_out = d_ffn1_pre @ params["blk0_ffn1_w"]

    # ln2
    d_h1_ln, d_ln2_w, d_ln2_b = _layernorm_backward(
        d_ln2_out, cache["h1"], cache["ln2_norm"], cache["ln2_inv"], params["blk0_ln2_w"])
    grads["blk0_ln2_w"] += d_ln2_w
    grads["blk0_ln2_b"] += d_ln2_b
    d_h1 = d_h1 + d_h1_ln

    # h1 = h0 + proj_out (residual)
    d_h0      = d_h1.copy()
    d_proj_out = d_h1.copy()

    # proj_out = attn_out @ proj_w.T
    grads["blk0_proj_w"] += np.einsum("bti,btj->ij", d_proj_out, cache["attn_out"])
    d_attn_out = d_proj_out @ params["blk0_proj_w"]

    # attention: d_attn_out = weights @ V
    Q, K, V = cache["Q"], cache["K"], cache["V"]
    weights = cache["attn_weights"]
    d_weights = d_attn_out @ np.swapaxes(V, -1, -2)                    # (B, T, T)
    d_V       = np.swapaxes(weights, -1, -2) @ d_attn_out              # (B, T, D)

    # softmax backward: d_scores = weights * (d_weights - sum(weights*d_weights, axis=-1))
    sum_wd = (weights * d_weights).sum(axis=-1, keepdims=True)
    d_scores = weights * (d_weights - sum_wd)
    d_scores = d_scores / np.sqrt(D)

    # scores = Q @ K.T
    d_Q = d_scores @ K                                                 # (B, T, D)
    d_K = np.swapaxes(d_scores, -1, -2) @ Q                            # (B, T, D)

    # qkv = [Q, K, V] = ln1_out @ qkv_w.T
    d_qkv = np.concatenate([d_Q, d_K, d_V], axis=-1)                   # (B, T, 3D)
    grads["blk0_qkv_w"] += np.einsum("bti,btj->ij", d_qkv, cache["ln1_out"])
    d_ln1_out = d_qkv @ params["blk0_qkv_w"]

    # ln1
    d_h0_ln, d_ln1_w, d_ln1_b = _layernorm_backward(
        d_ln1_out, cache["h0"], cache["ln1_norm"], cache["ln1_inv"], params["blk0_ln1_w"])
    grads["blk0_ln1_w"] += d_ln1_w
    grads["blk0_ln1_b"] += d_ln1_b
    d_h0 = d_h0 + d_h0_ln

    # h0 = tok_emb[ids] + pos_emb[pos]
    # d_pos_emb[t] = sum_b d_h0[b, t]
    grads["pos_emb"] += d_h0.sum(axis=0)
    # d_tok_emb[v] += sum over (b, t) where ids[b,t] == v
    ids = cache["ids"]
    np.add.at(grads["tok_emb"], ids.ravel(), d_h0.reshape(-1, D))

    return grads


def _backward_from_logits(d_logits: np.ndarray, params: dict, cache: dict) -> dict:
    """Analytic backward starting from dL/dlogits.

    ``logits = hidden @ head_w.T``, so dL/dhead_w = d_logits.T @ hidden and
    dL/dhidden = d_logits @ head_w.  Then defer to ``_backward_from_hidden``.
    """
    d_head_w = np.einsum("btv,btj->vj", d_logits, cache["hidden"])
    d_hidden = d_logits @ params["head_w"]                              # (B, T, D)
    grads = _backward_from_hidden(d_hidden, params, cache)
    grads["head_w"] = grads.get("head_w", np.zeros_like(params["head_w"])) + d_head_w
    return grads


# ---------------------------------------------------------------------------
# Training: analytic backward + Adam
# ---------------------------------------------------------------------------
#
# Manual Adam optimiser driven by the analytic backward pass defined above.
# This trains the tiny language model on the banana corpus to a low loss in
# a few seconds in pyodide; finite-difference gradients on the same model
# would be far too noisy to drive the loss down inside the pyodide budget.

def _stable_log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _ce_at_positions(logits, targets, positions_mask):
    """Average cross-entropy at the True positions of ``positions_mask``."""
    log_probs = _stable_log_softmax(logits)
    n = int(positions_mask.sum())
    if n == 0:
        return 0.0
    flat_logp = log_probs[positions_mask]
    flat_tgt  = targets[positions_mask]
    return -float(flat_logp[np.arange(n), flat_tgt].mean())


def _ce_dlogits_at_positions(logits, targets, positions_mask):
    """dL/dlogits for ``_ce_at_positions``.  Loss is averaged over masked positions."""
    probs = np.exp(_stable_log_softmax(logits))                        # (B, T, V)
    d_logits = np.zeros_like(probs)
    n = int(positions_mask.sum())
    if n == 0:
        return d_logits
    # At masked positions: (probs - one_hot(target)) / N
    # Implemented as scatter to avoid building a one-hot tensor.
    masked_probs = probs[positions_mask]                               # (n, V)
    masked_tgt   = targets[positions_mask]
    masked_probs[np.arange(n), masked_tgt] -= 1.0
    masked_probs /= n
    d_logits[positions_mask] = masked_probs
    return d_logits


def _ce_dlogits_next_token(logits, y_batch):
    """dL/dlogits for full next-token cross-entropy.  Targets are y_batch (B, T)."""
    B, T, V = logits.shape
    probs = np.exp(_stable_log_softmax(logits))
    flat_probs = probs.reshape(B * T, V)
    flat_probs[np.arange(B * T), y_batch.ravel()] -= 1.0
    flat_probs /= (B * T)
    return flat_probs.reshape(B, T, V)


def _adam_init(params: dict) -> dict:
    return {
        "m": {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()},
        "v": {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()},
        "t": 0,
    }


def _adam_step(params: dict, grads: dict, state: dict,
               lr: float = 0.05, b1: float = 0.9, b2: float = 0.999,
               eps: float = 1e-8) -> None:
    """In-place Adam update of ``params`` using ``grads`` and ``state``."""
    state["t"] += 1
    t = state["t"]
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    for k in params:
        g = grads[k].astype(np.float64)
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * (g * g)
        m_hat = state["m"][k] / bc1
        v_hat = state["v"][k] / bc2
        params[k] = (params[k] - lr * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)


def train_lm(params: dict, ids: np.ndarray, attention_fn, mask_strategy_fn,
             use_causal_mask: bool, n_steps: int = 200, batch_size: int = 32,
             lr: float = 0.05, seed: int = 0, verbose: bool = True
             ) -> tuple[dict, list]:
    """Train the tiny language model with analytic backprop + Adam.

    A fresh batch of windows is sampled every step (the banana corpus is
    small enough that this looks like full-batch training in practice).
    The loss curve has one entry per Adam step.

    Parameters
    ----------
    params              initial weight dict; updated in-place and also returned
    ids                 corpus encoded as a 1-D array of int ids
    attention_fn        student's ``attention(Q, K, V, attn_mask)`` — exercised
                        in the forward pass; backward uses an internal
                        equivalent computed alongside it
    mask_strategy_fn    ``function(batch, rng) -> (corrupted_batch, mask_positions)``
                        for masked language modelling, OR ``None`` for causal
                        next-token training
    use_causal_mask     True for causal language models; False otherwise
    n_steps             number of Adam steps
    batch_size          number of windows per step
    lr                  Adam learning rate
    """
    rng = np.random.default_rng(seed)
    context = params["pos_emb"].shape[0]
    vocab_size = params["tok_emb"].shape[0]
    if use_causal_mask:
        attn_mask = np.triu(np.full((context, context), -1e9), k=1).astype(np.float32)
    else:
        attn_mask = None
    state = _adam_init(params)
    losses = []

    for step in range(n_steps):
        # --- sample a fresh batch ---
        if mask_strategy_fn is None:
            starts = rng.integers(0, len(ids) - context - 1, size=batch_size)
            windows = np.stack([ids[s:s + context + 1] for s in starts])
            x_batch = windows[:, :-1]
            y_batch = windows[:, 1:]
            logits, _, cache = _forward_cache(params, x_batch, attention_fn, attn_mask)
            log_probs = _stable_log_softmax(logits)
            B, T, _   = logits.shape
            loss = -float(log_probs.reshape(B * T, -1)[np.arange(B * T), y_batch.ravel()].mean())
            d_logits = _ce_dlogits_next_token(logits, y_batch)
        else:
            starts = rng.integers(0, len(ids) - context, size=batch_size)
            clean_batch = np.stack([ids[s:s + context] for s in starts])
            corrupted_batch, mask_pos = mask_strategy_fn(clean_batch, rng)
            logits, _, cache = _forward_cache(params, corrupted_batch, attention_fn, attn_mask)
            loss = _ce_at_positions(logits, clean_batch, mask_pos)
            d_logits = _ce_dlogits_at_positions(logits, clean_batch, mask_pos)

        losses.append(loss)
        grads = _backward_from_logits(d_logits, params, cache)
        _adam_step(params, grads, state, lr=lr)

    if verbose:
        print(f"  initial loss: {losses[0]:.4f}  "
              f"(uniform baseline ~{np.log(vocab_size):.4f})")
        print(f"  final   loss: {losses[-1]:.4f}  ({n_steps} Adam steps)")
    return params, losses


def train_jepa_run(student_init: dict, teacher_init: dict, ids: np.ndarray,
                    attention_fn, mask_strategy_fn, ema_update_fn,
                    m: float = 0.99, n_steps: int = 80, batch_size: int = 32,
                    lr: float = 0.05, seed: int = 0) -> tuple[dict, list, list]:
    """Train a Joint-Embedding Predictive Architecture (JEPA) student/teacher pair.

    Per Adam step:
      1. sample a batch of windows
      2. corrupt with ``mask_strategy_fn`` to produce the student's input
      3. teacher hidden states are computed on the UNCORRUPTED input,
         frozen (no gradient flows into the teacher)
      4. student hidden states are computed on the corrupted input
      5. mean squared error (MSE) loss at masked positions; Adam updates
         the student
      6. exponential moving average (EMA) update of the teacher toward
         the freshly-updated student via ``ema_update_fn``
      7. record student embedding variance ACROSS the batch — the canonical
         collapse diagnostic

    Returns ``(final_student, loss_curve, variance_trace)``.  With EMA on
    (``m`` close to 1) the variance trace stays non-zero; with ``m == 0``
    (teacher copies student instantly) it collapses toward zero.
    """
    student = {k: v.copy() for k, v in student_init.items()}
    teacher = {k: v.copy() for k, v in teacher_init.items()}
    rng = np.random.default_rng(seed)
    context = student["pos_emb"].shape[0]
    state = _adam_init(student)
    losses = []
    variances = []

    for step in range(n_steps):
        starts = rng.integers(0, len(ids) - context, size=batch_size)
        clean_batch = np.stack([ids[s:s + context] for s in starts])
        corrupted_batch, mask_pos = mask_strategy_fn(clean_batch, rng)

        # Teacher: forward on UNCORRUPTED input, no gradient.
        _, teacher_h = forward(teacher, clean_batch, attention_fn, None)

        # Student: forward on CORRUPTED input, with cache for backward.
        _, student_h, cache = _forward_cache(student, corrupted_batch, attention_fn, None)

        # MSE loss at masked positions
        diff = (student_h - teacher_h)
        n_masked = int(mask_pos.sum())
        loss = float((diff[mask_pos] ** 2).mean()) if n_masked > 0 else 0.0
        losses.append(loss)

        # dL/dstudent_h = (2 / N) * diff at masked positions; zero elsewhere
        d_hidden = np.zeros_like(student_h)
        if n_masked > 0:
            d_hidden[mask_pos] = (2.0 / n_masked) * diff[mask_pos]
        grads = _backward_from_hidden(d_hidden, student, cache)

        _adam_step(student, grads, state, lr=lr)

        # EMA teacher update AFTER student step
        teacher = ema_update_fn(teacher, student, m)

        # Track student per-batch embedding variance with the freshly
        # updated student weights — the collapse diagnostic.
        _, post_step_h = forward(student, corrupted_batch, attention_fn, None)
        variances.append(float(post_step_h.var(axis=0).mean()))

    return student, losses, variances


# ---------------------------------------------------------------------------
# Sampling (uses student's attention_fn during forward)
# ---------------------------------------------------------------------------

def sample(params: dict, prompt_ids: np.ndarray, attention_fn,
           max_new_tokens: int = 16, temperature: float = 0.8,
           seed: int = 0) -> np.ndarray:
    """Temperature-sampled autoregressive generation from a causal LM."""
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
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(losses, title: str, baseline: float | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = np.arange(len(losses))
    ax.plot(xs, losses, color=_ACCENT, linewidth=1.4, marker="o",
            markersize=3, markerfacecolor=_ACCENT, markeredgecolor="none")
    if baseline is not None:
        ax.axhline(baseline, color=_TERRA, linewidth=0.8, linestyle="--",
                   label=f"uniform baseline ({baseline:.2f})")
        ax.legend(frameon=False, labelcolor=_TEXT)
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_attention_internals(sentence: str = "she slipped on a yellow banana peel and",
                              d_model: int = 8, seed: int = 0):
    """Show the two T x T matrices that define attention: scores and weights.

    Random Q and K projections on word embeddings, so the pattern itself is
    arbitrary - but the *flow* (Q . K^T / sqrt(D) -> softmax -> weights) is
    exactly what scaled dot-product attention computes.
    """
    words = sentence.strip().split()
    T = len(words)
    rng = np.random.default_rng(seed)
    X  = rng.standard_normal((T, d_model)).astype(np.float32) * 0.6
    Wq = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.4
    Wk = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.4
    Q = X @ Wq
    K = X @ Wk

    scores = (Q @ K.T) / np.sqrt(d_model)
    s_shift = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(s_shift)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    sc_lim = abs(scores).max()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    panels = [
        (axes[0], scores,  "Q . K^T / sqrt(D)  (scores)",  "RdBu_r", -sc_lim, sc_lim),
        (axes[1], weights, "softmax(scores)  (weights)",   "magma",   0.0,    weights.max()),
    ]
    for ax, M, title, cmap, vmin, vmax in panels:
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(T)); ax.set_xticklabels(words, fontsize=7, rotation=90)
        ax.set_yticks(range(T)); ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel("key position", fontsize=8)
        ax.set_ylabel("query position", fontsize=8)
        ax.set_title(title, fontsize=10, color=_GOLDEN)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    plt.tight_layout()
    plt.show()


def plot_attention_masks(context: int = 8):
    """Side-by-side: encoder (no mask) vs causal (upper-triangle blocked)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    bd = np.ones((context, context))
    cs = np.tril(np.ones((context, context)))
    for ax, M, title in [(axes[0], bd, "encoder (no mask)"),
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
    """Run the masked language model on a handful of windows and show the predictions.

    Predictions at masked positions are taken via argmax (greedy fill-in);
    correct word predictions are coloured normally, mistakes are flagged with
    a trailing ``*``.
    """
    rng = np.random.default_rng(seed)
    context = params["pos_emb"].shape[0]
    starts = rng.integers(0, len(ids) - context, size=n_examples)

    fig, ax = plt.subplots(figsize=(11, 0.6 + 1.0 * n_examples))
    ax.axis("off")
    ax.set_title("masked-token fill-ins  (_ marks where mask was, * after wrong predictions)",
                 fontsize=10, color=_GOLDEN, loc="left")

    y0 = 1.0
    for k, s in enumerate(starts):
        window = ids[s:s + context][None, :].copy()
        corrupted, mask_pos = mask_fn(window, rng)
        logits, _ = forward(params, corrupted, attention_fn, attn_mask=None)
        preds = logits.argmax(axis=-1)

        masked_view = decode_ids(corrupted[0], itos)
        true_view   = decode_ids(window[0], itos)
        recon_words = []
        for j in range(context):
            if mask_pos[0, j]:
                p = int(preds[0, j])
                t = int(window[0, j])
                w = itos.get(p, "?")
                if p != t:
                    recon_words.append(f"{w}*")
                else:
                    recon_words.append(w)
            else:
                recon_words.append(itos.get(int(window[0, j]), "?"))
        recon_view = " ".join(recon_words)

        y = y0 - (k + 1) / (n_examples + 1)
        ax.text(0.0,  y + 0.06, f"masked  : {masked_view}", family="monospace",
                fontsize=9, transform=ax.transAxes, color=_TEXT)
        ax.text(0.0,  y - 0.00, f"recon   : {recon_view}", family="monospace",
                fontsize=9, transform=ax.transAxes, color=_GOLDEN)
        ax.text(0.0,  y - 0.06, f"true    : {true_view}",  family="monospace",
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
    ax.text(0.0, 0.80, f"original : {decode_ids(window[0], itos)}",
            family="monospace", fontsize=9, transform=ax.transAxes, color=_TEXT)
    ax.text(0.0, 0.45, f"single   : {decode_ids(sing_corrupted[0], itos)}",
            family="monospace", fontsize=9, transform=ax.transAxes, color=_GOLDEN)
    ax.text(0.0, 0.10, f"span     : {decode_ids(span_corrupted[0], itos)}",
            family="monospace", fontsize=9, transform=ax.transAxes, color=_TERRA)
    ax.set_title("Single-token vs span masking on the same window",
                 fontsize=10, color=_GOLDEN, loc="left")
    plt.tight_layout()
    plt.show()


def plot_jepa_variance(variance_with_ema, variance_without_ema,
                       title: str = "embedding variance: with EMA vs without"):
    """Two variance traces on one axis: with EMA (healthy), without (collapse)."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = np.arange(len(variance_with_ema))
    ax.plot(xs, variance_with_ema, color=_ACCENT, linewidth=1.4, marker="o",
            markersize=3, markerfacecolor=_ACCENT, markeredgecolor="none",
            label="with EMA  (m = 0.99)")
    xs2 = np.arange(len(variance_without_ema))
    ax.plot(xs2, variance_without_ema, color=_TERRA, linewidth=1.4, marker="o",
            markersize=3, markerfacecolor=_TERRA, markeredgecolor="none",
            label="without EMA  (m = 0)")
    ax.axhline(0.0, color=_TEXT, linewidth=0.6, alpha=0.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean per-position embedding variance")
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 🏷️  visualisations (pure motivation, no model)
# ---------------------------------------------------------------------------

def show_one_sentence_many_pairs(sentence: str, n_pairs: int = 9):
    """Render one sentence as N (corrupted, target) training pairs.

    Demonstrates that masking each position in turn turns a single sentence
    into N supervision pairs — the corpus IS the label set.
    """
    words = sentence.split()
    n = min(n_pairs, len(words))

    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.45 * n))
    ax.axis("off")
    ax.set_title(f"one sentence → {n} training pairs   (mask each position in turn)",
                 fontsize=10, color=_GOLDEN, loc="left")

    header_y = 1.0 - 1 / (n + 2)
    ax.text(0.0,  header_y, "  masked input",  family="monospace",
            fontsize=9, transform=ax.transAxes, color=_TEXT)
    ax.text(0.66, header_y, "target", family="monospace",
            fontsize=9, transform=ax.transAxes, color=_TEXT)

    for k in range(n):
        masked = words.copy()
        target = masked[k]
        masked[k] = "_"
        y = 1.0 - (k + 2) / (n + 2)
        ax.text(0.0,  y, "  " + " ".join(masked), family="monospace",
                fontsize=9, transform=ax.transAxes, color=_GOLDEN)
        ax.text(0.66, y, target, family="monospace",
                fontsize=9, transform=ax.transAxes, color=_ACCENT)
    plt.tight_layout()
    plt.show()


def show_three_pretext_tasks(sentence: str):
    """Render the same sentence under three pretext tasks side-by-side.

    Demonstrates that masked-token, span-masked, and next-token-shifted
    are three different ways to read supervision off the same sentence.
    """
    words = sentence.split()
    n = len(words)

    # 1. Masked-token: hide one position in the middle.
    mask_idx = n // 2
    single_in  = words.copy(); single_in[mask_idx] = "_"
    single_target = words[mask_idx]

    # 2. Span-masked: hide three contiguous positions starting somewhere
    #    that catches a strong trigram (e.g. "yellow banana peel").
    if "yellow" in words:
        span_start = words.index("yellow")
    else:
        span_start = max(0, n // 2 - 1)
    span_end = min(span_start + 3, n)
    span_in = words.copy()
    for j in range(span_start, span_end):
        span_in[j] = "_"
    span_target = " ".join(words[span_start:span_end])

    # 3. Next-token shift: input is words[:-1], target is words[1:].
    causal_in     = words[:-1]
    causal_target = words[1:]

    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.axis("off")
    ax.set_title("three pretext tasks, same sentence",
                 fontsize=10, color=_GOLDEN, loc="left")

    rows = [
        ("masked token  ",  " ".join(single_in),                   single_target,  _GOLDEN),
        ("span masked   ",  " ".join(span_in),                     span_target,    _TERRA),
        ("next token    ",  " ".join(causal_in),                   " ".join(causal_target), _ACCENT),
    ]
    for k, (label, inp, tgt, color) in enumerate(rows):
        y_top = 0.92 - 0.32 * k
        ax.text(0.0,  y_top,         label + " input ",  family="monospace",
                fontsize=9, transform=ax.transAxes, color=_TEXT)
        ax.text(0.18, y_top,         inp,                family="monospace",
                fontsize=9, transform=ax.transAxes, color=color)
        ax.text(0.0,  y_top - 0.10,  label + " target",  family="monospace",
                fontsize=9, transform=ax.transAxes, color=_TEXT)
        ax.text(0.18, y_top - 0.10,  tgt,                family="monospace",
                fontsize=9, transform=ax.transAxes, color=color)
    plt.tight_layout()
    plt.show()


def plot_label_gap_magnitude():
    """Bar chart contrasting labelled-dataset and corpus-scale data magnitudes."""
    items = [
        ("MNIST",            7.0e4,    "labelled"),
        ("ImageNet",         1.4e7,    "labelled"),
        ("WMT (parallel)",   3.0e8,    "labelled"),
        ("Wikipedia",        4.0e9,    "corpus"),
        ("Common Crawl",     5.0e11,   "corpus"),
    ]
    names  = [it[0] for it in items]
    counts = np.array([it[1] for it in items])
    kinds  = [it[2] for it in items]
    colors = [_TERRA if k == "labelled" else _ACCENT for k in kinds]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(names, counts, color=colors, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("approximate token / image count  (log scale)")
    ax.set_title("Hand-labelled datasets vs raw corpora",
                 fontsize=10, color=_GOLDEN, loc="left")

    # Legend handles
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_TERRA,  label="labelled (annotation budget)"),
        Patch(facecolor=_ACCENT, label="corpus (read off the data)"),
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=_TEXT, loc="lower right")
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()
