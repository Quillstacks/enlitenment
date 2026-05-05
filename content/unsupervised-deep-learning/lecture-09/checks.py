"""Auto-checker helpers for lecture-09, Self-Supervised Learning."""

import numpy as np

_OK   = "✅"   # ✅
_FAIL = "❌"   # ❌
_NONE = "⬜"   # ⬜
_TOL  = 1e-3


def _close(a, b, tol=_TOL):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# Reference forward operations (mirrors viz_helpers.forward attention step)
# ---------------------------------------------------------------------------

def _ref_attention(Q, K, V, attn_mask=None):
    """Reference scaled dot-product attention."""
    d = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)
    if attn_mask is not None:
        scores = scores + attn_mask
    m = scores.max(axis=-1, keepdims=True)
    expv = np.exp(scores - m)
    weights = expv / expv.sum(axis=-1, keepdims=True)
    return weights @ V


def _ref_causal_mask(T):
    return np.triu(np.full((T, T), -np.inf), k=1).astype(np.float32)


def _stable_log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# 🎭  attention(Q, K, V, attn_mask=None)
# ---------------------------------------------------------------------------

def check_attention(fn):
    rng = np.random.default_rng(0)
    B, T, D = 2, 5, 4
    Q = rng.normal(size=(B, T, D)).astype(np.float32)
    K = rng.normal(size=(B, T, D)).astype(np.float32)
    V = rng.normal(size=(B, T, D)).astype(np.float32)

    # 1. unmasked
    got = fn(Q, K, V, None)
    if got is None:
        print(f"  {_NONE} attention: not implemented yet (expected shape {(B, T, D)})")
        return
    got = np.asarray(got)
    if got.shape != (B, T, D):
        print(f"  {_FAIL} attention shape {got.shape}, expected {(B, T, D)}")
        print(f"       Hint: output should be the weighted sum of V; same (B, T, D) as inputs.")
        return

    expected = _ref_attention(Q, K, V, None)
    if _close(got, expected, tol=1e-4):
        # 2. with causal mask -> different result
        mask = _ref_causal_mask(T)
        got_c = np.asarray(fn(Q, K, V, mask))
        expected_c = _ref_attention(Q, K, V, mask)
        if not _close(got_c, expected_c, tol=1e-4):
            print(f"  {_FAIL} attention without mask is correct, but the mask is being ignored")
            print(f"       Hint: add attn_mask to the scores BEFORE softmax (additive, "
                  f"-inf where blocked).")
            return
        print(f"  {_OK} attention: matches reference for both unmasked and causal cases")
        return

    # Diagnose common mistakes
    no_scale = (Q @ np.swapaxes(K, -1, -2))
    m = no_scale.max(axis=-1, keepdims=True)
    weights_ns = np.exp(no_scale - m)
    weights_ns = weights_ns / weights_ns.sum(axis=-1, keepdims=True)
    out_no_scale = weights_ns @ V
    if _close(got, out_no_scale, tol=1e-4):
        print(f"  {_FAIL} attention: forgot to divide scores by sqrt(d_head)")
        print(f"       Hint: scale = (Q @ K.T) / sqrt(D). Without scaling the softmax saturates "
              f"once D grows.")
        return

    # Maybe softmaxed K instead of the scores
    weights_no_softmax = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(D)
    out_no_softmax = weights_no_softmax @ V
    if _close(got, out_no_softmax, tol=1e-3):
        print(f"  {_FAIL} attention: returned scores @ V without softmax")
        print(f"       Hint: take softmax over the last axis of scores BEFORE multiplying by V.")
        return

    # Maybe mistook V for K
    qkv_swap = _ref_attention(Q, V, K, None)
    if _close(got, qkv_swap, tol=1e-3):
        print(f"  {_FAIL} attention: looks like K and V got swapped")
        print(f"       Hint: scores use K, the weighted sum uses V.")
        return

    print(f"  {_FAIL} attention: numerical mismatch with reference")
    print(f"       Hint: scale = (Q @ K.T) / sqrt(D); add mask if any; softmax along the last "
          f"axis; multiply by V.")


# ---------------------------------------------------------------------------
# 🎭  mask_tokens(ids, mask_id, vocab_size, frac, rng)
# ---------------------------------------------------------------------------

def check_mask_tokens(fn):
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 50, size=(8, 64)).astype(np.int64)
    out = fn(ids, mask_id=0, vocab_size=50, frac=0.15, rng=np.random.default_rng(0))
    if out is None:
        print(f"  {_NONE} mask_tokens: not implemented yet (expected (corrupted, mask_pos))")
        return

    if not (isinstance(out, tuple) and len(out) == 2):
        print(f"  {_FAIL} mask_tokens: should return a tuple (corrupted_ids, mask_positions)")
        return
    corrupted, mask_pos = out
    corrupted = np.asarray(corrupted); mask_pos = np.asarray(mask_pos)

    if corrupted.shape != ids.shape or mask_pos.shape != ids.shape:
        print(f"  {_FAIL} mask_tokens: shapes mismatch  "
              f"corrupted={corrupted.shape}  mask_pos={mask_pos.shape}  ids={ids.shape}")
        return

    if mask_pos.dtype != bool:
        print(f"  {_FAIL} mask_tokens: mask_pos should be a bool array (got {mask_pos.dtype})")
        return

    frac_chosen = mask_pos.mean()
    if abs(frac_chosen - 0.15) > 0.06:
        print(f"  {_FAIL} mask_tokens: chose {frac_chosen:.2%} of tokens, expected ~15%")
        print(f"       Hint: sample random ~ Uniform(0,1) per position and select where < frac.")
        return

    # Among chosen positions, ~80% should be replaced with mask_id (0).
    chosen_corrupted = corrupted[mask_pos]
    chosen_orig      = ids[mask_pos]
    p_mask = (chosen_corrupted == 0).mean()
    if not (0.6 < p_mask < 0.95):
        print(f"  {_FAIL} mask_tokens: of the chosen positions, only {p_mask:.0%} were "
              f"replaced with [MASK]; expected ~80%")
        print(f"       Hint: implement the BERT 80/10/10 mixture: 80% [MASK], 10% random "
              f"non-mask token, 10% unchanged.")
        return

    # Untouched positions should equal the original ids
    if not (corrupted[~mask_pos] == ids[~mask_pos]).all():
        print(f"  {_FAIL} mask_tokens: corrupted positions OUTSIDE the mask were modified")
        print(f"       Hint: only positions selected by the 'chosen' boolean should change.")
        return

    print(f"  {_OK} mask_tokens: ~{frac_chosen:.0%} chosen, of which ~{p_mask:.0%} are [MASK] "
          f"(BERT 80/10/10)")


# ---------------------------------------------------------------------------
# 🎭  masked_token_loss(logits, targets, mask_positions)
# ---------------------------------------------------------------------------

def check_masked_token_loss(fn):
    rng = np.random.default_rng(0)
    B, T, V = 4, 8, 12
    logits = rng.normal(size=(B, T, V)).astype(np.float32)
    targets = rng.integers(0, V, size=(B, T)).astype(np.int64)
    mask_pos = rng.random((B, T)) < 0.3

    got = fn(logits, targets, mask_pos)
    if got is None:
        print(f"  {_NONE} masked_token_loss: not implemented yet")
        return

    log_probs = _stable_log_softmax(logits)
    flat_logp = log_probs[mask_pos]
    flat_tgt  = targets[mask_pos]
    expected = -float(flat_logp[np.arange(len(flat_tgt)), flat_tgt].mean())

    if abs(float(got) - expected) < 1e-3:
        print(f"  {_OK} masked_token_loss = {float(got):.4f}  "
              f"(reference {expected:.4f}, predictions over {int(mask_pos.sum())} positions)")
        return

    # Mistake 1: averaged over ALL positions instead of only masked
    log_probs_full = log_probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
    full_avg = -float(log_probs_full.mean())
    if abs(float(got) - full_avg) < 1e-3:
        print(f"  {_FAIL} masked_token_loss = {float(got):.4f}, averaged over ALL positions "
              f"(expected only at masked, {expected:.4f})")
        print(f"       Hint: index logits and targets by mask_positions before computing CE.")
        return

    # Mistake 2: forgot the negative sign
    if abs(float(got) + expected) < 1e-3:
        print(f"  {_FAIL} masked_token_loss = {float(got):.4f}, missing the negative sign "
              f"(expected {expected:.4f})")
        print(f"       Hint: cross-entropy is the NEGATIVE log-probability of the target.")
        return

    # Mistake 3: used softmax probability instead of log-softmax
    probs = np.exp(log_probs)
    flat_p = probs[mask_pos]
    avg_prob = float(flat_p[np.arange(len(flat_tgt)), flat_tgt].mean())
    if abs(float(got) - avg_prob) < 1e-3:
        print(f"  {_FAIL} masked_token_loss = {float(got):.4f}, looks like the average "
              f"PROBABILITY rather than -log probability")
        print(f"       Hint: use log-softmax (or log of softmax) and negate.")
        return

    print(f"  {_FAIL} masked_token_loss = {float(got):.4f}, expected {expected:.4f}")
    print(f"       Hint: cross-entropy at masked positions = -mean(log_softmax(logits)[masked, target]).")


# ---------------------------------------------------------------------------
# 🧩  geometric_span_mask(ids, mask_id, mean_span, mask_frac, rng)
# ---------------------------------------------------------------------------

def check_geometric_span_mask(fn):
    ids = np.arange(1, 1 + 200, dtype=np.int64).reshape(4, 50)
    out = fn(ids, mask_id=0, mean_span=3.0, mask_frac=0.20,
             rng=np.random.default_rng(0))
    if out is None:
        print(f"  {_NONE} geometric_span_mask: not implemented yet")
        return
    if not (isinstance(out, tuple) and len(out) == 2):
        print(f"  {_FAIL} geometric_span_mask: return (corrupted_ids, mask_positions)")
        return
    corrupted, mask_pos = out
    corrupted = np.asarray(corrupted); mask_pos = np.asarray(mask_pos)

    if mask_pos.shape != ids.shape or corrupted.shape != ids.shape:
        print(f"  {_FAIL} geometric_span_mask: shapes mismatch")
        return

    if mask_pos.dtype != bool:
        print(f"  {_FAIL} geometric_span_mask: mask_positions should be a bool array")
        return

    frac = mask_pos.mean()
    if abs(frac - 0.20) > 0.10:
        print(f"  {_FAIL} geometric_span_mask: covered {frac:.0%} of tokens "
              f"(expected ~20%)")
        return

    # All masked positions should hold mask_id; unmasked should be untouched.
    if not (corrupted[mask_pos] == 0).all():
        print(f"  {_FAIL} geometric_span_mask: not all masked positions hold mask_id (0)")
        return
    if not (corrupted[~mask_pos] == ids[~mask_pos]).all():
        print(f"  {_FAIL} geometric_span_mask: positions outside the mask were modified")
        return

    # Spans should be CONTIGUOUS — average run length should exceed 1.
    runs = []
    for row in mask_pos:
        i = 0
        while i < len(row):
            if row[i]:
                j = i
                while j < len(row) and row[j]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
    avg_run = np.mean(runs) if runs else 0.0
    if avg_run < 1.4:
        print(f"  {_FAIL} geometric_span_mask: average masked-run length is {avg_run:.2f}; "
              f"that's basically per-token masking, not span masking")
        print(f"       Hint: pick a span START, then mask the next L contiguous positions, "
              f"where L is sampled geometrically.")
        return

    print(f"  {_OK} geometric_span_mask: covered {frac:.0%} with average span length "
          f"{avg_run:.1f}")


# ---------------------------------------------------------------------------
# 📜  causal_mask(T)
# ---------------------------------------------------------------------------

def check_causal_mask(fn):
    T = 5
    got = fn(T)
    if got is None:
        print(f"  {_NONE} causal_mask: not implemented yet (expected (5, 5) additive mask)")
        return
    got = np.asarray(got)

    if got.shape != (T, T):
        print(f"  {_FAIL} causal_mask shape {got.shape}, expected {(T, T)}")
        return

    # Lower triangle (incl diagonal) should be 0; strict upper triangle should be -inf.
    for i in range(T):
        for j in range(T):
            if j <= i:
                if not np.isclose(got[i, j], 0.0, atol=1e-3):
                    print(f"  {_FAIL} causal_mask[{i},{j}] = {got[i, j]}, expected 0 "
                          f"(allowed cell)")
                    print(f"       Hint: position i can attend to positions j <= i, so "
                          f"those cells should be 0 in the additive mask.")
                    return
            else:
                if not (np.isneginf(got[i, j]) or got[i, j] < -1e6):
                    print(f"  {_FAIL} causal_mask[{i},{j}] = {got[i, j]}, expected -inf "
                          f"(blocked cell)")
                    print(f"       Hint: positions j > i are 'in the future' and must "
                          f"contribute zero softmax weight; that's -inf added before softmax.")
                    return

    print(f"  {_OK} causal_mask: lower triangle 0, strict upper triangle blocked")


# ---------------------------------------------------------------------------
# 📜  next_token_loss(logits, targets)
# ---------------------------------------------------------------------------

def check_next_token_loss(fn):
    rng = np.random.default_rng(0)
    B, T, V = 3, 7, 11
    logits  = rng.normal(size=(B, T, V)).astype(np.float32)
    targets = rng.integers(0, V, size=(B, T)).astype(np.int64)

    got = fn(logits, targets)
    if got is None:
        print(f"  {_NONE} next_token_loss: not implemented yet")
        return

    # Reference: predict targets[:, 1:] from logits[:, :-1]
    log_probs = _stable_log_softmax(logits[:, :-1, :])
    tgt_shift = targets[:, 1:]
    flat = log_probs.reshape(-1, V)
    flat_tgt = tgt_shift.reshape(-1)
    expected = -float(flat[np.arange(len(flat_tgt)), flat_tgt].mean())

    if abs(float(got) - expected) < 1e-3:
        print(f"  {_OK} next_token_loss = {float(got):.4f}  (reference {expected:.4f})")
        return

    # Did they NOT shift the targets?
    log_probs_no_shift = _stable_log_softmax(logits)
    flat_ns = log_probs_no_shift.reshape(-1, V)
    no_shift = -float(flat_ns[np.arange(B * T), targets.reshape(-1)].mean())
    if abs(float(got) - no_shift) < 1e-3:
        print(f"  {_FAIL} next_token_loss = {float(got):.4f}, computed without shifting "
              f"(expected {expected:.4f})")
        print(f"       Hint: position t predicts token t+1, so use logits[:, :-1] vs "
              f"targets[:, 1:].")
        return

    print(f"  {_FAIL} next_token_loss = {float(got):.4f}, expected {expected:.4f}")
    print(f"       Hint: -mean(log_softmax(logits[:, :-1])[..., targets[:, 1:]]).")


# ---------------------------------------------------------------------------
# 📜  temperature_sample(logits, temperature, rng)
# ---------------------------------------------------------------------------

def check_temperature_sample(fn):
    # Logits chosen so position 5 wins overwhelmingly at low temperature.
    logits = np.array([0.1, -0.2, 0.0, 0.0, 0.5, 5.0, 0.1, 0.0], dtype=np.float32)
    n_trials = 400

    rng = np.random.default_rng(0)
    probe = fn(logits, 0.1, rng)
    if probe is None:
        print(f"  {_NONE} temperature_sample: not implemented yet")
        return
    samples_low = [int(probe)] + [int(fn(logits, 0.1, rng)) for _ in range(n_trials - 1)]
    counts_low = np.bincount(samples_low, minlength=len(logits))
    if counts_low.argmax() != 5 or counts_low[5] < 0.85 * n_trials:
        print(f"  {_FAIL} temperature_sample at T=0.1 is not concentrating on the peak")
        print(f"       Hint: divide logits by T BEFORE softmax. Low T sharpens the distribution.")
        return

    rng = np.random.default_rng(0)
    samples_high = [int(fn(logits, 2.0, rng)) for _ in range(n_trials)]
    counts_high = np.bincount(samples_high, minlength=len(logits))
    if counts_high.argmax() == 5 and counts_high[5] > 0.7 * n_trials:
        print(f"  {_FAIL} temperature_sample at T=2.0 still concentrates almost entirely on "
              f"the peak; expected a flatter distribution")
        print(f"       Hint: dividing by a LARGER T should flatten softmax, not sharpen it.")
        return

    print(f"  {_OK} temperature_sample: concentrates on peak at T=0.1, spreads at T=2.0")


# ---------------------------------------------------------------------------
# 🪞  ema_update(teacher_params, student_params, m)
# ---------------------------------------------------------------------------

def check_ema_update(fn):
    rng = np.random.default_rng(0)
    teacher = {"a": rng.normal(size=(3, 4)).astype(np.float32),
               "b": rng.normal(size=5).astype(np.float32)}
    student = {"a": rng.normal(size=(3, 4)).astype(np.float32),
               "b": rng.normal(size=5).astype(np.float32)}
    m = 0.99

    expected = {k: m * teacher[k] + (1 - m) * student[k] for k in teacher}
    got = fn(teacher, student, m)

    if got is None:
        print(f"  {_NONE} ema_update: not implemented yet")
        return
    if not isinstance(got, dict) or set(got.keys()) != set(teacher.keys()):
        print(f"  {_FAIL} ema_update: should return a dict with the same keys as the teacher")
        return

    if all(_close(got[k], expected[k], tol=1e-4) for k in teacher):
        print(f"  {_OK} ema_update: teacher = {m}*teacher + {1-m:.2f}*student per key")
        return

    # Did they swap the roles?
    swapped = {k: m * student[k] + (1 - m) * teacher[k] for k in teacher}
    if all(_close(got[k], swapped[k], tol=1e-4) for k in teacher):
        print(f"  {_FAIL} ema_update: looks like teacher and student were swapped")
        print(f"       Hint: the TEACHER updates slowly toward the STUDENT, not vice versa.")
        return

    print(f"  {_FAIL} ema_update: numerical mismatch")
    print(f"       Hint: for each key, teacher_new = m * teacher + (1 - m) * student.")


# ---------------------------------------------------------------------------
# 🪞  jepa_loss(student_h, teacher_h, mask_pos)
# ---------------------------------------------------------------------------

def check_jepa_loss(fn):
    rng = np.random.default_rng(0)
    B, T, D = 2, 6, 4
    student_h = rng.normal(size=(B, T, D)).astype(np.float32)
    teacher_h = rng.normal(size=(B, T, D)).astype(np.float32)
    mask_pos  = rng.random((B, T)) < 0.4

    got = fn(student_h, teacher_h, mask_pos)
    if got is None:
        print(f"  {_NONE} jepa_loss: not implemented yet")
        return

    diff = (student_h - teacher_h)[mask_pos]
    expected = float((diff ** 2).mean())

    if abs(float(got) - expected) < 1e-4:
        print(f"  {_OK} jepa_loss = {float(got):.4f}  (MSE at {int(mask_pos.sum())} masked positions)")
        return

    # Used all positions, not just masked
    full_mse = float(((student_h - teacher_h) ** 2).mean())
    if abs(float(got) - full_mse) < 1e-4:
        print(f"  {_FAIL} jepa_loss = {float(got):.4f}, averaged over ALL positions")
        print(f"       Hint: only the MASKED positions contribute; index by mask_pos first.")
        return

    print(f"  {_FAIL} jepa_loss = {float(got):.4f}, expected {expected:.4f}")
    print(f"       Hint: MSE over the masked positions only — mean of (student - teacher)**2 "
          f"at mask_pos.")
