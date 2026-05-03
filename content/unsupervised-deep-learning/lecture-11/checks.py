"""Auto-checker helpers for lecture-11, Reinforcement Learning."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3


def _close(a, b, tol=_TOL):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ---------------------------------------------------------------------------
# 🗺️  bellman_backup(V, P, R, gamma)
# ---------------------------------------------------------------------------
def check_bellman_backup(fn):
    """Tiny 3-state, 2-action MDP with a known one-step backup."""
    rng = np.random.default_rng(0)
    n_s, n_a = 3, 2
    P = np.zeros((n_s, n_a, n_s))
    # Deterministic transitions cycling through states.
    P[0, 0, 1] = 1.0; P[0, 1, 2] = 1.0
    P[1, 0, 2] = 1.0; P[1, 1, 0] = 1.0
    P[2, 0, 0] = 1.0; P[2, 1, 1] = 1.0
    R = rng.normal(size=(n_s, n_a))
    V = rng.normal(size=n_s)
    gamma = 0.9

    got = fn(V, P, R, gamma)
    if got is None:
        print(f"  {_NONE} bellman_backup: not implemented yet (expected shape {(n_s, n_a)})")
        return

    got = np.asarray(got, dtype=float)
    expected = R + gamma * (P @ V)        # broadcast: P @ V -> (n_s, n_a)

    if got.shape != (n_s, n_a):
        print(f"  {_FAIL} bellman_backup shape {got.shape}, expected {(n_s, n_a)}")
        return

    if _close(got, expected):
        print(f"  {_OK} bellman_backup: Q matches R + γ P V on the toy MDP")
        return

    # Forgot the discount factor
    if _close(got, R + (P @ V)):
        print(f"  {_FAIL} bellman_backup: missing the discount factor γ")
        print(f"       Hint: Q[s,a] = R[s,a] + γ · Σ_s' P[s,a,s'] V[s'].")
        return

    # Forgot the immediate reward
    if _close(got, gamma * (P @ V)):
        print(f"  {_FAIL} bellman_backup: missing the immediate reward R[s, a]")
        print(f"       Hint: Q decomposes into R + γ · expected next-state value.")
        return

    # Took max over actions instead of returning Q
    if got.shape != (n_s, n_a):
        pass
    if _close(got, expected.max(axis=1)) and got.shape == (n_s,):
        print(f"  {_FAIL} bellman_backup: returned V (max over actions) instead of Q")
        print(f"       Hint: keep both axes — value_iteration takes the max later.")
        return

    print(f"  {_FAIL} bellman_backup: numerical mismatch")
    print(f"       got shape {got.shape}, sample row got[0]={got[0]} expected {expected[0]}.")


# ---------------------------------------------------------------------------
# 🗺️  value_iteration(P, R, gamma, n_iters)
# ---------------------------------------------------------------------------
def check_value_iteration(fn):
    """Two-state MDP with a known closed-form V*."""
    n_s, n_a = 2, 2
    P = np.zeros((n_s, n_a, n_s))
    # Action 0 stays; action 1 swaps. Reward is a function of the CURRENT state:
    # state 0 always emits +1, state 1 always emits 0.
    P[0, 0, 0] = 1.0; P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0; P[1, 1, 0] = 1.0
    R = np.array([[1.0, 1.0],
                  [0.0, 0.0]])
    gamma = 0.5
    # Optimal: from state 0, stay (+1 every step) -> V*[0] = 1 / (1 - 0.5) = 2.
    # From state 1, swap to state 0 -> V*[1] = 0 + 0.5 * 2 = 1.
    expected = np.array([2.0, 1.0])

    got = fn(P, R, gamma, 200)
    if got is None:
        print(f"  {_NONE} value_iteration: not implemented yet (expected V ≈ [2.0, 1.0])")
        return

    got = np.asarray(got, dtype=float)
    if got.shape != (n_s,):
        print(f"  {_FAIL} value_iteration shape {got.shape}, expected {(n_s,)}")
        return

    if _close(got, expected, tol=1e-2):
        print(f"  {_OK} value_iteration: V = {[round(float(x), 3) for x in got]}  "
              f"(expected ≈ [2.0, 1.0])")
        return

    # Took min instead of max
    Vmin = np.array([0.0, 0.0])
    if _close(got, Vmin, tol=1e-2):
        print(f"  {_FAIL} value_iteration: looks like you took min over actions instead of max")
        print(f"       Hint: V[s] = max_a Q[s, a].")
        return

    # Returned Q instead of V (mismatched shape would have triggered above)
    print(f"  {_FAIL} value_iteration: got {got}, expected ≈ {expected}")
    print(f"       Hint: V_{{k+1}}[s] = max_a (R[s,a] + γ · P[s,a,:] @ V_k).")


# ---------------------------------------------------------------------------
# 🎯  td_target(r, gamma, q_next_max, done)
# ---------------------------------------------------------------------------
def check_td_target(fn):
    cases = [
        # (r,    gamma, q_next, done, expected)
        (1.0,   0.9,   2.0,    False, 1.0 + 0.9 * 2.0),
        (-1.0,  0.95,  0.5,    False, -1.0 + 0.95 * 0.5),
        (1.0,   0.9,   3.0,    True,  1.0),                # terminal: bootstrap dropped
        (0.0,   0.99,  0.0,    False, 0.0),
    ]
    got_first = fn(*cases[0][:4])
    if got_first is None:
        print(f"  {_NONE} td_target: not implemented yet")
        return

    fails = []
    for r, g, q, d, exp in cases:
        got = fn(r, g, q, d)
        if not (got is not None and abs(float(got) - exp) < 1e-6):
            fails.append((r, g, q, d, exp, got))

    if not fails:
        print(f"  {_OK} td_target: {_close(0,0)}  (terminal correctly drops the bootstrap)")
        print(f"       checks passed on 4 cases including done=True.")
        return

    # Likely mistake: ignored `done`
    if all(abs(float(got) - (r + g * q)) < 1e-6 for r, g, q, d, _, got in fails):
        print(f"  {_FAIL} td_target: ignored the `done` flag — bootstrapping past terminal")
        print(f"       Hint: target = r + (1 − done) · γ · max_a' Q(s', a').")
        return

    r, g, q, d, exp, got = fails[0]
    print(f"  {_FAIL} td_target({r}, {g}, {q}, done={d}) = {got}, expected {exp}")


# ---------------------------------------------------------------------------
# 🎯  q_learning_update(Q, s, a, r, s_next, done, alpha, gamma)
# ---------------------------------------------------------------------------
def check_q_learning_update(fn):
    Q = np.zeros((4, 3))
    Q[1] = [0.5, 1.0, -0.2]
    Q_pre = Q.copy()
    Q_after = fn(Q, 0, 2, 0.0, 1, False, 0.5, 0.9)

    if Q_after is None:
        print(f"  {_NONE} q_learning_update: not implemented yet")
        return

    Q_after = np.asarray(Q_after, dtype=float)
    expected = Q_pre.copy()
    target = 0.0 + 0.9 * Q_pre[1].max()
    expected[0, 2] = Q_pre[0, 2] + 0.5 * (target - Q_pre[0, 2])

    if _close(Q_after, expected):
        print(f"  {_OK} q_learning_update: Q[0, 2] = {Q_after[0, 2]:.4f}  "
              f"(expected {expected[0, 2]:.4f})")
        return

    # Forgot 1 - alpha factor (overshot)
    no_blend = Q_pre.copy(); no_blend[0, 2] = target
    if _close(Q_after, no_blend):
        print(f"  {_FAIL} q_learning_update: replaced Q[s, a] with the target (no learning rate)")
        print(f"       Hint: Q[s, a] ← Q[s, a] + α · (target − Q[s, a]).")
        return

    # Used Q[s_next, a] instead of max
    sa_target = 0.0 + 0.9 * Q_pre[1, 2]
    sa_blend = Q_pre.copy()
    sa_blend[0, 2] = Q_pre[0, 2] + 0.5 * (sa_target - Q_pre[0, 2])
    if _close(Q_after, sa_blend):
        print(f"  {_FAIL} q_learning_update: bootstrapped from Q[s_next, a] (the same action)")
        print(f"       Hint: Q-learning is OFF-policy — bootstrap from max_a' Q[s_next, a'].")
        return

    # Updated wrong cell
    if _close(Q_after, Q_pre):
        print(f"  {_FAIL} q_learning_update: nothing changed — did you write back to Q?")
        return

    print(f"  {_FAIL} q_learning_update: Q[0, 2] = {Q_after[0, 2]:.4f}, "
          f"expected {expected[0, 2]:.4f}")


# ---------------------------------------------------------------------------
# 📈  compute_returns(rewards, gamma)
# ---------------------------------------------------------------------------
def check_compute_returns(fn):
    rewards = [0.0, 0.0, 0.0, 1.0]
    gamma = 0.9
    expected = np.array([gamma ** 3, gamma ** 2, gamma, 1.0])

    got = fn(rewards, gamma)
    if got is None:
        print(f"  {_NONE} compute_returns: not implemented yet")
        return

    got = np.asarray(got, dtype=float)
    if got.shape != (4,):
        print(f"  {_FAIL} compute_returns shape {got.shape}, expected (4,)")
        return

    if _close(got, expected):
        print(f"  {_OK} compute_returns: {[round(float(x), 3) for x in got]}  "
              f"(expected {[round(float(x), 3) for x in expected]})")
        return

    # Returned the SAME total return at every timestep (forgot per-timestep discount)
    total = sum(gamma ** k * r for k, r in enumerate(rewards))
    if _close(got, np.full(4, total)):
        print(f"  {_FAIL} compute_returns: returned the episode return at every t")
        print(f"       Hint: G_t = Σ_{{k≥t}} γ^{{k−t}} r_k — the discount window slides with t.")
        return

    # Forgot the discount factor entirely
    no_disc = np.array([1.0, 1.0, 1.0, 1.0])
    if _close(got, no_disc):
        print(f"  {_FAIL} compute_returns: γ ignored — looks like cumulative undiscounted return")
        print(f"       Hint: each future reward is multiplied by γ^{{k−t}}.")
        return

    # Discounted forward instead of backward (G_t = Σ_{k>=t} γ^k r_k, no shift)
    fwd = np.array([sum(gamma ** k * rewards[k] for k in range(t, 4)) for t in range(4)])
    if _close(got, fwd):
        print(f"  {_FAIL} compute_returns: discount exponent uses absolute k, not k − t")
        print(f"       Hint: G_t starts at γ^0 · r_t — reset the exponent at each t.")
        return

    print(f"  {_FAIL} compute_returns: got {got}, expected {expected}")


# ---------------------------------------------------------------------------
# 📈  policy_gradient(theta, phis, actions, advantages)
# ---------------------------------------------------------------------------
def check_policy_gradient(fn):
    """Single-step toy: 2 actions, 2 features, one transition with advantage 1.0.
    The expected gradient is (1[k==a] − π(k|s)) · phi for each row k."""
    theta = np.zeros((2, 2))
    phi = np.array([1.0, 0.5])
    a = 0
    adv = 1.0
    got = fn(theta.copy(), [phi], [a], np.array([adv]))

    if got is None:
        print(f"  {_NONE} policy_gradient: not implemented yet")
        return

    got = np.asarray(got, dtype=float)
    if got.shape != theta.shape:
        print(f"  {_FAIL} policy_gradient shape {got.shape}, expected {theta.shape}")
        return

    # At theta = 0, π is uniform [0.5, 0.5]; gradient row 0: (1 - 0.5) · phi = 0.5 · phi.
    # Row 1: (0 - 0.5) · phi = -0.5 · phi.
    expected = np.array([[0.5, 0.25], [-0.5, -0.25]])

    if _close(got, expected):
        print(f"  {_OK} policy_gradient: gradient matches (1[k==a] − π(k|s)) · φ on the toy step")
        return

    # Common mistake: forgot to subtract π(k|s) (used raw indicator)
    raw_indicator = np.array([[1.0, 0.5], [0.0, 0.0]])
    if _close(got, raw_indicator):
        print(f"  {_FAIL} policy_gradient: used the raw indicator — missing the −π(k|s) term")
        print(f"       Hint: ∇log π(a|s) for action a equals (1[k=a] − π(k|s)) · φ — every "
              f"row gets a term, including the non-chosen actions.")
        return

    # Forgot the advantage scaling
    if _close(got, expected * 0.0):
        print(f"  {_FAIL} policy_gradient: zeroed everything — did you scale by advantage?")
        return

    # Returned the negative (treated it as a loss gradient)
    if _close(got, -expected):
        print(f"  {_FAIL} policy_gradient: sign flipped — return the gradient of E[adv · log π], "
              f"NOT a loss.  (The training loop ascends.)")
        return

    print(f"  {_FAIL} policy_gradient: got\n{got}\nexpected\n{expected}")


# ---------------------------------------------------------------------------
# ✂️  ppo_clipped_objective(ratios, advantages, eps_clip)
# ---------------------------------------------------------------------------
def check_ppo_clipped_objective(fn):
    ratios     = np.array([1.0,  1.5,  0.5,  1.5, 0.5])
    advantages = np.array([1.0,  1.0,  1.0, -1.0, -1.0])
    eps_clip   = 0.2

    got = fn(ratios, advantages, eps_clip)
    if got is None:
        print(f"  {_NONE} ppo_clipped_objective: not implemented yet")
        return

    clipped = np.clip(ratios, 1 - eps_clip, 1 + eps_clip)
    expected = float(np.mean(np.minimum(ratios * advantages, clipped * advantages)))

    if abs(float(got) - expected) < 1e-6:
        print(f"  {_OK} ppo_clipped_objective = {float(got):.4f}  "
              f"(reference {expected:.4f})")
        return

    # Forgot to clip
    no_clip = float(np.mean(ratios * advantages))
    if abs(float(got) - no_clip) < 1e-6:
        print(f"  {_FAIL} ppo_clipped_objective: did not clip — this is plain PG, not PPO")
        print(f"       Hint: take the elementwise min of (r·A) and (clip(r, 1−ε, 1+ε)·A).")
        return

    # Took max instead of min
    bad_max = float(np.mean(np.maximum(ratios * advantages, clipped * advantages)))
    if abs(float(got) - bad_max) < 1e-6:
        print(f"  {_FAIL} ppo_clipped_objective: used max instead of min")
        print(f"       Hint: PPO uses min so the clip is a PESSIMISTIC bound on the surrogate.")
        return

    # Forgot the advantage factor
    bad_no_adv = float(np.mean(np.minimum(ratios, clipped)))
    if abs(float(got) - bad_no_adv) < 1e-6:
        print(f"  {_FAIL} ppo_clipped_objective: missing the advantage multiplier")
        print(f"       Hint: both branches of min are r·A and clip(r)·A, not r and clip(r).")
        return

    # Sign flipped
    if abs(float(got) + expected) < 1e-6:
        print(f"  {_FAIL} ppo_clipped_objective: sign is flipped — return the surrogate to be "
              f"MAXIMISED (training negates internally).")
        return

    print(f"  {_FAIL} ppo_clipped_objective = {float(got):.4f}, expected {expected:.4f}")


# ---------------------------------------------------------------------------
# 🔭  pick_epsilon_greedy(Q_row, eps, rng)
# ---------------------------------------------------------------------------
def check_pick_epsilon_greedy(fn):
    Q_row = np.array([0.1, 0.9, 0.4])

    # eps = 0.0  -> always argmax
    rng = np.random.default_rng(0)
    greedy_picks = [int(fn(Q_row, 0.0, rng)) for _ in range(50)]
    if greedy_picks[0] is None:
        print(f"  {_NONE} pick_epsilon_greedy: not implemented yet")
        return
    if not all(p == 1 for p in greedy_picks):
        # Possibly broke ties weirdly; check majority
        from collections import Counter
        c = Counter(greedy_picks)
        most, freq = c.most_common(1)[0]
        if most != 1 or freq < 45:
            print(f"  {_FAIL} pick_epsilon_greedy(eps=0): not always picking argmax "
                  f"(picks={dict(c)}, expected mostly action 1)")
            print(f"       Hint: with eps=0 the rule degenerates to np.argmax(Q_row).")
            return

    # eps = 1.0 -> uniform over actions
    rng = np.random.default_rng(1)
    rand_picks = [int(fn(Q_row, 1.0, rng)) for _ in range(600)]
    counts = np.bincount(rand_picks, minlength=3)
    fracs = counts / counts.sum()
    if not (np.all(fracs > 0.25) and np.all(fracs < 0.42)):
        print(f"  {_FAIL} pick_epsilon_greedy(eps=1): action distribution {fracs} "
              f"is not uniform")
        print(f"       Hint: with eps=1 every action should appear ≈ 1/n_actions of the time.")
        return

    # eps = 0.5 -> mix; argmax should still be the modal action
    rng = np.random.default_rng(2)
    mix = [int(fn(Q_row, 0.5, rng)) for _ in range(400)]
    counts = np.bincount(mix, minlength=3)
    if int(np.argmax(counts)) != 1:
        print(f"  {_FAIL} pick_epsilon_greedy(eps=0.5): action 1 (the argmax) is not "
              f"the most-frequent pick (counts={counts.tolist()})")
        return

    print(f"  {_OK} pick_epsilon_greedy: argmax under eps=0, uniform under eps=1, "
          f"argmax-dominant under eps=0.5")
