"""Mini-Pong environment, training loops, and plot helpers for lecture-11.

Pure NumPy and SciPy. The MDP is small enough (6x5x2x5 = 300 non-terminal
states) that value iteration converges in milliseconds and tabular Q-learning
in well under a second; the policy-gradient methods reuse a 3-feature linear
softmax with 9 weights total, so PPO rolls out and optimises with finite
differences on a tiny scipy call.
"""

from __future__ import annotations

import sys
sys.path.insert(0, '../..')
from plot_style import *  # noqa: F401,F403

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

W = 7   # grid width  (paddle column = W - 1)
H = 5   # grid height
N_ACTIONS = 3                                    # 0 = up, 1 = stay, 2 = down
N_STATES  = (W - 1) * H * 2 * H + 1              # +1 for the absorbing terminal
TERMINAL_INDEX = N_STATES - 1
N_FEATURES = 3                                   # for the linear softmax policy

ACTION_NAMES = ('up', 'stay', 'down')


def state_index(state):
    """Pack a (bx, by, vy, py) tuple into a single integer in [0, N_STATES).

    The terminal slot (bx == W - 1, ball arrived at the paddle column) is
    always TERMINAL_INDEX regardless of the other components.
    """
    bx, by, vy, py = state
    if bx == W - 1:
        return TERMINAL_INDEX
    vy_idx = 0 if vy == -1 else 1
    return ((bx * H + by) * 2 + vy_idx) * H + py


def index_to_state(idx):
    """Inverse of state_index. Returns None for the terminal slot."""
    if idx == TERMINAL_INDEX:
        return None
    py = idx % H
    idx //= H
    vy = -1 if (idx % 2) == 0 else 1
    idx //= 2
    by = idx % H
    bx = idx // H
    return (bx, by, vy, py)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MiniPongEnv:
    """Tiny Pong-like catcher.

    Ball spawns at column 0, moves rightward one column per step and bounces
    off the top and bottom walls.  A 1-cell-tall paddle sits in column W-1
    and moves up / stay / down each step.  Reward is 0 at every step until
    the ball reaches the paddle column, at which point the agent receives
    +1 if ``py == by`` (catch) or -1 otherwise (miss); the episode then ends.

    The environment is fully deterministic given the initial state.  Only
    ``reset`` is stochastic (draws a random ball y and vy); set ``sparse=True``
    for the §🔭 exploration variant where misses score 0 instead of -1.
    """

    W = W
    H = H
    n_actions = N_ACTIONS

    def __init__(self, sparse: bool = False):
        self.sparse = sparse
        self.bx = 0
        self.by = H // 2
        self.vy = 1
        self.py = (H - 1) // 2
        self.done = False

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.bx = 0
        self.by = int(rng.integers(1, self.H - 1))
        self.vy = int(rng.choice([-1, 1]))
        self.py = (self.H - 1) // 2
        self.done = False
        return self.state()

    def state(self):
        return (self.bx, self.by, self.vy, self.py)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        # 1. Paddle moves first.
        if action == 0:
            self.py = max(0, self.py - 1)
        elif action == 2:
            self.py = min(self.H - 1, self.py + 1)
        # 2. Ball moves one column right; bounces off top / bottom walls.
        new_bx = self.bx + 1
        new_by = self.by + self.vy
        new_vy = self.vy
        if new_by < 0:
            new_by = 1; new_vy = 1
        elif new_by >= self.H:
            new_by = self.H - 2; new_vy = -1
        self.bx, self.by, self.vy = new_bx, new_by, new_vy
        # 3. Terminal at the paddle column.
        reward = 0.0
        if self.bx == self.W - 1:
            self.done = True
            if self.py == self.by:
                reward = 1.0
            else:
                reward = 0.0 if self.sparse else -1.0
        return self.state(), reward, self.done


# ---------------------------------------------------------------------------
# MDP enumeration -> (P, R) tensors for value iteration
# ---------------------------------------------------------------------------

def build_mdp(sparse: bool = False):
    """Enumerate every (state, action) and stack into transition / reward
    tensors.  Returns ``(P, R)`` of shapes ``(N_STATES, N_ACTIONS, N_STATES)``
    and ``(N_STATES, N_ACTIONS)``.  Transitions are deterministic, so each
    row of ``P[s, a, :]`` is a one-hot.
    """
    env = MiniPongEnv(sparse=sparse)
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))

    for bx in range(env.W - 1):
        for by in range(env.H):
            for vy in (-1, 1):
                for py in range(env.H):
                    s_idx = state_index((bx, by, vy, py))
                    for a in range(env.n_actions):
                        env.bx, env.by, env.vy, env.py = bx, by, vy, py
                        env.done = False
                        s_next, r, _ = env.step(a)
                        P[s_idx, a, state_index(s_next)] = 1.0
                        R[s_idx, a] = r
    # Terminal absorbs.
    for a in range(env.n_actions):
        P[TERMINAL_INDEX, a, TERMINAL_INDEX] = 1.0
    return P, R


# ---------------------------------------------------------------------------
# Black-box helpers: Bellman backup, TD target, discounted returns.
#
# Provided to students so the core notebook stays focused on the *structural*
# exercises (value_iteration, q_learning_update, policy_gradient).  Each one
# is a single line of arithmetic; the conceptual content lives in the
# functions that consume them.
# ---------------------------------------------------------------------------

def bellman_backup(V, P, R, gamma):
    """One-step Bellman backup.  ``Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] V[s']``.

    Inputs
    ------
    V : (N_STATES,)
    P : (N_STATES, N_ACTIONS, N_STATES) row-stochastic transition tensor
    R : (N_STATES, N_ACTIONS)
    gamma : float in (0, 1)

    Returns
    -------
    Q : (N_STATES, N_ACTIONS)
    """
    return R + gamma * (P @ V)


def td_target(r, gamma, q_next_max, done):
    """TD(0) bootstrap target.  Drops the bootstrap term at terminal transitions
    so we do not propagate value past an absorbing state.
    """
    return r + (1.0 - float(done)) * gamma * q_next_max


def compute_returns(rewards, gamma):
    """Per-timestep discounted return ``G_t = sum_{k>=t} gamma^(k - t) r_k``.

    Iterates from the last timestep backwards, which is the standard one-pass
    implementation; vectorised solutions also work but are not faster on
    episode lengths this small.
    """
    T = len(rewards)
    out = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def policy_iteration(P, R, gamma, n_iters: int = 20, eval_iters: int = 50):
    """Policy iteration: alternate policy evaluation and policy improvement.

    Starts from the all-zeros (always 'up') policy.  Each outer step solves
    ``V^pi`` iteratively under the current policy, then sets the new policy
    to be greedy with respect to the resulting Q.  Converges to the same
    ``V*`` as value iteration; the two differ in per-iteration cost and in
    how many outer iterations they take.

    Returns
    -------
    V  : (N_STATES,) value of the final policy (= V*)
    pi : (N_STATES,) the optimal policy as one action index per state.
    """
    n_states = P.shape[0]
    pi = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    state_idx = np.arange(n_states)
    for _ in range(n_iters):
        # Policy evaluation under the current pi.
        for _ in range(eval_iters):
            P_pi = P[state_idx, pi]      # (n_states, n_states)
            R_pi = R[state_idx, pi]      # (n_states,)
            V = R_pi + gamma * (P_pi @ V)
        # Policy improvement.
        Q = bellman_backup(V, P, R, gamma)
        new_pi = Q.argmax(axis=1)
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return V, pi


def initial_state_distribution():
    """Uniform distribution over the 6 starting states (bx=0, by in {1,2,3},
    vy in {-1, +1}, py = 2)."""
    initial_idxs = []
    for by in (1, 2, 3):
        for vy in (-1, 1):
            initial_idxs.append(state_index((0, by, vy, (H - 1) // 2)))
    d0 = np.zeros(N_STATES)
    for i in initial_idxs:
        d0[i] = 1.0 / len(initial_idxs)
    return d0


def expected_value_under(V):
    """E_{s0 ~ d0}[V(s0)] — convenient scalar summary of a value function."""
    return float(initial_state_distribution() @ V)


# ---------------------------------------------------------------------------
# Linear softmax policy
# ---------------------------------------------------------------------------

def predicted_impact_y(bx, by, vy):
    """Where the ball will land at column ``W - 1``, with reflection off the
    top and bottom walls.  Pure geometry: a straight line in the unfolded
    plane folded back into ``[0, H - 1]`` via the standard mirror trick.
    """
    steps = W - 1 - bx
    raw = by + vy * steps
    period = 2 * (H - 1)
    m = raw % period
    if m < 0:
        m += period
    return m if m <= H - 1 else period - m


def features(state):
    """Hand-crafted 3-D features for the linear softmax policy.

    [ (py - by) / (H - 1)              current paddle-ball vertical offset
    , (py - impact_y) / (H - 1)        offset to PREDICTED impact (bounce-aware)
    ,  1                                bias                                    ]

    The second feature is the engineered signal the chapter hints at: the
    linear policy alone cannot represent the bounce arithmetic (it is
    piecewise non-linear in ``by, vy, bx``); pre-computing impact_y lets the
    policy combine immediate tracking with anticipation.
    """
    bx, by, vy, py = state
    impact = predicted_impact_y(bx, by, vy)
    return np.array([(py - by) / (H - 1),
                     (py - impact) / (H - 1),
                     1.0])


def policy_probs(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Softmax over actions; ``theta`` has shape (N_ACTIONS, N_FEATURES)."""
    logits = theta @ phi
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def sample_action(theta, phi, rng):
    p = policy_probs(theta, phi)
    return int(rng.choice(N_ACTIONS, p=p))


def greedy_action(theta, phi):
    return int(np.argmax(theta @ phi))


# ---------------------------------------------------------------------------
# Rollouts and evaluation
# ---------------------------------------------------------------------------

def rollout_episode(env, action_fn, seed=None):
    """Run one episode under ``action_fn(state) -> action``; return a dict
    with per-step states, actions, rewards, and the final return.
    """
    s = env.reset(seed=seed)
    states, actions, rewards = [s], [], []
    while not env.done:
        a = int(action_fn(s))
        s, r, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return dict(states=states, actions=actions, rewards=rewards,
                total_return=float(sum(rewards)))


def evaluate_policy(action_fn, n_episodes=200, seed=0, sparse=False):
    """Average per-episode return under ``action_fn``."""
    env = MiniPongEnv(sparse=sparse)
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_episodes):
        ep = rollout_episode(env, action_fn, seed=int(rng.integers(0, 1 << 31)))
        returns.append(ep['total_return'])
    return float(np.mean(returns))


def greedy_action_from_Q(Q):
    return lambda s: int(np.argmax(Q[state_index(s)]))


# ---------------------------------------------------------------------------
# Q-learning training loop
# ---------------------------------------------------------------------------

def train_q_learning(q_update_fn, alpha=0.1, gamma=0.95, eps=0.1,
                     n_episodes=2000, use_replay=False, buffer_size=500,
                     batch_size=8, seed=0, sparse=False):
    """Tabular Q-learning with epsilon-greedy behaviour and (optional) replay.
    Calls the student's ``q_update_fn(Q, s, a, r, s', done, alpha, gamma)``.
    Returns ``(Q, episode_returns)``.
    """
    env = MiniPongEnv(sparse=sparse)
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    buffer: list = []
    episode_returns: list[float] = []

    for ep in range(n_episodes):
        s = env.reset(seed=int(rng.integers(0, 1 << 31)))
        s_idx = state_index(s)
        ep_return = 0.0
        while not env.done:
            if rng.random() < eps:
                a = int(rng.integers(0, N_ACTIONS))
            else:
                a = int(np.argmax(Q[s_idx]))
            s_next, r, done = env.step(a)
            sn_idx = state_index(s_next)
            transition = (s_idx, a, r, sn_idx, done)
            if use_replay:
                buffer.append(transition)
                if len(buffer) > buffer_size:
                    buffer.pop(0)
                k = min(batch_size, len(buffer))
                idxs = rng.integers(0, len(buffer), size=k)
                for i in idxs:
                    si, ai, ri, sni, di = buffer[int(i)]
                    res = q_update_fn(Q, si, ai, ri, sni, di, alpha, gamma)
                    if res is None:
                        return Q, episode_returns
                    Q = res
            else:
                res = q_update_fn(Q, s_idx, a, r, sn_idx, done, alpha, gamma)
                if res is None:
                    return Q, episode_returns
                Q = res
            s_idx = sn_idx
            ep_return += r
        episode_returns.append(ep_return)
    return Q, episode_returns


# ---------------------------------------------------------------------------
# REINFORCE training loop
# ---------------------------------------------------------------------------

def train_reinforce(returns_fn, gradient_fn, lr=0.15, gamma=0.95,
                     n_episodes=600, baseline=False, seed=0, sparse=False):
    """REINFORCE with optional running-average baseline.  Calls the student's
    ``returns_fn(rewards, gamma)`` and ``gradient_fn(theta, phis, actions, advs)``.

    Returns ``(theta, episode_returns, gradient_norms)``.
    """
    env = MiniPongEnv(sparse=sparse)
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_ACTIONS, N_FEATURES))
    baseline_val = 0.0
    episode_returns: list[float] = []
    grad_norms: list[float] = []

    for ep in range(n_episodes):
        s = env.reset(seed=int(rng.integers(0, 1 << 31)))
        phis: list = []
        actions: list[int] = []
        rewards: list[float] = []
        while not env.done:
            phi = features(s)
            a = sample_action(theta, phi, rng)
            s_next, r, _ = env.step(a)
            phis.append(phi)
            actions.append(a)
            rewards.append(r)
            s = s_next
        rs = returns_fn(rewards, gamma)
        if rs is None:
            return theta, episode_returns, grad_norms
        rs = np.asarray(rs, dtype=float)
        if baseline:
            advs = rs - baseline_val
            baseline_val = 0.9 * baseline_val + 0.1 * float(rs.mean())
        else:
            advs = rs
        grad = gradient_fn(theta, phis, actions, advs)
        if grad is None:
            return theta, episode_returns, grad_norms
        grad = np.asarray(grad, dtype=float)
        theta = theta + lr * grad
        episode_returns.append(float(sum(rewards)))
        grad_norms.append(float(np.linalg.norm(grad)))

    return theta, episode_returns, grad_norms


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

def _collect_batch(theta, env, returns_fn, gamma, batch_episodes, rng):
    """Roll out ``batch_episodes`` trajectories under the current ``theta``.
    Return flattened (phis, actions, advantages, log_probs_old) and per-episode
    returns.  Advantages are z-scored ACROSS the whole batch, not per episode —
    per-episode normalisation would flip the sign of every step in a losing
    episode.
    """
    all_phis, all_actions, all_returns, all_logp_old = [], [], [], []
    ep_rets: list[float] = []
    for _ in range(batch_episodes):
        s = env.reset(seed=int(rng.integers(0, 1 << 31)))
        phis, actions, rewards, logp_old = [], [], [], []
        while not env.done:
            phi = features(s)
            probs = policy_probs(theta, phi)
            a = int(rng.choice(N_ACTIONS, p=probs))
            logp_old.append(float(np.log(max(probs[a], 1e-12))))
            phis.append(phi); actions.append(a)
            s, r, _ = env.step(a)
            rewards.append(r)
        rs_raw = returns_fn(rewards, gamma)
        if rs_raw is None:
            return None
        rs = np.asarray(rs_raw, dtype=float)
        all_phis.extend(phis)
        all_actions.extend(actions)
        all_returns.extend(rs.tolist())
        all_logp_old.extend(logp_old)
        ep_rets.append(float(sum(rewards)))
    all_returns = np.asarray(all_returns, dtype=float)
    if all_returns.std() > 1e-8:
        advs = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
    else:
        advs = all_returns - all_returns.mean()
    return (np.asarray(all_phis), np.asarray(all_actions, dtype=int),
            advs, np.asarray(all_logp_old), ep_rets)


def _logp_under_theta(theta, phis, actions):
    """log π_θ(a | s) for each (phi, a) in the batch."""
    n = len(phis)
    out = np.empty(n)
    for i in range(n):
        p = policy_probs(theta, phis[i])
        out[i] = float(np.log(max(p[int(actions[i])], 1e-12)))
    return out


def train_ppo(ppo_loss_fn, returns_fn, eps_clip=0.2, lr=0.05, gamma=0.95,
              n_iters=120, batch_episodes=8, n_epochs=4, seed=0, sparse=False,
              average_last=10):
    """PPO with the student's clipped surrogate loss.  Each outer iteration:
       1. roll out a batch under the current θ_old,
       2. minimise ``-ppo_loss_fn`` for ``n_epochs`` scipy steps,
       3. set θ_old ← θ.

    The returned θ is the mean of the last ``average_last`` iterates — this
    smooths out the per-iteration L-BFGS noise that would otherwise leave the
    final θ at whatever the last batch happened to nudge it to.

    Returns ``(theta_avg, iter_returns)``.
    """
    env = MiniPongEnv(sparse=sparse)
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_ACTIONS, N_FEATURES))
    iter_returns: list[float] = []
    theta_history: list[np.ndarray] = []

    for it in range(n_iters):
        batch = _collect_batch(
            theta, env, returns_fn, gamma, batch_episodes, rng,
        )
        if batch is None:
            return theta, iter_returns
        phis, actions, advs, logp_old, ep_rets = batch
        iter_returns.append(float(np.mean(ep_rets)))

        def neg_loss(flat_theta):
            T = flat_theta.reshape(N_ACTIONS, N_FEATURES)
            logp_new = _logp_under_theta(T, phis, actions)
            ratios = np.exp(logp_new - logp_old)
            obj = ppo_loss_fn(ratios, advs, eps_clip)
            if obj is None:
                return 1e6
            return -float(obj)

        for _ in range(n_epochs):
            res = minimize(neg_loss, theta.ravel(), method='L-BFGS-B',
                           options={'maxiter': 6, 'disp': False})
            theta = res.x.reshape(N_ACTIONS, N_FEATURES)
        theta_history.append(theta.copy())

    tail = theta_history[-average_last:]
    theta_avg = np.mean(np.stack(tail, axis=0), axis=0)
    return theta_avg, iter_returns


# ---------------------------------------------------------------------------
# Visualisation: ASCII frame and trajectory trail
# ---------------------------------------------------------------------------

def render_frame(state) -> str:
    """One ASCII frame: paddle column at the right edge, ball as ●."""
    bx, by, vy, py = state
    rows = []
    for y in range(H):
        cells = []
        for x in range(W):
            if x == bx and y == by:
                cells.append('●')
            elif x == W - 1 and y == py:
                cells.append('█')
            else:
                cells.append('·')
        rows.append('  '.join(cells))
    return '\n'.join(rows)


def print_episode(states, actions=None, rewards=None, header=''):
    """Print every frame of an episode, separated by labels."""
    if header:
        print(header)
    for t, s in enumerate(states):
        a_str = '' if actions is None or t >= len(actions) else f'  →  action: {ACTION_NAMES[actions[t]]}'
        print(f'\n  t = {t}{a_str}')
        for line in render_frame(s).splitlines():
            print('  ' + line)
    if rewards:
        total = sum(rewards)
        verdict = 'CATCH (+1)' if total > 0 else ('MISS (-1)' if total < 0 else 'no signal')
        print(f'\n  episode return: {total:+.0f}   ({verdict})')


def plot_trajectory(states, title='episode trajectory'):
    """Static matplotlib view: ball trail with fading alpha + final paddle."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for t, (bx, by, vy, py) in enumerate(states):
        alpha = 0.25 + 0.75 * t / max(1, len(states) - 1)
        ax.scatter(bx, H - 1 - by, s=120, color=_GOLDEN, alpha=alpha,
                   edgecolor=_BORDER, linewidths=0.6, zorder=3)
    bx_end, by_end, _, py_end = states[-1]
    ax.plot([W - 1, W - 1], [H - 1 - py_end - 0.4, H - 1 - py_end + 0.4],
            color=_ACCENT, linewidth=8.0, solid_capstyle='butt',
            label=f'final paddle (py={py_end})')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)])
    ax.set_xlabel('column (bx)')
    ax.set_ylabel('row (by)')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower left', fontsize=9)
    ax.set_aspect('equal')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: V*, Q-table, learning curves
# ---------------------------------------------------------------------------

def smooth(xs, k=20):
    if len(xs) < k:
        return np.asarray(xs, dtype=float)
    xs = np.asarray(xs, dtype=float)
    pad = np.full(k - 1, xs[0])
    padded = np.concatenate([pad, xs])
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode='valid')


def plot_value_slice(V, title='V*  (slice: paddle centred, vy = +1)'):
    """Heatmap of V over (bx, by) for fixed py = H//2 and vy = +1.

    The bright diagonal you should see: V is highest at states the optimal
    paddle can reach in time, and dimmer at extreme by where the paddle
    cannot recover.
    """
    py = (H - 1) // 2
    grid = np.zeros((H, W - 1))
    for bx in range(W - 1):
        for by in range(H):
            grid[by, bx] = V[state_index((bx, by, 1, py))]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    im = ax.imshow(grid, cmap='magma', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(W - 1))
    ax.set_yticks(range(H))
    ax.set_xlabel('ball x  (bx)')
    ax.set_ylabel('ball y  (by)')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()


def plot_q_vs_vstar(Q, V_star, title='Tabular Q vs value-iteration V*'):
    """Scatter ``max_a Q(s, a)`` vs ``V*(s)`` over all non-terminal states.

    A tight diagonal means Q-learning has converged; the spread is the part
    of the value table that the sampled trajectories did not yet visit
    enough to estimate cleanly.
    """
    mask = np.arange(N_STATES) != TERMINAL_INDEX
    q_max = Q.max(axis=1)[mask]
    v = V_star[mask]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([-1, 1], [-1, 1], color=_TERRA, linestyle='--', linewidth=0.8,
            label='Q == V*')
    ax.scatter(v, q_max, s=10, color=_ACCENT, alpha=0.55, edgecolor='none')
    ax.set_xlabel('V*(s)  (value iteration)')
    ax.set_ylabel('max_a Q(s, a)  (Q-learning)')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(returns_dict, title='Learning curves', smooth_k=30,
                         ylabel='episode return'):
    """Plot one or more named return curves.  ``returns_dict`` maps label -> array."""
    palette = [_ACCENT, _GOLDEN, _TERRA, _SAGE, _STEEL, _LAVENDER]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, ys) in enumerate(returns_dict.items()):
        ys = np.asarray(ys, dtype=float)
        ax.plot(np.arange(len(ys)), smooth(ys, smooth_k),
                color=palette[i % len(palette)], linewidth=1.4, label=label)
    ax.set_xlabel('episode (or iteration)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_gradient_norms(norms_dict, title='Policy-gradient norms',
                         smooth_k=20):
    """Compare gradient-norm trajectories of REINFORCE with vs. without baseline."""
    palette = [_TERRA, _ACCENT]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for i, (label, ys) in enumerate(norms_dict.items()):
        ys = np.asarray(ys, dtype=float)
        ax.plot(np.arange(len(ys)), smooth(ys, smooth_k),
                color=palette[i % len(palette)], linewidth=1.2, label=label)
    ax.set_xlabel('episode')
    ax.set_ylabel('||∇θ J||')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    ax.legend(frameon=False, labelcolor=_TEXT, loc='upper right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_visitation(visit_counts, title='State visitation under the behaviour rule'):
    """Heatmap of how often each (bx, by) was visited during exploration."""
    py = (H - 1) // 2
    grid = np.zeros((H, W - 1))
    for bx in range(W - 1):
        for by in range(H):
            for vy in (-1, 1):
                idx = state_index((bx, by, vy, py))
                grid[by, bx] += visit_counts[idx]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    im = ax.imshow(grid, cmap='magma', aspect='auto')
    ax.set_xticks(range(W - 1))
    ax.set_yticks(range(H))
    ax.set_xlabel('ball x')
    ax.set_ylabel('ball y')
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Behaviour helpers used by the take-it-from-here exploration section
# ---------------------------------------------------------------------------

def collect_visitation(action_fn, n_episodes=200, seed=0, sparse=True):
    """Count how many times each state is visited under ``action_fn``."""
    env = MiniPongEnv(sparse=sparse)
    rng = np.random.default_rng(seed)
    counts = np.zeros(N_STATES, dtype=int)
    returns: list[float] = []
    for _ in range(n_episodes):
        s = env.reset(seed=int(rng.integers(0, 1 << 31)))
        counts[state_index(s)] += 1
        ep_r = 0.0
        while not env.done:
            a = int(action_fn(s))
            s, r, _ = env.step(a)
            counts[state_index(s)] += 1
            ep_r += r
        returns.append(ep_r)
    return counts, float(np.mean(returns))


def epsilon_greedy_using(picker_fn, Q, eps, rng):
    """Wrap the student's pick_epsilon_greedy into an action_fn for rollouts."""
    def action_fn(state):
        return int(picker_fn(Q[state_index(state)], eps, rng))
    return action_fn


# ---------------------------------------------------------------------------
# Visualisation: environment schematic
# ---------------------------------------------------------------------------

def plot_env_schematic(title='mini-Pong: state, actions, reward'):
    """Static schematic of the environment.

    Annotates the ball position, paddle column, the three actions the paddle
    can take, and the +1 / -1 reward zones at the rightmost column.  A picture
    that the chapter's prose alone cannot replace — the agent's job in one
    look.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    # Grid
    for x in range(W + 1):
        ax.axvline(x - 0.5, color=_BORDER, linewidth=0.4, alpha=0.45)
    for y in range(H + 1):
        ax.axhline(y - 0.5, color=_BORDER, linewidth=0.4, alpha=0.45)

    # Highlight the paddle column.
    ax.axvspan(W - 1.5, W - 0.5, color=_ACCENT, alpha=0.10)

    # Ball at (1, by=2)
    bx_demo, by_demo = 1, 2
    ax.scatter(bx_demo, H - 1 - by_demo, s=220, color=_GOLDEN,
               edgecolor=_BORDER, linewidths=0.8, zorder=3, label='ball  (bx, by)')

    # Velocity arrow (vy = +1 means down in board coords -> arrow goes downward).
    ax.annotate('', xy=(bx_demo + 0.9, H - 1 - by_demo - 0.9),
                xytext=(bx_demo, H - 1 - by_demo),
                arrowprops=dict(arrowstyle='->', color=_GOLDEN, lw=1.6))
    ax.text(bx_demo + 0.55, H - 1 - by_demo - 0.55, ' vx=+1, vy=+1',
            fontsize=8, color=_GOLDEN, va='center')

    # Paddle at (W-1, py=2)
    py_demo = 2
    ax.plot([W - 1, W - 1], [H - 1 - py_demo - 0.42, H - 1 - py_demo + 0.42],
            color=_ACCENT, linewidth=10, solid_capstyle='butt',
            label='paddle  (py)')

    # Action arrows next to the paddle.
    ax_off = W - 1 + 0.55
    ax.annotate('', xy=(ax_off, H - 1 - py_demo + 0.95),
                xytext=(ax_off, H - 1 - py_demo + 0.05),
                arrowprops=dict(arrowstyle='->', color=_SAGE, lw=1.4))
    ax.text(ax_off + 0.05, H - 1 - py_demo + 0.85, 'up',
            fontsize=9, color=_SAGE, va='center')
    ax.annotate('', xy=(ax_off, H - 1 - py_demo - 0.95),
                xytext=(ax_off, H - 1 - py_demo - 0.05),
                arrowprops=dict(arrowstyle='->', color=_TERRA, lw=1.4))
    ax.text(ax_off + 0.05, H - 1 - py_demo - 0.85, 'down',
            fontsize=9, color=_TERRA, va='center')
    ax.text(ax_off + 0.05, H - 1 - py_demo, 'stay',
            fontsize=9, color=_LAVENDER, va='center')

    # Reward zone label at the right.
    ax.text(W - 1, H + 0.05, '+1 if py = by   |   -1 (or 0 sparse) otherwise',
            fontsize=9, color=_GOLDEN, ha='right', va='bottom')

    # Origin label
    ax.text(-0.48, H - 0.6, 'bx = 0', fontsize=8, color=_TEXT, ha='left')

    ax.set_xlim(-0.6, W + 1.4)
    ax.set_ylim(-0.7, H + 0.6)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)])
    ax.set_xlabel('column (bx)')
    ax.set_ylabel('row (by)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.set_aspect('equal')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower left', fontsize=9)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: episode rollout as a strip of frames
# ---------------------------------------------------------------------------

def plot_episode_strip(states, actions=None, rewards=None,
                       title='episode rollout', max_frames=8):
    """Render successive episode frames as a horizontal strip of subplots.

    A graphical alternative to ``print_episode``'s ASCII rendering: each
    column is one timestep, ball and paddle drawn as scatter / bar.  Action
    chosen at that step (if provided) is annotated under each frame.
    """
    n = min(len(states), max_frames)
    states = list(states[:n])
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.3))
    if n == 1:
        axes = [axes]
    final_return = float(sum(rewards)) if rewards else 0.0
    for t, ax in enumerate(axes):
        bx, by, _vy, py = states[t]
        # Background grid.
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        # Light grid lines.
        for x in range(W):
            ax.axvline(x, color=_BORDER, linewidth=0.3, alpha=0.4)
        for y in range(H):
            ax.axhline(y, color=_BORDER, linewidth=0.3, alpha=0.4)
        # Paddle.
        ax.plot([W - 1, W - 1], [H - 1 - py - 0.4, H - 1 - py + 0.4],
                color=_ACCENT, linewidth=6, solid_capstyle='butt')
        # Ball.
        if 0 <= bx < W and 0 <= by < H:
            ax.scatter(bx, H - 1 - by, s=120, color=_GOLDEN,
                       edgecolor=_BORDER, linewidths=0.6, zorder=3)
        a_str = ''
        if actions is not None and t < len(actions):
            a_str = ACTION_NAMES[actions[t]]
        ax.set_title(f't={t}\n{a_str}', fontsize=8, color=_TEXT)
    verdict = ''
    if rewards:
        verdict = '  CATCH (+1)' if final_return > 0 else (
            '  MISS (-1)' if final_return < 0 else '  no signal')
    fig.suptitle(f'{title}{verdict}', fontsize=10, color=_GOLDEN, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: value-iteration convergence
# ---------------------------------------------------------------------------

def plot_value_iteration_convergence(P, R, gamma=0.95, n_iters=80,
                                       title='Value iteration: a γ-contraction'):
    """Plot ``max_s |V_k(s) - V*(s)|`` per iteration and overlay the
    ``γ^k`` envelope.

    Bellman is a γ-contraction in the sup-norm; the gap to V* shrinks at
    least geometrically with rate γ.  Plotting both makes the theorem
    visible: the empirical curve hugs the envelope until floating-point
    bottoms out.
    """
    # Build V* with many iterations as ground truth.
    V_star = np.zeros(P.shape[0])
    for _ in range(400):
        V_star = (R + gamma * (P @ V_star)).max(axis=1)

    V = np.zeros(P.shape[0])
    errors = []
    init_err = float(np.max(np.abs(V - V_star)))
    for k in range(n_iters):
        Q = R + gamma * (P @ V)
        V = Q.max(axis=1)
        errors.append(float(np.max(np.abs(V - V_star))))

    ks = np.arange(1, n_iters + 1)
    envelope = init_err * (gamma ** ks)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ks, errors, color=_ACCENT, linewidth=1.6,
            label='empirical  max_s |V_k - V*|')
    ax.plot(ks, envelope, color=_TERRA, linewidth=1.0, linestyle='--',
            label=f'γ^k envelope  (γ = {gamma})')
    ax.set_yscale('log')
    ax.set_xlabel('iteration k')
    ax.set_ylabel('sup-norm error  (log scale)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='upper right')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: optimal-policy arrows over the state space
# ---------------------------------------------------------------------------

_ACTION_ARROW = {0: '↑', 1: '·', 2: '↓'}
_ACTION_COLOR = lambda: {0: _SAGE, 1: _LAVENDER, 2: _TERRA}


def plot_policy_arrows(Q, title='Optimal policy  (slice: paddle centred, vy = +1)'):
    """Per-state arrow at every (bx, by) for ``argmax_a Q(s, a)``.

    Reads the strategy directly: at each ball position the picture shows
    where the paddle is told to move.  The bands of "up" near the bottom and
    "down" near the top reveal the policy is tracking the impact prediction,
    not just the current ball row.
    """
    py = (H - 1) // 2
    colors = _ACTION_COLOR()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for bx in range(W - 1):
        for by in range(H):
            s_idx = state_index((bx, by, 1, py))
            a = int(np.argmax(Q[s_idx]))
            ax.text(bx, H - 1 - by, _ACTION_ARROW[a],
                    fontsize=18, color=colors[a],
                    ha='center', va='center')
    # Faint paddle column highlight.
    ax.axvspan(W - 1.5, W - 0.5, color=_ACCENT, alpha=0.08)
    ax.axhline(H - 1 - py, color=_ACCENT, linestyle='--', linewidth=0.6,
               alpha=0.5, label=f'paddle row  (py = {py})')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)])
    ax.set_xlabel('ball x  (bx)')
    ax.set_ylabel('ball y  (by)')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.set_aspect('equal')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower left', fontsize=9)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: discounted-return decomposition
# ---------------------------------------------------------------------------

def plot_returns_decomposition(rewards, gamma=0.95,
                                title='Discounted return  G_t  along an episode'):
    """Stacked bars: per-step reward, then discounted future return.

    For mini-Pong the only non-zero reward sits at the terminal step, so
    G_t = γ^(T-1-t) · r_T.  Drawing the curve makes the chapter's "credit
    assignment" intuition concrete: the reward at the end is reflected
    backward through every step that led there, with weights that decay
    by γ per step.
    """
    rewards = np.asarray(rewards, dtype=float)
    T = len(rewards)
    Gs = np.zeros(T)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        Gs[t] = G
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(ts - 0.18, rewards, width=0.36, color=_TERRA, alpha=0.85,
           label='immediate reward  r_t')
    ax.bar(ts + 0.18, Gs, width=0.36, color=_ACCENT, alpha=0.85,
           label=f'discounted return  G_t  (γ = {gamma})')
    ax.axhline(0, color=_BORDER, linewidth=0.5)
    ax.set_xticks(ts)
    ax.set_xlabel('timestep  t')
    ax.set_ylabel('value')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower left', fontsize=9)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: PPO clip-surrogate landscape
# ---------------------------------------------------------------------------

def plot_ppo_clip_landscape(eps_clip=0.2,
                              title='PPO clipped surrogate  L(r, A)  vs ratio'):
    """Twin panels: L_clip(r, A=+1) and L_clip(r, A=-1) as r varies.

    Visualises the chapter's "min(unclipped, clipped) is what makes the
    bound pessimistic" claim.  The flat plateaus on each side appear at
    different r ranges depending on the sign of A — exactly the asymmetry
    the min enforces.
    """
    rs = np.linspace(0.0, 2.0, 401)
    rs_clip = np.clip(rs, 1.0 - eps_clip, 1.0 + eps_clip)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, A, panel_title in zip(axes, (+1.0, -1.0),
                                   (f'A > 0  (good action)',
                                    f'A < 0  (bad action)')):
        unclipped = rs * A
        clipped = rs_clip * A
        L = np.minimum(unclipped, clipped)
        ax.plot(rs, unclipped, color=_TERRA, linestyle='--', linewidth=1.0,
                label='r · A  (unclipped)')
        ax.plot(rs, clipped, color=_GOLDEN, linestyle=':', linewidth=1.0,
                label='clip(r) · A')
        ax.plot(rs, L, color=_ACCENT, linewidth=2.0, label='L = min(·, ·)')
        ax.axvspan(1.0 - eps_clip, 1.0 + eps_clip, color=_ACCENT, alpha=0.08)
        ax.axvline(1.0, color=_BORDER, linewidth=0.5)
        ax.axhline(0, color=_BORDER, linewidth=0.5)
        ax.set_xlabel('importance ratio  r = π_θ / π_old')
        ax.set_title(panel_title, fontsize=10, color=_GOLDEN, loc='left')
        tufte_axis(ax)
    axes[0].set_ylabel('per-sample objective')
    axes[1].legend(frameon=False, labelcolor=_TEXT, loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=10, color=_GOLDEN, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: action-probability heatmap of the trained linear softmax
# ---------------------------------------------------------------------------

def plot_policy_action_probs(theta, title='π(a | s) under the trained linear softmax'):
    """Three heatmaps side by side: P(up | s), P(stay | s), P(down | s)
    over the (bx, by) slice with paddle centred and vy = +1.

    Reveals what the policy *does* at each ball position in pictures: a
    competent paddle has up-mass concentrated where the ball is above the
    paddle and down-mass where it is below.
    """
    py = (H - 1) // 2
    grids = [np.zeros((H, W - 1)) for _ in range(N_ACTIONS)]
    for bx in range(W - 1):
        for by in range(H):
            phi = features((bx, by, 1, py))
            p = policy_probs(theta, phi)
            for a in range(N_ACTIONS):
                grids[a][by, bx] = p[a]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, ax in enumerate(axes):
        im = ax.imshow(grids[a], cmap='magma', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(W - 1))
        ax.set_yticks(range(H))
        ax.set_xlabel('ball x' if a == 1 else '')
        ax.set_ylabel('ball y' if a == 0 else '')
        ax.set_title(f'P({ACTION_NAMES[a]} | s)', fontsize=10, color=_GOLDEN, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=10, color=_GOLDEN, y=1.04)
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: REINFORCE return vs running baseline vs advantage
# ---------------------------------------------------------------------------

def plot_baseline_advantage_trace(returns_no_b, returns_b, gamma=0.95,
                                    title='Baseline turns return into advantage'):
    """Two stacked panels.  Top: per-episode return for the two runs +
    the running-average baseline that the second run subtracts.  Bottom:
    the implied per-episode advantage for each run (return - baseline,
    where the no-baseline run subtracts zero).

    Makes the variance-reduction story the chapter tells in prose readable
    in one frame: the no-baseline curve oscillates with full amplitude;
    the baselined curve is centred near zero with much smaller amplitude.
    """
    returns_no_b = np.asarray(returns_no_b, dtype=float)
    returns_b    = np.asarray(returns_b, dtype=float)
    n = min(len(returns_no_b), len(returns_b))
    returns_no_b = returns_no_b[:n]
    returns_b    = returns_b[:n]

    # Reconstruct the running baseline that train_reinforce uses.
    b_trace = np.zeros(n)
    val = 0.0
    for i, r in enumerate(returns_b):
        b_trace[i] = val
        val = 0.9 * val + 0.1 * float(r)

    adv_no_b = returns_no_b
    adv_b    = returns_b - b_trace

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True)
    ax = axes[0]
    ax.plot(returns_no_b, color=_TERRA, linewidth=0.7, alpha=0.45,
            label='return  (no baseline)')
    ax.plot(smooth(returns_no_b, 30), color=_TERRA, linewidth=1.6,
            label='smoothed return  (no baseline)')
    ax.plot(b_trace, color=_GOLDEN, linewidth=1.6,
            label='running baseline  b̄_t')
    ax.set_ylabel('per-episode value')
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right', fontsize=9)
    tufte_axis(ax)

    ax = axes[1]
    ax.axhline(0, color=_BORDER, linewidth=0.6)
    ax.plot(adv_no_b, color=_TERRA, linewidth=0.6, alpha=0.5,
            label='advantage  (no baseline)  =  return')
    ax.plot(adv_b, color=_ACCENT, linewidth=0.6, alpha=0.6,
            label='advantage  (with baseline)  =  return − b̄')
    ax.plot(smooth(adv_b, 30), color=_ACCENT, linewidth=1.6,
            label='smoothed advantage  (with baseline)')
    ax.set_xlabel('episode')
    ax.set_ylabel('advantage')
    ax.legend(frameon=False, labelcolor=_TEXT, loc='lower right', fontsize=9)
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation: random vs learned baseline curves on sparse rewards
# ---------------------------------------------------------------------------

def plot_sparse_summary_bars(eps_values, returns_under_greedy,
                              title='Greedy-extracted return vs ε  (sparse reward)'):
    """One bar per ε: greedy-extracted return after Q-learning.

    Companion to the three visitation heatmaps in §🔭.  Where the heatmaps
    show what each rule *saw*, this bar chart shows what each rule *learned*
    — the U-shape (or its absence) makes the exploration trade-off legible
    at a glance.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs = np.arange(len(eps_values))
    palette = [_TERRA, _ACCENT, _GOLDEN, _SAGE, _STEEL, _LAVENDER]
    bars = ax.bar(xs, returns_under_greedy,
                  color=[palette[i % len(palette)] for i in range(len(xs))],
                  alpha=0.9, edgecolor=_BORDER, linewidth=0.6)
    for b, v in zip(bars, returns_under_greedy):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06),
                f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top',
                color=_TEXT, fontsize=9)
    ax.axhline(0, color=_BORDER, linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'ε = {e:.2f}' for e in eps_values])
    ax.set_ylabel('greedy return  (avg over 500 eps)')
    ax.set_ylim(min(-0.05, min(returns_under_greedy) - 0.1),
                max(1.05, max(returns_under_greedy) + 0.15))
    ax.set_title(title, fontsize=10, color=_GOLDEN, loc='left')
    tufte_axis(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Playback: chain rallies until first miss, render as embedded animation
# ---------------------------------------------------------------------------

def play_until_miss(env, action_fn, max_rallies=20, seed=0):
    """Chain rallies under ``action_fn`` until the first miss or ``max_rallies``.

    Each rally is one episode of the underlying env (terminates on hit or
    miss); consecutive rallies are concatenated into one continuous state
    stream so ``animate_episode`` can render it as a single video. Returns a
    dict with ``states``, ``actions``, ``rewards``, ``rally_boundaries`` (the
    state index of each rally's terminal frame), ``hits`` (catches before the
    miss), and ``ended_on_miss`` (False if we hit ``max_rallies`` first).
    """
    rng = np.random.default_rng(seed)
    states, actions, rewards = [], [], []
    rally_boundaries = []
    hits = 0
    ended_on_miss = False
    for _ in range(max_rallies):
        ep = rollout_episode(env, action_fn,
                             seed=int(rng.integers(0, 1 << 31)))
        states.extend(ep['states'])
        actions.extend(ep['actions'])
        rewards.extend(ep['rewards'])
        rally_boundaries.append(len(states) - 1)
        if ep['rewards'][-1] == 1.0:
            hits += 1
        else:
            ended_on_miss = True
            break
    return dict(
        states=states, actions=actions, rewards=rewards,
        rally_boundaries=rally_boundaries, hits=hits,
        ended_on_miss=ended_on_miss,
    )


def animate_episode(playback, fps=4, title='mini-Pong rollout',
                    figsize=(5.4, 3.0), dpi=72):
    """Embed a playback as an HTML5 animation in the notebook.

    ``playback`` may be either a ``rollout_episode`` dict (single rally) or a
    ``play_until_miss`` dict (chained rallies). Uses ``FuncAnimation`` and
    ``to_jshtml`` so no ffmpeg is required, which keeps it Pyodide-friendly.
    Return value is an ``IPython.display.HTML`` object: place the call as the
    last expression of a notebook cell to render it.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    states  = playback['states']
    rewards = playback.get('rewards', [])
    rally_boundaries = playback.get('rally_boundaries',
                                    [len(states) - 1])
    hits_total       = playback.get('hits', None)
    ended_on_miss    = playback.get('ended_on_miss', None)
    n_frames         = len(states)

    # Per-rally outcomes (True = hit, False = miss), used by per-frame text.
    rally_outcomes, r_cursor, prev_b = [], 0, -1
    for b in rally_boundaries:
        n_steps = b - prev_b - 1
        terminal_reward = rewards[r_cursor + n_steps - 1] if n_steps > 0 else 0.0
        rally_outcomes.append(terminal_reward == 1.0)
        r_cursor += n_steps
        prev_b = b

    frame_info = []
    for t in range(n_frames):
        completed = sum(1 for b in rally_boundaries if b < t)
        hits_so_far = sum(1 for i, b in enumerate(rally_boundaries)
                          if b < t and rally_outcomes[i])
        in_rally = min(completed + 1, len(rally_boundaries))
        info = f'rally {in_rally}    catches: {hits_so_far}'
        if t == n_frames - 1 and ended_on_miss:
            info = f'MISS in rally {in_rally}    catches: {hits_so_far}'
        elif t == n_frames - 1 and ended_on_miss is False:
            info = f'cleared {hits_total} rallies without missing'
        frame_info.append(info)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)])
    ax.set_xlabel('column (bx)')
    ax.set_ylabel('row (by)')
    ax.set_aspect('equal')
    ax.axvspan(W - 1.5, W - 0.5, color=_ACCENT, alpha=0.08)
    ax.set_title(title, fontsize=10, color=_GOLDEN)
    tufte_axis(ax)

    ball    = ax.scatter([], [], s=160, color=_GOLDEN,
                         edgecolor=_BORDER, linewidths=0.6, zorder=3)
    paddle, = ax.plot([], [], color=_ACCENT, linewidth=8.0,
                      solid_capstyle='butt', zorder=2)
    info_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                        fontsize=9, color=_TEXT, va='top', ha='left')

    def init():
        ball.set_offsets(np.zeros((0, 2)))
        paddle.set_data([], [])
        info_text.set_text('')
        return ball, paddle, info_text

    def update(t):
        bx, by, _, py = states[t]
        ball.set_offsets([[bx, H - 1 - by]])
        paddle.set_data([W - 1, W - 1],
                        [H - 1 - py - 0.4, H - 1 - py + 0.4])
        info_text.set_text(frame_info[t])
        return ball, paddle, info_text

    interval_ms = max(40, int(1000 / max(1, fps)))
    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=init, interval=interval_ms,
                         blit=True, repeat=False)
    html = anim.to_jshtml(default_mode='once')
    plt.close(fig)
    return HTML(html)


# ---------------------------------------------------------------------------
# Playback: async live-play loop for Pyodide / modern Jupyter
# ---------------------------------------------------------------------------

def _draw_pong_frame(state, hits, rally, title):
    """One static matplotlib frame in the same visual language as
    ``animate_episode``: ball, paddle, shaded paddle column, overlay text.
    Returned figure is meant to be embedded via ``_frame_to_html`` so it can
    exceed the notebook cell's default image width.
    """
    bx, by, _, py = state
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=120)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)])
    ax.tick_params(labelsize=12)
    ax.set_aspect('equal')
    ax.axvspan(W - 1.5, W - 0.5, color=_ACCENT, alpha=0.08)
    ax.scatter(bx, H - 1 - by, s=520, color=_GOLDEN,
               edgecolor=_BORDER, linewidths=0.8, zorder=3)
    ax.plot([W - 1, W - 1],
            [H - 1 - py - 0.4, H - 1 - py + 0.4],
            color=_ACCENT, linewidth=18.0, solid_capstyle='butt', zorder=2)
    ax.text(0.02, 0.96, f'rally {rally}    catches: {hits}',
            transform=ax.transAxes, fontsize=14, color=_TEXT,
            va='top', ha='left')
    ax.set_title(title, fontsize=14, color=_GOLDEN)
    tufte_axis(ax)
    plt.tight_layout()
    return fig


def _frame_to_html(fig, width_px=600):
    """Render a matplotlib figure to a width-pinned HTML <img>.

    Jupyter's default inline backend embeds the PNG at its natural
    ``figsize * dpi`` pixel size, but the surrounding cell CSS often caps
    image width to the column. Pinning the display width via HTML lets the
    live-play frame use the full content area regardless of the theme.
    """
    import base64
    import io
    from IPython.display import HTML

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return HTML(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:{width_px}px;max-width:100%;display:block;">'
    )


async def play_live(env, action_fn, fps=4, max_rallies=20, seed=0,
                    title='mini-Pong: live'):
    """Live-play chained rallies, one frame at a time, in the notebook.

    This coroutine is async because the Pyodide kernel runs on the browser's
    single JavaScript event loop: a synchronous ``time.sleep`` blocks that
    loop, so matplotlib never gets a chance to repaint mid-loop and the cell
    only shows the final frame. ``await asyncio.sleep`` yields control back
    to the event loop between frames, which lets the displayed figure update
    in real time. Top-level ``await`` works in Pyodide and in modern Jupyter,
    so call this as the last line of a cell::

        await play_live(env, action_fn)

    Returns the total number of catches achieved before the first miss, or
    before ``max_rallies`` is exhausted.
    """
    import asyncio
    from IPython.display import display, clear_output

    rng = np.random.default_rng(seed)
    hits = 0
    delay = 1.0 / max(1, fps)

    for rally in range(1, max_rallies + 1):
        s = env.reset(seed=int(rng.integers(0, 1 << 31)))
        terminal_reward = 0.0
        while True:
            fig = _draw_pong_frame(s, hits, rally, title)
            html = _frame_to_html(fig)
            clear_output(wait=True)
            display(html)
            plt.close(fig)
            await asyncio.sleep(delay)
            a = int(action_fn(s))
            s, r, done = env.step(a)
            if done:
                terminal_reward = r
                break
        if terminal_reward == 1.0:
            hits += 1
            continue
        # Miss: render a final annotated frame and bail out.
        fig = _draw_pong_frame(s, hits, rally,
                               f'MISS in rally {rally}')
        html = _frame_to_html(fig)
        clear_output(wait=True)
        display(html)
        plt.close(fig)
        return hits

    # Cleared all max_rallies without missing.
    fig = _draw_pong_frame(s, hits, max_rallies,
                           f'cleared {hits} rallies')
    html = _frame_to_html(fig)
    clear_output(wait=True)
    display(html)
    plt.close(fig)
    return hits
