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
        rs = np.asarray(returns_fn(rewards, gamma), dtype=float)
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
        phis, actions, advs, logp_old, ep_rets = _collect_batch(
            theta, env, returns_fn, gamma, batch_episodes, rng,
        )
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
