"""traffic_ai/controllers/rl_controllers.py

Reinforcement learning controllers for traffic signal optimization.

Controllers
-----------
- QLearningController  : Tabular Q-learning with discretized state space.
- DQNController        : Double DQN with dueling architecture, prioritized replay,
                         multi-step returns, cosine-annealing LR, gradient clipping.
- PPOController        : PPO with clipped surrogate + GAE (unchanged).
- A2CController        : Advantage Actor-Critic (synchronous, no clip).
- SACController        : Soft Actor-Critic for discrete actions (twin Q + auto alpha).
- RecurrentPPOController : PPO with LSTM actor/critic for temporal pattern capture.

All controllers implement the BaseController interface:
  reset(n), compute_actions(obs_dict, step), select_action(obs), update(...)
"""
from __future__ import annotations

import collections
import logging
import random
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase

logger = logging.getLogger(__name__)

# Expanded action space (Problem 4):
# action = phase_idx * N_DURATIONS + duration_idx
# phase_idx   ∈ {0=NS, 1=EW}
# duration_idx ∈ index into GREEN_DURATIONS
GREEN_DURATIONS: list[int] = [15, 20, 25, 30, 35, 40, 45, 60]
N_DURATIONS: int = len(GREEN_DURATIONS)   # 8
N_PHASES: int = 2
N_ACTIONS: int = N_PHASES * N_DURATIONS   # 16
STATE_DIM: int = 6


def _action_to_phase(action: int) -> int:
    """Extract phase index (0 or 1) from a 16-action integer."""
    return int(action) // N_DURATIONS


def _obs_to_vec(obs: dict[str, float]) -> np.ndarray:
    """6-feature observation vector for RL controllers.

    Features (Problem 4 expanded observation space)
    ------------------------------------------------
    0  phase_elapsed_norm   : phase_elapsed / 60
    1  phase_ns             : 1.0 if current phase is NS, else 0.0
    2  queue_ns_norm        : queue_ns / 120
    3  queue_ew_norm        : queue_ew / 120
    4  time_of_day_norm     : time_of_day in [0, 1]
    5  upstream_queue_norm  : avg upstream queue / 120
    """
    return np.array(
        [
            obs.get("phase_elapsed", 0.0) / 60.0,
            obs.get("phase_ns", float(obs.get("current_phase", 0.0)) == 0.0),
            obs.get("queue_ns", 0.0) / 120.0,
            obs.get("queue_ew", 0.0) / 120.0,
            obs.get("time_of_day_normalized",
                    (float(obs.get("step", obs.get("sim_step", 0.0))) % 86400.0) / 86400.0),
            obs.get("upstream_queue", 0.0) / 120.0,
        ],
        dtype=np.float32,
    )


# ===========================================================================
# Tabular Q-Learning Controller
# ===========================================================================

class QLearningController(BaseController):
    """Tabular Q-learning with discretized state space.

    State: (queue_ns_bucket, queue_ew_bucket, current_phase)
    Action: 0 (NS green) or 1 (EW green)

    Simpler than DQN but with guaranteed convergence on small state spaces.
    Primarily useful for educational demonstrations and ablation comparisons.
    """

    QUEUE_BUCKETS: int = 6
    QUEUE_MAX: float = 60.0

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        super().__init__(name="q_learning")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        # Q table: (q_ns_bucket, q_ew_bucket, current_phase, action)
        # action ∈ [0, N_ACTIONS) = 16 (phase × duration)
        self._q: np.ndarray = np.zeros(
            (self.QUEUE_BUCKETS, self.QUEUE_BUCKETS, 2, N_ACTIONS), dtype=np.float64
        )
        self._prev_state: dict[int, tuple[int, int, int]] | None = None
        self._prev_action: dict[int, int] | None = None
        self._current_phase: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._prev_state = None
        self._prev_action = None

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            full_action = self._act(iid, obs)
            # Extract phase from joint (phase, duration) action
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        full_action = self._act(0, obs)
        return _action_to_phase(full_action)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        # action may be phase-only (0/1) from external callers or full (0-15) from env
        # Clamp to valid range for safety
        safe_action = max(0, min(int(action), N_ACTIONS - 1))
        s = self._discretize(obs)
        s_next = self._discretize(next_obs)
        q_next = 0.0 if done else float(np.max(self._q[s_next]))
        td_target = reward + self.gamma * q_next
        self._q[s][safe_action] += self.alpha * (td_target - self._q[s][safe_action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _act(self, iid: int, obs: dict[str, float]) -> int:
        if float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(0, N_ACTIONS))
        s = self._discretize(obs)
        return int(np.argmax(self._q[s]))

    def _discretize(self, obs: dict[str, float]) -> tuple[int, int, int]:
        def bucket(val: float) -> int:
            idx = int(val / self.QUEUE_MAX * self.QUEUE_BUCKETS)
            return min(idx, self.QUEUE_BUCKETS - 1)
        return (
            bucket(obs.get("queue_ns", 0.0)),
            bucket(obs.get("queue_ew", 0.0)),
            int(obs.get("current_phase", 0.0)) % 2,
        )

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"q": self._q, "epsilon": self.epsilon}, str(path))

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "QLearningController":
        import joblib
        ctrl = cls(**kwargs)
        state = joblib.load(str(path))
        ctrl._q = state["q"]
        ctrl.epsilon = state["epsilon"]
        return ctrl


# ===========================================================================
# Dueling Double DQN Controller (research-grade)
# ===========================================================================

class DQNController(BaseController):
    """Dueling Double DQN with prioritized experience replay and multi-step returns.

    Improvements over vanilla DQN:
    - **Double DQN:** online net selects actions, target net evaluates values —
      eliminating overestimation bias (van Hasselt et al., 2016).
    - **Dueling architecture:** shared feature layers split into a value stream
      V(s) and advantage stream A(s,a), merged as Q = V + A - mean(A). Allows
      learning state value independently of action selection (Wang et al., 2016).
    - **Prioritized Experience Replay:** samples transitions proportional to
      |TD error|^alpha, with IS-weight correction for gradient bias (Schaul et
      al., 2016). More sample-efficient than uniform replay.
    - **3-step returns:** accumulate rewards over 3 steps before bootstrapping,
      reducing variance and propagating value information faster.
    - **Cosine-annealing LR:** smoothly decays learning rate from 3e-4 to 1e-5.
    - **Gradient clipping:** max_norm=1.0 prevents exploding gradients.
    - **Epsilon schedule:** decays from 1.0 to 0.01 over 80% of update_steps,
      then holds at 0.01 for stable exploitation.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 3e-4,
        lr_min: float = 1e-5,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        buffer_size: int = 30_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        n_step: int = 3,
        update_steps: int = 50_000,
        seed: int = 42,
    ) -> None:
        super().__init__(name="dqn")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self._step_count = 0
        self._train_steps = 0
        self._update_steps = max(update_steps, 1)
        self._epsilon_anneal_end = int(0.8 * update_steps)
        self._current_phase: dict[int, int] = {}
        # n-step buffer per intersection
        self._nstep_buf: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = collections.deque(maxlen=n_step)

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        random.seed(seed)

        # Dueling network architecture
        class DuelingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(STATE_DIM, 128), nn.ReLU(),
                    nn.Linear(128, 128), nn.ReLU(),
                )
                self.value = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 1),
                )
                self.advantage = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, N_ACTIONS),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                feat = self.shared(x)
                v = self.value(feat)
                a = self.advantage(feat)
                return v + a - a.mean(dim=-1, keepdim=True)

        self._online = DuelingNet()
        self._target = DuelingNet()
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optim = torch.optim.Adam(self._online.parameters(), lr=lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optim, T_max=update_steps, eta_min=lr_min
        )

        from traffic_ai.rl_models.replay_buffer import PrioritizedReplayBuffer
        self._per: PrioritizedReplayBuffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=update_steps,
        )
        self._rng = np.random.default_rng(seed)

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            full_action = self._act(_obs_to_vec(obs))
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        full_action = self._act(_obs_to_vec(obs))
        return _action_to_phase(full_action)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        self._nstep_buf.append((_obs_to_vec(obs), action, reward, _obs_to_vec(next_obs), done))
        if len(self._nstep_buf) == self.n_step:
            # Accumulate n-step return
            n_reward, n_next, n_done = self._compute_nstep_return()
            s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
            self._per.push(s0, a0, n_reward, n_next, n_done)

        if len(self._per) >= self.batch_size:
            self._learn()
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

    def pretrain_from_demonstrations(
        self, states: np.ndarray, actions: np.ndarray, epochs: int = 1
    ) -> None:
        """Pre-train the online network on imitation data."""
        import torch
        import torch.nn.functional as F
        X_t = torch.tensor(states, dtype=torch.float32)
        y_t = torch.tensor(actions, dtype=torch.long)
        self._online.train()
        for start in range(0, len(X_t), 256):
            end = start + 256
            logits = self._online(X_t[start:end])
            loss = F.cross_entropy(logits, y_t[start:end])
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        self._target.load_state_dict(self._online.state_dict())

    def _act(self, state: np.ndarray) -> int:
        # Epsilon annealing: linear from 1.0→0.01 over first 80% of update_steps
        if self._train_steps < self._epsilon_anneal_end:
            frac = self._train_steps / self._epsilon_anneal_end
            self.epsilon = max(self.epsilon_min, 1.0 - frac * (1.0 - self.epsilon_min))
        else:
            self.epsilon = self.epsilon_min

        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        import torch
        self._online.eval()
        with torch.no_grad():
            q = self._online(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q).item())

    def _compute_nstep_return(self) -> tuple[float, np.ndarray, bool]:
        """Compute accumulated discounted reward over n-step buffer."""
        gamma_n = 1.0
        n_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self._nstep_buf):
            n_reward += gamma_n * r
            gamma_n *= self.gamma
            if d:
                return n_reward, self._nstep_buf[i][3], True
        return n_reward, self._nstep_buf[-1][3], self._nstep_buf[-1][4]

    def _learn(self) -> None:
        import torch
        transitions, indices, weights_np = self._per.sample(self.batch_size, self._rng)
        s = torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
        a = torch.tensor([t.action for t in transitions], dtype=torch.long)
        r = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        ns = torch.tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32)
        d = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        w = torch.tensor(weights_np, dtype=torch.float32)

        self._online.train()
        q_vals = self._online(s).gather(1, a.unsqueeze(-1)).squeeze(-1)

        # Double DQN: online selects action, target evaluates it
        with torch.no_grad():
            best_actions = self._online(ns).argmax(dim=1, keepdim=True)
            next_q = self._target(ns).gather(1, best_actions).squeeze(-1)
            target = r + (self.gamma ** self.n_step) * next_q * (1 - d)

        td_errors = (q_vals - target).detach().abs().cpu().numpy()
        self._per.update_priorities(indices, td_errors)

        loss = (w * (q_vals - target) ** 2).mean()
        self._optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=1.0)
        self._optim.step()
        self._scheduler.step()
        self._train_steps += 1

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"online": self._online.state_dict(), "target": self._target.state_dict(), "epsilon": self.epsilon},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "DQNController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._online.load_state_dict(state["online"])
        ctrl._target.load_state_dict(state["target"])
        ctrl.epsilon = state["epsilon"]
        return ctrl


# ===========================================================================
# PPO Controller (unchanged)
# ===========================================================================

class PPOController(BaseController):
    """Proximal Policy Optimization with actor-critic architecture.

    Uses clipped surrogate objective and GAE for stable, sample-efficient
    on-policy training. Generally the most reliable RL default.
    """

    STATE_DIM: int = STATE_DIM

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 4,
        rollout_len: int = 128,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__(name="ppo")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.rollout_len = rollout_len
        self.entropy_coef = entropy_coef
        self._current_phase: dict[int, int] = {}

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            return nn.Sequential(*layers)

        self._actor = mlp(self.STATE_DIM, N_ACTIONS)
        self._critic = mlp(self.STATE_DIM, 1)
        params = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optim = torch.optim.Adam(params, lr=lr)
        self._rollout: list[dict[str, Any]] = []
        self._last_obs: np.ndarray | None = None
        self._last_action: int = 0
        self._last_log_prob: float = 0.0

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._rollout = []

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            vec = _obs_to_vec(obs)
            full_action, _ = self._sample_action(vec)
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        vec = _obs_to_vec(obs)
        self._last_obs = vec
        full_action, log_prob = self._sample_action(vec)
        self._last_action = full_action
        self._last_log_prob = log_prob
        return _action_to_phase(full_action)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        vec = _obs_to_vec(obs)
        next_vec = _obs_to_vec(next_obs)
        _, log_prob = self._sample_action(vec)
        self._rollout.append(
            {"obs": vec, "action": action, "reward": reward,
             "next_obs": next_vec, "done": done, "log_prob": log_prob}
        )
        if len(self._rollout) >= self.rollout_len:
            self._ppo_update()
            self._rollout = []

    def _sample_action(self, state: np.ndarray) -> tuple[int, float]:
        import torch
        import torch.nn.functional as F
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self._actor.eval()
        with torch.no_grad():
            logits = self._actor(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())

    def _ppo_update(self) -> None:
        import torch
        import torch.nn.functional as F
        if not self._rollout:
            return
        obs_t = torch.tensor(np.stack([r["obs"] for r in self._rollout]), dtype=torch.float32)
        acts_t = torch.tensor([r["action"] for r in self._rollout], dtype=torch.long)
        rews_t = torch.tensor([r["reward"] for r in self._rollout], dtype=torch.float32)
        next_obs_t = torch.tensor(np.stack([r["next_obs"] for r in self._rollout]), dtype=torch.float32)
        dones_t = torch.tensor([r["done"] for r in self._rollout], dtype=torch.float32)
        old_log_probs_t = torch.tensor([r["log_prob"] for r in self._rollout], dtype=torch.float32)

        with torch.no_grad():
            values = self._critic(obs_t).squeeze(-1)
            next_values = self._critic(next_obs_t).squeeze(-1)
        advantages = torch.zeros_like(rews_t)
        gae = 0.0
        for t in reversed(range(len(self._rollout))):
            delta = rews_t[t] + self.gamma * next_values[t] * (1 - dones_t[t]) - values[t]
            gae = float(delta.item()) + self.gamma * self.gae_lambda * gae * (1 - float(dones_t[t].item()))
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            self._actor.train()
            self._critic.train()
            logits = self._actor(obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_vals = self._critic(obs_t).squeeze(-1)
            critic_loss = F.mse_loss(critic_vals, returns)
            loss = actor_loss + 0.5 * critic_loss
            self._optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 0.5)
            self._optim.step()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "PPOController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._actor.load_state_dict(state["actor"])
        ctrl._critic.load_state_dict(state["critic"])
        return ctrl


# ===========================================================================
# A2C Controller (Advantage Actor-Critic)
# ===========================================================================

class A2CController(BaseController):
    """Synchronous Advantage Actor-Critic (A2C) signal controller.

    Similar to PPO but without the clipped objective. Uses direct policy
    gradient with GAE (lambda=0.95) advantage estimation. Simpler than PPO
    and faster per update, but less stable for large learning rates.

    Key differences from PPO:
    - No policy ratio clipping → more aggressive but less stable updates.
    - Entropy bonus (coef=0.01) maintains exploration throughout training.
    - Single update per rollout (no update_epochs loop).
    - Separate optimizers for actor and critic.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr_actor: float = 7e-4,
        lr_critic: float = 1e-3,
        entropy_coef: float = 0.01,
        rollout_len: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__(name="a2c")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.rollout_len = rollout_len
        self._current_phase: dict[int, int] = {}

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            return nn.Sequential(*layers)

        self._actor = mlp(STATE_DIM, N_ACTIONS)
        self._critic = mlp(STATE_DIM, 1)
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=lr_actor)
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=lr_critic)
        self._rollout: list[dict[str, Any]] = []

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._rollout = []

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            full_action = self._sample_action(_obs_to_vec(obs))
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        return _action_to_phase(self._sample_action(_obs_to_vec(obs)))

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        safe_action = max(0, min(int(action), N_ACTIONS - 1))
        self._rollout.append(
            {"obs": _obs_to_vec(obs), "action": safe_action, "reward": reward,
             "next_obs": _obs_to_vec(next_obs), "done": done}
        )
        if len(self._rollout) >= self.rollout_len or done:
            self._a2c_update()
            self._rollout = []

    def _sample_action(self, state: np.ndarray) -> int:
        import torch
        import torch.nn.functional as F
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self._actor.eval()
        with torch.no_grad():
            logits = self._actor(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item())

    def _a2c_update(self) -> None:
        import torch
        import torch.nn.functional as F
        if not self._rollout:
            return
        obs_t = torch.tensor(np.stack([r["obs"] for r in self._rollout]), dtype=torch.float32)
        acts_t = torch.tensor([r["action"] for r in self._rollout], dtype=torch.long)
        rews_t = torch.tensor([r["reward"] for r in self._rollout], dtype=torch.float32)
        next_obs_t = torch.tensor(np.stack([r["next_obs"] for r in self._rollout]), dtype=torch.float32)
        dones_t = torch.tensor([r["done"] for r in self._rollout], dtype=torch.float32)

        with torch.no_grad():
            values = self._critic(obs_t).squeeze(-1)
            next_values = self._critic(next_obs_t).squeeze(-1)
        advantages = torch.zeros_like(rews_t)
        gae = 0.0
        for t in reversed(range(len(self._rollout))):
            delta = rews_t[t] + self.gamma * next_values[t] * (1 - dones_t[t]) - values[t]
            gae = float(delta.item()) + self.gamma * self.gae_lambda * gae * (1 - float(dones_t[t].item()))
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor update: vanilla policy gradient (no clip)
        self._actor.train()
        logits = self._actor(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(acts_t)
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        self._actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
        self._actor_optim.step()

        # Critic update: MSE on value estimates
        self._critic.train()
        critic_vals = self._critic(obs_t).squeeze(-1)
        critic_loss = F.mse_loss(critic_vals, returns.detach())
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "A2CController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._actor.load_state_dict(state["actor"])
        ctrl._critic.load_state_dict(state["critic"])
        return ctrl


# ===========================================================================
# SAC Controller (Soft Actor-Critic, discrete variant)
# ===========================================================================

class SACController(BaseController):
    """Soft Actor-Critic for discrete action spaces.

    SAC maximises a trade-off between expected return and entropy of the policy,
    encouraging thorough exploration. The discrete variant outputs a categorical
    distribution over actions and uses twin Q-networks to prevent value
    overestimation. Entropy temperature alpha is tuned automatically to maintain
    a target entropy of -log(1/|A|) * 0.98.

    Key properties vs PPO/A2C:
    - Off-policy (replay buffer) → sample efficient.
    - Automatic entropy tuning → robust to hyperparameters.
    - Twin Q-networks → reduced overestimation.
    - Best suited to environments where exploration is critical.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        gamma: float = 0.99,
        lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        buffer_size: int = 50_000,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__(name="sac")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self._current_phase: dict[int, int] = {}
        self._buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            collections.deque(maxlen=buffer_size)
        )
        self._rng = np.random.default_rng(seed)

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        def mlp(in_d: int, out_d: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_d
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, out_d))
            return nn.Sequential(*layers)

        # Policy network outputs logits over N_ACTIONS
        self._policy = mlp(STATE_DIM, N_ACTIONS)
        # Twin Q-networks
        self._q1 = mlp(STATE_DIM, N_ACTIONS)
        self._q2 = mlp(STATE_DIM, N_ACTIONS)
        self._q1_target = mlp(STATE_DIM, N_ACTIONS)
        self._q2_target = mlp(STATE_DIM, N_ACTIONS)
        self._q1_target.load_state_dict(self._q1.state_dict())
        self._q2_target.load_state_dict(self._q2.state_dict())
        self._q1_target.eval()
        self._q2_target.eval()

        self._policy_optim = torch.optim.Adam(self._policy.parameters(), lr=lr)
        self._q_optim = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()), lr=lr
        )

        # Automatic entropy temperature (learnable log_alpha)
        import math
        self._target_entropy = -math.log(1.0 / N_ACTIONS) * 0.98
        self._log_alpha = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            full_action = self._sample_action(_obs_to_vec(obs))
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        return _action_to_phase(self._sample_action(_obs_to_vec(obs)))

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        safe_action = max(0, min(int(action), N_ACTIONS - 1))
        self._buffer.append((_obs_to_vec(obs), safe_action, reward, _obs_to_vec(next_obs), done))
        if len(self._buffer) >= self.batch_size:
            self._sac_update()

    def _sample_action(self, state: np.ndarray) -> int:
        import torch
        import torch.nn.functional as F
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self._policy.eval()
        with torch.no_grad():
            logits = self._policy(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item())

    def _sac_update(self) -> None:
        import torch
        import torch.nn.functional as F

        batch_idx = self._rng.choice(len(self._buffer), size=self.batch_size, replace=False)
        buf_list = list(self._buffer)
        batch = [buf_list[i] for i in batch_idx]
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        ns = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        alpha = self._log_alpha.exp().item()

        # Compute target Q using soft value
        with torch.no_grad():
            next_logits = self._policy(ns)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)
            q1_next = self._q1_target(ns)
            q2_next = self._q2_target(ns)
            min_q_next = torch.min(q1_next, q2_next)
            # Soft V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=-1)
            target_q = r + self.gamma * (1 - d) * v_next

        # Q-network update
        q1_pred = self._q1(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        q2_pred = self._q2(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        self._q_optim.zero_grad()
        q_loss.backward()
        self._q_optim.step()

        # Policy update
        logits = self._policy(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_s = self._q1(s)
            q2_s = self._q2(s)
            min_q_s = torch.min(q1_s, q2_s)
        policy_loss = (probs * (alpha * log_probs - min_q_s)).sum(dim=-1).mean()
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        # Alpha (entropy temperature) update
        entropy_diff = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = self._log_alpha * (entropy_diff - self._target_entropy).detach()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        # Soft target update (Polyak averaging)
        for p, tp in zip(self._q1.parameters(), self._q1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self._q2.parameters(), self._q2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"policy": self._policy.state_dict(),
             "q1": self._q1.state_dict(), "q2": self._q2.state_dict(),
             "log_alpha": self._log_alpha.item()},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "SACController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._policy.load_state_dict(state["policy"])
        ctrl._q1.load_state_dict(state["q1"])
        ctrl._q2.load_state_dict(state["q2"])
        ctrl._log_alpha.data.fill_(state["log_alpha"])
        return ctrl


# ===========================================================================
# Recurrent PPO Controller (LSTM actor/critic)
# ===========================================================================

class RecurrentPPOController(BaseController):
    """PPO with LSTM actor and critic for temporal traffic pattern capture.

    Unlike feedforward PPO, this controller maintains hidden state across
    timesteps, enabling it to recognise patterns like "traffic has been building
    for 5 minutes — a surge is coming." Uses the same clipped surrogate
    objective and GAE as PPOController but with LSTM networks.

    Architecture:
    - LSTM hidden size: 64, with 1 layer.
    - Sequence training length: 16 steps (truncated BPTT).
    - Hidden states carried across steps; reset at episode boundaries.

    Best used with variable demand profiles (event_surge, incident_response)
    where history matters for predicting near-future congestion.
    """

    LSTM_HIDDEN: int = 64
    SEQ_LEN: int = 16

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 4,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__(name="recurrent_ppo")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self._current_phase: dict[int, int] = {}

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        class LSTMActor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(STATE_DIM, RecurrentPPOController.LSTM_HIDDEN, batch_first=True)
                self.head = nn.Linear(RecurrentPPOController.LSTM_HIDDEN, N_ACTIONS)

            def forward(
                self,
                x: "torch.Tensor",
                hidden: "Optional[tuple[torch.Tensor, torch.Tensor]]" = None,
            ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]":
                out, hidden_new = self.lstm(x, hidden)
                return self.head(out), hidden_new

        class LSTMCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(STATE_DIM, RecurrentPPOController.LSTM_HIDDEN, batch_first=True)
                self.head = nn.Linear(RecurrentPPOController.LSTM_HIDDEN, 1)

            def forward(
                self,
                x: "torch.Tensor",
                hidden: "Optional[tuple[torch.Tensor, torch.Tensor]]" = None,
            ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]":
                out, hidden_new = self.lstm(x, hidden)
                return self.head(out), hidden_new

        self._actor = LSTMActor()
        self._critic = LSTMCritic()
        params = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optim = torch.optim.Adam(params, lr=lr)

        # Hidden state per intersection
        self._actor_hidden: dict[int, Any] = {}
        self._critic_hidden: dict[int, Any] = {}

        # Rollout buffer stores (obs, action, reward, next_obs, done, log_prob)
        self._rollout: list[dict[str, Any]] = []

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._actor_hidden = {}
        self._critic_hidden = {}
        self._rollout = []

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            full_action, _ = self._act_recurrent(iid, _obs_to_vec(obs))
            phase_idx = _action_to_phase(full_action)
            actions[iid] = "NS" if phase_idx == 0 else "EW"
            self._current_phase[iid] = phase_idx
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        """Return phase index (0=NS, 1=EW) — satisfies BaseController interface."""
        full_action, _ = self._act_recurrent(0, _obs_to_vec(obs))
        return _action_to_phase(full_action)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        vec = _obs_to_vec(obs)
        next_vec = _obs_to_vec(next_obs)
        safe_action = max(0, min(int(action), N_ACTIONS - 1))
        _, log_prob = self._act_recurrent(0, vec, return_log_prob=True)
        self._rollout.append(
            {"obs": vec, "action": safe_action, "reward": reward,
             "next_obs": next_vec, "done": done, "log_prob": log_prob}
        )
        if done or len(self._rollout) >= self.SEQ_LEN * 2:
            self._rppo_update()
            self._rollout = []
            self._actor_hidden = {}
            self._critic_hidden = {}

    def _act_recurrent(
        self, iid: int, state: np.ndarray, return_log_prob: bool = False
    ) -> tuple[int, float]:
        import torch
        import torch.nn.functional as F

        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
        hidden = self._actor_hidden.get(iid, None)
        self._actor.eval()
        with torch.no_grad():
            logits, new_hidden = self._actor(s, hidden)
            self._actor_hidden[iid] = (new_hidden[0].detach(), new_hidden[1].detach())
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            log_prob = float(dist.log_prob(a).item()) if return_log_prob else 0.0
        return int(a.item()), log_prob

    def _rppo_update(self) -> None:
        import torch
        import torch.nn.functional as F
        if not self._rollout:
            return

        obs_t = torch.tensor(np.stack([r["obs"] for r in self._rollout]), dtype=torch.float32).unsqueeze(0)
        acts_t = torch.tensor([r["action"] for r in self._rollout], dtype=torch.long)
        rews_t = torch.tensor([r["reward"] for r in self._rollout], dtype=torch.float32)
        next_obs_t = torch.tensor(np.stack([r["next_obs"] for r in self._rollout]), dtype=torch.float32).unsqueeze(0)
        dones_t = torch.tensor([r["done"] for r in self._rollout], dtype=torch.float32)
        old_lp_t = torch.tensor([r["log_prob"] for r in self._rollout], dtype=torch.float32)

        with torch.no_grad():
            v_out, _ = self._critic(obs_t)
            values = v_out.squeeze(0).squeeze(-1)
            nv_out, _ = self._critic(next_obs_t)
            next_values = nv_out.squeeze(0).squeeze(-1)

        advantages = torch.zeros_like(rews_t)
        gae = 0.0
        for t in reversed(range(len(self._rollout))):
            delta = rews_t[t] + self.gamma * next_values[t] * (1 - dones_t[t]) - values[t]
            gae = float(delta.item()) + self.gamma * self.gae_lambda * gae * (1 - float(dones_t[t].item()))
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            self._actor.train()
            self._critic.train()
            a_out, _ = self._actor(obs_t)
            logits = a_out.squeeze(0)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - old_lp_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            c_out, _ = self._critic(obs_t)
            critic_vals = c_out.squeeze(0).squeeze(-1)
            critic_loss = F.mse_loss(critic_vals, returns.detach())
            loss = actor_loss + 0.5 * critic_loss
            self._optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 0.5)
            self._optim.step()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "RecurrentPPOController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._actor.load_state_dict(state["actor"])
        ctrl._critic.load_state_dict(state["critic"])
        return ctrl
