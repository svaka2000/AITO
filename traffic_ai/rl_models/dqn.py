from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np

from traffic_ai.rl_models.environment import SignalControlEnv
from traffic_ai.rl_models.q_learning import QLearningPolicy, train_q_learning

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DQNetwork(nn.Module):
        def __init__(self, input_dim: int = 6, output_dim: int = 16) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


@dataclass(slots=True)
class DQNPolicy:
    network: Any

    def act(self, features: np.ndarray) -> int:
        if TORCH_AVAILABLE and self.network is not None:
            obs = torch.tensor(features[:6], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.network(obs)
            full_action = int(torch.argmax(q, dim=-1).item())
            # Return phase index (0=NS, 1=EW) from the 16-action encoding
            return full_action // 8
        return int(features[2] < features[3])  # queue_ns_norm < queue_ew_norm → EW


def train_dqn(
    env: SignalControlEnv,
    episodes: int = 220,
    gamma: float = 0.97,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DQNPolicy, list[float], Any]:
    rng = np.random.default_rng(seed)
    if not TORCH_AVAILABLE:
        fallback_policy, rewards = train_q_learning(env, episodes=max(20, episodes // 2), seed=seed)
        return DQNPolicy(network=fallback_policy), rewards, None

    torch.manual_seed(seed)
    device = torch.device("cpu")
    obs_dim = env.observation_dim
    n_act = env.n_actions
    policy_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=20_000)
    rewards: list[float] = []

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / max(episodes, 1)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_act))
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch_idx = rng.choice(len(memory), size=batch_size, replace=False)
                batch = [memory[i] for i in batch_idx]
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
                rewards_t = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1).values
                    target = rewards_t + gamma * next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        rewards.append(float(total_reward))
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return DQNPolicy(network=policy_net.eval()), rewards, policy_net.eval()

