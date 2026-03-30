from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from traffic_ai.rl_models.environment import SignalControlEnv
from traffic_ai.rl_models.q_learning import train_q_learning

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

    class PolicyNet(nn.Module):
        def __init__(self, input_dim: int = 6, output_dim: int = 16) -> None:
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, output_dim),
                nn.Softmax(dim=-1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)


@dataclass(slots=True)
class PolicyGradientPolicy:
    network: Any

    def act(self, features: np.ndarray) -> int:
        if TORCH_AVAILABLE and self.network is not None:
            with torch.no_grad():
                probs = self.network(torch.tensor(features[:6], dtype=torch.float32).unsqueeze(0))
            full_action = int(torch.argmax(probs, dim=-1).item())
            return full_action // 8  # extract phase (0 or 1)
        return int(features[2] < features[3])  # queue_ns_norm < queue_ew_norm → EW


def train_policy_gradient(
    env: SignalControlEnv,
    episodes: int = 220,
    gamma: float = 0.98,
    lr: float = 2e-3,
) -> tuple[PolicyGradientPolicy, list[float], Any]:
    if not TORCH_AVAILABLE:
        fallback_policy, rewards = train_q_learning(env, episodes=max(20, episodes // 2))
        return PolicyGradientPolicy(network=fallback_policy), rewards, None

    obs_dim = env.observation_dim
    n_act = env.n_actions
    policy = PolicyNet(input_dim=obs_dim, output_dim=n_act)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        total_reward = 0.0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            next_state, reward, done, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))
            total_reward += reward
            state = next_state

        discounted: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            discounted.append(running)
        discounted.reverse()
        discounted_t = torch.tensor(discounted, dtype=torch.float32)
        discounted_t = (discounted_t - discounted_t.mean()) / (discounted_t.std() + 1e-8)

        loss = 0.0
        for log_prob, ret in zip(log_probs, discounted_t):
            loss = loss - log_prob * ret
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_history.append(float(total_reward))

    return PolicyGradientPolicy(network=policy.eval()), reward_history, policy.eval()

