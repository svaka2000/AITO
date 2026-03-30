from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_ai.rl_models.dqn import train_dqn
from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv
from traffic_ai.rl_models.policy_gradient import train_policy_gradient
from traffic_ai.rl_models.q_learning import train_q_learning
from traffic_ai.utils.io_utils import save_model, write_dataframe

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class RLTrainingResult:
    policies: dict[str, Any]
    reward_history: pd.DataFrame


def train_rl_policy_suite(
    output_dir: Path,
    seed: int = 42,
    quick_run: bool = True,
    full_run: bool = False,
) -> RLTrainingResult:
    if full_run:
        episodes = 2000
    elif quick_run:
        episodes = 35
    else:
        episodes = 260
    env = SignalControlEnv(EnvConfig(seed=seed))
    # DQN full-run uses 500-step episodes to match the evaluation horizon and prevent
    # the agent from exploiting episode boundaries to ignore EW-direction starvation.
    dqn_env = SignalControlEnv(EnvConfig(seed=seed, step_limit=500)) if full_run else env
    policies: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    q_policy, q_rewards = train_q_learning(env, episodes=episodes)
    policies["q_learning"] = q_policy
    rows.extend({"algorithm": "q_learning", "episode": i, "reward": r} for i, r in enumerate(q_rewards))
    save_model(q_policy, output_dir / "models" / "q_learning_policy.joblib")

    dqn_policy, dqn_rewards, dqn_network = train_dqn(dqn_env, episodes=episodes)
    policies["dqn"] = dqn_policy
    rows.extend({"algorithm": "dqn", "episode": i, "reward": r} for i, r in enumerate(dqn_rewards))
    if torch is not None and dqn_network is not None:
        torch.save(dqn_network.state_dict(), output_dir / "models" / "dqn_policy.pt")

    pg_policy, pg_rewards, pg_network = train_policy_gradient(env, episodes=episodes)
    policies["policy_gradient"] = pg_policy
    rows.extend({"algorithm": "policy_gradient", "episode": i, "reward": r} for i, r in enumerate(pg_rewards))
    if torch is not None and pg_network is not None:
        torch.save(pg_network.state_dict(), output_dir / "models" / "policy_gradient.pt")

    history = pd.DataFrame(rows)
    write_dataframe(history, output_dir / "results" / "rl_training_history.csv")
    return RLTrainingResult(policies=policies, reward_history=history)
