"""traffic_ai/training/trainer.py

Unified training interface for any controller on any synthetic dataset.

For RL controllers, arrival-rate statistics are extracted from the dataset to
configure a matching ``EnvConfig`` / ``SimulatorConfig`` so the agent learns
for the specific traffic pattern encoded in the data.

For ML (supervised) controllers, the dataset is pivoted into a feature matrix
and passed directly to the controller's ``fit()`` method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from traffic_ai.config.settings import Settings


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Outcome of a ModelTrainer.train() call."""

    controller_name: str
    controller_type: str
    training_time_seconds: float
    episodes_trained: int
    final_accuracy: float | None
    reward_history: list[float]
    loss_history: list[float]
    evaluation_metrics: dict[str, float]
    model_path: Path | None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

_ML_FEATURE_COLS = [
    "queue_ns",
    "queue_ew",
    "speed_kph",
    "occupancy",
    "hour_mod_sec",
    "is_rush_hour",
    "day_of_week",
]

_RL_CONTROLLERS = {
    "q_learning", "dqn", "policy_gradient",
    "a2c", "sac", "recurrent_ppo",
}
_ML_CONTROLLERS = {
    "random_forest", "xgboost", "gradient_boosting", "mlp",
}


def _extract_ml_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the 7-feature matrix and label array expected by ML controllers.

    The synthetic dataset has one row per (timestamp, location, direction).
    We aggregate it to per-(timestamp, location) rows first.

    Returns
    -------
    (X, y)
        X : shape (n_pairs, 7) float64
        y : shape (n_pairs,) int32  — 0 or 1
    """
    # Pivot to per-intersection rows
    ns_mask = df["direction"].isin(["N", "S"])
    ew_mask = df["direction"].isin(["E", "W"])

    group_cols = ["timestamp", "location_id"]

    ns_q = df[ns_mask].groupby(group_cols)["queue_length"].sum().rename("queue_ns")
    ew_q = df[ew_mask].groupby(group_cols)["queue_length"].sum().rename("queue_ew")
    speed = df.groupby(group_cols)["speed_kph"].mean().rename("speed_kph")
    occ = df.groupby(group_cols)["occupancy"].mean().rename("occupancy")

    agg = (
        df.groupby(group_cols)
        .agg(
            hour_of_day=("hour_of_day", "first"),
            is_rush_hour=("is_rush_hour", "first"),
            day_of_week=("day_of_week", "first"),
            optimal_phase=("optimal_phase", "first"),
        )
    )

    wide = pd.concat([ns_q, ew_q, speed, occ, agg], axis=1).dropna()
    wide["hour_mod_sec"] = (wide["hour_of_day"] * 3600.0) % 86400.0

    X = wide[_ML_FEATURE_COLS].to_numpy(dtype=np.float64)
    y = wide["optimal_phase"].to_numpy(dtype=np.int32)
    return X, y


def _extract_rl_env_params(df: pd.DataFrame) -> dict[str, float]:
    """Extract arrival-rate statistics from a synthetic dataset.

    Returns
    -------
    dict with keys arrival_rate_ns, arrival_rate_ew (vehicles/step, not per lane)
    """
    ns_mask = df["direction"].isin(["N", "S"])
    ew_mask = df["direction"].isin(["E", "W"])

    mean_vc_ns = float(df.loc[ns_mask, "vehicle_count"].mean())
    mean_vc_ew = float(df.loc[ew_mask, "vehicle_count"].mean())

    # Scale to approximate veh/step at EnvConfig.step_limit scale
    # The existing EnvConfig default uses arrival_rate_ns=1.7, arrival_rate_ew=1.4
    # We normalise the dataset mean to that range.
    scale = 1.7 / max(mean_vc_ns, 0.1)
    arrival_ns = float(np.clip(mean_vc_ns * scale, 0.5, 4.0))
    arrival_ew = float(np.clip(mean_vc_ew * scale, 0.5, 4.0))

    return {"arrival_rate_ns": arrival_ns, "arrival_rate_ew": arrival_ew}


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Train any supported controller on a synthetic dataset.

    Supported controller types
    --------------------------
    RL  : q_learning, dqn, policy_gradient, a2c, sac, recurrent_ppo
    ML  : random_forest, xgboost, gradient_boosting, mlp
    """

    def train(
        self,
        controller_type: str,
        dataset: pd.DataFrame,
        config: dict[str, Any],
        settings: Settings,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """Train a controller on the provided dataset.

        Parameters
        ----------
        controller_type:
            One of the supported type strings (case-insensitive).
        dataset:
            Synthetic dataset DataFrame as returned by
            :class:`~traffic_ai.data_pipeline.synthetic_generator.SyntheticDatasetGenerator`.
        config:
            Hyperparameter overrides (episodes, lr, n_estimators, etc.).
        settings:
            Project settings (used for seed and output directory).
        progress_callback:
            Optional callable(fraction: float, status: str).

        Returns
        -------
        TrainingResult
        """
        ctype = controller_type.lower().strip()
        t0 = time.perf_counter()

        def _prog(frac: float, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(frac, msg)

        if ctype in _RL_CONTROLLERS:
            return self._train_rl(ctype, dataset, config, settings, _prog, t0)
        if ctype in _ML_CONTROLLERS:
            return self._train_ml(ctype, dataset, config, settings, _prog, t0)

        raise ValueError(
            f"Unknown controller type '{controller_type}'. "
            f"Valid types: {sorted(_RL_CONTROLLERS | _ML_CONTROLLERS)}"
        )

    # ------------------------------------------------------------------
    # RL training
    # ------------------------------------------------------------------

    def _train_rl(
        self,
        ctype: str,
        df: pd.DataFrame,
        config: dict[str, Any],
        settings: Settings,
        prog: Callable,
        t0: float,
    ) -> TrainingResult:
        prog(0.05, "Extracting traffic statistics from dataset…")
        env_params = _extract_rl_env_params(df)
        seed = config.get("seed", settings.seed)
        episodes = int(config.get("episodes", 200))

        prog(0.15, f"Configuring simulation environment for {ctype}…")

        from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv

        env_cfg = EnvConfig(
            arrival_rate_ns=env_params["arrival_rate_ns"],
            arrival_rate_ew=env_params["arrival_rate_ew"],
            step_limit=int(config.get("step_limit", 220)),
            seed=seed,
        )
        env = SignalControlEnv(env_cfg)

        reward_history: list[float] = []
        loss_history: list[float] = []

        prog(0.25, f"Training {ctype} for {episodes} episodes…")

        if ctype == "dqn":
            from traffic_ai.rl_models.dqn import train_dqn

            lr = float(config.get("lr", 1e-3))
            batch = int(config.get("batch_size", 128))
            policy, reward_history, _ = train_dqn(
                env, episodes=episodes, lr=lr, batch_size=batch, seed=seed
            )
            controller_name = "DQN"

        elif ctype == "q_learning":
            from traffic_ai.rl_models.q_learning import train_q_learning

            policy, reward_history = train_q_learning(
                env,
                episodes=episodes,
                alpha=float(config.get("alpha", 0.15)),
                gamma=float(config.get("gamma", 0.95)),
                seed=seed,
            )
            controller_name = "Q-Learning"

        elif ctype == "policy_gradient":
            from traffic_ai.rl_models.policy_gradient import train_policy_gradient

            policy, reward_history, _ = train_policy_gradient(
                env,
                episodes=episodes,
                lr=float(config.get("lr", 2e-3)),
                gamma=float(config.get("gamma", 0.98)),
            )
            controller_name = "Policy Gradient"

        elif ctype == "a2c":
            from traffic_ai.rl_models.a2c import train_a2c
            from traffic_ai.simulation_engine.demand import ALL_DEMAND_PROFILES

            demand_profile = config.get("demand_profile", "rush_hour")
            if demand_profile not in ALL_DEMAND_PROFILES:
                demand_profile = "rush_hour"
            ctrl = train_a2c(
                n_intersections=int(config.get("n_intersections", 4)),
                demand_profile=demand_profile,
                steps_per_episode=int(config.get("steps_per_episode", 300)),
                n_episodes=episodes,
                seed=seed,
            )
            controller_name = "A2C"
            policy = ctrl
            reward_history = []

        elif ctype == "sac":
            from traffic_ai.rl_models.sac import train_sac

            demand_profile = config.get("demand_profile", "rush_hour")
            ctrl = train_sac(
                n_intersections=int(config.get("n_intersections", 4)),
                demand_profile=demand_profile,
                steps_per_episode=int(config.get("steps_per_episode", 300)),
                n_episodes=episodes,
                seed=seed,
            )
            controller_name = "SAC"
            policy = ctrl
            reward_history = []

        elif ctype == "recurrent_ppo":
            from traffic_ai.rl_models.recurrent_ppo import train_recurrent_ppo

            demand_profile = config.get("demand_profile", "rush_hour")
            ctrl = train_recurrent_ppo(
                n_intersections=int(config.get("n_intersections", 4)),
                demand_profile=demand_profile,
                steps_per_episode=int(config.get("steps_per_episode", 300)),
                n_episodes=episodes,
                seed=seed,
            )
            controller_name = "Recurrent PPO"
            policy = ctrl
            reward_history = []

        else:
            raise ValueError(f"Unsupported RL type: {ctype}")

        prog(0.85, "Evaluating trained policy…")
        eval_metrics = self._evaluate_rl(policy, env, seed)

        elapsed = time.perf_counter() - t0
        prog(1.0, "Done.")
        return TrainingResult(
            controller_name=controller_name,
            controller_type=ctype,
            training_time_seconds=elapsed,
            episodes_trained=episodes,
            final_accuracy=None,
            reward_history=reward_history,
            loss_history=loss_history,
            evaluation_metrics=eval_metrics,
            model_path=None,
        )

    @staticmethod
    def _evaluate_rl(policy: Any, env: Any, seed: int) -> dict[str, float]:
        """Run 3 evaluation episodes and return mean metrics."""
        import numpy as _np

        total_rewards = []
        total_queues = []
        total_wait = []
        for ep in range(3):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_queue = 0.0
            ep_wait = 0.0
            steps = 0
            while not done:
                # Support both policy.act() and compute_actions() interfaces
                if hasattr(policy, "act"):
                    action = policy.act(obs)
                elif hasattr(policy, "compute_actions"):
                    actions = policy.compute_actions({0: {"queue_ns": float(obs[0]), "queue_ew": float(obs[1]), "total_queue": float(obs[2]), "phase_elapsed": float(obs[3]), "phase_ns": 1.0 - float(obs[4]), "phase_ew": float(obs[4]), "sim_step": float(steps), "arrivals": 0.0, "departures": 0.0, "wait_sec": 0.0, "emergency_active": 0.0, "intersection_id": 0.0}}, step=steps)
                    action = 0 if actions.get(0, "NS") == "NS" else 1
                else:
                    action = int(_np.argmax([float(obs[0]), float(obs[1])]))
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                ep_queue += float(info.get("total_queue", 0.0))
                ep_wait += float(info.get("queue_ns", 0.0) + info.get("queue_ew", 0.0))
                steps += 1
            total_rewards.append(ep_reward)
            total_queues.append(ep_queue / max(steps, 1))
            total_wait.append(ep_wait / max(steps, 1))

        return {
            "avg_episode_reward": float(_np.mean(total_rewards)),
            "avg_queue_length": float(_np.mean(total_queues)),
            "avg_wait_proxy": float(_np.mean(total_wait)),
        }

    # ------------------------------------------------------------------
    # ML training
    # ------------------------------------------------------------------

    def _train_ml(
        self,
        ctype: str,
        df: pd.DataFrame,
        config: dict[str, Any],
        settings: Settings,
        prog: Callable,
        t0: float,
    ) -> TrainingResult:
        prog(0.10, "Extracting features from dataset…")
        X, y = _extract_ml_features(df)

        if len(X) < 10:
            raise ValueError(
                "Dataset too small for ML training (need at least 10 aggregated rows)."
            )

        seed = config.get("seed", settings.seed)
        cv_folds = int(config.get("cv_folds", 3))

        prog(0.25, f"Instantiating {ctype} controller…")

        if ctype == "random_forest":
            from traffic_ai.controllers.ml_controllers import RandomForestController

            n_est = int(config.get("n_estimators", 100))
            ctrl = RandomForestController(n_estimators=n_est, seed=seed)
            controller_name = "Random Forest"

        elif ctype == "xgboost":
            from traffic_ai.controllers.ml_controllers import XGBoostController

            ctrl = XGBoostController(seed=seed)
            controller_name = "XGBoost"

        elif ctype == "gradient_boosting":
            from traffic_ai.controllers.ml_controllers import GradientBoostingController

            ctrl = GradientBoostingController(seed=seed)
            controller_name = "Gradient Boosting"

        elif ctype == "mlp":
            from traffic_ai.controllers.ml_controllers import MLPController

            ctrl = MLPController(seed=seed)
            controller_name = "Neural Network MLP"

        else:
            raise ValueError(f"Unsupported ML type: {ctype}")

        prog(0.45, f"Training {controller_name} with {cv_folds}-fold CV…")

        # ML controllers fit with string labels ("NS" / "EW").
        # Our labels are int32: 0 = NS green, 1 = EW green.
        y_str = np.where(y == 0, "NS", "EW")
        fit_metrics = ctrl.fit(X, y_str, cv_folds=cv_folds)

        prog(0.80, "Evaluating on held-out set…")
        accuracy = float(fit_metrics.get("cv_mean", 0.0))

        # Quick held-out evaluation (last 20% of data)
        split = max(1, int(len(X) * 0.8))
        X_test, y_test = X[split:], y[split:]
        if len(X_test) > 0:
            correct = sum(
                ctrl.select_action(
                    {
                        "queue_ns": float(row[0]),
                        "queue_ew": float(row[1]),
                        "avg_speed": float(row[2]),
                        "lane_occupancy": float(row[3]),
                        "step": float(row[4]),
                        "is_rush_hour": float(row[5]),
                        "day_of_week": float(row[6]),
                    }
                )
                == ("NS" if int(label) == 0 else "EW")
                for row, label in zip(X_test, y_test)
            )
            test_acc = correct / len(X_test)
        else:
            test_acc = accuracy

        elapsed = time.perf_counter() - t0
        prog(1.0, "Done.")
        return TrainingResult(
            controller_name=controller_name,
            controller_type=ctype,
            training_time_seconds=elapsed,
            episodes_trained=0,
            final_accuracy=test_acc,
            reward_history=[],
            loss_history=[],
            evaluation_metrics={
                "cv_mean_accuracy": accuracy,
                "cv_std": float(fit_metrics.get("cv_std", 0.0)),
                "test_accuracy": test_acc,
            },
            model_path=None,
        )
