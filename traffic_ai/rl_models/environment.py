"""traffic_ai/rl_models/environment.py

Single-intersection MDP for RL pretraining, built on the canonical
TrafficNetworkSimulator.

Expanded action space (Problem 4)
----------------------------------
Action is now a joint (phase, green_duration) pair encoded as a single
integer in [0, 15]:

    action = phase_idx * 8 + duration_idx

where:
    phase_idx      : 0 = NS green,  1 = EW green
    duration_idx   : index into GREEN_DURATIONS = [15, 20, 25, 30, 35, 40, 45, 60]

This gives 2 × 8 = 16 discrete actions.

When ``select_action`` / ``compute_actions`` are called by controllers
*outside* of training, only the phase component (action // 8) is returned
so that the BaseController interface contract (returning 0 or 1) is preserved.

Observation space (6 floats)
-----------------------------
    [phase_elapsed_norm, cycle_length_norm, queue_ns_norm, queue_ew_norm,
     time_of_day_normalized, upstream_queue_norm]

Reward
------
    reward = -0.12 * total_queue
             - 0.05 * |queue_ns - queue_ew|
             - switch_cost
             - 0.02 * cycle_length          # cycle_length_penalty (Problem 4)

where switch_cost = 2.0 if phase changed, else 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.simulation_engine.types import SignalPhase


# Green duration choices (seconds) — must stay in sync with RL controllers
GREEN_DURATIONS: list[int] = [15, 20, 25, 30, 35, 40, 45, 60]
N_PHASES = 2
N_DURATIONS = len(GREEN_DURATIONS)
N_ACTIONS = N_PHASES * N_DURATIONS  # 16


@dataclass(slots=True)
class EnvConfig:
    max_queue: float = 120.0
    switch_penalty: float = 2.0
    cycle_length_penalty: float = 0.02   # penalty weight for long green phases (Problem 4)
    step_limit: int = 220
    seed: int = 42
    demand_profile: str = "rush_hour"


class SignalControlEnv:
    """Single-intersection MDP wrapping the canonical simulation engine.

    The environment uses ``TrafficNetworkSimulator`` with one intersection
    so that all physics (HCM signal timing, EPA emissions, demand profiles)
    are identical to the benchmarking engine used by the experiment runner.

    Action encoding
    ---------------
    ``action`` ∈ [0, N_ACTIONS) = phase_idx * N_DURATIONS + duration_idx
    ``phase_idx``   = action // N_DURATIONS   → 0 (NS) or 1 (EW)
    ``duration_idx`` = action %  N_DURATIONS  → index into GREEN_DURATIONS
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        if config is None:
            config = EnvConfig()
        self.config = config

        engine_cfg = SimulatorConfig(
            steps=config.step_limit,
            intersections=1,
            lanes_per_direction=2,
            step_seconds=1.0,
            max_queue_per_lane=int(config.max_queue // 2),  # per lane
            demand_profile=config.demand_profile,
            demand_scale=1.0,
            seed=config.seed,
        )
        self._engine = TrafficNetworkSimulator(engine_cfg)
        self.step_idx: int = 0
        self._current_phase: int = 0       # 0=NS, 1=EW
        self._current_duration: int = GREEN_DURATIONS[3]  # 30 s default
        self._phase_hold_remaining: int = 0
        self._last_obs: dict[str, float] = {}

    @property
    def observation_dim(self) -> int:
        return 6

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def reset(self) -> np.ndarray:
        raw = self._engine.reset_env()
        self.step_idx = 0
        obs0 = raw.get(0, {})
        self._current_phase = 0
        self._current_duration = GREEN_DURATIONS[3]
        self._phase_hold_remaining = 0
        self._last_obs = obs0
        return self._encode_obs(obs0)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        """Advance one simulation step.

        Parameters
        ----------
        action:
            Integer in [0, N_ACTIONS).  Encodes both the desired phase and the
            green duration to hold that phase.
        """
        self.step_idx += 1

        # Decode action
        phase_idx = int(action) // N_DURATIONS
        duration_idx = int(action) % N_DURATIONS
        new_phase: int = phase_idx
        new_duration: int = GREEN_DURATIONS[duration_idx]

        # Switch cost
        switch_cost = 0.0
        if new_phase != self._current_phase:
            switch_cost = self.config.switch_penalty
        self._current_phase = new_phase
        self._current_duration = new_duration

        # Apply to engine
        phase_str: SignalPhase = "NS" if new_phase == 0 else "EW"
        engine_actions = {0: phase_str}
        raw_obs, _, engine_done, _ = self._engine.step_env(engine_actions)

        obs = raw_obs.get(0, {})
        self._last_obs = obs

        queue_ns = obs.get("queue_ns", 0.0)
        queue_ew = obs.get("queue_ew", 0.0)
        total_queue = queue_ns + queue_ew

        # Reward: queue minimisation + fairness + switch cost + cycle length penalty
        reward = (
            -0.12 * total_queue
            - 0.05 * abs(queue_ns - queue_ew)
            - switch_cost
            - self.config.cycle_length_penalty * float(new_duration)  # cycle_length_penalty
        )

        done = self.step_idx >= self.config.step_limit or engine_done
        info = {
            "queue_ns": queue_ns,
            "queue_ew": queue_ew,
            "total_queue": total_queue,
            "phase": float(new_phase),
            "duration": float(new_duration),
        }
        return self._encode_obs(obs), float(reward), done, info

    def _encode_obs(self, obs: dict[str, float]) -> np.ndarray:
        """Encode engine observation dict to 6-float RL state vector.

        Features
        --------
        0  phase_elapsed_norm    : phase_elapsed / 60
        1  cycle_length_norm     : current_duration / 60
        2  queue_ns_norm         : queue_ns / max_queue
        3  queue_ew_norm         : queue_ew / max_queue
        4  time_of_day_norm      : time_of_day in [0, 1]
        5  upstream_queue_norm   : upstream_queue / max_queue
        """
        max_q = float(self.config.max_queue)
        return np.array(
            [
                obs.get("phase_elapsed", 0.0) / 60.0,
                float(self._current_duration) / 60.0,
                obs.get("queue_ns", 0.0) / max_q,
                obs.get("queue_ew", 0.0) / max_q,
                obs.get("time_of_day_normalized", 0.0),
                obs.get("upstream_queue", 0.0) / max(max_q, 1.0),
            ],
            dtype=np.float32,
        )
