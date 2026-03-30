from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class RLPolicyController(BaseController):
    def __init__(self, policy: object, name: str = "rl_policy", min_green: int = 6) -> None:
        super().__init__(name=name)
        self.policy = policy
        self.min_green = min_green
        self.current_phase: dict[int, SignalPhase] = {}
        self.green_elapsed: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self.current_phase = {i: "NS_THROUGH" for i in range(n_intersections)}
        self.green_elapsed = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for intersection_id, obs in observations.items():
            phase = self.current_phase[intersection_id]
            elapsed = self.green_elapsed[intersection_id] + 1
            if elapsed < self.min_green:
                actions[intersection_id] = phase
                self.green_elapsed[intersection_id] = elapsed
                continue

            phase_idx = obs.get("current_phase_idx", 0.0)
            features = np.array(
                [
                    obs.get("phase_elapsed", 0.0) / 60.0,
                    float(phase_idx) / 3.0,
                    obs.get("queue_ns_through", obs.get("queue_ns", 0.0)) / 120.0,
                    obs.get("queue_ew_through", obs.get("queue_ew", 0.0)) / 120.0,
                    obs.get("queue_ns_left", 0.0) / 120.0,
                    obs.get("queue_ew_left", 0.0) / 120.0,
                    obs.get("time_of_day_normalized",
                            (float(obs.get("step", obs.get("sim_step", 0.0))) % 86400.0) / 86400.0),
                    0.0,  # upstream_queue always 0 — policy trained on 1-intersection (no upstream)
                ],
                dtype=np.float32,
            )
            action = self._policy_action(features)
            from traffic_ai.simulation_engine.types import IDX_TO_PHASE
            phase_idx_out = int(action) % 4
            target: SignalPhase = IDX_TO_PHASE.get(phase_idx_out, "NS_THROUGH")
            if target != phase:
                elapsed = 0
            self.current_phase[intersection_id] = target
            self.green_elapsed[intersection_id] = elapsed
            actions[intersection_id] = target
        return actions

    def _policy_action(self, features: np.ndarray) -> int:
        if hasattr(self.policy, "act"):
            return int(self.policy.act(features))
        if hasattr(self.policy, "predict"):
            pred = self.policy.predict(features.reshape(1, -1))
            return int(pred[0])
        if callable(self.policy):
            return int(self.policy(features))
        if torch is not None and isinstance(self.policy, torch.nn.Module):
            with torch.no_grad():
                logits = self.policy(torch.tensor(features).unsqueeze(0))
                return int(torch.argmax(logits, dim=-1).item())
        return int(features[0] >= features[1])
