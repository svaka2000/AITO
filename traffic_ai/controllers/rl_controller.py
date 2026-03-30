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
        self.current_phase = {i: "NS" for i in range(n_intersections)}
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

            phase_int = 0 if obs.get("current_phase", "NS") == "NS" else 1
            hour = obs.get("hour_of_day", obs.get("step", 0.0) / 3600.0 % 24.0)
            features = np.array(
                [
                    obs.get("phase_elapsed", 0.0) / 60.0,
                    float(phase_int == 0),  # phase_ns flag
                    obs.get("queue_ns", 0.0) / 120.0,
                    obs.get("queue_ew", 0.0) / 120.0,
                    float(hour) / 24.0,  # time_of_day_normalized
                    obs.get("upstream_queue", 0.0) / 120.0,
                ],
                dtype=np.float32,
            )
            action = self._policy_action(features)
            target: SignalPhase = "NS" if int(action) == 0 else "EW"
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
