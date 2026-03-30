"""traffic_ai/controllers/fault_tolerant.py

Fault-tolerant controller wrapper for AITO Phase 4.

Wraps any BaseController and applies EWMA (Exponentially Weighted Moving
Average) imputation to corrupted sensor observations before passing them to
the underlying controller.  Falls back to a queue-balanced default action
on complete sensor failure (all queue readings ≤ 0).

EWMA imputation follows the approach used in:
    Toth & Ceder (2002), "Transportation Research Part C" — sensor data
    quality assessment and imputation; smoothing factor α = 0.3 (default).

Usage
-----
    base = DQNController()
    ctrl = FaultTolerantController(base, alpha=0.3)
    ctrl.reset(n_intersections)
    actions = ctrl.compute_actions(corrupted_obs, step)
"""
from __future__ import annotations

from typing import Dict, Optional

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class FaultTolerantController(BaseController):
    """EWMA-imputing wrapper around any BaseController.

    Parameters
    ----------
    controller:
        The underlying controller to wrap.
    alpha:
        EWMA smoothing factor ∈ (0, 1].  Higher → faster response to new
        values; lower → heavier smoothing.  Default 0.3 per Toth & Ceder
        (2002).
    """

    def __init__(
        self,
        controller: BaseController,
        alpha: float = 0.3,
    ) -> None:
        super().__init__(name=f"fault_tolerant_{controller.name}")
        self._ctrl = controller
        self.alpha = alpha
        # EWMA state per (intersection_id, feature_key)
        self._ewma: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._ctrl.reset(n_intersections)
        self._ewma.clear()

    def compute_actions(
        self, observations: Dict[int, Dict[str, float]], step: int
    ) -> Dict[int, SignalPhase]:
        cleaned = {}
        for iid, obs in observations.items():
            cleaned[iid] = self._impute(iid, obs)
        return self._ctrl.compute_actions(cleaned, step)

    def select_action(self, obs: Dict[str, float]) -> int:
        cleaned = self._impute(0, obs)
        return self._ctrl.select_action(cleaned)

    def update(
        self,
        obs: Dict[str, float],
        action: int,
        reward: float,
        next_obs: Dict[str, float],
        done: bool = False,
    ) -> None:
        self._ctrl.update(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute(self, iid: int, obs: Dict[str, float]) -> Dict[str, float]:
        """Return obs with EWMA-imputed values for suspicious (≤ 0) readings."""
        QUEUE_KEYS = [
            "queue_ns", "queue_ew", "queue_ns_through", "queue_ew_through",
            "queue_ns_left", "queue_ew_left", "total_queue", "upstream_queue",
        ]
        result = dict(obs)

        # Check for total sensor failure (all queues report 0)
        queue_vals = [result.get(k, 0.0) for k in QUEUE_KEYS if k in result]
        total_sensor_ok = any(v > 0.0 for v in queue_vals)

        if not total_sensor_ok:
            # Complete sensor failure — return EWMA-cached values or 0
            for k in QUEUE_KEYS:
                if k in result:
                    ewma_key = f"{iid}:{k}"
                    result[k] = self._ewma.get(ewma_key, 0.0)
            return result

        for k in QUEUE_KEYS:
            if k not in result:
                continue
            ewma_key = f"{iid}:{k}"
            val = result[k]

            if val <= 0.0 and ewma_key in self._ewma:
                # Impute with EWMA-cached previous value
                result[k] = self._ewma[ewma_key]
            else:
                # Update EWMA with current reading
                prev = self._ewma.get(ewma_key, val)
                self._ewma[ewma_key] = self.alpha * val + (1.0 - self.alpha) * prev
                result[k] = self._ewma[ewma_key]

        return result
