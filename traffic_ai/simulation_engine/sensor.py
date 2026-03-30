"""traffic_ai/simulation_engine/sensor.py

Sensor fault model for AITO Phase 4.

Simulates three realistic loop-detector failure modes observed in NGSIM and
Caltrans PeMS datasets (Chen et al., 2001, "Freeway Loop Detector Errors"):

    stuck      — sensor returns the same value for ≥ stuck_window consecutive
                 steps (frozen counter failure; ~12% of detector outages)
    noise      — Gaussian noise added to the reading (calibration drift)
    dropout    — sensor returns 0 for a random burst of steps (data loss)

Usage
-----
    fault_model = SensorFaultModel(seed=42)
    corrupted_obs = fault_model.apply(raw_obs, step=t, intersection_id=iid)

The ``FaultTolerantController`` (in controllers/fault_tolerant.py) consumes
the corrupted observations and applies EWMA imputation before passing them
to the wrapped controller.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SensorFaultModel:
    """Probabilistic sensor fault injection for traffic loop detectors.

    Parameters
    ----------
    stuck_prob:
        Per-step probability that a sensor enters the stuck state.
        Source: ~2 % per 5-min interval from PeMS quality flags.
    noise_std:
        Standard deviation of Gaussian noise added in the noisy state
        (expressed as fraction of max_queue = 120 vehicles).
    dropout_prob:
        Per-step probability that a sensor drops out (returns 0).
    stuck_window:
        Minimum consecutive steps a sensor stays stuck before recovery.
    seed:
        RNG seed for reproducibility.
    """

    stuck_prob: float = 0.02
    noise_std: float = 0.05          # 5 % of max_queue → ~6 vehicles (120 max)
    dropout_prob: float = 0.01
    stuck_window: int = 5
    seed: int = 42

    # Internal state — not included in __init__ args
    _rng: random.Random = field(init=False, repr=False)
    _stuck_value: Dict[str, float] = field(init=False, repr=False, default_factory=dict)
    _stuck_remaining: Dict[str, int] = field(init=False, repr=False, default_factory=dict)
    _dropout_remaining: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._stuck_value = {}
        self._stuck_remaining = {}
        self._dropout_remaining = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        obs: Dict[str, float],
        step: int,
        intersection_id: int = 0,
    ) -> Dict[str, float]:
        """Return a potentially corrupted copy of *obs*.

        Only the queue-related features are corrupted; metadata fields
        (intersection_id, sim_step, phase_elapsed, emergency_active) are
        left untouched to preserve simulation integrity.
        """
        corrupted = dict(obs)
        CORRUPTIBLE = [
            "queue_ns", "queue_ew", "queue_ns_through", "queue_ew_through",
            "queue_ns_left", "queue_ew_left", "total_queue", "upstream_queue",
        ]

        for key in CORRUPTIBLE:
            if key not in corrupted:
                continue
            tag = f"{intersection_id}:{key}"
            val = corrupted[key]

            # --- dropout ---
            if self._dropout_remaining.get(tag, 0) > 0:
                corrupted[key] = 0.0
                self._dropout_remaining[tag] -= 1
                continue

            if self._rng.random() < self.dropout_prob:
                burst = self._rng.randint(1, 3)
                self._dropout_remaining[tag] = burst
                corrupted[key] = 0.0
                continue

            # --- stuck ---
            if self._stuck_remaining.get(tag, 0) > 0:
                corrupted[key] = self._stuck_value[tag]
                self._stuck_remaining[tag] -= 1
                continue

            if self._rng.random() < self.stuck_prob:
                self._stuck_value[tag] = val
                self._stuck_remaining[tag] = self.stuck_window
                corrupted[key] = val
                continue

            # --- noise ---
            noise = self._rng.gauss(0.0, self.noise_std * 120.0)
            corrupted[key] = max(0.0, val + noise)

        return corrupted

    def reset(self) -> None:
        """Clear all fault state (call between episodes)."""
        self._stuck_value.clear()
        self._stuck_remaining.clear()
        self._dropout_remaining.clear()
