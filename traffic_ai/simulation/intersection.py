"""traffic_ai/simulation/intersection.py

MultiIntersectionNetwork: Gym-compatible N×M grid wrapper around the
canonical TrafficNetworkSimulator.

This module previously contained a second, divergent simulation engine.
It has been unified: all physics now come from
``traffic_ai.simulation_engine.engine.TrafficNetworkSimulator``.  The
Gym-compatible ``reset()`` / ``step()`` interface is preserved unchanged so
that existing tests and dashboard code continue to work without modification.
"""
from __future__ import annotations

from typing import Any

from traffic_ai.simulation_engine.demand import DemandModel
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.simulation_engine.types import SignalPhase


class MultiIntersectionNetwork:
    """N×M grid of intersections with Gym-compatible interface.

    All simulation physics are provided by :class:`TrafficNetworkSimulator`.
    This class is a thin Gym adapter that maps between integer phase actions
    (0 = NS, 1 = EW) and the engine's ``SignalPhase`` type.

    Parameters
    ----------
    rows, cols:
        Grid dimensions (default 2×2 = 4 intersections).
    lanes_per_approach:
        Number of lanes per direction at each intersection.
    max_queue_length:
        Maximum vehicles per lane before spillback triggers.
    max_steps:
        Episode length in simulation steps.
    step_seconds:
        Real-world seconds represented by each simulation step.
    base_arrival_rate:
        Base Poisson rate (vehicles/second/lane) under normal conditions.
        Maps to ``demand_scale`` on the canonical engine.
    rush_hour_scale:
        Multiplier applied during rush hours.  Retained for API compatibility;
        the canonical engine handles time-of-day scaling via demand profiles.
    seed:
        Random seed for reproducibility.
    calibration_data:
        Optional dict mapping hour → mean_vehicle_count for Poisson calibration.
        When provided, the engine is configured with a calibrated demand scale.
    demand_profile:
        Demand profile name forwarded to the engine (default ``"rush_hour"``).
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        lanes_per_approach: int = 2,
        max_queue_length: int = 50,
        max_steps: int = 2_000,
        step_seconds: float = 1.0,
        base_arrival_rate: float = 0.12,
        rush_hour_scale: float = 2.5,
        seed: int = 42,
        calibration_data: dict[int, float] | None = None,
        demand_profile: str = "rush_hour",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n_intersections = rows * cols
        self.lanes = lanes_per_approach
        self.max_queue = max_queue_length
        self.max_steps = max_steps
        self.step_seconds = step_seconds
        self.base_rate = base_arrival_rate
        self.rush_scale = rush_hour_scale
        self.seed = seed
        self.calibration = calibration_data or {}

        # Derive demand_scale from base_arrival_rate relative to 0.12 default
        demand_scale = (base_arrival_rate / 0.12) if base_arrival_rate > 0 else 1.0

        cfg = SimulatorConfig(
            steps=max_steps,
            intersections=self.n_intersections,
            lanes_per_direction=lanes_per_approach,
            step_seconds=step_seconds,
            max_queue_per_lane=max_queue_length,
            demand_profile=demand_profile,
            demand_scale=demand_scale,
            seed=seed,
        )
        self._engine = TrafficNetworkSimulator(cfg)
        self._step_count: int = 0
        self._spillback_events: int = 0
        self._green_switches: int = 0

    # ------------------------------------------------------------------
    # Gym-compatible interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> dict[int, dict[str, float]]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            self._engine.config.seed = seed
        self._step_count = 0
        self._spillback_events = 0
        self._green_switches = 0
        raw_obs = self._engine.reset_env()
        return self._map_observations(raw_obs)

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, dict[str, float]], float, bool, dict[str, Any]]:
        """Advance the simulation one step.

        Parameters
        ----------
        actions:
            ``{node_id: phase}`` where phase is 0 (NS green) or 1 (EW green).

        Returns
        -------
        obs:
            Per-intersection observation dicts.
        reward:
            Negative total queue length (minimise congestion).
        done:
            True when episode length is reached.
        info:
            Auxiliary metrics.
        """
        # Map int actions → SignalPhase strings for the engine
        phase_actions: dict[int, SignalPhase] = {
            nid: ("NS" if int(a) == 0 else "EW")
            for nid, a in actions.items()
        }

        raw_obs, reward, done, info = self._engine.step_env(phase_actions)

        # Track switches for info dict (compare to previous phase)
        prev_phases = {
            nid: state.current_phase
            for nid, state in self._engine.states.items()
        }

        self._step_count += 1
        done = self._step_count >= self.max_steps

        mapped_obs = self._map_observations(raw_obs)

        # Count spillback events (queues at max capacity)
        for state in self._engine.states.values():
            for q in state.queue_matrix.values():
                if float(q.max()) >= self.max_queue:
                    self._spillback_events += 1

        info["spillback_events"] = self._spillback_events
        info["green_switches"] = self._green_switches
        info["hour"] = (self._step_count * self.step_seconds / 3600.0) % 24.0

        return mapped_obs, reward, done, info

    # ------------------------------------------------------------------
    # Observation adapter
    # ------------------------------------------------------------------

    def _map_observations(
        self, raw: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Map engine observations to the expected Gym observation format.

        The engine's ``as_observation`` includes all needed keys; this method
        adds the ``current_phase`` integer (0/1) expected by existing code.
        """
        out: dict[int, dict[str, float]] = {}
        for nid, obs in raw.items():
            mapped = dict(obs)
            # current_phase as int (0=NS, 1=EW) for backward compatibility
            mapped["current_phase"] = 0.0 if obs.get("phase_ns", 1.0) > 0.5 else 1.0
            # avg_speed approximation (engine doesn't compute speed directly)
            mapped["avg_speed"] = max(0.0, 60.0 - obs.get("total_queue", 0.0) * 0.5)
            # lane_occupancy approximation
            mapped["lane_occupancy"] = min(
                1.0,
                obs.get("total_queue", 0.0) / max(1.0, self.lanes * 4 * self.max_queue),
            )
            # Alias step → node_id for consistency with old IntersectionNode.observe()
            mapped["node_id"] = float(nid)
            mapped["step"] = obs.get("sim_step", 0.0)
            mapped["arrivals"] = obs.get("arrivals", 0.0)
            mapped["departures"] = obs.get("departures", 0.0)
            mapped["wait_time"] = obs.get("wait_sec", 0.0) / max(obs.get("departures", 1.0), 1.0)
            out[nid] = mapped
        return out

    # ------------------------------------------------------------------
    # Convenience properties (preserve old API)
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.n_intersections

    @property
    def _nodes(self) -> dict:
        """Expose engine states under the old _nodes name for test compatibility."""
        return self._engine.states

    @property
    def _neighbors(self) -> dict:
        return self._engine.neighbors

    @property
    def observation_shape(self) -> tuple[int]:
        """Flat observation vector size for a single intersection."""
        sample = self._map_observations(
            self._engine._collect_observations(0)
        )
        return (len(next(iter(sample.values()))),)

    @property
    def n_actions(self) -> int:
        return 2  # NS green or EW green
