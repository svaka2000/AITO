from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from traffic_ai.simulation_engine.demand import DemandModel
from traffic_ai.simulation_engine.emissions import EmissionsCalculator
from traffic_ai.simulation_engine.types import (
    Direction,
    IntersectionState,
    SignalPhase,
    SimulationResult,
    StepMetrics,
)


class ControllerLike(Protocol):
    name: str

    def reset(self, n_intersections: int) -> None: ...

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]: ...


@dataclass(slots=True)
class SimulatorConfig:
    steps: int = 2000
    intersections: int = 4
    lanes_per_direction: int = 2
    step_seconds: float = 1.0
    max_queue_per_lane: int = 60
    demand_profile: str = "rush_hour"
    demand_scale: float = 1.0
    seed: int = 42
    # HCM 7th edition signal timing defaults
    min_green_sec: int = 7          # HCM 7th ed. minimum green interval (seconds)
    yellow_sec: int = 3             # HCM 7th ed. yellow change interval (seconds)
    all_red_sec: int = 1            # HCM 7th ed. all-red clearance interval (seconds)
    saturation_flow_rate: float = 1800.0  # veh/hr/lane — HCM 7th ed. default for LT+TH+RT
    # HCM-aligned turning movement factor (fraction of departures forwarded downstream)
    turning_movement_factor: float = 0.60  # replaces hardcoded 0.65; HCM-aligned default


class TrafficNetworkSimulator:
    """Canonical physics engine for the entire platform.

    All controllers, RL training, benchmarking, and the dashboard live
    simulation run through this single engine.  Two operating modes:

    * **Batch mode** – ``run(controller)`` runs a full episode and returns a
      ``SimulationResult``.  Used by the experiment runner and benchmarks.

    * **Step-by-step mode** – ``reset_env()`` + ``step_env(actions)`` expose a
      Gym-compatible interface used by ``MultiIntersectionNetwork`` and
      ``SignalControlEnv``.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.demand = DemandModel(
            profile=config.demand_profile,  # type: ignore[arg-type]
            scale=config.demand_scale,
            step_seconds=config.step_seconds,
            seed=config.seed,
        )
        self._emissions_calc = EmissionsCalculator()
        self.intersection_ids = list(range(config.intersections))
        self.neighbors = self._build_directional_neighbors()
        self.states = self._init_intersections()
        # Step-by-step mode tracking
        self._env_step: int = 0

    # =========================================================================
    # Batch mode: run a full episode
    # =========================================================================

    def run(self, controller: ControllerLike, steps: int | None = None) -> SimulationResult:
        total_steps = steps or self.config.steps
        controller.reset(len(self.states))
        self.states = self._init_intersections()
        step_logs: list[StepMetrics] = []

        baseline_reference_queue = None
        for step in range(total_steps):
            self.demand.tick_emergency(step)
            self.demand.tick_incident(step)
            self._apply_emergency_events(step)

            observations = self._collect_observations(step)
            actions = controller.compute_actions(observations, step)
            actions = self._override_emergency_actions(actions)

            self._advance_step(actions, step)
            metrics = self._compute_step_metrics(step)
            if baseline_reference_queue is None:
                baseline_reference_queue = max(metrics.total_queue, 1.0)
            metrics.delay_reduction_pct = max(
                -100.0,
                min(100.0, (baseline_reference_queue - metrics.total_queue) / baseline_reference_queue * 100.0),
            )
            step_logs.append(metrics)

        summaries = self._intersection_summaries()
        aggregate = self._aggregate_metrics(step_logs)
        return SimulationResult(
            controller_name=controller.name,
            step_metrics=step_logs,
            intersection_summaries=summaries,
            aggregate=aggregate,
        )

    # =========================================================================
    # Step-by-step mode: Gym-compatible interface
    # =========================================================================

    def reset_env(self) -> dict[int, dict[str, float]]:
        """Reset all intersection states for step-by-step operation.

        Returns
        -------
        Initial observations for all intersections.
        """
        self.states = self._init_intersections()
        self._env_step = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.demand = DemandModel(
            profile=self.config.demand_profile,  # type: ignore[arg-type]
            scale=self.config.demand_scale,
            step_seconds=self.config.step_seconds,
            seed=self.config.seed,
        )
        return self._collect_observations(0)

    def step_env(
        self,
        actions: dict[int, SignalPhase],
    ) -> tuple[dict[int, dict[str, float]], float, bool, dict[str, Any]]:
        """Advance the simulation one step.

        Parameters
        ----------
        actions:
            ``{intersection_id: phase}`` where phase is ``"NS"`` or ``"EW"``.

        Returns
        -------
        (obs, reward, done, info)
            Gym-compatible tuple.
        """
        step = self._env_step
        self.demand.tick_emergency(step)
        self.demand.tick_incident(step)
        self._apply_emergency_events(step)
        actions = self._override_emergency_actions(actions)

        self._advance_step(actions, step)
        metrics = self._compute_step_metrics(step)

        self._env_step += 1
        done = self._env_step >= self.config.steps

        total_q = metrics.total_queue
        obs = self._collect_observations(self._env_step)
        reward = -float(total_q)
        info: dict[str, Any] = {
            "step": self._env_step,
            "total_queue": total_q,
            "avg_wait": metrics.avg_wait_sec,
            "throughput": metrics.throughput,
            "emissions_co2_kg": metrics.emissions_co2_kg,
            "fuel_gallons": metrics.fuel_gallons,
        }
        return obs, reward, done, info

    # =========================================================================
    # Emergency handling
    # =========================================================================

    def _apply_emergency_events(self, step: int) -> None:
        """Assign incoming emergency events to random intersections."""
        events = self.demand.pop_emergency_events()
        for ev in events:
            iid = int(self.rng.integers(0, len(self.states)))
            state = self.states[iid]
            state.emergency_active = True
            state.emergency_direction = ev.direction
            state.emergency_steps_remaining = ev.steps_remaining

    def _override_emergency_actions(
        self, actions: dict[int, SignalPhase]
    ) -> dict[int, SignalPhase]:
        """Force green for emergency vehicle direction, decrement counter."""
        overridden = dict(actions)
        for iid, state in self.states.items():
            if state.emergency_active and state.emergency_steps_remaining > 0:
                ev_dir = state.emergency_direction
                required_phase: SignalPhase = "NS" if ev_dir in ("N", "S") else "EW"
                overridden[iid] = required_phase
                state.emergency_steps_remaining -= 1
                if state.emergency_steps_remaining <= 0:
                    state.emergency_active = False
                    state.emergency_direction = ""
        return overridden

    # =========================================================================
    # Orchestrator: sequences one simulation tick
    # =========================================================================

    def _advance_step(self, actions: dict[int, SignalPhase], step: int) -> None:
        """Orchestrates one simulation tick: inflow → signals → arrivals → service."""
        self._apply_pending_inflow()
        self._update_signal_phases(actions)
        self._generate_stochastic_arrivals(step)
        self._service_and_propagate()

    # =========================================================================
    # SRP sub-steps
    # =========================================================================

    def _apply_pending_inflow(self) -> None:
        """Moves vehicles queued from the previous tick into a randomly chosen receiving lane."""
        for state in self.states.values():
            for direction, incoming in state.pending_inflow.items():
                lane_index = int(self.rng.integers(0, self.config.lanes_per_direction))
                state.queue_matrix[direction][lane_index] = min(
                    state.max_queue_per_lane,
                    state.queue_matrix[direction][lane_index] + incoming,
                )
            state.pending_inflow = {}

    def _update_signal_phases(self, actions: dict[int, SignalPhase]) -> None:
        """Commits phase decisions with HCM 7th edition minimum green enforcement.

        Phase changes are deferred until:
        1. The current phase has been green for at least ``min_green_sec`` steps,
           AND
        2. Any pending yellow+all-red transition has completed.

        When a phase change is approved, a transition interval of
        ``yellow_sec + all_red_sec`` steps begins.  During this window the
        old phase remains nominally active but ``transition_steps_remaining > 0``
        causes ``_service_intersection`` to suppress all departures (all-red
        clearance semantics).
        """
        for intersection_id, state in self.states.items():
            requested = actions.get(intersection_id, state.current_phase)

            # --- Complete any in-progress yellow/all-red transition ---
            if state.transition_steps_remaining > 0:
                state.transition_steps_remaining -= 1
                if state.transition_steps_remaining == 0:
                    # Transition complete: switch to the target phase
                    state.current_phase = state.target_phase
                    state.phase_elapsed = 0
                    state.phase_changes += 1
                continue  # no further action while in transition

            # --- Steady-state: evaluate whether a phase change is allowed ---
            if requested != state.current_phase:
                # HCM 7th ed.: enforce minimum green before phase change
                if state.phase_elapsed >= self.config.min_green_sec:
                    # Begin yellow + all-red clearance interval
                    transition = self.config.yellow_sec + self.config.all_red_sec
                    state.transition_steps_remaining = transition
                    state.target_phase = requested
                    # phase_elapsed keeps counting during transition (not reset yet)
                else:
                    # Minimum green not satisfied — hold current phase
                    state.phase_elapsed += 1
            else:
                state.phase_elapsed += 1

    def _generate_stochastic_arrivals(self, step: int) -> None:
        """Adds Poisson-distributed arrivals to each lane based on the current demand profile."""
        for state in self.states.values():
            for direction in ["N", "S", "E", "W"]:
                self._sample_arrivals_for_direction(state, direction, step)

    def _service_and_propagate(self) -> None:
        """Clears green-phase queues; forwards turning_movement_factor of departures downstream.

        The propagation fraction defaults to 0.60 (HCM-aligned; replaces the
        previously hardcoded 0.65 which had no cited source).
        """
        flow_to_neighbors: dict[tuple[int, Direction], float] = defaultdict(float)
        for intersection_id, state in self.states.items():
            departures_by_direction = self._service_intersection(state)
            for direction, departed in departures_by_direction.items():
                neighbor_id = self.neighbors[intersection_id].get(direction)
                if neighbor_id is None:
                    continue
                # HCM-aligned turning movement factor (configurable, default 0.60)
                transfer = departed * self.config.turning_movement_factor
                if transfer <= 0:
                    continue
                incoming_direction = self._opposite(direction)
                flow_to_neighbors[(neighbor_id, incoming_direction)] += transfer

        for (neighbor_id, direction), volume in flow_to_neighbors.items():
            self.states[neighbor_id].pending_inflow[direction] = (
                self.states[neighbor_id].pending_inflow.get(direction, 0.0) + volume
            )

    def _sample_arrivals_for_direction(
        self, state: IntersectionState, direction: Direction, step: int
    ) -> None:
        rate = self.demand.arrival_rate_per_lane(step, direction)
        for lane in range(self.config.lanes_per_direction):
            arrivals = float(self.rng.poisson(rate * self.config.step_seconds))
            state.queue_matrix[direction][lane] = min(
                state.max_queue_per_lane,
                state.queue_matrix[direction][lane] + arrivals,
            )
            state.total_arrivals += int(arrivals)

    def _service_intersection(self, state: IntersectionState) -> dict[Direction, float]:
        """Service vehicles at the intersection for one simulation step.

        During yellow + all-red clearance (``transition_steps_remaining > 0``)
        no vehicles depart — this models the all-red clearance interval per
        HCM 7th edition.

        Saturation flow rate: 1800 veh/hr/lane (HCM 7th ed. default for shared
        through/turn movements) = 0.5 veh/s/lane.
        """
        departures: dict[Direction, float] = {d: 0.0 for d in ["N", "S", "E", "W"]}

        # HCM 7th ed. all-red clearance: suppress all departures during transition
        if state.transition_steps_remaining > 0:
            state.cumulative_wait_sec += state.total_queue * self.config.step_seconds
            state.cumulative_stopped_vehicles += state.total_queue
            return departures

        green_directions = ["N", "S"] if state.current_phase == "NS" else ["E", "W"]
        # HCM 7th ed. saturation flow rate: 1800 veh/hr/lane = 0.5 veh/s/lane
        saturation_rate_per_lane = self.config.saturation_flow_rate / 3600.0
        non_compliance = self.demand.noncompliance_rate()

        for direction in ["N", "S", "E", "W"]:
            svc_multiplier = self.demand.service_rate_multiplier(direction)
            for lane in range(self.config.lanes_per_direction):
                queue = state.queue_matrix[direction][lane]
                if direction in green_directions:
                    capacity = float(
                        self.rng.poisson(saturation_rate_per_lane * svc_multiplier * self.config.step_seconds)
                    )
                    moved = min(queue, capacity)
                elif non_compliance > 0.0:
                    # High-density profile: some vehicles run red lights
                    red_rate = saturation_rate_per_lane * non_compliance * 0.2 * svc_multiplier
                    capacity = float(self.rng.poisson(red_rate * self.config.step_seconds))
                    moved = min(queue, capacity)
                else:
                    moved = 0.0
                state.queue_matrix[direction][lane] -= moved
                departures[direction] += moved

        total_queue = state.total_queue
        state.cumulative_wait_sec += total_queue * self.config.step_seconds
        state.cumulative_stopped_vehicles += total_queue
        state.total_departures += int(sum(departures.values()))
        return departures

    # =========================================================================
    # Observations and metrics
    # =========================================================================

    def _collect_observations(self, step: int) -> dict[int, dict[str, float]]:
        """Collect per-intersection observations, including upstream queue averages."""
        obs: dict[int, dict[str, float]] = {}
        for intersection_id, state in self.states.items():
            neighbor_queues = [
                self.states[nid].total_queue
                for nid in self.neighbors[intersection_id].values()
                if nid is not None and nid in self.states
            ]
            upstream_queue = float(np.mean(neighbor_queues)) if neighbor_queues else 0.0
            obs[intersection_id] = state.as_observation(step, upstream_queue=upstream_queue)
        return obs

    def _compute_step_metrics(self, step: int) -> StepMetrics:
        queues = np.array([state.total_queue for state in self.states.values()], dtype=np.float64)
        total_queue = float(queues.sum())
        total_departures = float(sum(state.total_departures for state in self.states.values()))
        total_wait = float(sum(state.cumulative_wait_sec for state in self.states.values()))
        avg_wait = total_wait / max(total_departures, 1.0)
        throughput = float(
            sum(state.total_departures for state in self.states.values())
        ) / max(step + 1, 1)
        total_phase_changes = int(sum(state.phase_changes for state in self.states.values()))

        # EPA MOVES2014b idle emission factor for light-duty gasoline vehicles.
        # idle_co2_rate_per_sec = 0.000457 kg/s/vehicle (= 0.0274 kg/min / 60)
        idle_co2_rate_per_sec = 0.000457
        emissions_co2_kg = total_queue * idle_co2_rate_per_sec * self.config.step_seconds

        # EPA-based detailed fuel and CO₂ from EmissionsCalculator (includes
        # stop-start penalty and moving fuel components)
        fuel_gallons, co2_kg = self._emissions_calc.compute_step(
            total_queue=total_queue,
            departures=total_departures,
            phase_changes=total_phase_changes,
            step_seconds=self.config.step_seconds,
        )

        fuel_proxy = total_queue * 0.12 + throughput * 0.04
        fairness = self._fairness_score(queues)
        efficiency = throughput / (1.0 + total_queue / max(len(self.states), 1))

        return StepMetrics(
            step=step,
            total_queue=total_queue,
            avg_wait_sec=avg_wait,
            throughput=throughput,
            # emissions_proxy now mirrors the EPA MOVES2014b value (not fabricated)
            emissions_proxy=emissions_co2_kg,
            fuel_proxy=fuel_proxy,
            fairness=fairness,
            efficiency_score=efficiency,
            delay_reduction_pct=0.0,
            fuel_gallons=fuel_gallons,
            co2_kg=co2_kg,
            emissions_co2_kg=emissions_co2_kg,
        )

    @staticmethod
    def _fairness_score(values: np.ndarray) -> float:
        if len(values) == 0:
            return 1.0
        mean = float(values.mean())
        if mean <= 1e-9:
            return 1.0
        diffsum = np.abs(values[:, None] - values[None, :]).sum()
        gini = diffsum / (2.0 * len(values) ** 2 * mean)
        return float(max(0.0, 1.0 - gini))

    def _intersection_summaries(self) -> list[dict[str, float]]:
        summaries: list[dict[str, float]] = []
        for intersection_id, state in self.states.items():
            total_departures = max(state.total_departures, 1)
            summaries.append(
                {
                    "intersection_id": float(intersection_id),
                    "total_arrivals": float(state.total_arrivals),
                    "total_departures": float(state.total_departures),
                    "mean_wait_sec": state.cumulative_wait_sec / total_departures,
                    "queue_ns": state.queue_ns,
                    "queue_ew": state.queue_ew,
                    "phase_changes": float(state.phase_changes),
                }
            )
        return summaries

    def _aggregate_metrics(self, logs: list[StepMetrics]) -> dict[str, float]:
        if not logs:
            return {}
        avg = lambda name: float(np.mean([getattr(item, name) for item in logs]))
        return {
            "average_wait_time": avg("avg_wait_sec"),
            "average_queue_length": avg("total_queue"),
            "average_throughput": avg("throughput"),
            "average_emissions_proxy": avg("emissions_proxy"),
            "average_fuel_proxy": avg("fuel_proxy"),
            "average_fairness": avg("fairness"),
            "average_efficiency_score": avg("efficiency_score"),
            "delay_reduction_pct": avg("delay_reduction_pct"),
            "max_queue_length": float(max(item.total_queue for item in logs)),
            "total_fuel_gallons": float(sum(item.fuel_gallons for item in logs)),
            "total_co2_kg": float(sum(item.co2_kg for item in logs)),
            "total_emissions_co2_kg": float(sum(item.emissions_co2_kg for item in logs)),
        }

    # =========================================================================
    # Initialisation helpers
    # =========================================================================

    def _init_intersections(self) -> dict[int, IntersectionState]:
        states: dict[int, IntersectionState] = {}
        for intersection_id in self.intersection_ids:
            queue_matrix = {
                "N": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "S": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "E": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "W": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
            }
            states[intersection_id] = IntersectionState(
                intersection_id=intersection_id,
                lanes_per_direction=self.config.lanes_per_direction,
                max_queue_per_lane=self.config.max_queue_per_lane,
                queue_matrix=queue_matrix,
                pending_inflow={},
            )
        return states

    def _build_directional_neighbors(self) -> dict[int, dict[Direction, int | None]]:
        side = int(math.ceil(math.sqrt(self.config.intersections)))
        mapping: dict[int, dict[Direction, int | None]] = {}
        for idx in range(self.config.intersections):
            row, col = divmod(idx, side)
            candidates = {
                "N": (row - 1, col),
                "S": (row + 1, col),
                "W": (row, col - 1),
                "E": (row, col + 1),
            }
            mapping[idx] = {}
            for direction, (r, c) in candidates.items():
                if 0 <= r < side and 0 <= c < side:
                    nid = r * side + c
                    mapping[idx][direction] = nid if nid < self.config.intersections else None
                else:
                    mapping[idx][direction] = None
        return mapping

    @staticmethod
    def _opposite(direction: Direction) -> Direction:
        lookup: dict[Direction, Direction] = {"N": "S", "S": "N", "E": "W", "W": "E"}
        return lookup[direction]
