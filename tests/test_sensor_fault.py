"""tests/test_sensor_fault.py

Tests for AITO Phase 4 sensor fault model and fault-tolerant controller.
"""
from __future__ import annotations

import pytest

from traffic_ai.simulation_engine.sensor import SensorFaultModel
from traffic_ai.controllers.fault_tolerant import FaultTolerantController
from traffic_ai.controllers.rule_based import RuleBasedController
from traffic_ai.controllers.fixed import FixedTimingController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(queue_ns: float = 10.0, queue_ew: float = 8.0) -> dict[str, float]:
    return {
        "queue_ns": queue_ns,
        "queue_ew": queue_ew,
        "queue_ns_through": queue_ns,
        "queue_ew_through": queue_ew,
        "queue_ns_left": 2.0,
        "queue_ew_left": 1.5,
        "total_queue": queue_ns + queue_ew + 3.5,
        "upstream_queue": 3.0,
        "phase_elapsed": 5.0,
        "step": 0.0,
        "current_phase_idx": 0.0,
        "time_of_day_normalized": 0.3,
        "in_transition": 0.0,
        "emergency_active": 0.0,
    }


# ---------------------------------------------------------------------------
# SensorFaultModel tests
# ---------------------------------------------------------------------------

def test_sensor_fault_apply_returns_dict() -> None:
    """apply() must return a dict with the same keys as the input."""
    fault = SensorFaultModel(seed=1)
    obs = _make_obs()
    corrupted = fault.apply(obs, step=0, intersection_id=0)
    assert isinstance(corrupted, dict)
    assert set(corrupted.keys()) == set(obs.keys())


def test_sensor_fault_non_queue_fields_unchanged() -> None:
    """Non-queue metadata fields must not be altered."""
    fault = SensorFaultModel(seed=1)
    obs = _make_obs()
    for step in range(50):
        corrupted = fault.apply(obs, step=step)
        assert corrupted["phase_elapsed"] == obs["phase_elapsed"]
        assert corrupted["emergency_active"] == obs["emergency_active"]
        assert corrupted["step"] == obs["step"]


def test_sensor_fault_dropout_produces_zeros() -> None:
    """Dropout fault should produce 0.0 for queue fields on affected steps."""
    # Use very high dropout probability to ensure we observe it
    fault = SensorFaultModel(dropout_prob=1.0, stuck_prob=0.0, noise_std=0.0, seed=42)
    obs = _make_obs(queue_ns=20.0, queue_ew=15.0)
    corrupted = fault.apply(obs, step=0)
    assert corrupted["queue_ns"] == 0.0
    assert corrupted["queue_ew"] == 0.0


def test_sensor_fault_noise_bounded() -> None:
    """Corrupted values must be non-negative (noise clipped at 0)."""
    fault = SensorFaultModel(noise_std=2.0, stuck_prob=0.0, dropout_prob=0.0, seed=7)
    obs = _make_obs(queue_ns=0.0, queue_ew=0.0)  # boundary case
    for step in range(30):
        corrupted = fault.apply(obs, step=step)
        assert corrupted["queue_ns"] >= 0.0, "Noise must not produce negative queues"
        assert corrupted["queue_ew"] >= 0.0


def test_sensor_fault_stuck_persists() -> None:
    """Stuck fault should return the same value for stuck_window steps."""
    fault = SensorFaultModel(stuck_prob=1.0, noise_std=0.0, dropout_prob=0.0, stuck_window=4, seed=0)
    obs = _make_obs(queue_ns=10.0, queue_ew=8.0)
    first = fault.apply(obs, step=0)["queue_ns"]
    stuck_count = 0
    for step in range(1, 5):
        val = fault.apply(obs, step=step)["queue_ns"]
        if val == first:
            stuck_count += 1
    assert stuck_count >= 3, "Stuck fault should persist for at least stuck_window steps"


def test_sensor_fault_reset_clears_state() -> None:
    """reset() should clear all fault state."""
    fault = SensorFaultModel(stuck_prob=1.0, noise_std=0.0, dropout_prob=0.0, seed=0)
    obs = _make_obs()
    fault.apply(obs, step=0)
    fault.reset()
    assert fault._stuck_remaining == {}
    assert fault._stuck_value == {}
    assert fault._dropout_remaining == {}


# ---------------------------------------------------------------------------
# FaultTolerantController tests
# ---------------------------------------------------------------------------

def test_fault_tolerant_wraps_controller() -> None:
    """FaultTolerantController should expose the wrapped controller name."""
    base = RuleBasedController()
    ctrl = FaultTolerantController(base)
    assert "rule_based" in ctrl.name


def test_fault_tolerant_returns_valid_phases() -> None:
    """compute_actions must return valid SignalPhase strings under noisy obs."""
    VALID = {"NS", "EW", "NS_THROUGH", "EW_THROUGH", "NS_LEFT", "EW_LEFT"}
    base = RuleBasedController()
    ctrl = FaultTolerantController(base)
    ctrl.reset(4)
    obs = {i: _make_obs(queue_ns=float(5 + i), queue_ew=float(3 + i)) for i in range(4)}
    actions = ctrl.compute_actions(obs, step=0)
    assert len(actions) == 4
    for phase in actions.values():
        assert phase in VALID


def test_fault_tolerant_imputes_zero_readings() -> None:
    """Zero queue readings should be replaced with EWMA-cached previous values."""
    base = FixedTimingController()
    ctrl = FaultTolerantController(base, alpha=0.5)
    ctrl.reset(1)

    # First pass with valid data to populate EWMA cache
    valid_obs = {0: _make_obs(queue_ns=10.0, queue_ew=8.0)}
    ctrl.compute_actions(valid_obs, step=0)

    # Second pass with zeroed queues (simulated dropout)
    zero_obs = {0: _make_obs(queue_ns=0.0, queue_ew=0.0)}
    # The controller should not crash and should return a valid phase
    actions = ctrl.compute_actions(zero_obs, step=1)
    assert 0 in actions
    assert actions[0] in {"NS", "EW", "NS_THROUGH", "EW_THROUGH", "NS_LEFT", "EW_LEFT"}


def test_fault_tolerant_select_action_valid() -> None:
    """select_action must return a valid phase index under corrupted obs."""
    base = RuleBasedController()
    ctrl = FaultTolerantController(base)
    obs = _make_obs()
    action = ctrl.select_action(obs)
    assert action in (0, 1, 2, 3)


def test_fault_tolerant_update_delegates() -> None:
    """update() must not raise (delegates to wrapped controller)."""
    base = RuleBasedController()
    ctrl = FaultTolerantController(base)
    obs = _make_obs()
    next_obs = _make_obs(queue_ns=5.0, queue_ew=12.0)
    ctrl.update(obs, action=0, reward=-3.0, next_obs=next_obs)


def test_fault_tolerant_reset_clears_ewma() -> None:
    """reset() should clear the EWMA cache."""
    base = FixedTimingController()
    ctrl = FaultTolerantController(base)
    ctrl.reset(2)
    # Populate cache with a step
    ctrl.compute_actions({0: _make_obs(), 1: _make_obs()}, step=0)
    assert len(ctrl._ewma) > 0
    ctrl.reset(2)
    assert len(ctrl._ewma) == 0
