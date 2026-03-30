"""tests/test_explainability.py

Tests for AITO Phase 6 explainability engine.
"""
from __future__ import annotations

import pytest

from traffic_ai.explainability.explainer import DecisionExplainer, ExplanationResult
from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.rule_based import RuleBasedController


def _make_obs(queue_ns=10.0, queue_ew=5.0, phase_elapsed=10.0) -> dict:
    return {
        "queue_ns": queue_ns,
        "queue_ew": queue_ew,
        "queue_ns_through": queue_ns,
        "queue_ew_through": queue_ew,
        "queue_ns_left": 2.0,
        "queue_ew_left": 1.0,
        "total_queue": queue_ns + queue_ew + 3.0,
        "phase_elapsed": phase_elapsed,
        "current_phase_idx": 0.0,
        "upstream_queue": 3.0,
        "time_of_day_normalized": 0.3,
        "in_transition": 0.0,
        "emergency_active": 0.0,
        "step": 0.0,
    }


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

def test_explain_returns_explanation_result() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    result = explainer.explain(obs, action=0)
    assert isinstance(result, ExplanationResult)


def test_explain_natural_language_non_empty() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    for action in range(4):
        result = explainer.explain(obs, action=action)
        assert isinstance(result.natural_language, str)
        assert len(result.natural_language) > 0


def test_explain_feature_importances_sum_to_one() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    result = explainer.explain(obs, action=0)
    total = sum(result.feature_importances.values())
    assert abs(total - 1.0) < 1e-6 or total == 0.0


def test_explain_dominant_feature_in_importances() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    result = explainer.explain(obs, action=0)
    assert result.dominant_feature in result.feature_importances


def test_explain_phase_label_set() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    for action in range(4):
        result = explainer.explain(obs, action=action)
        assert len(result.phase_label) > 0


def test_explain_action_stored() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    for action in range(4):
        result = explainer.explain(obs, action=action)
        assert result.action == action


# ---------------------------------------------------------------------------
# Semantic / content tests
# ---------------------------------------------------------------------------

def test_emergency_explanation_mentions_emergency() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    obs["emergency_active"] = 1.0
    for action in range(2):
        result = explainer.explain(obs, action=action)
        assert "emergency" in result.phase_rationale.lower()


def test_high_phase_elapsed_triggers_switch_warning() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs(phase_elapsed=60.0)
    result = explainer.explain(obs, action=0)
    # Natural language should mention the long elapsed time
    assert "60" in result.natural_language or "switch" in result.natural_language.lower()


def test_left_turn_action_explains_left_queue() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    obs["queue_ns_left"] = 20.0  # large left turn queue
    result = explainer.explain(obs, action=2)  # NS_LEFT phase
    assert "left" in result.phase_rationale.lower()


# ---------------------------------------------------------------------------
# Controller-backed explainer
# ---------------------------------------------------------------------------

def test_explainer_with_controller() -> None:
    ctrl = FixedTimingController()
    explainer = DecisionExplainer(controller=ctrl)
    obs = _make_obs()
    action = ctrl.select_action(obs)
    result = explainer.explain(obs, action=action)
    assert isinstance(result, ExplanationResult)
    assert result.action == action


def test_explainer_feature_importances_non_negative() -> None:
    explainer = DecisionExplainer()
    obs = _make_obs()
    result = explainer.explain(obs, action=1)
    for k, v in result.feature_importances.items():
        assert v >= 0.0, f"Importance for {k} is negative: {v}"
