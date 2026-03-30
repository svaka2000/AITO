"""tests/test_shadow_mode.py

Tests for AITO Phase 5 shadow mode runner.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from traffic_ai.shadow.shadow_runner import ShadowModeRunner, ShadowReport
from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.rule_based import RuleBasedController
from traffic_ai.simulation_engine.engine import SimulatorConfig


def _small_config(steps: int = 20) -> SimulatorConfig:
    return SimulatorConfig(steps=steps, intersections=2, seed=42)


# ---------------------------------------------------------------------------
# ShadowModeRunner tests
# ---------------------------------------------------------------------------

def test_shadow_run_returns_report() -> None:
    """run() must return a ShadowReport with sensible fields."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=15),
    )
    report = runner.run()
    assert isinstance(report, ShadowReport)
    assert report.production_controller == "fixed_timing"
    assert report.candidate_controller == "rule_based"
    assert report.steps > 0


def test_shadow_report_agreement_rate_in_range() -> None:
    """agreement_rate must be in [0, 1]."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=20),
    )
    report = runner.run()
    assert 0.0 <= report.agreement_rate <= 1.0


def test_shadow_report_queue_non_negative() -> None:
    """Queue averages must be non-negative."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=20),
    )
    report = runner.run()
    assert report.prod_avg_queue >= 0.0
    assert report.cand_avg_queue_est >= 0.0


def test_shadow_same_controller_full_agreement() -> None:
    """Using the same controller for production and candidate → 100% agreement."""
    runner = ShadowModeRunner(
        production=FixedTimingController(cycle_seconds=30),
        candidate=FixedTimingController(cycle_seconds=30),
        config=_small_config(steps=15),
    )
    report = runner.run()
    assert report.agreement_rate == pytest.approx(1.0)


def test_shadow_step_records_populated() -> None:
    """step_records must be non-empty after running."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=10),
    )
    report = runner.run()
    assert len(report.step_records) > 0
    rec = report.step_records[0]
    assert "production_phase" in rec
    assert "candidate_phase" in rec
    assert "agreed" in rec


def test_shadow_save_report(tmp_path: Path) -> None:
    """save_report() must write a valid JSON file."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=10),
    )
    report = runner.run()
    out_path = tmp_path / "shadow_report.json"
    runner.save_report(report, out_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "production_controller" in data
    assert "agreement_rate" in data
    assert "step_records" in data


def test_shadow_generated_at_set() -> None:
    """generated_at must be a non-empty ISO timestamp string."""
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=RuleBasedController(),
        config=_small_config(steps=5),
    )
    report = runner.run()
    assert report.generated_at != ""
    assert "T" in report.generated_at   # ISO format check
