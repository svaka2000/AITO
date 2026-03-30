from __future__ import annotations

from pathlib import Path

from traffic_ai.config.settings import ensure_runtime_dirs, load_settings
from traffic_ai.experiments import ExperimentRunner


def _temp_settings(tmp_path: Path):
    settings = load_settings("traffic_ai/config/default_config.yaml")
    settings.payload["project"]["output_dir"] = str(tmp_path / "artifacts")
    settings.payload["data"]["raw_dir"] = str(tmp_path / "raw")
    settings.payload["data"]["processed_dir"] = str(tmp_path / "processed")
    settings.payload["simulation"]["steps"] = 120
    ensure_runtime_dirs(settings)
    return settings


def test_runner_ingest_only(tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    runner = ExperimentRunner(settings=settings, quick_run=True)
    artifacts = runner.run(
        ingest_only=True,
        include_kaggle=False,
        include_public=False,
    )
    assert artifacts.summary_csv.exists()
    assert artifacts.step_metrics_csv.exists()


def test_runner_pretrain_only(tmp_path: Path) -> None:
    """--pretrain-only trains RL models and exits without running CV benchmark."""
    settings = _temp_settings(tmp_path)
    runner = ExperimentRunner(settings=settings, quick_run=True)
    artifacts = runner.run(
        pretrain_only=True,
        include_kaggle=False,
        include_public=False,
    )
    # Should exit early — only the stub artifact is written
    assert artifacts.trained_model_dir.exists()


def test_rl_controller_zeros_upstream_queue() -> None:
    """RLPolicyController always passes upstream_queue=0 to match 1-intersection training."""
    import numpy as np
    from traffic_ai.controllers.rl_controller import RLPolicyController

    captured: list[np.ndarray] = []

    def mock_policy(features: np.ndarray) -> int:
        captured.append(features.copy())
        return 0

    ctrl = RLPolicyController(policy=mock_policy, name="test_rl", min_green=1)
    ctrl.reset(1)
    obs = {0: {
        "phase_elapsed": 5.0,
        "current_phase_idx": 0.0,
        "queue_ns_through": 20.0,
        "queue_ew_through": 10.0,
        "queue_ns_left": 3.0,
        "queue_ew_left": 2.0,
        "time_of_day_normalized": 0.5,
        "upstream_queue": 80.0,  # non-zero — should be zeroed by controller
    }}
    ctrl.compute_actions(obs, step=10)
    assert len(captured) == 1
    assert captured[0][7] == 0.0, "upstream_queue feature must be zeroed to match training distribution"

