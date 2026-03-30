"""traffic_ai/ml_models/controller_models.py

Supervised controller training using imitation learning.

Problem 6 fix: replace heuristic labels (action = int(ns_queue < ew_queue))
with labels derived from AdaptiveRuleController runs across a full simulation.
This is behavioural cloning — the ML models learn to imitate the best
non-ML baseline rather than a trivial queue-comparison heuristic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass(slots=True)
class ControllerTrainingData:
    X: np.ndarray
    y: np.ndarray


def generate_controller_training_data(
    seed: int = 42,
    n_steps: int = 500,
) -> ControllerTrainingData:
    """Generate (state, action) training pairs via imitation learning.

    Runs the :class:`~traffic_ai.controllers.adaptive_rule.AdaptiveRuleController`
    on the canonical simulation engine and records its decisions as training
    labels.  This is imitation learning (behavioural cloning): ML models learn
    to reproduce the adaptive controller's decisions rather than a trivial
    heuristic.

    Parameters
    ----------
    seed:
        Random seed for the simulation.
    n_steps:
        Number of simulation steps to run.  More steps give richer labels.

    Returns
    -------
    ControllerTrainingData
        ``X`` : (N, 7) float32 feature matrix.
        ``y`` : (N,) int64 action array (0=NS, 1=EW).
    """
    from traffic_ai.controllers.adaptive_rule import AdaptiveRuleController
    from traffic_ai.controllers.ml_controllers import generate_imitation_labels
    from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator

    cfg = SimulatorConfig(
        steps=n_steps,
        intersections=4,
        lanes_per_direction=2,
        step_seconds=1.0,
        demand_profile="rush_hour",
        seed=seed,
    )
    simulator = TrafficNetworkSimulator(cfg)
    rule_ctrl = AdaptiveRuleController(min_green=15, max_green=75, queue_threshold=6.0)

    try:
        X, y = generate_imitation_labels(simulator, rule_ctrl, n_steps=n_steps)
        logger.info(
            "Generated %d imitation labels from AdaptiveRuleController "
            "(n_steps=%d, intersections=%d).",
            len(y), n_steps, cfg.intersections,
        )
        return ControllerTrainingData(X=X, y=y)
    except Exception as exc:
        logger.warning(
            "Imitation label generation failed (%s); falling back to heuristic labels.", exc
        )
        return _heuristic_fallback(seed=seed)


def _heuristic_fallback(seed: int = 42, n_samples: int = 3_000) -> ControllerTrainingData:
    """Fallback: heuristic labels used only if imitation learning fails."""
    rng = np.random.default_rng(seed)
    queue_ns = rng.gamma(shape=2.2, scale=7.0, size=n_samples)
    queue_ew = rng.gamma(shape=2.1, scale=6.5, size=n_samples)
    total_queue = queue_ns + queue_ew
    phase_elapsed = rng.integers(0, 80, size=n_samples).astype(float)
    phase_ns = rng.integers(0, 2, size=n_samples).astype(float)
    phase_ew = 1.0 - phase_ns
    sim_step = rng.integers(0, 12_000, size=n_samples).astype(float)
    wait_sec = np.clip(total_queue * rng.normal(4.2, 0.9, size=n_samples), 0, 600)
    y = (queue_ns < queue_ew).astype(int)
    X = np.column_stack(
        [queue_ns, queue_ew, total_queue, phase_elapsed, phase_ns, phase_ew, sim_step, wait_sec]
    ).astype(np.float32)
    return ControllerTrainingData(X=X, y=y)


def train_supervised_controller_models(seed: int = 42, quick_run: bool = False) -> dict[str, Any]:
    """Train sklearn classifiers on imitation learning labels.

    Parameters
    ----------
    seed:
        Random seed for both data generation and model training.
    quick_run:
        When True, uses fewer simulation steps for faster iteration.
    """
    n_steps = 200 if quick_run else 500
    data = generate_controller_training_data(seed=seed, n_steps=n_steps)
    X, y = data.X, data.y

    models: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=90, max_depth=10, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=2, random_state=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 32),
            max_iter=120,
            random_state=seed,
            early_stopping=True,
        ),
        "timeseries": LogisticRegression(max_iter=1200),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=90,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )
    else:
        models["xgboost"] = GradientBoostingClassifier(random_state=seed)

    for name, model in models.items():
        try:
            model.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to train %s: %s", name, exc)

    return models
