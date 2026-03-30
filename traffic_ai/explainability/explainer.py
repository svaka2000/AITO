"""traffic_ai/explainability/explainer.py

Decision explainability engine for AITO Phase 6.

Provides 1-3 sentence natural language explanations for controller decisions,
feature importance scores, and phase transition rationale.

Designed for use in the dashboard, shadow mode log, and CLI output.

Architecture
------------
    DecisionExplainer:
        - explain(obs, action) → ExplanationResult
          * natural_language: str (1-3 sentences)
          * feature_importances: dict[str, float] (ranked by contribution)
          * dominant_feature: str (top driver)
          * phase_rationale: str (why this phase was selected)

Feature importance method: gradient-free sensitivity analysis —
each feature is zeroed out and the change in the recommended action's
score (or queue heuristic) is measured.  Importance = |delta_score|.

Reference
---------
    Ribeiro et al. (2016) "Why Should I Trust You?": Explaining the
    Predictions of Any Classifier, KDD 2016.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Feature metadata for natural language generation
# ---------------------------------------------------------------------------

_FEATURE_LABELS: dict[str, str] = {
    "queue_ns": "northbound/southbound through queue",
    "queue_ew": "eastbound/westbound through queue",
    "queue_ns_through": "northbound/southbound through queue",
    "queue_ew_through": "eastbound/westbound through queue",
    "queue_ns_left": "northbound/southbound left-turn queue",
    "queue_ew_left": "eastbound/westbound left-turn queue",
    "total_queue": "total intersection queue",
    "phase_elapsed": "time since last phase change",
    "upstream_queue": "upstream intersection backpressure",
    "time_of_day_normalized": "time of day",
    "in_transition": "signal transition in progress",
    "emergency_active": "emergency vehicle preemption",
    "current_phase_idx": "current signal phase",
}

_PHASE_LABELS: dict[int, str] = {
    0: "NS_THROUGH (northbound/southbound green)",
    1: "EW_THROUGH (eastbound/westbound green)",
    2: "NS_LEFT (northbound/southbound protected left turn)",
    3: "EW_LEFT (eastbound/westbound protected left turn)",
}


@dataclass
class ExplanationResult:
    """Structured explanation for a single controller decision."""
    natural_language: str
    feature_importances: Dict[str, float]
    dominant_feature: str
    phase_rationale: str
    action: int
    phase_label: str


class DecisionExplainer:
    """Generate human-readable explanations for traffic signal decisions.

    Works with any controller — ML-based, RL-based, or rule-based —
    because it uses a model-agnostic sensitivity analysis approach.

    Parameters
    ----------
    controller:
        Any BaseController instance.  If None, uses a pure queue-heuristic
        baseline to compute sensitivity.
    """

    def __init__(self, controller=None) -> None:
        self.controller = controller

    def explain(
        self,
        obs: Dict[str, float],
        action: int,
        step: int = 0,
    ) -> ExplanationResult:
        """Generate an explanation for the given action given the observation.

        Parameters
        ----------
        obs:
            Observation dict from IntersectionState.as_observation().
        action:
            Phase index (0-3) selected by the controller.
        step:
            Current simulation step (for context).

        Returns
        -------
        ExplanationResult
        """
        importances = self._compute_importances(obs, action)
        dominant = max(importances, key=importances.get) if importances else "queue_ns"

        phase_label = _PHASE_LABELS.get(action, f"phase {action}")
        rationale = self._phase_rationale(obs, action)
        nl = self._natural_language(obs, action, dominant, rationale)

        return ExplanationResult(
            natural_language=nl,
            feature_importances=importances,
            dominant_feature=dominant,
            phase_rationale=rationale,
            action=action,
            phase_label=phase_label,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_importances(
        self, obs: Dict[str, float], action: int
    ) -> Dict[str, float]:
        """Sensitivity-based feature importance: zero out each feature and
        measure impact on the baseline score for the selected action."""
        FEATURES = [
            "queue_ns", "queue_ew", "queue_ns_left", "queue_ew_left",
            "phase_elapsed", "upstream_queue", "time_of_day_normalized",
        ]
        baseline_score = self._score(obs, action)
        importances: Dict[str, float] = {}

        for feat in FEATURES:
            if feat not in obs:
                continue
            perturbed = dict(obs)
            perturbed[feat] = 0.0
            delta = abs(baseline_score - self._score(perturbed, action))
            importances[feat] = delta

        # Normalise to [0, 1]
        total = sum(importances.values()) or 1.0
        return {k: v / total for k, v in sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )}

    def _score(self, obs: Dict[str, float], action: int) -> float:
        """Estimate a scalar score for the action given obs.

        If a controller is available, use it; otherwise use a queue heuristic.
        The score represents how strongly obs supports the given action.
        """
        if self.controller is not None:
            try:
                selected = self.controller.select_action(obs)
                # Score = 1.0 if controller agrees, else closeness of queues
                return 1.0 if selected == action else 0.0
            except Exception:
                pass

        # Queue-based heuristic score
        q = [
            obs.get("queue_ns", 0.0) + obs.get("queue_ns_through", 0.0),
            obs.get("queue_ew", 0.0) + obs.get("queue_ew_through", 0.0),
            obs.get("queue_ns_left", 0.0),
            obs.get("queue_ew_left", 0.0),
        ]
        total = sum(q) or 1.0
        return q[action] / total if action < len(q) else 0.0

    def _phase_rationale(self, obs: Dict[str, float], action: int) -> str:
        """One-sentence rationale for the phase selection."""
        q_ns = obs.get("queue_ns", obs.get("queue_ns_through", 0.0))
        q_ew = obs.get("queue_ew", obs.get("queue_ew_through", 0.0))
        q_ns_l = obs.get("queue_ns_left", 0.0)
        q_ew_l = obs.get("queue_ew_left", 0.0)
        elapsed = obs.get("phase_elapsed", 0.0)
        emerg = obs.get("emergency_active", 0.0)

        if emerg > 0.5:
            return "Emergency vehicle preemption active — clearing the emergency axis."

        if action == 0:
            if q_ns > q_ew:
                return f"NS queue ({q_ns:.0f} veh) exceeds EW queue ({q_ew:.0f} veh) — NS throughput prioritised."
            return f"NS through phase held (phase_elapsed={elapsed:.0f}s, queue balance within threshold)."
        elif action == 1:
            if q_ew > q_ns:
                return f"EW queue ({q_ew:.0f} veh) exceeds NS queue ({q_ns:.0f} veh) — EW throughput prioritised."
            return f"EW through phase held (phase_elapsed={elapsed:.0f}s, queue balance within threshold)."
        elif action == 2:
            return f"NS left-turn queue ({q_ns_l:.0f} veh) requires protected phase to prevent starvation."
        elif action == 3:
            return f"EW left-turn queue ({q_ew_l:.0f} veh) requires protected phase to prevent starvation."
        return "Phase selected based on queue balance."

    def _natural_language(
        self, obs: Dict[str, float], action: int, dominant: str, rationale: str
    ) -> str:
        """1-3 sentence natural language explanation."""
        phase_label = _PHASE_LABELS.get(action, f"phase {action}")
        feature_label = _FEATURE_LABELS.get(dominant, dominant)
        q_ns = obs.get("queue_ns", obs.get("queue_ns_through", 0.0))
        q_ew = obs.get("queue_ew", obs.get("queue_ew_through", 0.0))
        elapsed = obs.get("phase_elapsed", 0.0)

        sentences = [
            f"Selected {phase_label}.",
            f"Key driver: {feature_label} (highest feature influence). {rationale}",
        ]
        if elapsed > 45:
            sentences.append(
                f"Phase has been active for {elapsed:.0f}s — a switch may be due soon."
            )
        elif q_ns > 0 and q_ew > 0:
            ratio = q_ns / max(q_ew, 1.0)
            if ratio > 1.5 or ratio < 0.67:
                sentences.append(
                    f"Queue imbalance detected (NS={q_ns:.0f}, EW={q_ew:.0f} veh) — "
                    "fairness constraint may trigger early switch."
                )
        return " ".join(sentences)
