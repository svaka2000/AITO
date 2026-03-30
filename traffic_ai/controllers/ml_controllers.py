"""traffic_ai/controllers/ml_controllers.py

Supervised ML controllers for traffic signal optimization.
All controllers extend BaseController and can be trained on processed data.

Controllers
-----------
- RandomForestController
- XGBoostController
- GradientBoostingController
- MLPController (PyTorch 3-layer MLP)
- LSTMForecastController (2-layer LSTM for 15-min-ahead queue prediction)
- ImitationLearningController (behavioural cloning on RL state/action pairs)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Imitation learning: generate labels from AdaptiveRuleController
# ---------------------------------------------------------------------------

def generate_imitation_labels(
    simulator: "Any",
    rule_controller: "Any",
    n_steps: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``rule_controller`` on ``simulator`` and return imitation labels.

    This implements **imitation learning** (behavioural cloning): instead of
    training ML controllers on trivial heuristic labels
    (``action = int(ns_queue < ew_queue)``), we train them to imitate the
    best non-ML baseline — the ``AdaptiveRuleController``.

    The AdaptiveRuleController uses queue-threshold logic with minimum/maximum
    green constraints and produces substantially better labels than simple
    queue-balance comparisons.

    Parameters
    ----------
    simulator:
        A :class:`~traffic_ai.simulation_engine.engine.TrafficNetworkSimulator`
        instance (or compatible).
    rule_controller:
        An :class:`~traffic_ai.controllers.adaptive_rule.AdaptiveRuleController`
        instance (or any BaseController).
    n_steps:
        Number of simulation steps to run; more steps → richer label set.

    Returns
    -------
    (X, y)
        ``X`` : ``(n_steps, 7)`` float32 feature matrix (using ``_extract_features``).
        ``y`` : ``(n_steps,)`` integer array of actions (0=NS, 1=EW).
    """
    from traffic_ai.simulation_engine.engine import TrafficNetworkSimulator
    from traffic_ai.simulation_engine.types import SignalPhase

    n_intersections = len(simulator.states)
    rule_controller.reset(n_intersections)
    simulator.states = simulator._init_intersections()

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for step in range(n_steps):
        simulator.demand.tick_emergency(step)
        simulator.demand.tick_incident(step)
        simulator._apply_emergency_events(step)

        observations = simulator._collect_observations(step)
        actions = rule_controller.compute_actions(observations, step)
        actions = simulator._override_emergency_actions(actions)

        for iid, obs in observations.items():
            feat = _extract_features(obs)
            phase: SignalPhase = actions.get(iid, "NS")
            label = 0 if phase == "NS" else 1
            X_rows.append(feat)
            y_rows.append(label)

        simulator._advance_step(actions, step)

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Feature vector helpers
# ---------------------------------------------------------------------------
_FEATURE_KEYS: list[str] = [
    "queue_ns", "queue_ew", "avg_speed", "lane_occupancy",
    "step", "is_rush_hour", "day_of_week",
]

_RUSH_HOURS_START_END: list[tuple[int, int]] = [(7, 9), (16, 19)]


def _extract_features(obs: dict[str, float]) -> np.ndarray:
    """Extract the standard 7-feature vector from an observation dict."""
    hour = float(obs.get("step", 0)) * 1.0 / 3600.0 % 24.0
    is_rush = int(
        any(s <= hour < e for s, e in _RUSH_HOURS_START_END)
    )
    return np.array(
        [
            obs.get("queue_ns", 0.0),
            obs.get("queue_ew", 0.0),
            obs.get("avg_speed", 30.0),
            obs.get("lane_occupancy", 0.5),
            float(obs.get("step", 0)) % 86400.0,
            float(is_rush),
            float(obs.get("day_of_week", 0.0)),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Random Forest Controller
# ---------------------------------------------------------------------------

class RandomForestController(BaseController):
    """Signal controller backed by a Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(name="random_forest")
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        self._fitted = False
        self._le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict[str, float]:
        self._le.fit(["EW", "NS"])
        y_enc = self._le.transform(y)
        scores = cross_val_score(self._model, X, y_enc, cv=cv_folds, scoring="accuracy")
        self._model.fit(X, y_enc)
        self._fitted = True
        return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        phase = self._predict(obs)
        return 0 if phase == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        if not self._fitted:
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"
        feat = _extract_features(obs).reshape(1, -1)
        pred = int(self._model.predict(feat)[0])
        return str(self._le.inverse_transform([pred])[0])  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self._model, "le": self._le, "fitted": self._fitted}, fh)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "RandomForestController":
        ctrl = cls(**kwargs)
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        ctrl._model = state["model"]
        ctrl._le = state["le"]
        ctrl._fitted = state["fitted"]
        return ctrl

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_ if self._fitted else np.zeros(7)


# ---------------------------------------------------------------------------
# XGBoost Controller
# ---------------------------------------------------------------------------

class XGBoostController(BaseController):
    """Signal controller backed by XGBoost."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, seed: int = 42) -> None:
        super().__init__(name="xgboost")
        try:
            from xgboost import XGBClassifier  # type: ignore
            self._model: Any = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
        except ImportError:
            logger.warning("XGBoost not available; falling back to GradientBoosting.")
            self._model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed
            )
        self._fitted = False
        self._le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict[str, float]:
        self._le.fit(["EW", "NS"])
        y_enc = self._le.transform(y)
        scores = cross_val_score(self._model, X, y_enc, cv=cv_folds, scoring="accuracy")
        self._model.fit(X, y_enc)
        self._fitted = True
        return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        return 0 if self._predict(obs) == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        if not self._fitted:
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"
        feat = _extract_features(obs).reshape(1, -1)
        pred = int(self._model.predict(feat)[0])
        return str(self._le.inverse_transform([pred])[0])  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self._model, "le": self._le, "fitted": self._fitted}, fh)

    @property
    def feature_importances(self) -> np.ndarray:
        if not self._fitted:
            return np.zeros(7)
        return getattr(self._model, "feature_importances_", np.zeros(7))


# ---------------------------------------------------------------------------
# Gradient Boosting Controller
# ---------------------------------------------------------------------------

class GradientBoostingController(BaseController):
    """Signal controller backed by sklearn GradientBoostingClassifier."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, seed: int = 42) -> None:
        super().__init__(name="gradient_boosting")
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        self._fitted = False
        self._le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict[str, float]:
        self._le.fit(["EW", "NS"])
        y_enc = self._le.transform(y)
        scores = cross_val_score(self._model, X, y_enc, cv=cv_folds, scoring="accuracy")
        self._model.fit(X, y_enc)
        self._fitted = True
        return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        return 0 if self._predict(obs) == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        if not self._fitted:
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"
        feat = _extract_features(obs).reshape(1, -1)
        pred = int(self._model.predict(feat)[0])
        return str(self._le.inverse_transform([pred])[0])  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self._model, "le": self._le, "fitted": self._fitted}, fh)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_ if self._fitted else np.zeros(7)


# ---------------------------------------------------------------------------
# MLP Controller (PyTorch 3-layer: 128→64→32)
# ---------------------------------------------------------------------------

class MLPController(BaseController):
    """PyTorch 3-layer MLP (128→64→32) signal controller."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden: tuple[int, ...] = (128, 64, 32),
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(name="mlp")
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self._fitted = False
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self._net = nn.Sequential(*layers)
        self._optim = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
    ) -> dict[str, float]:
        import torch
        le = LabelEncoder().fit(["EW", "NS"])
        y_enc = torch.tensor(le.transform(y), dtype=torch.long)
        X_t = torch.tensor(X, dtype=torch.float32)
        self._le = le
        losses: list[float] = []
        self._net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                idx = perm[start : start + batch_size]
                logits = self._net(X_t[idx])
                loss = self._loss_fn(logits, y_enc[idx])
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
                losses.append(float(loss.item()))
        self._fitted = True
        return {"final_loss": float(losses[-1]) if losses else 0.0}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        return 0 if self._predict(obs) == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        if not self._fitted:
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"
        import torch
        feat = torch.tensor(_extract_features(obs), dtype=torch.float32).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feat)
            pred = int(torch.argmax(logits, dim=-1).item())
        le = getattr(self, "_le", None)
        if le is not None:
            label = str(le.inverse_transform([pred])[0])
            return label  # type: ignore[return-value]
        return "NS" if pred == 0 else "EW"

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "fitted": self._fitted}, str(path))

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "MLPController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._net.load_state_dict(state["state_dict"])
        ctrl._fitted = state["fitted"]
        return ctrl


# ---------------------------------------------------------------------------
# LSTM Forecast Controller
# ---------------------------------------------------------------------------

class LSTMForecastController(BaseController):
    """2-layer LSTM that predicts queue length 15 min ahead, then selects
    the phase that minimises predicted queue.

    The LSTM is trained on sequences of observations at 15-minute intervals.
    """

    SEQ_LEN: int = 12  # 12 × 5-min bins = 60 min history window

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(name="lstm_forecast")
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self._head = nn.Linear(hidden_dim, 1)
        self._optim = torch.optim.Adam(
            list(self._lstm.parameters()) + list(self._head.parameters()), lr=lr
        )
        self._loss_fn = nn.MSELoss()
        self._fitted = False
        self._history: list[list[float]] = []

    def fit(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train LSTM on (N, seq_len, features) sequences predicting scalar targets."""
        import torch
        X_t = torch.tensor(sequences, dtype=torch.float32)
        y_t = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        losses: list[float] = []
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                idx = perm[start : start + batch_size]
                out, _ = self._lstm(X_t[idx])
                pred = self._head(out[:, -1, :])
                loss = self._loss_fn(pred, y_t[idx])
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
                losses.append(float(loss.item()))
        self._fitted = True
        return {"final_loss": float(losses[-1]) if losses else 0.0}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        return 0 if self._predict(obs) == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        feat = [
            obs.get("queue_ns", 0.0),
            obs.get("queue_ew", 0.0),
            obs.get("total_queue", 0.0),
            obs.get("phase_elapsed", 0.0),
        ]
        self._history.append(feat)
        if len(self._history) > self.SEQ_LEN:
            self._history = self._history[-self.SEQ_LEN :]

        if not self._fitted or len(self._history) < self.SEQ_LEN:
            # Fall back to queue balance heuristic
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"

        import torch
        seq = torch.tensor([self._history], dtype=torch.float32)
        self._lstm.eval()
        self._head.eval()
        with torch.no_grad():
            out, _ = self._lstm(seq)
            pred_queue = float(self._head(out[:, -1, :]).item())

        # Simple policy: if predicted queue large → keep NS green, else EW
        current_ns = obs.get("queue_ns", 0.0)
        current_ew = obs.get("queue_ew", 0.0)
        return "NS" if (current_ns + pred_queue) >= current_ew else "EW"

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lstm": self._lstm.state_dict(),
                "head": self._head.state_dict(),
                "fitted": self._fitted,
            },
            str(path),
        )


# ---------------------------------------------------------------------------
# Imitation Learning Controller
# ---------------------------------------------------------------------------

class ImitationLearningController(BaseController):
    """Behavioural cloning controller trained on RL state/action pairs.

    Uses a 3-layer MLP identical to MLPController, trained via standard
    cross-entropy on (state, action) pairs from the Urban Traffic Light
    Control dataset.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden: tuple[int, ...] = (128, 64, 32),
        lr: float = 1e-3,
        n_actions: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__(name="imitation_learning")
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self._net = nn.Sequential(*layers)
        self._optim = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._fitted = False

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Train via behavioural cloning.

        Parameters
        ----------
        states:
            (N, feature_dim) array of observation vectors.
        actions:
            (N,) integer array of expert actions (0=NS, 1=EW).
        """
        import torch
        X_t = torch.tensor(states, dtype=torch.float32)
        y_t = torch.tensor(actions, dtype=torch.long)
        losses: list[float] = []
        self._net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                idx = perm[start : start + batch_size]
                logits = self._net(X_t[idx])
                loss = self._loss_fn(logits, y_t[idx])
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
                losses.append(float(loss.item()))
        self._fitted = True
        return {"final_loss": float(losses[-1]) if losses else 0.0}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        return {iid: self._predict(obs) for iid, obs in observations.items()}

    def select_action(self, obs: dict[str, float]) -> int:
        return 0 if self._predict(obs) == "NS" else 1

    def _predict(self, obs: dict[str, float]) -> SignalPhase:
        if not self._fitted:
            return "NS" if obs.get("queue_ns", 0) >= obs.get("queue_ew", 0) else "EW"
        import torch
        feat = torch.tensor(_extract_features(obs), dtype=torch.float32).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feat)
            action = int(torch.argmax(logits, dim=-1).item())
        return "NS" if action == 0 else "EW"

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "fitted": self._fitted}, str(path))

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "ImitationLearningController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._net.load_state_dict(state["state_dict"])
        ctrl._fitted = state["fitted"]
        return ctrl
