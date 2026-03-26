"""traffic_ai/data_pipeline/dataset_store.py

Filesystem-backed CRUD store for saved synthetic traffic datasets.

Each dataset lives in its own subdirectory under ``base_dir``:

    data/synthetic_datasets/
    └── my_rush_hour_dataset/
        ├── data.csv          # The full DataFrame
        ├── config.json       # SyntheticDatasetConfig as dict
        └── metadata.json     # Row counts, class balance, time range, etc.

Writes are atomic (write to .tmp then os.replace) so concurrent processes
cannot observe partially-written files.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_ai.data_pipeline.synthetic_generator import (
    SyntheticDatasetConfig,
    SyntheticDatasetResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(data: dict[str, Any], path: Path) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


def _config_from_dict(d: dict[str, Any]) -> SyntheticDatasetConfig:
    """Reconstruct a SyntheticDatasetConfig from a plain dict (JSON round-trip)."""
    fields = {f.name for f in dataclasses.fields(SyntheticDatasetConfig)}
    filtered = {k: v for k, v in d.items() if k in fields}
    return SyntheticDatasetConfig(**filtered)


# ---------------------------------------------------------------------------
# DatasetStore
# ---------------------------------------------------------------------------


class DatasetStore:
    """CRUD operations for saved synthetic datasets.

    Parameters
    ----------
    base_dir:
        Root directory for all dataset subdirectories.
        Defaults to ``data/synthetic_datasets`` relative to the CWD.
    """

    def __init__(self, base_dir: Path | str = Path("data/synthetic_datasets")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, name: str, result: SyntheticDatasetResult) -> Path:
        """Persist a generated dataset to disk.

        Writes three files atomically:
        - ``data.csv``     — the full DataFrame
        - ``config.json``  — the generation config
        - ``metadata.json`` — row count, class balance, timestamps, etc.

        Parameters
        ----------
        name:
            Human-readable dataset name (will be sanitized for filesystem).
        result:
            The :class:`SyntheticDatasetResult` to persist.

        Returns
        -------
        Path
            Directory path where the dataset was saved.
        """
        safe = self._safe_name(name)
        dataset_dir = self.base_dir / safe
        dataset_dir.mkdir(parents=True, exist_ok=True)

        _write_csv_atomic(result.dataframe, dataset_dir / "data.csv")

        config_dict = dataclasses.asdict(result.config)
        _write_json_atomic(config_dict, dataset_dir / "config.json")

        metadata = dict(result.metadata)
        metadata["saved_at"] = datetime.utcnow().isoformat()
        metadata["safe_name"] = safe
        _write_json_atomic(metadata, dataset_dir / "metadata.json")

        return dataset_dir

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return a list of summary dicts for every saved dataset.

        Each summary contains: ``name``, ``safe_name``, ``rows``,
        ``description``, ``demand_profile``, ``label_strategy``,
        ``time_range_start``, ``time_range_end``, ``class_balance``,
        ``saved_at``.

        Datasets with missing or corrupt metadata are silently skipped.
        """
        summaries: list[dict[str, Any]] = []
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "metadata.json"
            cfg_path = d / "config.json"
            if not meta_path.exists() or not cfg_path.exists():
                continue
            try:
                meta = _read_json(meta_path)
                cfg = _read_json(cfg_path)
                summaries.append(
                    {
                        "name": cfg.get("name", d.name),
                        "safe_name": d.name,
                        "description": cfg.get("description", ""),
                        "rows": meta.get("rows", 0),
                        "demand_profile": meta.get("demand_profile", "—"),
                        "label_strategy": meta.get("label_strategy", "—"),
                        "time_range_start": meta.get("time_range_start", ""),
                        "time_range_end": meta.get("time_range_end", ""),
                        "class_balance": meta.get("class_balance", {}),
                        "saved_at": meta.get("saved_at", ""),
                        "n_intersections": meta.get("n_intersections", "—"),
                    }
                )
            except Exception:
                continue
        return summaries

    def load(
        self, name: str
    ) -> tuple[pd.DataFrame, SyntheticDatasetConfig, dict[str, Any]]:
        """Load a saved dataset by its human-readable or safe name.

        Parameters
        ----------
        name:
            Dataset name as returned by :meth:`list_datasets`.

        Returns
        -------
        (dataframe, config, metadata)
        """
        dataset_dir = self._resolve_dir(name)
        df = pd.read_csv(dataset_dir / "data.csv")
        config = _config_from_dict(_read_json(dataset_dir / "config.json"))
        metadata = _read_json(dataset_dir / "metadata.json")
        return df, config, metadata

    def delete(self, name: str) -> bool:
        """Delete a dataset directory.

        Returns
        -------
        bool
            True if the dataset existed and was deleted, False otherwise.
        """
        try:
            dataset_dir = self._resolve_dir(name)
            shutil.rmtree(dataset_dir)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a dataset.

        Returns
        -------
        bool
            True if successful, False if source not found or target exists.
        """
        try:
            src = self._resolve_dir(old_name)
            dst = self.base_dir / self._safe_name(new_name)
            if dst.exists():
                return False
            src.rename(dst)
            # Update the name field inside config.json
            cfg_path = dst / "config.json"
            if cfg_path.exists():
                cfg_data = _read_json(cfg_path)
                cfg_data["name"] = new_name
                _write_json_atomic(cfg_data, cfg_path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def duplicate(self, name: str, new_name: str) -> bool:
        """Duplicate a dataset under a new name.

        Returns
        -------
        bool
            True if successful, False if source not found or target exists.
        """
        try:
            src = self._resolve_dir(name)
            dst = self.base_dir / self._safe_name(new_name)
            if dst.exists():
                return False
            shutil.copytree(src, dst)
            cfg_path = dst / "config.json"
            if cfg_path.exists():
                cfg_data = _read_json(cfg_path)
                cfg_data["name"] = new_name
                _write_json_atomic(cfg_data, cfg_path)
            meta_path = dst / "metadata.json"
            if meta_path.exists():
                meta = _read_json(meta_path)
                meta["saved_at"] = datetime.utcnow().isoformat()
                meta["safe_name"] = self._safe_name(new_name)
                _write_json_atomic(meta, meta_path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def export_csv(self, name: str) -> Path:
        """Return the path to the dataset's CSV file for direct download.

        Parameters
        ----------
        name:
            Dataset name.

        Returns
        -------
        Path
            Absolute path to ``data.csv``.
        """
        dataset_dir = self._resolve_dir(name)
        return dataset_dir / "data.csv"

    def get_config(self, name: str) -> SyntheticDatasetConfig:
        """Load just the generation config for a saved dataset.

        Useful for the 'Edit & Regenerate' dashboard action.
        """
        dataset_dir = self._resolve_dir(name)
        return _config_from_dict(_read_json(dataset_dir / "config.json"))

    def exists(self, name: str) -> bool:
        """Return True if a dataset with this name is already saved."""
        try:
            self._resolve_dir(name)
            return True
        except (FileNotFoundError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dir(self, name: str) -> Path:
        """Resolve dataset name → directory, checking both safe and raw forms."""
        safe = self._safe_name(name)
        # Prefer exact safe-name match
        candidate = self.base_dir / safe
        if candidate.is_dir() and (candidate / "metadata.json").exists():
            return candidate
        # Fall back: search by config name field
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if cfg_path.exists():
                try:
                    cfg = _read_json(cfg_path)
                    if cfg.get("name", "") == name:
                        return d
                except Exception:
                    continue
        raise FileNotFoundError(f"Dataset '{name}' not found in {self.base_dir}")

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a dataset name for filesystem safety."""
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())[:100]
