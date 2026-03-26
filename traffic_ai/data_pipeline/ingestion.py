from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from traffic_ai.config.settings import Settings


@dataclass(slots=True)
class SourceResult:
    name: str
    path: Path
    rows: int
    columns: list[str]


class DataIngestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest_all(
        self,
        include_kaggle: bool = True,
        include_public: bool = True,
        include_local_csv: bool = True,
        synthetic_dataset_name: str | None = None,
    ) -> list[SourceResult]:
        """Ingest all enabled data sources.

        Parameters
        ----------
        synthetic_dataset_name:
            If provided, load this named dataset from the DatasetStore
            (``data/synthetic_datasets/``) and include it in the pipeline
            instead of (or in addition to) other sources.
        """
        artifacts: list[SourceResult] = []
        if include_local_csv:
            artifacts.extend(self._ingest_local_csv_files())
        if include_public:
            artifacts.extend(self._ingest_public_sources())
        if include_kaggle:
            artifacts.extend(self._ingest_kaggle_sources())

        # Optional: inject a saved synthetic dataset from the Data Studio
        if synthetic_dataset_name is not None:
            studio_artifact = self._ingest_studio_dataset(synthetic_dataset_name)
            if studio_artifact is not None:
                artifacts.append(studio_artifact)

        if not artifacts:
            artifacts.append(self._generate_synthetic_dataset())
            return artifacts

        total_rows = sum(item.rows for item in artifacts)
        has_synthetic = any(item.name == "synthetic_urban_traffic" for item in artifacts)
        if total_rows < 2500 and not has_synthetic:
            artifacts.append(self._generate_synthetic_dataset())
        return artifacts

    def _ingest_studio_dataset(self, name: str) -> SourceResult | None:
        """Load a named dataset from the DatasetStore and write it to raw_dir."""
        try:
            from traffic_ai.data_pipeline.dataset_store import DatasetStore

            store_dir = Path(self.settings.get("project.synthetic_datasets_dir", "data/synthetic_datasets"))
            store = DatasetStore(base_dir=store_dir)
            df, _, _ = store.load(name)
            safe = re.sub(r"[^a-zA-Z0-9_\\-]", "_", name)
            target = self.raw_dir / f"studio_{safe}.csv"
            df.to_csv(target, index=False)
            return SourceResult(name=f"studio_{safe}", path=target, rows=len(df), columns=list(df.columns))
        except Exception:
            return None

    def _ingest_local_csv_files(self) -> list[SourceResult]:
        items: list[SourceResult] = []
        for csv_path in Path(".").glob("*.csv"):
            if csv_path.parent == self.raw_dir:
                continue
            target = self.raw_dir / f"local_{csv_path.name}"
            shutil.copy2(csv_path, target)
            df = self._safe_read_csv(target)
            if df is None:
                continue
            items.append(
                SourceResult(
                    name=target.stem,
                    path=target,
                    rows=len(df),
                    columns=list(df.columns),
                )
            )
        return items

    def _ingest_public_sources(self) -> list[SourceResult]:
        sources = self.settings.get("data.public_sources", [])
        artifacts: list[SourceResult] = []
        for source in sources:
            if not source.get("enabled", True):
                continue
            name = source["name"]
            url = source["url"]
            safe_name = re.sub(r"[^a-zA-Z0-9_\\-]", "_", name)
            target = self.raw_dir / f"public_{safe_name}.csv"
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                target.write_bytes(response.content)
            except Exception:
                continue
            frame = self._safe_read_csv(target)
            if frame is None:
                continue
            artifacts.append(
                SourceResult(
                    name=name,
                    path=target,
                    rows=len(frame),
                    columns=list(frame.columns),
                )
            )
        return artifacts

    def _ingest_kaggle_sources(self) -> list[SourceResult]:
        datasets = self.settings.get("data.kaggle_datasets", [])
        if not datasets:
            return []
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

            api = KaggleApi()
            api.authenticate()
        except Exception:
            return []

        outputs: list[SourceResult] = []
        for entry in datasets:
            dataset_id = entry["id"]
            if not entry.get("enabled", True):
                continue
            safe_id = re.sub(r"[^a-zA-Z0-9_\\-]", "_", dataset_id)
            dataset_dir = self.raw_dir / f"kaggle_{safe_id}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            try:
                api.dataset_download_files(
                    dataset=dataset_id,
                    path=str(dataset_dir),
                    unzip=True,
                    quiet=True,
                )
            except Exception:
                continue

            for csv_path in dataset_dir.rglob("*.csv"):
                frame = self._safe_read_csv(csv_path)
                if frame is None:
                    continue
                canonical = self.raw_dir / f"{safe_id}_{csv_path.name}"
                frame.to_csv(canonical, index=False)
                outputs.append(
                    SourceResult(
                        name=f"kaggle_{dataset_id}",
                        path=canonical,
                        rows=len(frame),
                        columns=list(frame.columns),
                    )
                )
        return outputs

    def _generate_synthetic_dataset(self) -> SourceResult:
        seed = int(self.settings.get("reproducibility.seed", 42))
        rng = np.random.default_rng(seed)
        periods = 6_000
        timestamp = pd.date_range("2025-01-01", periods=periods, freq="5min")
        directions = rng.choice(["N", "S", "E", "W"], size=periods)
        base = pd.DataFrame(
            {
                "timestamp": timestamp,
                "location_id": rng.integers(1, 8, size=periods),
                "direction": directions,
                "vehicle_count": rng.poisson(lam=12, size=periods),
                "speed_kph": np.clip(rng.normal(34, 9, size=periods), 3, 80),
                "occupancy": np.clip(rng.normal(0.32, 0.2, size=periods), 0, 1),
                "signal_phase": rng.choice(["NS", "EW"], size=periods),
                "queue_length": rng.poisson(lam=6, size=periods),
                "avg_wait_sec": np.clip(rng.normal(35, 20, size=periods), 0, 200),
            }
        )
        rush_mask = base["timestamp"].dt.hour.isin([7, 8, 9, 16, 17, 18])
        base.loc[rush_mask, "vehicle_count"] = (
            base.loc[rush_mask, "vehicle_count"] * 1.6
        ).round()
        base.loc[rush_mask, "queue_length"] = (
            base.loc[rush_mask, "queue_length"] * 1.5
        ).round()

        target = self.raw_dir / "synthetic_urban_traffic.csv"
        base.to_csv(target, index=False)
        return SourceResult(
            name="synthetic_urban_traffic",
            path=target,
            rows=len(base),
            columns=list(base.columns),
        )

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
