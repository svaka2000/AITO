"""traffic_ai/data_pipeline/pems_connector.py

Caltrans PeMS (Performance Measurement System) data connector.

PeMS provides free per-lane volume, speed, and occupancy data at 5-minute
intervals for California freeway detectors.  See https://pems.dot.ca.gov

Usage
-----
    from traffic_ai.data_pipeline.pems_connector import PeMSConnector

    connector = PeMSConnector(station_id=400456)
    df = connector.fetch(date_from="2024-01-15", date_to="2024-01-22")
    calibration = connector.calibration_by_hour(df)

If the PeMS API key is missing or the API is unavailable, the connector
falls back to synthetic data and logs a clear warning.

Default calibration target
--------------------------
PeMS Station 400456 — I-5 near downtown San Diego, CA (Caltrans District 11).
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_STATION: int = 400456           # I-5 near downtown San Diego
_PEMS_API_BASE: str = "https://pems.dot.ca.gov"
_ENV_VAR_API_KEY: str = "PEMS_API_KEY"  # name of the env-var that holds the key

# Unified schema column names required by DatasetPreprocessor
_UNIFIED_SCHEMA: list[str] = [
    "timestamp",
    "station_id",
    "lane",
    "volume",         # vehicles per 5-min interval
    "occupancy",      # 0-1 fraction of time loop is occupied
    "speed_mph",
    "arrival_rate",   # derived: vehicles per second
    "hour_of_day",
    "day_of_week",
    "is_rush_hour",
    "is_weekend",
    "rolling_mean_volume_1h",
    "rolling_mean_speed_1h",
    "queue_proxy",    # (1 - occupancy) proxy for downstream queue
    "optimal_phase",  # 0=NS, 1=EW derived label for ML training
]


# ---------------------------------------------------------------------------
# PeMSConnector
# ---------------------------------------------------------------------------

class PeMSConnector:
    """Fetch and normalise PeMS loop detector data.

    Parameters
    ----------
    station_id:
        PeMS detector station ID (default 400456 = I-5 near downtown San Diego).
    api_key:
        PeMS API key.  If ``None``, reads from the ``PEMS_API_KEY`` environment
        variable.  When absent, the connector falls back to synthetic data.
    cache_dir:
        Directory for caching raw CSV responses.  Defaults to
        ``data/raw/pems/``.
    """

    def __init__(
        self,
        station_id: int = _DEFAULT_STATION,
        api_key: str | None = None,
        cache_dir: str | Path = "data/raw/pems",
    ) -> None:
        self.station_id = station_id
        self._api_key: str | None = api_key or os.environ.get(_ENV_VAR_API_KEY)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self._api_key:
            warnings.warn(
                f"PeMS API key not found.  Set the {_ENV_VAR_API_KEY!r} environment "
                "variable or pass api_key=... to PeMSConnector.  "
                "Falling back to synthetic traffic data.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(
        self,
        date_from: str | date,
        date_to: str | date,
        lanes: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch PeMS detector data for a date range.

        If the API key is absent or the request fails, returns a synthetic
        DataFrame calibrated to typical I-5 San Diego volume patterns.

        Parameters
        ----------
        date_from, date_to:
            Inclusive date range (``"YYYY-MM-DD"`` or :class:`datetime.date`).
        lanes:
            Specific lane numbers to return.  ``None`` = all lanes.

        Returns
        -------
        DataFrame with columns matching ``_UNIFIED_SCHEMA``.
        """
        if not self._api_key:
            logger.warning(
                "PeMS API key missing — returning synthetic fallback data for "
                "station %d (%s → %s).", self.station_id, date_from, date_to
            )
            return self._synthetic_fallback(date_from, date_to)

        try:
            raw = self._fetch_from_api(date_from, date_to)
            df = self._normalise(raw, lanes=lanes)
            logger.info(
                "Fetched %d rows from PeMS station %d.", len(df), self.station_id
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PeMS API request failed (%s) — falling back to synthetic data "
                "for station %d.", exc, self.station_id
            )
            return self._synthetic_fallback(date_from, date_to)

    def calibration_by_hour(self, df: pd.DataFrame) -> dict[int, float]:
        """Derive per-hour mean arrival rate from PeMS data.

        Returns a dict mapping ``hour_of_day`` (0-23) →
        mean vehicle count per 5-minute interval that can be passed to
        ``MultiIntersectionNetwork(calibration_data=...)`` or used to
        calibrate the ``DemandModel``.

        Parameters
        ----------
        df:
            Output of :meth:`fetch`.

        Returns
        -------
        dict mapping hour → mean_count_per_5min_interval.
        """
        if df.empty or "hour_of_day" not in df.columns:
            return {}
        grouped = df.groupby("hour_of_day")["volume"].mean()
        return {int(h): float(v) for h, v in grouped.items()}

    # ------------------------------------------------------------------
    # API fetch (requires valid key)
    # ------------------------------------------------------------------

    def _fetch_from_api(
        self,
        date_from: str | date,
        date_to: str | date,
    ) -> pd.DataFrame:
        """Download raw PeMS clearinghouse data for the station.

        PeMS clearinghouse endpoint (type=station_5min):
            GET /clearinghouse?
                    type=station_5min
                    &district_id=11
                    &station_id={id}
                    &start_time={date_from}
                    &end_time={date_to}
                    &format=text/csv
                    &user={api_key}

        Each row contains: Timestamp, Station, District, Freeway, Direction,
        Lane Type, Station Length, Samples, % Observed, and per-lane triples
        (Volume, Occupancy, Speed).
        """
        import requests

        d_from = str(date_from)
        d_to = str(date_to)
        cache_key = self.cache_dir / f"station_{self.station_id}_{d_from}_{d_to}.csv"

        if cache_key.exists():
            logger.info("Loading cached PeMS data from %s.", cache_key)
            return pd.read_csv(cache_key)

        url = _PEMS_API_BASE + "/clearinghouse"
        params: dict[str, Any] = {
            "type": "station_5min",
            "district_id": 11,  # Caltrans District 11 (San Diego)
            "station_id": self.station_id,
            "start_time": d_from,
            "end_time": d_to,
            "format": "text/csv",
            "user": self._api_key,
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        from io import StringIO
        raw = pd.read_csv(StringIO(response.text))
        raw.to_csv(cache_key, index=False)
        return raw

    # ------------------------------------------------------------------
    # Normalisation: map raw PeMS columns → 15-column unified schema
    # ------------------------------------------------------------------

    def _normalise(
        self, raw: pd.DataFrame, lanes: list[int] | None = None
    ) -> pd.DataFrame:
        """Normalise raw PeMS output to the 15-column unified schema.

        PeMS 5-min files have the following structure (columns vary by
        district / version, but core fields are standard):
            Timestamp, Station, District, Freeway, Direction, Lane Type,
            Station Length, Samples, % Observed,
            [Lane N Volume, Lane N Occupancy, Lane N Speed] × N_lanes
        """
        rows: list[dict[str, Any]] = []

        # Detect timestamp column (case-insensitive)
        ts_col = next(
            (c for c in raw.columns if "timestamp" in c.lower() or "time" in c.lower()),
            raw.columns[0],
        )

        # Detect per-lane volume/occupancy/speed columns
        vol_cols = [c for c in raw.columns if "lane" in c.lower() and "vol" in c.lower()]
        occ_cols = [c for c in raw.columns if "lane" in c.lower() and "occ" in c.lower()]
        spd_cols = [c for c in raw.columns if "lane" in c.lower() and ("spd" in c.lower() or "speed" in c.lower())]

        if not vol_cols:
            # Fallback: assume single-lane aggregated columns
            vol_cols = [c for c in raw.columns if "vol" in c.lower()]
            occ_cols = [c for c in raw.columns if "occ" in c.lower()]
            spd_cols = [c for c in raw.columns if "spd" in c.lower() or "speed" in c.lower()]

        n_lanes = max(len(vol_cols), 1)

        for _, row_data in raw.iterrows():
            ts = pd.to_datetime(row_data.get(ts_col, "2024-01-01"), errors="coerce")
            if pd.isna(ts):
                continue
            hour = ts.hour
            dow = ts.weekday()
            is_rush = 1 if (7 <= hour < 9) or (16 <= hour < 19) else 0
            is_weekend = 1 if dow >= 5 else 0

            for lane_idx in range(n_lanes):
                if lanes is not None and lane_idx not in lanes:
                    continue

                vol_val = float(row_data.get(vol_cols[lane_idx] if lane_idx < len(vol_cols) else vol_cols[0], 0.0) or 0.0)
                occ_val = float(row_data.get(occ_cols[lane_idx] if lane_idx < len(occ_cols) else occ_cols[0], 0.0) or 0.0)
                spd_val = float(row_data.get(spd_cols[lane_idx] if lane_idx < len(spd_cols) else spd_cols[0], 30.0) or 30.0)

                # Arrival rate: volume per 5-min interval → vehicles per second
                arrival_rate = vol_val / 300.0

                # Simple label: 0=NS, 1=EW based on hour-of-day heuristic
                # (NS gets priority during morning rush from residential areas)
                optimal_phase = 0 if (7 <= hour < 9) else 1

                rows.append({
                    "timestamp": ts,
                    "station_id": self.station_id,
                    "lane": lane_idx + 1,
                    "volume": vol_val,
                    "occupancy": max(0.0, min(1.0, occ_val)),
                    "speed_mph": max(0.0, spd_val),
                    "arrival_rate": arrival_rate,
                    "hour_of_day": hour,
                    "day_of_week": dow,
                    "is_rush_hour": is_rush,
                    "is_weekend": is_weekend,
                    "rolling_mean_volume_1h": vol_val,   # placeholder; computed below
                    "rolling_mean_speed_1h": spd_val,
                    "queue_proxy": max(0.0, 1.0 - occ_val),
                    "optimal_phase": optimal_phase,
                })

        if not rows:
            return pd.DataFrame(columns=_UNIFIED_SCHEMA)

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Compute rolling means (12 × 5-min = 1 hour)
        df["rolling_mean_volume_1h"] = (
            df.groupby("lane")["volume"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )
        df["rolling_mean_speed_1h"] = (
            df.groupby("lane")["speed_mph"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )

        return df[_UNIFIED_SCHEMA]

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    def _synthetic_fallback(
        self,
        date_from: str | date,
        date_to: str | date,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic data that mimics typical I-5 San Diego patterns.

        Uses a Gaussian-mixture demand model calibrated to approximate
        real PeMS Station 400456 volume patterns:
            - Morning peak: ~1,800 veh/hr at 8 AM
            - Evening peak: ~2,100 veh/hr at 5:30 PM
            - Off-peak:     ~600 veh/hr
        """
        rng = np.random.default_rng(seed)

        d_from = pd.to_datetime(str(date_from))
        d_to = pd.to_datetime(str(date_to))
        timestamps = pd.date_range(d_from, d_to, freq="5min")

        rows: list[dict[str, Any]] = []
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            dow = ts.dayofweek
            is_weekend = 1 if dow >= 5 else 0

            # Calibrated to PeMS 400456 (I-5 San Diego) hourly volumes
            morning = np.exp(-((hour - 8.0) ** 2) / 2.0)
            evening = np.exp(-((hour - 17.5) ** 2) / 2.0)
            base_rate = 600.0  # veh/hr off-peak
            weekend_factor = 0.70 if is_weekend else 1.0
            hourly_vol = base_rate * weekend_factor * (
                1.0 + 2.0 * morning + 2.5 * evening
            )
            vol_5min = max(0.0, float(rng.poisson(hourly_vol / 12.0)))
            occ = min(1.0, max(0.0, vol_5min / 150.0 + rng.normal(0, 0.02)))
            spd = max(5.0, 65.0 - occ * 60.0 + rng.normal(0, 3.0))
            is_rush = 1 if (7 <= ts.hour < 9) or (16 <= ts.hour < 19) else 0
            optimal_phase = 0 if (7 <= ts.hour < 9) else 1

            rows.append({
                "timestamp": ts,
                "station_id": self.station_id,
                "lane": 1,
                "volume": vol_5min,
                "occupancy": occ,
                "speed_mph": spd,
                "arrival_rate": vol_5min / 300.0,
                "hour_of_day": ts.hour,
                "day_of_week": dow,
                "is_rush_hour": is_rush,
                "is_weekend": is_weekend,
                "rolling_mean_volume_1h": vol_5min,
                "rolling_mean_speed_1h": spd,
                "queue_proxy": max(0.0, 1.0 - occ),
                "optimal_phase": optimal_phase,
            })

        if not rows:
            return pd.DataFrame(columns=_UNIFIED_SCHEMA)

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        window = 12  # 1 hour
        df["rolling_mean_volume_1h"] = df["volume"].rolling(window, min_periods=1).mean()
        df["rolling_mean_speed_1h"] = df["speed_mph"].rolling(window, min_periods=1).mean()
        return df[_UNIFIED_SCHEMA]
