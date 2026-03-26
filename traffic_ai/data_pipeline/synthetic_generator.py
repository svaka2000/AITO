"""traffic_ai/data_pipeline/synthetic_generator.py

Highly configurable synthetic traffic dataset generator.

Generates physically plausible time-series traffic data using the DemandModel
for base arrival rates, with support for rush-hour patterns, special scenarios
(incidents, weather, events, school zones, emergency vehicles), and multiple
label strategies for training signal controllers.

All data generation is fully vectorized with NumPy — no Python row-level loops
over the sample array (except the small adaptive_rule pair-level loop which
iterates over intersection×timestamp combinations, not individual rows).

Performance target: 100,000 samples with heuristic labels in < 5 seconds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from traffic_ai.simulation_engine.demand import DemandModel


_DIRECTIONS: list[str] = ["N", "S", "E", "W"]
_IS_NS: np.ndarray = np.array([True, True, False, False], dtype=bool)


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SyntheticDatasetConfig:
    """Every parameter the user can tweak from the dashboard."""

    name: str
    description: str = ""
    n_samples: int = 10_000
    time_span_days: int = 30
    sampling_interval_minutes: int = 5
    seed: int = 42

    # Intersection configuration
    n_intersections: int = 4
    lanes_per_direction: int = 2
    grid_rows: int = 2
    grid_cols: int = 2

    # Traffic volume parameters
    base_arrival_rate: float = 0.12
    peak_multiplier: float = 2.5
    volume_noise_std: float = 0.15

    # Temporal patterns
    morning_rush_center: float = 8.0
    morning_rush_width: float = 1.5
    evening_rush_center: float = 17.5
    evening_rush_width: float = 1.5
    weekend_reduction: float = 0.7
    overnight_min: float = 0.15

    demand_profile: str = "rush_hour"

    # Special scenarios
    include_incidents: bool = False
    incident_frequency_per_day: float = 0.5
    include_weather: bool = False
    weather_frequency_per_day: float = 0.3
    include_events: bool = False
    event_hour: float = 19.0
    include_school_zones: bool = False
    include_emergency_vehicles: bool = False
    emergency_frequency_per_day: float = 2.0

    signal_compliance_rate: float = 1.0
    ns_ew_ratio: float = 1.0

    label_strategy: str = "optimal"


@dataclass(slots=True)
class SyntheticDatasetResult:
    """Result of synthetic dataset generation."""

    config: SyntheticDatasetConfig
    dataframe: pd.DataFrame
    metadata: dict[str, Any]
    generation_time_seconds: float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticDatasetGenerator:
    """Generate realistic synthetic traffic datasets using vectorized NumPy.

    Parameters
    ----------
    config:
        Complete dataset configuration from the dashboard.
    """

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> SyntheticDatasetResult:
        """Generate the synthetic dataset.

        Parameters
        ----------
        progress_callback:
            Optional callable(fraction: float, status: str) called at key stages.

        Returns
        -------
        SyntheticDatasetResult
        """
        t0 = time.perf_counter()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        def _progress(frac: float, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(frac, msg)

        _progress(0.0, "Initializing time grid…")

        # ── 1. Time grid ────────────────────────────────────────────────────
        steps_per_day = max(1, int(24 * 60 / cfg.sampling_interval_minutes))
        max_ts = cfg.time_span_days * steps_per_day
        rows_per_ts = cfg.n_intersections * 4  # 4 directions per intersection
        n_timestamps = min(max_ts, max(1, cfg.n_samples // rows_per_ts))

        timestamps = pd.date_range(
            start="2025-01-01",
            periods=n_timestamps,
            freq=f"{cfg.sampling_interval_minutes}min",
        )

        # ── 2. Row-index expansion ──────────────────────────────────────────
        # Each row = (timestamp × intersection × direction)
        ts_idx = np.repeat(np.arange(n_timestamps), rows_per_ts)
        loc_idx = np.tile(
            np.repeat(np.arange(cfg.n_intersections), 4), n_timestamps
        )
        dir_idx = np.tile(np.arange(4), n_timestamps * cfg.n_intersections)

        # Cap at n_samples (random subsample, preserve ordering)
        n_total = len(ts_idx)
        if n_total > cfg.n_samples:
            keep = np.sort(rng.choice(n_total, size=cfg.n_samples, replace=False))
            ts_idx = ts_idx[keep]
            loc_idx = loc_idx[keep]
            dir_idx = dir_idx[keep]
        n = len(ts_idx)

        _progress(0.08, "Computing time features…")

        # ── 3. Time features (vectorized) ───────────────────────────────────
        ts_arr = timestamps[ts_idx]
        hours = (
            ts_arr.hour.values.astype(np.float64)
            + ts_arr.minute.values.astype(np.float64) / 60.0
        )
        dow = ts_arr.dayofweek.values.astype(np.int32)  # 0=Mon … 6=Sun
        is_weekend = dow >= 5
        is_rush = (
            ((hours >= 7.0) & (hours <= 9.0))
            | ((hours >= 16.0) & (hours <= 18.0))
        ).astype(np.int8)
        is_overnight = (hours >= 2.0) & (hours <= 5.0)
        is_ns_row = _IS_NS[dir_idx]

        _progress(0.18, "Computing demand-model arrival rates…")

        # ── 4. Arrival rates via DemandModel (pre-computed rate table) ───────
        arrival_rates = self._compute_arrival_rates(hours, dir_idx, steps_per_day)

        # Directional ratio: boost NS, reduce EW (or vice-versa)
        ratio = max(cfg.ns_ew_ratio, 0.01)
        arrival_rates = np.where(is_ns_row, arrival_rates * ratio, arrival_rates / ratio)

        # Weekend reduction
        arrival_rates = np.where(is_weekend, arrival_rates * cfg.weekend_reduction, arrival_rates)

        # Overnight floor
        overnight_floor = cfg.base_arrival_rate * cfg.overnight_min
        arrival_rates = np.where(is_overnight, np.maximum(arrival_rates, overnight_floor), arrival_rates)

        _progress(0.30, "Sampling vehicle counts…")

        # ── 5. Vehicle counts (Poisson, vectorized) ─────────────────────────
        interval_sec = cfg.sampling_interval_minutes * 60.0
        mean_veh = np.maximum(0.01, arrival_rates * interval_sec * cfg.lanes_per_direction)
        vehicle_count = rng.poisson(mean_veh).astype(np.float64)

        if cfg.volume_noise_std > 0.0:
            noise = rng.normal(1.0, cfg.volume_noise_std, size=n)
            vehicle_count = np.maximum(0.0, vehicle_count * noise)

        _progress(0.40, "Generating queue and kinematic data…")

        # ── 6. Queue length ─────────────────────────────────────────────────
        queue_mean = np.maximum(0.5, vehicle_count * 0.45)
        queue_length = rng.poisson(queue_mean).astype(np.float64)

        # ── 7. Occupancy (ratio of queue to capacity) ────────────────────────
        capacity = cfg.lanes_per_direction * 60.0
        occupancy = np.clip(queue_length / max(capacity, 1.0), 0.0, 1.0)

        # ── 8. Speed (inversely correlated with occupancy) ──────────────────
        base_speed = 60.0 - 40.0 * occupancy
        speed_kph = np.clip(rng.normal(base_speed, 5.0 + 5.0 * occupancy, size=n), 3.0, 100.0)

        # ── 9. Average wait ─────────────────────────────────────────────────
        avg_wait_sec = np.clip(
            queue_length * (cfg.sampling_interval_minutes * 0.8)
            + rng.normal(0.0, 5.0, size=n),
            0.0, 600.0,
        )

        # ── 10. Current signal phase (random, independent of optimal) ────────
        current_phase_int = rng.integers(0, 2, size=n)
        signal_phase = np.where(current_phase_int == 0, "NS", "EW")

        _progress(0.52, "Applying special scenarios…")

        # ── 11. Scenario overlays ────────────────────────────────────────────
        vehicle_count, speed_kph, occupancy, queue_length, avg_wait_sec = (
            self._apply_scenarios(
                hours=hours,
                ts_idx=ts_idx,
                n_timestamps=n_timestamps,
                steps_per_day=steps_per_day,
                is_ns_row=is_ns_row,
                vehicle_count=vehicle_count,
                speed_kph=speed_kph,
                occupancy=occupancy,
                queue_length=queue_length,
                avg_wait_sec=avg_wait_sec,
                rng=rng,
            )
        )

        # Re-clip after modifications
        vehicle_count = np.maximum(0.0, vehicle_count)
        speed_kph = np.clip(speed_kph, 3.0, 100.0)
        occupancy = np.clip(occupancy, 0.0, 1.0)
        queue_length = np.maximum(0.0, queue_length)
        avg_wait_sec = np.clip(avg_wait_sec, 0.0, 600.0)

        _progress(0.68, "Generating labels…")

        # ── 12. Labels ───────────────────────────────────────────────────────
        pair_idx = ts_idx * cfg.n_intersections + loc_idx
        n_pairs = n_timestamps * cfg.n_intersections
        optimal_phase = self._generate_labels(
            queue_length=queue_length,
            is_ns_row=is_ns_row,
            pair_idx=pair_idx,
            n_pairs=n_pairs,
            ts_idx=ts_idx,
            rng=rng,
        )

        _progress(0.82, "Building DataFrame and rolling features…")

        # ── 13. Build DataFrame ──────────────────────────────────────────────
        df = pd.DataFrame(
            {
                "timestamp": ts_arr,
                "location_id": (loc_idx + 1).astype(str),
                "direction": [_DIRECTIONS[d] for d in dir_idx],
                "vehicle_count": vehicle_count,
                "speed_kph": speed_kph,
                "occupancy": occupancy,
                "signal_phase": signal_phase,
                "queue_length": queue_length,
                "avg_wait_sec": avg_wait_sec,
                "optimal_phase": optimal_phase.astype(np.int32),
                "hour_of_day": hours,
                "day_of_week": dow.astype(np.int32),
                "is_rush_hour": is_rush.astype(np.int32),
                "is_weekend": is_weekend.astype(np.int32),
            }
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Rolling means over vehicle_count (temporal smoothing)
        roll_15 = max(1, 15 // cfg.sampling_interval_minutes)
        roll_60 = max(1, 60 // cfg.sampling_interval_minutes)
        df["rolling_mean_15min"] = (
            df["vehicle_count"].rolling(roll_15, min_periods=1).mean()
        )
        df["rolling_mean_60min"] = (
            df["vehicle_count"].rolling(roll_60, min_periods=1).mean()
        )

        _progress(0.95, "Finalizing metadata…")

        # ── 14. Metadata ─────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        class_bal = df["optimal_phase"].value_counts(normalize=True).to_dict()
        metadata: dict[str, Any] = {
            "rows": len(df),
            "columns": list(df.columns),
            "class_balance": {str(k): float(v) for k, v in class_bal.items()},
            "time_range_start": str(df["timestamp"].min()),
            "time_range_end": str(df["timestamp"].max()),
            "n_intersections": cfg.n_intersections,
            "demand_profile": cfg.demand_profile,
            "label_strategy": cfg.label_strategy,
            "generation_time_seconds": round(elapsed, 3),
        }

        _progress(1.0, "Done.")
        return SyntheticDatasetResult(
            config=cfg,
            dataframe=df,
            metadata=metadata,
            generation_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_arrival_rates(
        self,
        hours: np.ndarray,
        dir_idx: np.ndarray,
        steps_per_day: int,
    ) -> np.ndarray:
        """Pre-compute a rate table from DemandModel then broadcast to all rows.

        Makes at most ``steps_per_day × 4`` DemandModel calls (typically ≤ 1,152)
        regardless of n_samples, keeping the generator fast.
        """
        cfg = self.config
        step_seconds = cfg.sampling_interval_minutes * 60.0

        # Map each row's hour to a bucket index [0, steps_per_day)
        hour_buckets = (hours * (60.0 / cfg.sampling_interval_minutes)).astype(int) % steps_per_day
        unique_buckets = np.unique(hour_buckets)

        scale = cfg.base_arrival_rate / 0.12  # normalize against DemandModel base
        demand = DemandModel(
            profile=cfg.demand_profile,
            scale=scale,
            step_seconds=step_seconds,
            seed=cfg.seed,
        )

        rate_table = np.full((steps_per_day, 4), cfg.base_arrival_rate, dtype=np.float64)
        for bucket in unique_buckets:
            step = int(bucket)
            for d_i, d_name in enumerate(_DIRECTIONS):
                rate_table[bucket, d_i] = demand.arrival_rate_per_lane(step, d_name)

        return rate_table[hour_buckets, dir_idx]

    def _apply_scenarios(
        self,
        hours: np.ndarray,
        ts_idx: np.ndarray,
        n_timestamps: int,
        steps_per_day: int,
        is_ns_row: np.ndarray,
        vehicle_count: np.ndarray,
        speed_kph: np.ndarray,
        occupancy: np.ndarray,
        queue_length: np.ndarray,
        avg_wait_sec: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply all enabled special-scenario overlays (vectorized)."""
        cfg = self.config

        # ── Incidents ────────────────────────────────────────────────────────
        if cfg.include_incidents:
            p = cfg.incident_frequency_per_day / steps_per_day
            evt_ts = rng.random(n_timestamps) < p
            mask = evt_ts[ts_idx]
            vehicle_count = np.where(mask, vehicle_count * 3.0, vehicle_count)
            speed_kph = np.where(mask, speed_kph * 0.40, speed_kph)
            occupancy = np.where(mask, occupancy * 1.80, occupancy)
            queue_length = np.where(mask, queue_length * 3.0, queue_length)
            avg_wait_sec = np.where(mask, avg_wait_sec * 3.0, avg_wait_sec)

        # ── Weather (rain, persists in blocks) ───────────────────────────────
        if cfg.include_weather:
            p = cfg.weather_frequency_per_day / steps_per_day
            block = max(1, steps_per_day // 12)  # ~2-hour blocks
            rain_ts = self._block_event_mask(n_timestamps, p, block, rng)
            mask = rain_ts[ts_idx]
            vehicle_count = np.where(mask, vehicle_count * 1.30, vehicle_count)
            speed_kph = np.where(mask, speed_kph * 0.85, speed_kph)
            occupancy = np.where(mask, occupancy * 1.10, occupancy)

        # ── Stadium / concert event surge ────────────────────────────────────
        if cfg.include_events:
            pre = (hours >= cfg.event_hour - 1.0) & (hours <= cfg.event_hour + 0.5)
            post = (hours > cfg.event_hour + 0.5) & (hours <= cfg.event_hour + 1.5)
            vehicle_count = np.where(pre, vehicle_count * 4.0, vehicle_count)
            vehicle_count = np.where(post, vehicle_count * 3.5, vehicle_count)
            queue_length = np.where(pre | post, queue_length * 3.0, queue_length)
            avg_wait_sec = np.where(pre | post, avg_wait_sec * 3.5, avg_wait_sec)

        # ── School zone (N/S directions, morning + afternoon) ────────────────
        if cfg.include_school_zones:
            morn = (hours >= 7.50) & (hours <= 8.25)
            aft = (hours >= 14.75) & (hours <= 15.50)
            school = (morn | aft) & is_ns_row
            vehicle_count = np.where(school, vehicle_count * 2.5, vehicle_count)
            queue_length = np.where(school, queue_length * 2.5, queue_length)
            avg_wait_sec = np.where(school, avg_wait_sec * 2.0, avg_wait_sec)

        # ── Emergency vehicles ────────────────────────────────────────────────
        if cfg.include_emergency_vehicles:
            p = cfg.emergency_frequency_per_day / steps_per_day
            evt_ts = rng.random(n_timestamps) < p
            mask = evt_ts[ts_idx]
            # Brief clearance on one direction, brief spike on other
            vehicle_count = np.where(mask & is_ns_row, vehicle_count * 0.3, vehicle_count)
            speed_kph = np.where(mask, speed_kph * 1.15, speed_kph)

        # ── Non-compliance (developing-world) ─────────────────────────────────
        if cfg.signal_compliance_rate < 1.0:
            noncompliance = 1.0 - cfg.signal_compliance_rate
            # Some EW vehicles depart even on red → reduce EW queue variability
            extra_ew = rng.binomial(
                queue_length.astype(np.int64),
                noncompliance * 0.2 * (~is_ns_row).astype(float),
            )
            queue_length = np.maximum(0.0, queue_length - extra_ew)

        return vehicle_count, speed_kph, occupancy, queue_length, avg_wait_sec

    @staticmethod
    def _block_event_mask(
        n: int, prob_start: float, block_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a boolean mask where events persist in contiguous blocks."""
        mask = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if rng.random() < prob_start:
                length = int(rng.integers(1, block_size * 2 + 1))
                end = min(n, i + length)
                mask[i:end] = True
                i = end
            else:
                i += 1
        return mask

    def _generate_labels(
        self,
        queue_length: np.ndarray,
        is_ns_row: np.ndarray,
        pair_idx: np.ndarray,
        n_pairs: int,
        ts_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate optimal_phase labels using the configured strategy."""
        cfg = self.config

        # ── Fixed alternating ─────────────────────────────────────────────────
        if cfg.label_strategy == "fixed":
            return (ts_idx // 30 % 2).astype(np.int32)

        # ── Aggregate queue per (timestamp × intersection) pair ───────────────
        q_ns = np.zeros(n_pairs, dtype=np.float64)
        q_ew = np.zeros(n_pairs, dtype=np.float64)
        np.add.at(q_ns, pair_idx[is_ns_row], queue_length[is_ns_row])
        np.add.at(q_ew, pair_idx[~is_ns_row], queue_length[~is_ns_row])

        # ── Queue balance heuristic ───────────────────────────────────────────
        if cfg.label_strategy == "queue_balance":
            # 0 = NS green (NS more congested), 1 = EW green
            per_pair = (q_ns <= q_ew).astype(np.int32)
            return per_pair[pair_idx]

        # ── Simulation-based optimal ──────────────────────────────────────────
        if cfg.label_strategy == "optimal":
            sat = 0.45 * cfg.lanes_per_direction
            interval_sec = cfg.sampling_interval_minutes * 60.0
            service_cap = rng.poisson(sat * interval_sec, size=n_pairs).astype(np.float64)

            svc_ns = np.minimum(q_ns, service_cap)
            svc_ew = np.minimum(q_ew, service_cap)
            rem_if_ns = (q_ns - svc_ns) + q_ew
            rem_if_ew = q_ns + (q_ew - svc_ew)

            # 0 = NS green clears more, 1 = EW green clears more
            per_pair = (rem_if_ns > rem_if_ew).astype(np.int32)
            return per_pair[pair_idx]

        # ── Adaptive rule controller ──────────────────────────────────────────
        if cfg.label_strategy == "adaptive_rule":
            from traffic_ai.controllers.rule_based import RuleBasedController

            ctrl = RuleBasedController(min_green=1, max_green=10_000, threshold=0.5)
            per_pair = np.array(
                [
                    ctrl.select_action({"queue_ns": float(q_ns[i]), "queue_ew": float(q_ew[i])})
                    for i in range(n_pairs)
                ],
                dtype=np.int32,
            )
            return per_pair[pair_idx]

        # Fallback — queue balance
        per_pair = (q_ns <= q_ew).astype(np.int32)
        return per_pair[pair_idx]
