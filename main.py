from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from traffic_ai.config.settings import load_settings
from traffic_ai.experiments import ExperimentRunner
from traffic_ai.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Traffic Signal Optimization Research Platform"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="traffic_ai/config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Use reduced training/simulation budget for faster iteration",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run data ingestion and preprocessing only",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Disable Kaggle dataset ingestion",
    )
    parser.add_argument(
        "--skip-public",
        action="store_true",
        help="Disable public dataset ingestion",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--pems-station",
        type=int,
        default=None,
        metavar="STATION_ID",
        help=(
            "Caltrans PeMS station ID to use for demand calibration "
            "(default: 400456 = I-5 near downtown San Diego). "
            "Requires PEMS_API_KEY env var. Falls back to synthetic data when absent."
        ),
    )
    return parser.parse_args()


def _maybe_calibrate_from_pems(settings: object, pems_station: int | None) -> None:
    """Attempt PeMS calibration and store the calibration dict in settings."""
    station_id = pems_station or int(
        getattr(settings, "get", lambda k, d: d)("data.pems.station_id", 400456)
    )
    try:
        from traffic_ai.data_pipeline.pems_connector import PeMSConnector
        connector = PeMSConnector(station_id=station_id)
        df = connector.fetch("2024-01-15", "2024-01-22")
        calibration = connector.calibration_by_hour(df)
        if calibration:
            settings.payload.setdefault("data", {}).setdefault("pems", {})
            settings.payload["data"]["pems"]["calibration_by_hour"] = calibration
            logger.info(
                "PeMS calibration loaded for station %d (%d hours).",
                station_id, len(calibration),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PeMS calibration skipped: %s", exc)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.output_dir:
        settings.payload["project"]["output_dir"] = args.output_dir
        settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(settings.seed)

    # Attempt PeMS demand calibration (falls back gracefully if unavailable)
    if args.pems_station is not None or True:  # always try; connector handles fallback
        _maybe_calibrate_from_pems(settings, args.pems_station)

    runner = ExperimentRunner(settings=settings, quick_run=args.quick_run)
    artifacts = runner.run(
        ingest_only=args.ingest_only,
        include_kaggle=not args.skip_kaggle,
        include_public=not args.skip_public,
    )

    summary = {
        "summary_csv": str(artifacts.summary_csv),
        "step_metrics_csv": str(artifacts.step_metrics_csv),
        "significance_csv": str(artifacts.significance_csv),
        "ablation_csv": str(artifacts.ablation_csv),
        "model_metrics_csv": str(artifacts.model_metrics_csv),
        "generated_plots": [str(p) for p in artifacts.generated_plots],
        "trained_model_dir": str(artifacts.trained_model_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

