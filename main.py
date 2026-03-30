from __future__ import annotations

# Load .env before any other imports so PEMS_USERNAME, PEMS_PASSWORD, etc. are set
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

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
        description="AITO — AI Traffic Optimization | Professional Engineering Platform"
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
        "--full-run",
        action="store_true",
        help="Train RL agents for 2000 episodes (overnight quality run)",
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
    parser.add_argument(
        "--pems-calibrate",
        action="store_true",
        help=(
            "Test PeMS connection and pull calibration data for the given station. "
            "Uses PEMS_USERNAME / PEMS_PASSWORD from environment or .env file."
        ),
    )
    parser.add_argument(
        "--shadow-mode",
        action="store_true",
        help=(
            "Run in shadow mode: production controller drives the simulation; "
            "candidate controller logs counterfactual recommendations only. "
            "Requires --shadow-production and --shadow-candidate."
        ),
    )
    parser.add_argument(
        "--shadow-production",
        type=str,
        default="fixed_timing",
        metavar="CONTROLLER",
        help="Production controller name for shadow mode (default: fixed_timing)",
    )
    parser.add_argument(
        "--shadow-candidate",
        type=str,
        default="dqn",
        metavar="CONTROLLER",
        help="Candidate AI controller name for shadow mode (default: dqn)",
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


def _run_pems_calibrate(args: argparse.Namespace, settings: object) -> None:
    """Test PeMS connection and pull calibration data for the given station."""
    import os
    # Load .env if present
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    station_id = args.pems_station or 400456
    username = os.environ.get("PEMS_USERNAME", "")
    password = os.environ.get("PEMS_PASSWORD", "")

    if not username or not password:
        logger.error("PEMS_USERNAME / PEMS_PASSWORD not set. Add them to .env or environment.")
        return

    logger.info("Connecting to Caltrans PeMS — station %d as %s …", station_id, username)
    print(f"\nPeMS Calibration Test")
    print(f"  Station  : {station_id}")
    print(f"  Username : {username}")
    print(f"  Password : {'*' * len(password)}")

    try:
        from traffic_ai.data_pipeline.pems_connector import PeMSConnector
        connector = PeMSConnector(station_id=station_id, username=username, password=password)
        df = connector.fetch("2024-01-15", "2024-01-22")
        calibration = connector.calibration_by_hour(df)

        if calibration:
            print(f"\n✓ Calibration loaded: {len(calibration)} hourly demand profiles")
            for hour in sorted(calibration.keys())[:5]:
                print(f"  Hour {hour:02d}: {calibration[hour]:.3f} veh/step")
            if len(calibration) > 5:
                print(f"  ... ({len(calibration) - 5} more hours)")
            # Save calibration to artifacts
            output_dir = Path(getattr(settings, "output_dir", "artifacts"))
            output_dir.mkdir(parents=True, exist_ok=True)
            cal_path = output_dir / "pems_calibration.json"
            import json as _json
            cal_path.write_text(_json.dumps({"station_id": station_id, "calibration_by_hour": calibration}, indent=2))
            print(f"\n✓ Saved to {cal_path}")
        else:
            print("\n⚠ Connection succeeded but no calibration data returned (synthetic fallback active)")
    except Exception as e:
        logger.error("PeMS calibration failed: %s", e)
        print(f"\n✗ PeMS connection failed: {e}")
        print("  → Synthetic fallback will be used in simulation runs")


def _run_shadow_mode(args: argparse.Namespace, settings: object) -> None:
    """Execute shadow mode and write report to artifacts/shadow_report.json."""
    from traffic_ai.shadow.shadow_runner import ShadowModeRunner
    from traffic_ai.simulation_engine.engine import SimulatorConfig

    _CTRL_MAP: dict[str, type] = {}
    try:
        from traffic_ai.controllers.fixed import FixedTimingController
        from traffic_ai.controllers.rule_based import RuleBasedController
        from traffic_ai.controllers.rl_controllers import (
            DQNController, PPOController, QLearningController,
        )
        _CTRL_MAP = {
            "fixed_timing": FixedTimingController,
            "rule_based": RuleBasedController,
            "q_learning": QLearningController,
            "dqn": DQNController,
            "ppo": PPOController,
        }
    except ImportError as e:
        logger.error("Could not import controllers for shadow mode: %s", e)
        return

    prod_name = args.shadow_production.lower()
    cand_name = args.shadow_candidate.lower()

    if prod_name not in _CTRL_MAP:
        logger.error("Unknown production controller %r. Choose from: %s", prod_name, list(_CTRL_MAP))
        return
    if cand_name not in _CTRL_MAP:
        logger.error("Unknown candidate controller %r. Choose from: %s", cand_name, list(_CTRL_MAP))
        return

    prod_ctrl = _CTRL_MAP[prod_name]()
    cand_ctrl = _CTRL_MAP[cand_name]()

    sim_steps = 300 if not getattr(args, "quick_run", False) else 60
    cfg = SimulatorConfig(steps=sim_steps, intersections=4, seed=42)

    logger.info("Shadow mode: production=%s, candidate=%s, steps=%d", prod_name, cand_name, sim_steps)
    runner = ShadowModeRunner(production=prod_ctrl, candidate=cand_ctrl, config=cfg)
    report = runner.run()

    output_dir = Path(getattr(settings, "output_dir", "artifacts"))
    report_path = output_dir / "shadow_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_report(report, report_path)

    logger.info(
        "Shadow report saved to %s  |  agreement=%.1f%%  |  est_queue_reduction=%.1f%%",
        report_path,
        report.agreement_rate * 100,
        report.estimated_queue_reduction_pct,
    )
    print(json.dumps({
        "shadow_report": str(report_path),
        "production_controller": report.production_controller,
        "candidate_controller": report.candidate_controller,
        "agreement_rate": report.agreement_rate,
        "estimated_queue_reduction_pct": report.estimated_queue_reduction_pct,
    }, indent=2))


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.output_dir:
        settings.payload["project"]["output_dir"] = args.output_dir
        settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(settings.seed)

    # PeMS calibration test
    if args.pems_calibrate:
        _run_pems_calibrate(args, settings)
        return

    # Shadow mode: run AI evaluation without touching production traffic
    if args.shadow_mode:
        _run_shadow_mode(args, settings)
        return

    # Attempt PeMS demand calibration (falls back gracefully if unavailable)
    if args.pems_station is not None or True:  # always try; connector handles fallback
        _maybe_calibrate_from_pems(settings, args.pems_station)

    runner = ExperimentRunner(settings=settings, quick_run=args.quick_run, full_run=args.full_run)
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

