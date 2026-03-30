from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_ai.config.settings import Settings
from traffic_ai.controllers import (
    AdaptiveRuleController,
    build_baseline_controllers,
    build_rl_controllers,
    build_supervised_controllers,
    merge_controller_sets,
)
from traffic_ai.data_pipeline import TrafficDataPipeline
from traffic_ai.data_pipeline.pipeline import DataPipelineResult
from traffic_ai.metrics import (
    aggregate_experiment_rows,
    compute_system_efficiency_score,
    controller_bootstrap_table,
    pairwise_significance_tests,
    simulation_result_to_step_dataframe,
    simulation_result_to_summary_row,
)
from traffic_ai.ml_models import SupervisedModelSuite, train_supervised_controller_models
from traffic_ai.rl_models import train_rl_policy_suite
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.utils.io_utils import write_dataframe, write_json
from traffic_ai.visualization import (
    plot_controller_performance,
    plot_learning_curves,
    plot_model_metrics_table,
    plot_queue_and_wait_curves,
    plot_traffic_heatmap,
)


@dataclass(slots=True)
class ExperimentArtifacts:
    summary_csv: Path
    step_metrics_csv: Path
    significance_csv: Path
    ablation_csv: Path
    model_metrics_csv: Path
    generated_plots: list[Path]
    trained_model_dir: Path


class ExperimentRunner:
    def __init__(self, settings: Settings, quick_run: bool = False) -> None:
        self.settings = settings
        self.quick_run = quick_run
        self.output_dir = settings.output_dir
        self.result_dir = self.output_dir / "results"
        self.plot_dir = self.output_dir / "plots"
        self.model_dir = self.output_dir / "models"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Orchestrator: coordinates every phase of the experiment in sequence.
    # -------------------------------------------------------------------------
    def run(
        self,
        ingest_only: bool = False,
        include_kaggle: bool = True,
        include_public: bool = True,
    ) -> ExperimentArtifacts:
        """Orchestrates the full experiment pipeline end-to-end:
        data ingestion → model training → simulation → statistics → artifact generation.
        """
        data_result = self._run_data_pipeline(include_kaggle, include_public)

        if ingest_only:
            return self._early_exit_artifacts()

        supervised_metrics_df, supervised_controller_models = self._train_supervised_models(data_result)
        rl_result = self._train_rl_policies()
        controllers = self._build_controller_set(supervised_controller_models, rl_result)
        all_steps, grouped_summary = self._run_cross_validated_simulations(controllers)
        significance, bootstrap, ablation = self._compute_statistical_analysis(all_steps, grouped_summary)

        return self._save_and_plot_artifacts(
            grouped_summary=grouped_summary,
            all_steps=all_steps,
            significance=significance,
            bootstrap=bootstrap,
            ablation=ablation,
            supervised_metrics_df=supervised_metrics_df,
            rl_result=rl_result,
            data_result=data_result,
        )

    # -------------------------------------------------------------------------
    # SRP: Runs the full data pipeline (ingest → clean → engineer → preprocess).
    # -------------------------------------------------------------------------
    def _run_data_pipeline(
        self,
        include_kaggle: bool,
        include_public: bool,
    ) -> DataPipelineResult:
        """Pulls raw traffic data from all sources and prepares a modeling-ready dataset."""
        pipeline = TrafficDataPipeline(self.settings)
        return pipeline.run(
            include_kaggle=include_kaggle,
            include_public=include_public,
            include_local_csv=True,
        )

    # -------------------------------------------------------------------------
    # SRP: Returns a minimal artifact bundle when only data ingestion was requested.
    # -------------------------------------------------------------------------
    def _early_exit_artifacts(self) -> ExperimentArtifacts:
        """Writes a placeholder CSV and returns stub artifacts for ingest-only mode."""
        dummy = self.result_dir / "ingestion_only.csv"
        write_dataframe(pd.DataFrame({"status": ["completed"]}), dummy)
        return ExperimentArtifacts(
            summary_csv=dummy,
            step_metrics_csv=dummy,
            significance_csv=dummy,
            ablation_csv=dummy,
            model_metrics_csv=dummy,
            generated_plots=[],
            trained_model_dir=self.model_dir,
        )

    # -------------------------------------------------------------------------
    # SRP: Trains supervised ML models (Random Forest, XGBoost, MLP) on traffic data.
    # -------------------------------------------------------------------------
    def _train_supervised_models(
        self, data_result: DataPipelineResult
    ) -> tuple[pd.DataFrame, object]:
        """Fits and evaluates supervised classifiers; also trains signal-controller wrappers."""
        supervised_suite = SupervisedModelSuite(seed=self.settings.seed)
        _, supervised_metrics_df = supervised_suite.train_and_evaluate(
            dataset=data_result.prepared_dataset,
            model_dir=self.model_dir,
            cv_folds=self._cv_folds(),
            quick_mode=self.quick_run,
        )
        # Use imitation learning labels from AdaptiveRuleController (Problem 6)
        supervised_controller_models = train_supervised_controller_models(
            seed=self.settings.seed, quick_run=self.quick_run
        )
        return supervised_metrics_df, supervised_controller_models

    # -------------------------------------------------------------------------
    # SRP: Trains the full RL policy suite (Q-Learning, DQN, PPO, A2C, SAC, etc.).
    # -------------------------------------------------------------------------
    def _train_rl_policies(self) -> object:
        """Runs the RL training loop for every configured algorithm and returns learned policies."""
        return train_rl_policy_suite(
            output_dir=self.output_dir,
            seed=self.settings.seed,
            quick_run=self.quick_run,
        )

    # -------------------------------------------------------------------------
    # SRP: Assembles the full controller roster from all three families.
    # -------------------------------------------------------------------------
    def _build_controller_set(
        self, supervised_controller_models: object, rl_result: object
    ) -> list:
        """Merges fixed/adaptive baselines, supervised wrappers, and trained RL policies."""
        return merge_controller_sets(
            build_baseline_controllers(self.settings),
            build_supervised_controllers(supervised_controller_models),
            build_rl_controllers(rl_result.policies),
        )

    # -------------------------------------------------------------------------
    # SRP: Runs every controller over every CV fold and aggregates the results.
    # -------------------------------------------------------------------------
    def _run_cross_validated_simulations(
        self, controllers: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Benchmarks all controllers across independent random-seed folds of the simulator."""
        step_frames: list[pd.DataFrame] = []
        summary_rows: list[dict[str, float | str]] = []

        for fold in range(self._cv_folds()):
            simulator = TrafficNetworkSimulator(self._simulator_config(seed_offset=fold))
            for controller in controllers:
                sim_result = simulator.run(controller, steps=self._sim_steps())

                step_df = simulation_result_to_step_dataframe(sim_result)
                step_df["fold"] = fold
                step_frames.append(step_df)

                row = simulation_result_to_summary_row(sim_result)
                row["fold"] = fold
                summary_rows.append(row)

        all_steps = pd.concat(step_frames, ignore_index=True)
        all_summary = aggregate_experiment_rows(summary_rows)
        grouped_summary = (
            all_summary.groupby("controller", as_index=False)
            .mean(numeric_only=True)
            .sort_values("average_wait_time")
        )
        grouped_summary["system_efficiency_score"] = compute_system_efficiency_score(grouped_summary)
        return all_steps, grouped_summary

    # -------------------------------------------------------------------------
    # SRP: Runs all statistical analyses on the simulation results.
    # -------------------------------------------------------------------------
    def _compute_statistical_analysis(
        self, all_steps: pd.DataFrame, grouped_summary: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Computes Holm-Bonferroni pairwise significance, bootstrap CI, and ablation results."""
        significance = pairwise_significance_tests(
            all_steps,
            metric_col="avg_wait_sec",
            alpha=float(self.settings.get("experiments.significance_alpha", 0.05)),
        )
        bootstrap = controller_bootstrap_table(
            grouped_summary,
            metric="average_wait_time",
            n_bootstrap=int(self.settings.get("experiments.n_bootstrap", 300)),
            seed=self.settings.seed,
        )
        ablation = self._run_ablation_study()
        return significance, bootstrap, ablation

    # -------------------------------------------------------------------------
    # Orchestrator: delegates CSV writes, plot generation, and manifest writing.
    # -------------------------------------------------------------------------
    def _save_and_plot_artifacts(
        self,
        grouped_summary: pd.DataFrame,
        all_steps: pd.DataFrame,
        significance: pd.DataFrame,
        bootstrap: pd.DataFrame,
        ablation: pd.DataFrame,
        supervised_metrics_df: pd.DataFrame,
        rl_result: object,
        data_result: DataPipelineResult,
    ) -> ExperimentArtifacts:
        """Coordinates output persistence: delegates to CSV writer, plot generator, and manifest writer."""
        csv_paths = self._save_result_csvs(
            grouped_summary, all_steps, significance, bootstrap, ablation, supervised_metrics_df
        )
        generated_plots = self._generate_plots(grouped_summary, all_steps, rl_result, supervised_metrics_df)
        self._write_run_manifest(data_result)

        return ExperimentArtifacts(
            summary_csv=csv_paths["summary"],
            step_metrics_csv=csv_paths["steps"],
            significance_csv=csv_paths["significance"],
            ablation_csv=csv_paths["ablation"],
            model_metrics_csv=csv_paths["model_metrics"],
            generated_plots=generated_plots,
            trained_model_dir=self.model_dir,
        )

    # -------------------------------------------------------------------------
    # SRP: Writes every simulation result DataFrame to the results directory.
    # -------------------------------------------------------------------------
    def _save_result_csvs(
        self,
        grouped_summary: pd.DataFrame,
        all_steps: pd.DataFrame,
        significance: pd.DataFrame,
        bootstrap: pd.DataFrame,
        ablation: pd.DataFrame,
        supervised_metrics_df: pd.DataFrame,
    ) -> dict[str, Path]:
        """Atomically writes all result CSVs; returns a dict of named Path objects."""
        summary_csv = write_dataframe(grouped_summary, self.result_dir / "controller_summary.csv")
        step_csv = write_dataframe(all_steps, self.result_dir / "controller_step_metrics.csv")
        significance_csv = write_dataframe(significance, self.result_dir / "significance_tests.csv")
        ablation_csv = write_dataframe(ablation, self.result_dir / "ablation_study.csv")
        model_metrics_csv = write_dataframe(
            supervised_metrics_df, self.result_dir / "supervised_model_metrics.csv"
        )
        write_dataframe(bootstrap, self.result_dir / "bootstrap_wait_time_ci.csv")
        return {
            "summary": summary_csv,
            "steps": step_csv,
            "significance": significance_csv,
            "ablation": ablation_csv,
            "model_metrics": model_metrics_csv,
        }

    # -------------------------------------------------------------------------
    # SRP: Generates every publication-quality visualization for the experiment.
    # -------------------------------------------------------------------------
    def _generate_plots(
        self,
        grouped_summary: pd.DataFrame,
        all_steps: pd.DataFrame,
        rl_result: object,
        supervised_metrics_df: pd.DataFrame,
    ) -> list[Path]:
        """Calls each specialized plot function and returns the list of written file paths."""
        return [
            plot_controller_performance(grouped_summary, self.plot_dir),
            plot_queue_and_wait_curves(all_steps, self.plot_dir),
            plot_traffic_heatmap(all_steps, self.plot_dir),
            plot_learning_curves(rl_result.reward_history, self.plot_dir),
            plot_model_metrics_table(supervised_metrics_df, self.plot_dir),
        ]

    # -------------------------------------------------------------------------
    # SRP: Persists a JSON manifest documenting experiment settings and data provenance.
    # -------------------------------------------------------------------------
    def _write_run_manifest(self, data_result: DataPipelineResult) -> None:
        """Records configuration, CV folds, sim steps, and all source/cleaned file paths."""
        write_json(
            {
                "quick_run": self.quick_run,
                "cv_folds": self._cv_folds(),
                "sim_steps": self._sim_steps(),
                "source_files": [str(item.path) for item in data_result.source_results],
                "cleaned_files": [str(path) for path in data_result.cleaned_files],
            },
            self.result_dir / "run_manifest.json",
        )

    def _run_ablation_study(self) -> pd.DataFrame:
        """Grid-searches AdaptiveRuleController hyperparameters to measure sensitivity."""
        rows: list[dict[str, float | str]] = []
        queue_thresholds = [2.0, 6.0, 10.0]
        min_greens = [8, 15, 25]
        for threshold in queue_thresholds:
            for min_green in min_greens:
                controller = AdaptiveRuleController(
                    min_green=min_green,
                    max_green=75,
                    queue_threshold=threshold,
                )
                simulator = TrafficNetworkSimulator(self._simulator_config(seed_offset=int(threshold + min_green)))
                result = simulator.run(controller, steps=max(300, int(self._sim_steps() * 0.6)))
                row = simulation_result_to_summary_row(result)
                row.update({"ablation_queue_threshold": threshold, "ablation_min_green": min_green})
                rows.append(row)
        return pd.DataFrame(rows).sort_values("average_wait_time")

    def _simulator_config(self, seed_offset: int = 0) -> SimulatorConfig:
        """Builds a SimulatorConfig from settings, offset by fold index for reproducibility."""
        return SimulatorConfig(
            steps=self._sim_steps(),
            intersections=int(self.settings.get("simulation.intersections", 4)),
            lanes_per_direction=int(self.settings.get("simulation.lanes_per_direction", 2)),
            step_seconds=float(self.settings.get("simulation.step_seconds", 1.0)),
            max_queue_per_lane=int(self.settings.get("simulation.max_queue_per_lane", 60)),
            demand_profile=str(self.settings.get("simulation.demand_profile", "rush_hour")),
            demand_scale=float(self.settings.get("simulation.demand_scale", 1.0)),
            seed=self.settings.seed + seed_offset,
        )

    def _sim_steps(self) -> int:
        """Returns the number of simulation ticks, capped for quick-run mode."""
        base = int(self.settings.get("simulation.steps", 2000))
        return min(base, 260) if self.quick_run else base

    def _cv_folds(self) -> int:
        """Returns the number of cross-validation folds based on run mode."""
        if self.quick_run:
            return max(2, int(self.settings.get("experiments.quick_run.cv_folds", 3)) - 1)
        return int(self.settings.get("experiments.full_run.cv_folds", 5))
