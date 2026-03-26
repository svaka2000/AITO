# PLANS.md

## Objective
Deliver a modular, reproducible AI traffic optimization research platform with classical control, supervised ML, and RL comparisons.

## Phase 1: Core Skeleton — DONE
- [x] Create package structure and artifact directories.
- [x] Add repository operating rules (`AGENTS.md`).
- [x] Add configuration (`traffic_ai/config/settings.py`, `default_config.yaml`) and reproducibility primitives (`traffic_ai/utils/reproducibility.py`).

## Phase 2: Data Platform — DONE
- [x] Kaggle ingestion via API client (`traffic_ai/data/ingestion.py`).
- [x] Public dataset ingestion: Metro Interstate (UCI), synthetic fallback.
- [x] Common schema normalization to 15-column unified schema (`traffic_ai/data/preprocessing.py`).
- [x] Derived features: `is_rush_hour`, `hour_of_day`, `day_of_week`, rolling means (15-min, 60-min).
- [x] Train/validation/test 70/15/15 stratified split; saved to `data/processed/`.

## Phase 3: Simulation Engine — DONE
- [x] Multi-lane intersection simulator (`traffic_ai/simulation_engine/`) — existing engine.
- [x] `MultiIntersectionNetwork` N×M grid with Gym-compatible step/reset (`traffic_ai/simulation/intersection.py`).
- [x] Poisson arrivals with rush-hour 2.5× scaling, vehicle spillback.
- [x] Queue dynamics: max queue cap enforced per lane.
- [x] `EpisodeMetricsTracker` with per-step/per-episode CSV export (`traffic_ai/simulation/metrics.py`).

## Phase 4: Controllers — DONE
- [x] Unified `BaseController` interface with `select_action`, `update`, `compute_actions` (`traffic_ai/controllers/base.py`).
- [x] `FixedTimingController` (30s default cycle) — `traffic_ai/controllers/fixed.py`.
- [x] `RuleBasedController` (queue-balance adaptive) — `traffic_ai/controllers/rule_based.py`.
- [x] Supervised ML controllers: RF, XGBoost, GradientBoosting, MLP (PyTorch 128→64→32), LSTMForecast, ImitationLearning — `traffic_ai/controllers/ml_controllers.py`.
- [x] RL controllers: QLearning (tabular), DQN (replay buffer + target network), PPO (actor-critic) — `traffic_ai/controllers/rl_controllers.py`.

## Phase 5: Research Workflow — DONE
- [x] Experiment runner with cross-validation and ablation sweeps (`traffic_ai/experiments/runner.py`).
- [x] Statistical significance testing (Mann-Whitney U / bootstrap) via `traffic_ai/metrics/statistics.py`.
- [x] New publication-quality plots: controller comparison, feature importance, ablation heatmap, queue over time (`traffic_ai/visualization/plots.py`).

## Phase 6: Interfaces — DONE
- [x] FastAPI service: `POST /run`, `GET /results/{run_id}`, `GET /plots/{run_id}/{plot_name}`, `GET /health` — `traffic_ai/api/server.py`.
- [x] Streamlit dashboard with live grid animation, metrics charts, results panel, controller switcher — `traffic_ai/dashboard/streamlit_app.py`.

## Phase 7: Validation — DONE
- [x] Unit tests: `tests/test_simulation.py` (8 tests), `tests/test_controllers.py` (22 tests), `tests/test_metrics.py` (9 tests).
- [x] Existing test suites: `test_data_pipeline.py`, `test_experiment_runner.py`, `test_simulation_and_controllers.py`.
- [x] End-to-end smoke run: `python main.py --quick-run` passes cleanly.
- [x] `pytest -q` — 66 tests pass.

## Phase 8: Research-Grade Upgrades — DONE
### WS1: Upgraded DQN
- [x] Double DQN: online net selects action, target net evaluates (eliminates overestimation bias).
- [x] Dueling Architecture: shared 6→128→128 + V(s) value stream + A(s,a) advantage stream.
- [x] Prioritized Experience Replay (`traffic_ai/rl_models/replay_buffer.py`): |TD error|^α sampling + IS-weight correction + β annealing.
- [x] N-step returns (3-step buffer) before bootstrapping.
- [x] RewardShaper class (`traffic_ai/rl_models/rewards.py`): composite reward with queue/throughput/wait/phase-change/emergency/fairness terms.
- [x] Cosine-annealing LR (3e-4 → 1e-5), gradient clipping (max_norm=1.0), ε: 1.0→0.01 over 80% of training.

### WS2: New RL Controllers
- [x] `A2CController`: GAE(λ=0.95), separate actor/critic optimizers, entropy_coef=0.01.
- [x] `SACController`: twin Q-networks, learnable log_α temperature, Polyak update (τ=0.005), off-policy 50k replay.
- [x] `MADDPGController` (`traffic_ai/controllers/maddpg_controller.py`): per-intersection actors + centralized critics, neighbor-augmented observations, Gumbel-Softmax.
- [x] `RecurrentPPOController`: LSTM actor/critic (hidden=64), SEQ_LEN=16 BPTT, per-intersection hidden states.
- [x] Training functions: `traffic_ai/rl_models/a2c.py`, `sac.py`, `maddpg.py`, `recurrent_ppo.py`.

### WS3: 8 New Demand Profiles
- [x] `DemandModel` extended: weekend, school_zone, event_surge, construction, emergency_priority, high_density_developing, incident_response, weather_degraded.
- [x] `service_rate_multiplier()`, `tick_emergency()`, `tick_incident()`, `noncompliance_rate()` methods.
- [x] `IntersectionState`: emergency_active, emergency_direction, emergency_steps_remaining fields.
- [x] Engine: `_apply_emergency_events()`, `_override_emergency_actions()`, demand side-effect ticks.

### WS4: Environmental Impact Tracking
- [x] `EmissionsCalculator` (`traffic_ai/simulation_engine/emissions.py`): EPA MOVES3 idle fuel (0.16 gal/hr), CO₂ (8.887 kg/gal), stop-start penalty, annualise().
- [x] `StepMetrics`: fuel_gallons + co2_kg fields.
- [x] Engine computes fuel/CO₂ per step; aggregate includes total_fuel_gallons + total_co2_kg.
- [x] Dashboard Environmental Impact tab: per-controller fuel/CO₂ bars, tree-years equivalent KPI.

### WS5: Controller Info Cards
- [x] 14 controller expandable glassmorphism cards in dashboard (architecture, strengths, weaknesses).
- [x] `CONTROLLER_INFO` dict wired into `_render_controller_cards()`.

### WS6: Enhanced Statistics
- [x] `statistics.py`: Holm-Bonferroni step-down correction, Cohen's d, `bootstrap_median_difference()`, `statistical_power_analysis()`, median bootstrap CI.
- [x] Dashboard Statistics tab: live correction method switcher, bootstrap CI table with median.

### WS7: UI/UX Overhaul
- [x] Hero: animated pulse-border, updated subtitle with all new features, 8 pills.
- [x] Glassmorphism `ctrl-info-card` with hover + fadeSlideIn animation.
- [x] Environmental metric cards with green-tinted glassmorphism.
- [x] All 11 demand profiles in Live Simulation selectbox with descriptions.
- [x] 4 new RL controllers (A2C, SAC, MADDPG, RecurrentPPO) wired into Live Simulation.
- [x] Fuel/CO₂/tree-years KPI row in Live Simulation results.
- [x] `CONTROLLER_DISPLAY_NAMES` updated for 14 controllers.
- [x] `plotly>=5.20.0` added to requirements.txt.

## Phase 9: Synthetic Data Studio — DONE

### Backend
- [x] `SyntheticDatasetGenerator` (`traffic_ai/data_pipeline/synthetic_generator.py`): `SyntheticDatasetConfig` (35+ parameters), `SyntheticDatasetResult`, fully vectorised NumPy generation (no Python row-loops over samples), pre-computed DemandModel rate table (≤1,152 calls regardless of n_samples).
- [x] 4 label strategies: `optimal` (1-step simulation comparison), `queue_balance` (heuristic), `fixed` (alternating), `adaptive_rule` (RuleBasedController).
- [x] 5 special-scenario overlays: incidents (queue ×3, speed ×0.4), weather (volume ×1.3, speed ×0.85), event surges (×4 pre, ×3.5 post), school-zone concentration, emergency-vehicle clearance.
- [x] `DatasetStore` (`traffic_ai/data_pipeline/dataset_store.py`): atomic CRUD via `os.replace()`, `save`, `load`, `list_datasets`, `delete`, `rename`, `duplicate`, `export_csv`, `get_config`, `_resolve_dir`, `_safe_name`.
- [x] `ModelTrainer` + `TrainingResult` (`traffic_ai/training/trainer.py`): dispatches RL (EnvConfig parameterised from dataset arrival-rate stats) and ML (feature extraction → `ctrl.fit(X, y_str)`); progress callbacks at each stage.
- [x] `DataIngestor.ingest_all()` extended with `synthetic_dataset_name` parameter; `_ingest_studio_dataset()` loads and copies saved datasets into the pipeline.
- [x] `default_config.yaml`: added `synthetic_datasets_dir: data/synthetic_datasets`.

### Dashboard
- [x] "Data Studio" fourth top-level tab added to `run_dashboard()`.
- [x] 4A — Dataset Manager: glassmorphism `.dataset-card` cards (View / Dup / CSV / Del buttons), empty-state CTA, 3-per-row grid.
- [x] 4B — Generator panel: 7 sections (Basics, Network, Volume, Temporal, Scenarios, Labels, Preview & Generate), estimated generation-time display, live progress bar, toast on completion.
- [x] 4C — Dataset Detail: Plotly line + histogram + hour×day heatmap, Download CSV, Edit & Regenerate (pre-fills sliders), Train Model shortcut.
- [x] 4D — Training Workbench: controller selector with info card, dataset selector, RL/ML-adaptive config, `st.status()` live training, reward-curve plot, evaluation metrics.
- [x] 4E — Model Comparison: bar chart of last result's evaluation metrics.
- [x] `.dataset-card` glassmorphism CSS with `transform: scale(1.02)` hover + `transition: all 0.2s ease`.
- [x] `CONTROLLER_DISPLAY_NAMES` extended with 10 Data Studio trainer keys.
- [x] Sidebar Data Studio shortcut label added.

### Tests
- [x] `tests/test_synthetic_generator.py`: 24 tests — columns, row caps, all 4 label strategies, scenario injection (incidents/weather/events shift distributions), full DatasetStore CRUD roundtrip, ModelTrainer DQN 5-episode smoke, Random Forest smoke.

