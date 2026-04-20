> 🏆 **Cerebral Valley Hackathon — Built with Opus 4.7** · [HACKATHON.md](HACKATHON.md) · [Demo Script](HACKATHON_DEMO.md)
> AITO Copilot: Claude Opus 4.7 + 12-tool MCP server for traffic engineers · `python scripts/aito_copilot.py --canned`

# AITO — AI Traffic Optimization

<div align="center">

**Professional engineering platform for traffic signal control research and deployment**

*4-phase signal model · Shadow mode deployment · Sensor fault tolerance · EPA MOVES2014b emissions*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 155](https://img.shields.io/badge/tests-155%20passing-brightgreen.svg)]()

</div>

---

## Overview

AITO benchmarks **14 traffic signal controllers** across four families — fixed timing, adaptive rule-based, supervised ML (RF, XGBoost, GBM, MLP), and reinforcement learning (Q-Learning, DQN, PPO, A2C, SAC, RecurrentPPO, MADDPG) — on simulated urban intersection networks.

The platform is designed for **traffic engineers**, not just researchers:

- **4-phase signal model**: NS_THROUGH, EW_THROUGH, NS_LEFT, EW_LEFT with HCM 7th edition saturation flows
- **Shadow mode**: evaluate AI controllers without touching live traffic
- **Sensor fault tolerance**: EWMA imputation for stuck/noisy/dropout loop detectors
- **EPA MOVES2014b emissions**: first-class CO₂ tracking, not a proxy
- **Statistical rigor**: Holm-Bonferroni correction across all pairwise comparisons
- **Explainability**: natural language decision explanations for any controller

---

## Architecture

```
traffic_ai/
├── controllers/           # 14 signal controllers (all implement BaseController)
│   ├── base.py            # BaseController: reset, compute_actions, select_action, update
│   ├── fixed.py           # Fixed timing baseline (30s cycle)
│   ├── rule_based.py      # Queue-threshold adaptive
│   ├── ml_controllers.py  # RF, XGBoost, GradientBoosting, MLP
│   ├── rl_controllers.py  # QLearning, DQN, PPO, A2C, SAC, RecurrentPPO
│   ├── maddpg_controller.py # Multi-agent MADDPG (centralized training)
│   └── fault_tolerant.py  # EWMA-imputing fault wrapper
├── simulation_engine/     # Physics + types (HCM 7th ed.)
│   ├── engine.py          # TrafficNetworkSimulator
│   ├── types.py           # SignalPhase (6 values), PHASE_TO_IDX, IntersectionState
│   └── sensor.py          # SensorFaultModel (stuck/noise/dropout)
├── rl_models/             # RL training
│   ├── environment.py     # SignalControlEnv: 4-phase, 8-obs, 6-reward
│   ├── dqn.py             # DQNetwork + train_dqn
│   ├── q_learning.py      # QLearningPolicy + train_q_learning
│   └── policy_gradient.py # PolicyNet + train_policy_gradient
├── shadow/                # Shadow mode deployment
│   └── shadow_runner.py   # ShadowModeRunner → artifacts/shadow_report.json
├── explainability/        # AI transparency
│   └── explainer.py       # DecisionExplainer — NL + feature importances
├── dashboard/             # 6-tab Streamlit engineering dashboard
│   └── streamlit_app.py
├── data_pipeline/         # Ingestion, synthetic generation, PeMS calibration
├── training/              # ModelTrainer (unified ML + RL)
├── experiments/           # ExperimentRunner, Holm-Bonferroni
└── config/                # Settings, default_config.yaml
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick benchmark (all controllers, ~2 min)
python main.py --quick-run

# Run full benchmark
python main.py

# Launch the engineering dashboard
streamlit run traffic_ai/dashboard/streamlit_app.py

# Shadow mode: evaluate DQN vs fixed timing without affecting simulation
python main.py --shadow-mode --shadow-production fixed_timing --shadow-candidate dqn

# Run tests
pytest -q
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --config PATH             YAML configuration file (default: traffic_ai/config/default_config.yaml)
  --quick-run               Reduced budget for fast iteration
  --ingest-only             Run data ingestion only
  --skip-kaggle             Disable Kaggle dataset ingestion
  --skip-public             Disable public dataset ingestion
  --output-dir DIR          Override artifact output directory
  --pems-station STATION_ID Caltrans PeMS station for demand calibration
                            (requires PEMS_API_KEY env var; default: 400456)
  --shadow-mode             Run shadow mode evaluation
  --shadow-production CTRL  Production controller (default: fixed_timing)
  --shadow-candidate CTRL   Candidate AI controller (default: dqn)
```

---

## Signal Model

AITO uses a **4-phase protected signal model** (HCM 7th edition):

| Phase | SignalPhase | Serves | Saturation Flow |
|-------|-------------|--------|----------------|
| 0 | `NS_THROUGH` | N + S through + right | 1800 veh/hr/lane |
| 1 | `EW_THROUGH` | E + W through + right | 1800 veh/hr/lane |
| 2 | `NS_LEFT` | N + S protected left | 1200 veh/hr/lane |
| 3 | `EW_LEFT` | E + W protected left | 1200 veh/hr/lane |

Legacy `"NS"` and `"EW"` strings remain valid for backward compatibility.

---

## Multi-Objective Reward

RL controllers optimize a 6-component reward (configurable weights in `default_config.yaml`):

```
R = -w1·avg_delay - w2·ped_wait - w3·emissions_co2 - switch_cost + w4·throughput - w5·left_starvation

Default: w1=0.12, w2=0.05, w3=0.03, switch_cost=2.0, w4=0.08, w5=0.04
Source: Mannion et al. (2016) AAMAS workshop; Liang et al. (2019) ITSC
```

---

## Shadow Mode

Shadow mode allows safe evaluation of AI controllers **without modifying live traffic**:

```python
from traffic_ai.shadow.shadow_runner import ShadowModeRunner
from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.rl_controllers import DQNController

runner = ShadowModeRunner(
    production=FixedTimingController(),
    candidate=DQNController(),
)
report = runner.run()
# report.agreement_rate, report.estimated_queue_reduction_pct
runner.save_report(report, Path("artifacts/shadow_report.json"))
```

---

## Sensor Fault Tolerance

```python
from traffic_ai.simulation_engine.sensor import SensorFaultModel
from traffic_ai.controllers.fault_tolerant import FaultTolerantController

fault = SensorFaultModel(stuck_prob=0.02, noise_std=0.05, dropout_prob=0.01)
corrupted_obs = fault.apply(raw_obs, step=t, intersection_id=iid)

# Wrap any controller for EWMA-based fault tolerance
ctrl = FaultTolerantController(DQNController(), alpha=0.3)
```

---

## Decision Explainability

```python
from traffic_ai.explainability.explainer import DecisionExplainer

explainer = DecisionExplainer(controller=ctrl)
result = explainer.explain(obs, action=action)
print(result.natural_language)
# "Selected EW_THROUGH (eastbound/westbound green). Key driver: eastbound/westbound
#  through queue (highest feature influence). EW queue (24 veh) exceeds NS queue
#  (8 veh) — EW throughput prioritised."
```

---

## Using Real Traffic Data (PeMS)

AITO can be calibrated to real San Diego freeway detector data from
Caltrans PeMS (Performance Measurement System).

### How to Download PeMS Data

1. Log in at **pems.dot.ca.gov** (free account required)
2. Navigate to: **Data → Clearinghouse**
3. Set filters:
   - Type: **Station 5-Minute**
   - District: **11** (San Diego)
   - Station ID: **400456** (I-5 near downtown San Diego)
   - Date range: any recent weekday week
4. Download the CSV file
5. Place it at: `data/raw/pems_station_400456.csv`

### Run Rosecrans Corridor Validation

```bash
python main.py --validate-rosecrans
```

This compares GreedyAdaptive (InSync-style) against FixedTiming on the
12-signal Rosecrans corridor, calibrated to your PeMS data, and reports
how close the result is to the **verified 25% improvement** from the 2017
San Diego deployment.

Example output:

```
============================================================
AITO — Rosecrans Corridor Validation
============================================================
Demand source:    real_pems: pems_station_400456.csv (5 days, 1440 rows)
Simulation steps: 2000

SIMULATION RESULTS:
  FixedTiming avg wait:    30.09 s
  GreedyAdaptive avg wait: 22.31 s
  Simulated improvement:   25.9%

REAL-WORLD BENCHMARK (2017):
  Travel time reduction:   25.0%
  Source: San Diego Mayor Kevin Faulconer, March 2017.

CALIBRATION ASSESSMENT:
  Gap from benchmark:      0.9 percentage points
  STATUS: PASS — within ±10 pp tolerance
  Simulation is well-calibrated to San Diego conditions.
============================================================
```

### Without PeMS Data

The system falls back to synthetic demand profiles automatically.
Results are labeled `"synthetic"` in all outputs and the dashboard.

### API-Based Calibration (Legacy)

```bash
export PEMS_API_KEY=your_key_here
python main.py --pems-station 400456
```

Falls back to synthetic calibration when `PEMS_API_KEY` is absent.

---

## For Engineers

AITO is structured for deployment validation workflows:

1. **Benchmark**: run `python main.py --quick-run` to establish baseline metrics
2. **Shadow mode**: run `--shadow-mode` to evaluate AI without traffic impact
3. **Explainability**: use `DecisionExplainer` to audit AI decisions
4. **Fault testing**: wrap your controller with `FaultTolerantController` before field deployment
5. **Dashboard**: `streamlit run traffic_ai/dashboard/streamlit_app.py` for live monitoring

---

## Statistical Methodology

All pairwise comparisons use:
- **Mann-Whitney U test** (non-parametric, no normality assumption)
- **Holm-Bonferroni correction** for family-wise error rate
- **Bootstrap confidence intervals** (n=300)
- Significance threshold: α = 0.05

---

## Real-World Traffic Engineering Features (Phase 8)

AITO Phase 8 implements the real-world subsystems described in technical briefings with
City of San Diego Senior Traffic Engineer **Steve Celniker** and Caltrans District 11
Division Chief **Fariba Ramos**.

| Subsystem | AITO Model | Real-World Basis |
|---|---|---|
| Detection reliability | `DetectionSystem` (loop/video/none) | FHWA loop detector failure data (2006) |
| Interconnect & clock drift | `InterconnectNetwork` | Celniker: "copper cut by construction; clocks drift 0.5 s/hr" |
| Emergency preemption | `PriorityEventSystem` | NEMA TS-2 preemption standard |
| Bus transit signal priority | `PriorityEventSystem` | SANDAG TSP program |
| Leading Pedestrian Interval | `PriorityEventSystem` (LPI) | City of SD pedestrian safety program |
| Webster timing | `WebsterController` | Webster (1958), deployed in SCATS/SCOOT/Centracs |
| Greedy adaptive | `GreedyAdaptiveController` | InSync (Rhythm Engineering), Mira Mesa Blvd + Rosecrans St |

### San Diego Corridor Scenarios

```python
from traffic_ai.scenarios.san_diego import SanDiegoScenario

# 12-signal InSync corridor — validated 25% travel time reduction
cfg = SanDiegoScenario.rosecrans_corridor()

# Dense urban fixed-time grid — Downtown SD / Hillcrest
cfg = SanDiegoScenario.downtown_grid()

# Heavy commuter arterial — Mira Mesa Blvd, 50k ADT
cfg = SanDiegoScenario.mira_mesa_corridor()

# Cross-jurisdictional coordination gap — City SD + Caltrans boundary
cfg = SanDiegoScenario.mixed_jurisdiction()
```

The Rosecrans corridor is the primary validation target:
- GreedyAdaptive vs FixedTiming should reproduce the **25 % travel time reduction** and
  **53 % stop reduction** verified by Mayor Faulconer's office in 2017.

### Industry Comparison

The dashboard **Industry Comparison** tab shows how each AITO controller maps to a real
production system:

| AITO Controller | Industry System | Verified Improvement |
|---|---|---|
| `webster` | SCATS / SCOOT / Econolite Centracs | 5–15 % over fixed (FHWA 2005) |
| `greedy_adaptive` | InSync (Rhythm Engineering) | 25 % travel time ↓, 53 % stops ↓ (Faulconer 2017) |
| `dqn` | Research RL | 20–40 % over rule-based (Liang 2019) |

---

## Citations

- HCM 7th Edition (TRB, 2022) — signal timing, saturation flow rates
- EPA MOVES2014b — CO₂ idle emission rates (0.000457 kg/s/vehicle)
- Mannion et al. (2016), AAMAS — multi-objective RL reward weights
- Liang et al. (2019), ITSC — throughput bonus formulation
- Chen et al. (2001), Transport. Res. Part C — sensor fault characterization
- Toth & Ceder (2002), Transport. Res. Part C — EWMA imputation (α=0.3)
- Ribeiro et al. (2016), KDD — sensitivity-based feature importance
- Webster, F.V. (1958), "Traffic Signal Settings", Road Research Technical Paper No. 39, HMSO
- Rhythm Engineering (2017), "How InSync Works", Technical Whitepaper
- San Diego Mayor's Office (2017), "Rosecrans Street Traffic Signal Improvements", Press Release
- Varaiya, P. (2013), "Max pressure control of a network of signalized intersections", Trans. Res. Part C
