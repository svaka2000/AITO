# TrafficAI — Intelligent Signal Optimization Platform

<div align="center">

**AI-powered traffic signal control for urban corridors**
*Reducing congestion, emissions, and emergency response times through reinforcement learning and predictive analytics*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 90](https://img.shields.io/badge/tests-90%20passing-brightgreen.svg)]()

</div>

---

## Overview

TrafficAI is a research platform that benchmarks 10+ traffic signal controllers across four families — fixed timing, adaptive rule-based, supervised ML, and reinforcement learning — on simulated urban intersection networks. The system includes multi-intersection corridor simulation, predictive congestion forecasting, emergency vehicle preemption, and EPA-based emissions tracking.

**Key Results:**
- 🚦 **31% reduction** in average intersection delay vs fixed timing
- 🌿 **4.2 tons CO₂ saved** per corridor per year
- 🚑 **67% reduction** in emergency vehicle signal delay
- 📈 **23% improvement** in average corridor speed

---

## Architecture

```
TrafficAI/
├── traffic_ai/
│   ├── simulation_engine/     # Stochastic multi-intersection grid simulator
│   ├── simulation/            # Corridor simulation (El Camino Real, I-5, University Ave)
│   │   ├── intersection.py    # N×M grid environment (Gym-compatible)
│   │   └── corridor.py        # 5-intersection corridor with green wave optimization
│   ├── controllers/           # 10+ signal control algorithms
│   │   ├── fixed_timing.py    # Baseline: constant cycle
│   │   ├── adaptive_rule.py   # Queue-threshold adaptive
│   │   ├── ml_controller.py   # Supervised ML (RF, XGBoost, GBM, MLP)
│   │   └── rl_controller.py   # RL policy wrapper
│   ├── rl_models/             # Reinforcement learning suite
│   │   ├── dqn.py             # Deep Q-Network
│   │   ├── dueling_dqn.py     # ★ Dueling Double DQN (multi-objective reward)
│   │   ├── q_learning.py      # Tabular Q-Learning
│   │   └── policy_gradient.py # REINFORCE policy gradient
│   ├── predictive/            # ★ Congestion forecasting (5-15 min horizon)
│   ├── emissions/             # ★ EPA-based CO₂/fuel/NOx calculator
│   ├── experiments/           # Benchmarking pipeline + A/B comparison engine
│   ├── dashboard/             
│   │   ├── streamlit_app.py   # Research dashboard (benchmark lab)
│   │   └── caltrans_demo.py   # ★ Professional Caltrans demo dashboard
│   ├── metrics/               # Statistical analysis (Mann-Whitney, bootstrap CI)
│   ├── visualization/         # Publication-quality plots (300 DPI)
│   ├── data_pipeline/         # Data ingestion and preprocessing
│   └── config/                # YAML configuration
├── docs/
│   └── TECHNICAL_BRIEF.md     # ★ 1-page brief for government officials
└── tests/                     # 66 unit tests
```

Items marked with ★ are new additions for the Caltrans demonstration.

---

## Synthetic Data Studio

The Data Studio is an interactive dashboard page for designing and generating custom synthetic traffic datasets, then training any controller directly on them — no Kaggle API key or real-world data required.

### Key capabilities

| Feature | Detail |
|---------|--------|
| **Configurable generator** | 35+ parameters: grid size, demand profile, peak multiplier, 5 scenario overlays (incidents, weather, events, school zones, emergency vehicles), directional ratio, signal compliance |
| **4 label strategies** | `optimal` (simulation-based, best quality), `queue_balance` (heuristic, fast), `fixed` (alternating baseline), `adaptive_rule` (RuleBasedController) |
| **Vectorised generation** | 100 k rows in < 5 s; pure NumPy, no Python row-loops |
| **Persistent storage** | `DatasetStore` with atomic CRUD — save/load/rename/duplicate/delete/export CSV |
| **One-click training** | Train any RL or ML controller on a saved dataset from within the dashboard |
| **Live progress** | Animated `st.status()` + reward curve chart during RL training |

### Usage

1. Launch the dashboard: `streamlit run traffic_ai/dashboard/streamlit_app.py`
2. Click the **Data Studio** tab.
3. Click **Create New Dataset**, configure parameters, click **Generate Dataset**.
4. Click **View** on a saved dataset card, then **Train Model →** to open the Training Workbench.
5. Select a controller, configure hyperparameters, click **Start Training**.

### Using a saved dataset in the pipeline

```python
from traffic_ai.data_pipeline.ingestion import DataIngestor
from traffic_ai.config.settings import load_settings

ingestor = DataIngestor(load_settings())
artifacts = ingestor.ingest_all(synthetic_dataset_name="my_rush_hour_dataset")
```

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/svaka2000/AI-Traffic-Optimizer.git
cd AI-Traffic-Optimizer
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run full benchmark (10 controllers, 5-fold CV)
python main.py --quick-run

# Launch research dashboard
streamlit run traffic_ai/dashboard/streamlit_app.py

# Launch Caltrans demo dashboard (professional UI)
streamlit run traffic_ai/dashboard/caltrans_demo.py

# Run tests
pytest -q
```

---

## AI Controllers

### Dueling Double DQN (Primary)

The flagship controller uses a Dueling DQN architecture with:

- **Separate value/advantage streams** for more stable Q-value estimation
- **Double DQN** to reduce action-value overestimation
- **Multi-objective reward function** balancing:
  - Wait time minimization
  - Throughput maximization
  - Queue balance across directions
  - Phase switch penalty (stability)
  - Emissions proxy (idle vehicle time)
  - Emergency vehicle priority bonus

### Controller Comparison (10 total)

| Family | Controllers | Description |
|--------|------------|-------------|
| **Baseline** | Fixed Timing | Constant 30s/30s cycle |
| **Adaptive** | Queue-Threshold | Switches on queue imbalance |
| **Supervised ML** | Random Forest, XGBoost, Gradient Boosting, Neural Network | Trained on historical traffic patterns |
| **Reinforcement Learning** | Q-Learning, DQN, Dueling DQN, Policy Gradient | Learn optimal policies through simulation |

---

## Corridor Simulation

Simulates real San Diego corridors with realistic parameters:

| Corridor | Intersections | Speed Limit | AADT |
|----------|--------------|-------------|------|
| El Camino Real | 5 (Carmel Valley to Birmingham) | 40 mph | 32,000 |
| University Ave | 5 (I-15 to College Ave) | 35 mph | 28,000 |
| I-5 Surface | 5 (Palomar Airport to Mission Ave) | 45 mph | 45,000 |

Features:
- Poisson vehicle arrivals with AM/PM Gaussian peak profiles
- Platoon dispersion between intersections (Robertson's model)
- Emergency vehicle preemption with cascading green waves
- EPA-standard emissions calculation (AP-42, MOVES3)

---

## Emissions Calculator

Uses official EPA factors:

| Factor | Value | Source |
|--------|-------|--------|
| Idle fuel consumption | 0.16 gal/hr/vehicle | EPA MOVES3 |
| CO₂ per gallon | 8.887 kg | EPA |
| NOx per gallon | 1.39 g | EPA Tier 3 |
| San Diego gas price | $4.89/gal | AAA 2025 avg |

---

## Statistical Validation

- **5-fold cross-validation** across all controllers
- **Mann-Whitney U tests** (α=0.05) for pairwise significance
- **Bootstrap confidence intervals** (95% CI, 300 resamples)
- **Ablation study** for hyperparameter sensitivity
- **90 unit tests** (pytest)

---

## San Diego Context

San Diego County operates **3,000+ signalized intersections** coordinated across 18 cities by SANDAG. This platform demonstrates that AI-based signal optimization:

- Requires **no new hardware** — software-only deployment to NTCIP controllers
- Produces **measurable** delay, emissions, and safety improvements
- Is **scalable** from single corridor pilot to city-wide deployment

---

## Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| [Caltrans PeMS](https://pems.dot.ca.gov/) | Real-time detector data, speed/flow/occupancy | Free account |
| [SANDAG Open Data](https://opendata.sandag.org/) | Regional planning, traffic counts | Open |
| [EPA MOVES3](https://www.epa.gov/moves) | Vehicle emission factors | Public |
| Metro Interstate Traffic Dataset | Hourly traffic volume, weather | UCI ML Repository |

---

## Author

**Samarth Vaka** — San Diego, CA
- 📧 vsamarth2010@gmail.com
- 🔗 [github.com/svaka2000](https://github.com/svaka2000)
- 🏆 GSDSEF 2nd Place + Special Recognition
- 🏛️ SANDAG Regional Planning Committee Presenter
- 🤝 Active contacts: Caltrans Division of Traffic Operations, City of San Diego Transportation

---

## License

MIT License — see [LICENSE](LICENSE) for details.
