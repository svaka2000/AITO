# AI Traffic Signal Optimization
## A Comparative Analysis of Machine Learning and Reinforcement Learning Controllers for Reducing Urban Intersection Congestion

**Author:** Samarth Vaka | GSDSEF Senior Division

---

## Abstract

Urban traffic congestion costs U.S. drivers an estimated $87 billion annually in lost productivity and wasted fuel. This research investigates whether artificial intelligence can optimize traffic signal timing to reduce intersection delays. A stochastic multi-intersection simulation engine was developed modeling Poisson vehicle arrivals, rush-hour demand scaling, queue spillback, and network-level vehicle propagation across a configurable grid. Ten signal controllers spanning four families — fixed timing (baseline), adaptive rule-based, supervised machine learning (Random Forest, XGBoost, Gradient Boosting, Neural Network), and reinforcement learning (Q-Learning, Deep Q-Network, Policy Gradient) — were benchmarked across 5-fold cross-validation with 2,000 simulation steps per fold. Statistical significance was validated using Mann-Whitney U tests (α=0.05).

---

## Research Question

Can AI-powered traffic signal controllers — using supervised machine learning and reinforcement learning — reduce vehicle wait times and improve intersection throughput compared to fixed-timing systems?

## Hypothesis

Adaptive ML/RL controllers will outperform fixed-timing and rule-based baselines by dynamically responding to real-time queue conditions, resulting in lower average wait times and higher throughput.

---

## Controllers Tested (10 total across 4 families)

| Family | Controllers |
|--------|------------|
| Fixed Timing (Baseline) | Fixed 30s cycle |
| Adaptive Rule-Based | Queue-threshold adaptive |
| Supervised ML | Random Forest, XGBoost, Gradient Boosting, Neural Network (MLP) |
| Reinforcement Learning | Q-Learning, Deep Q-Network (DQN), Policy Gradient |

---

## Statistical Methods

- **5-fold cross-validation** — results averaged across folds to reduce variance
- **Mann-Whitney U tests** — non-parametric pairwise significance testing (α=0.05)
- **Bootstrap confidence intervals** — 95% CI via 300 bootstrap resamples
- **Ablation study** — adaptive controller hyperparameter sensitivity analysis
- **Reproducible seeded randomness** — global seed = 42

---

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python main.py --quick-run
```

## Full Benchmark Run

```bash
python main.py --output-dir artifacts --config traffic_ai/config/default_config.yaml
```

## Interactive Dashboard

```bash
streamlit run traffic_ai/dashboard/streamlit_app.py
```

## Run Tests (66 unit tests)

```bash
pytest -q
```

---

## Artifacts Generated

| Path | Contents |
|------|----------|
| `artifacts/results/controller_summary.csv` | Primary results table |
| `artifacts/results/significance_tests.csv` | Pairwise Mann-Whitney p-values |
| `artifacts/results/ablation_study.csv` | Hyperparameter sensitivity |
| `artifacts/plots/` | 5 publication-quality figures (300 DPI) |
| `artifacts/models/` | Trained model files (.joblib, .pt) |

---

## Project Structure

```
traffic_ai/
├── simulation_engine/   # Stochastic 2x2 grid simulator
├── controllers/         # 10 signal control algorithms
├── ml_models/           # Supervised learning suite (RF, XGBoost, GBM, MLP)
├── rl_models/           # RL suite (Q-learning, DQN, Policy Gradient)
├── experiments/         # CV, ablation, statistical testing pipeline
├── metrics/             # Metric calculation and Mann-Whitney tests
├── visualization/       # Publication-quality matplotlib/seaborn figures (300 DPI)
├── dashboard/           # Interactive Streamlit web UI
├── config/              # YAML configuration
└── data_pipeline/       # Data ingestion and preprocessing
```

---

## San Diego Relevance

San Diego County operates over 3,000 signalized intersections coordinated across 18 cities by SANDAG. Software-based signal optimization requires no new hardware — only updated timing algorithms deployed to existing controllers. This research demonstrates that AI-based approaches are technically viable and warrant real-world piloting.

---

## GitHub

[github.com/svaka2000](https://github.com/svaka2000)
