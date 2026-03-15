# AI Traffic Signal Optimization
### Greater San Diego Science & Engineering Fair — Senior Division

**Author:** Samarth Vaka | vsamarth2010@gmail.com | github.com/svaka2000

---

## Abstract

Urban traffic congestion costs U.S. drivers an estimated $87 billion annually in lost productivity and wasted fuel. This research investigates whether artificial intelligence can optimize traffic signal timing to reduce intersection delays. A stochastic multi-intersection simulation engine was developed modeling Poisson vehicle arrivals, rush-hour demand scaling, queue spillback, and network-level vehicle propagation across a configurable grid. Ten signal controllers spanning four families — fixed timing (baseline), adaptive rule-based, supervised machine learning (Random Forest, XGBoost, Gradient Boosting, Neural Network), and reinforcement learning (Q-Learning, Deep Q-Network, Policy Gradient) — were benchmarked across 5-fold cross-validation with 2,000 simulation steps per fold. Statistical significance was validated using Mann-Whitney U tests (α=0.05).

---

## Research Question
Can AI-powered traffic signal controllers reduce vehicle wait times and improve intersection throughput compared to fixed-timing systems?

## Hypothesis
Adaptive ML/RL controllers will outperform fixed-timing baselines by dynamically responding to real-time queue conditions.

---

## Methods

**Simulation Engine:** Stochastic 2×2 intersection grid with:
- Poisson vehicle arrivals
- Rush-hour demand scaling (2.5× peak)
- Queue spillback and vehicle propagation
- 2,000 simulation steps per trial

**Controllers Tested (10 total):**

| Family | Controllers |
|--------|------------|
| Fixed Timing (Baseline) | Fixed 30-second cycle |
| Adaptive Rule-Based | Queue-threshold adaptive controller |
| Supervised ML | Random Forest, XGBoost, Gradient Boosting, Neural Network |
| Reinforcement Learning | Q-Learning, Deep Q-Network (DQN), Policy Gradient |

**Statistical Validation:**
- 5-fold cross-validation
- Mann-Whitney U tests (α=0.05, non-parametric)
- Bootstrap confidence intervals (95% CI, 300 resamples)
- Ablation study on hyperparameters

**Metrics:** Average wait time (s), queue length (vehicles), throughput (vehicles/step), fairness score, system efficiency score

---

## Key Results

Full benchmark run (5-fold cross-validation, 50 episodes). Ranked by average wait time (lower is better). Fixed timing (20.36s) is the baseline.

| Rank | Controller | Avg Wait (s) | vs. Fixed Timing | Efficiency Score |
|------|-----------|-------------|-----------------|-----------------|
| 1 | **Random Forest (ML)** | **15.40** | **−24.4%** | 0.714 |
| 2 | XGBoost (ML) | 17.05 | −16.3% | **0.746** |
| 3 | Logistic Regression | 17.31 | −14.9% | 0.660 |
| 4 | Gradient Boosting (ML) | 17.32 | −14.9% | **0.787** |
| 5 | Q-Learning (RL) | 19.54 | −4.0% | 0.770 |
| 6 | **Fixed Timing (Baseline)** | **20.36** | — | 0.735 |
| 7 | Adaptive Rule | 21.07 | +3.5% | 0.622 |
| 8 | Neural Network (MLP) | 27.52 | +35.2% | 0.576 |
| 9 | Policy Gradient (RL) | 35.80 | +75.8% | 0.463 |
| 10 | DQN (RL) | 49.64 | +143.8% | 0.250 |

All pairwise comparisons validated using Mann-Whitney U tests (α = 0.05). 5 of 10 controllers significantly outperformed the fixed-timing baseline.

---

## Conclusion

- **ML controllers collectively outperform fixed timing**: Random Forest (−24.4%), XGBoost (−16.3%), Logistic Regression (−14.9%), Gradient Boosting (−14.9%) all significantly beat the baseline.
- **Random Forest achieved the lowest average wait time of 15.40 seconds** — a 24.4% reduction from the 20.36s fixed-timing baseline, statistically significant (p < 0.001).
- **Q-Learning (RL)** also outperformed fixed timing (−4.0%), demonstrating that simpler RL algorithms can converge effectively within a practical training budget.
- **Deep RL (DQN, Policy Gradient) underperformed**, reflecting the substantially larger training budgets required for these methods. This is an important practical finding: algorithm complexity does not guarantee performance with limited computational resources.
- **Gradient Boosting** achieved the highest system efficiency score (0.787), indicating the best composite network utilization.

The results strongly support the hypothesis that AI-driven signal controllers can reduce urban intersection delays compared to fixed-timing systems, particularly through supervised machine learning approaches.

**Limitations:** Results reflect simulation, not real-world deployment. Deep RL controllers would benefit from extended training. Simulation parameters (Poisson arrivals, 2×2 grid) are simplified relative to real urban networks.

---

## San Diego Relevance

- San Diego County: 3,000+ signalized intersections across 18 cities
- SANDAG coordinates regional transportation planning
- Software-only optimization: no new hardware required
- Potential impact: millions of vehicle-hours saved annually

---

## Reproducibility

All code, data, and artifacts are publicly available:
- GitHub: github.com/svaka2000
- To reproduce: `python main.py --quick-run` → `pytest -q` (66 tests)
- Dashboard: `streamlit run traffic_ai/dashboard/streamlit_app.py`

---

## References

1. INRIX. (2023). *Global Traffic Scorecard*. INRIX Research.
2. SANDAG. (2023). *2021 Regional Transportation Plan*. San Diego Association of Governments.
3. Ault, J., et al. (2021). *Reinforcement Learning Benchmarks for Traffic Signal Control*. NeurIPS Datasets and Benchmarks.
4. Wei, H., et al. (2019). *A Survey on Traffic Signal Control Methods*. arXiv:1904.08117.
5. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529-533.
