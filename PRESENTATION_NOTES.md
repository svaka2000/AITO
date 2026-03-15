# GSDSEF 3-Minute Speaking Notes
## AI Traffic Signal Optimization — Samarth Vaka

---

### Hook (15 seconds)
"Every day, San Diego drivers waste millions of hours sitting at red lights — even when there's no cross traffic. The signal doesn't know any better because it's running on a fixed timer set decades ago. What if we could teach it to think?"

---

### Problem (30 seconds)
"Traffic congestion costs U.S. drivers $87 billion per year in lost time and wasted fuel. The problem isn't just inconvenience — it's emissions, emergency response times, and economic productivity. Most intersections today run on fixed-timing systems: a 30-second green, flip, repeat, regardless of whether anyone is waiting. These systems can't respond to rush hour, accidents, or any real-world variation."

---

### Approach (45 seconds)
"My approach: build a research platform that fairly tests whether AI can do better. I developed a stochastic multi-intersection traffic simulator — modeling Poisson vehicle arrivals, rush-hour demand scaling, and queue spillback — and benchmarked **10 different signal controllers** across **4 families**:
- Fixed timing — the baseline
- Adaptive rule-based — uses queue thresholds
- Supervised machine learning — Random Forest, XGBoost, Gradient Boosting, and Neural Network — trained on historical traffic patterns
- Reinforcement learning — Q-Learning, Deep Q-Network, and Policy Gradient — agents that learn by trial and reward

To make the comparison rigorous, I used **5-fold cross-validation**, **Mann-Whitney U statistical significance tests**, and **bootstrap confidence intervals**. Every experiment is seeded and reproducible, and the entire platform has 66 unit tests."

---

### Results (45 seconds)
"The results clearly supported the hypothesis. **Random Forest** achieved the lowest average wait time of **15.4 seconds** — a **24.4% reduction** from the fixed-timing baseline of 20.4 seconds. That's statistically significant at p < 0.001 using Mann-Whitney U testing.

In total, **5 out of 10 controllers beat fixed timing**: all four supervised ML models (Random Forest, XGBoost, Gradient Boosting, Logistic Regression), plus Q-Learning RL. Gradient Boosting achieved the highest overall efficiency score of 0.787.

The deep RL controllers — DQN and Policy Gradient — underperformed, reflecting that complex RL algorithms need substantially more training episodes to converge. This is itself an important finding: algorithm sophistication doesn't automatically translate to performance within a fixed computational budget. Q-Learning, the simpler RL approach, succeeded where DQN did not."

---

### Significance (30 seconds)
"Why does this matter for San Diego? The county has over 3,000 signalized intersections coordinated by SANDAG across 18 cities. The key finding isn't just that AI beats fixed timing in simulation — it's that **software-based optimization requires no new hardware**. You update the algorithm, not the infrastructure. Even a 10% reduction in average wait times across San Diego's network could save millions of vehicle-hours annually, with direct reductions in fuel consumption and CO₂ emissions."

---

### Close (15 seconds)
"The full simulation platform, all 66 tests, and every generated artifact are available on GitHub at github.com/svaka2000. I'm happy to walk through the code, the statistical methodology, or the dashboard — what would you like to explore?"

---

## Common Judge Questions & Answers

**Q: How does your simulation compare to real traffic?**
A: My stochastic model uses Poisson vehicle arrivals, rush-hour demand scaling (2.5×), and queue spillback propagation — these capture the key dynamics. It's simplified compared to real traffic, but that's intentional: controlled simulation is the standard approach for fair algorithm comparison. I also used real-world traffic datasets from Kaggle to train the supervised ML models.

**Q: Why not use real traffic data for the whole experiment?**
A: Real traffic data is great for training — and I did use it for the ML controllers. But for controlled comparison of 10 algorithms, you need a simulation where you can hold all variables constant except the controller. Otherwise you can't attribute performance differences to the algorithm.

**Q: What makes this better than existing traffic systems?**
A: I'm not claiming it's production-ready — this is simulation research. What I'm demonstrating is that AI approaches can significantly reduce wait times in a controlled setting, which suggests real-world piloting is warranted. The next step would be testing on actual sensor data from a San Diego corridor.

**Q: How do you know the results aren't random?**
A: Three layers of rigor: (1) 5-fold cross-validation to average out variance, (2) Mann-Whitney U tests for statistical significance at α=0.05 — a non-parametric test that makes no normality assumptions, and (3) reproducible seeded experiments — anyone can re-run and get identical results.

**Q: What would you do next?**
A: Three things: (1) integrate real-time sensor data from San Diego's existing loop detectors, (2) extend to multi-objective optimization balancing wait time, emissions, and fairness simultaneously, and (3) test on an actual corridor from SANDAG's network to validate simulation findings in the field.
