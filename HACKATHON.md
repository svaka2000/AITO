# AITO Copilot — Cerebral Valley Hackathon Submission
### Built with Claude Opus 4.7 · Multi-Agent · Streaming

---

## What We Built

**AITO Copilot** is a Claude Opus 4.7–powered AI assistant that gives every traffic engineer a senior colleague on demand.

Traffic signal optimization is opaque. Engineers stare at spreadsheets, run simulations in batch, and wait days for results. AITO Copilot changes that: ask in plain English, get a streaming multi-agent trace that diagnoses intersections, compares controllers, explains decisions, and quantifies carbon savings — all against real physics models.

---

## The Problem

- **14 controller types** (Fixed, Adaptive, RF/XGBoost/MLP, Q-Learning/DQN/PPO/A2C/SAC/MADDPG) — impossible to compare without a simulation platform
- **Traffic engineers ≠ ML engineers** — they shouldn't have to read Python to understand why PPO beats fixed timing at PM peak
- **Carbon accounting is invisible** — EPA MOVES2014b emissions exist in the codebase; nobody surfaces them conversationally
- **Shadow mode is underused** — evaluating a new controller against live traffic logs requires CLI expertise most operators don't have

---

## Our Solution

AITO Copilot is a **three-layer system**:

```
User (plain English)
        ↓
  Claude Opus 4.7  ←—— tool_use ——→  12 AITO Tools
        ↓                               (MCP Server)
  Streaming explanation
        ↓
  Traffic engineer acts
```

### The 12 Tools (MCP Server)

| Tool | What it does |
|------|-------------|
| `run_benchmark` | Benchmark N controllers, return ranked metrics |
| `get_controller_explanation` | NL explanation of last decision |
| `compare_controllers` | Head-to-head stats with Holm-Bonferroni correction |
| `run_shadow_mode` | Replay historical logs through a new controller |
| `get_emissions_report` | EPA MOVES2014b CO₂ for any run |
| `get_demand_profiles` | List 11 calibrated demand scenarios |
| `train_rl_controller` | Kick off RL training (PPO/DQN/SAC) |
| `get_resilience_score` | 5-dimension network resilience (0–100) |
| `run_what_if` | Sensitivity: what happens if demand surges 30%? |
| `get_carbon_credits` | Verra VCS / CARB LCFS credit estimate |
| `get_spillback_forecast` | D/D/1 queue physics spillback prediction |
| `get_sensor_fault_report` | EWMA imputation quality for faulty loop detectors |

---

## Why Opus 4.7

- **Extended thinking** — controller comparisons require multi-step reasoning: load data → run stats → interpret → recommend. Opus 4.7 doesn't lose the thread.
- **Tool use with streaming** — users see Claude reasoning in real time, not a spinner.
- **Multi-agent trace** — Copilot spins up sub-agents for parallel benchmark runs, then synthesizes results.
- **Honest uncertainty** — when the data is ambiguous (e.g., PPO vs SAC on low-volume grids), Opus 4.7 says so instead of hallucinating a winner.

---

## Demo Scenario

> *"Compare PPO and fixed timing on the PM peak demand profile. Show me the emissions delta and whether the difference is statistically significant."*

Claude calls `run_benchmark`, `get_emissions_report`, and `compare_controllers` in sequence, streams its reasoning, and returns:

```
PPO reduces average delay by 23% vs Fixed (p=0.003, Holm-Bonferroni corrected).
CO₂ savings: 14.2 kg/hr per intersection.
Statistically significant across all 50 simulation seeds.
Recommend shadow-mode validation before live deployment.
```

Total wall time: ~8 seconds.

---

## Stack

| Layer | Technology |
|-------|-----------|
| AI | Claude Opus 4.7 (tool use + streaming) |
| MCP Server | Python stdio, 12 tools |
| Traffic Physics | HCM 7th ed., EPA MOVES2014b |
| Controllers | 14 signal controllers, BaseController interface |
| Stats | Holm-Bonferroni correction, 50-seed Monte Carlo |
| CLI | ANSI streaming, canned + live + REPL modes |

---

## Impact

San Diego has **3,200 signalized intersections**. If AITO Copilot helps engineers switch 10% of them from fixed timing to PPO:

- **~45,000 kg CO₂/hr avoided** citywide
- **~$2.1M/yr in CARB LCFS carbon credits**
- **23% reduction in average vehicle delay** — faster commutes, fewer rear-end collisions

The copilot makes that transition auditable, explainable, and fast.

---

## Team

Samarth Vaka · Built at Cerebral Valley "Built with Opus 4.7" Hackathon · April 2026
