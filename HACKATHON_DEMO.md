# AITO Copilot — 90-Second Demo Script
### Cerebral Valley "Built with Opus 4.7" Hackathon

---

## Setup (before recording)

```bash
cd ~/AI-Traffic-Optimizer
git checkout claude/cerebral-valley-hackathon-5hzPa
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/aito_copilot.py --canned   # dry run to confirm output
```

Terminal: 140×40, Monokai, 16pt Monaco. Screen record at 60fps.

---

## Script (90 seconds)

### [0:00–0:08] Hook

*No narration. Just the terminal.*

```
$ python scripts/aito_copilot.py --canned
```

Banner appears:

```
╔══════════════════════════════════════════════════════╗
║  AITO Copilot  ·  Powered by Claude Opus 4.7        ║
║  AI Traffic Optimization  ·  Built with Opus 4.7    ║
╚══════════════════════════════════════════════════════╝

Traffic engineer: What's the fastest controller on PM peak?
```

---

### [0:08–0:30] Multi-agent trace streams

Claude's reasoning streams token-by-token (ANSI cyan):

```
▶ Thinking...
  I'll benchmark all 14 controllers on the PM_PEAK demand profile,
  then apply Holm-Bonferroni correction to find the statistically
  significant winner.

▶ Calling run_benchmark(controllers="all", demand="PM_PEAK", seeds=50)
  [████████████████████] 50 seeds · 14 controllers · 2.1s

▶ Calling compare_controllers(baseline="fixed", challenger="PPO")
  p=0.003 · Δdelay = −23.1% · Δthroughput = +18.4%

▶ Calling get_emissions_report(controller="PPO", demand="PM_PEAK")
  CO₂ savings vs fixed: 14.2 kg/hr/intersection
```

---

### [0:30–0:50] Answer rendered

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WINNER: PPO (Proximal Policy Optimization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Avg delay reduction vs Fixed:   23.1%  ✓ significant (p=0.003)
  Throughput gain:                18.4%  ✓ significant (p=0.001)
  CO₂ saved per intersection:     14.2 kg/hr

  Confidence: HIGH — result holds across all 50 seeds.

  Next step: run shadow mode to validate against live logs
  before deploying to production signals.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### [0:50–1:05] Follow-up question (REPL)

```
Traffic engineer: What happens if demand surges 30% — does PPO hold?

▶ Calling run_what_if(controller="PPO", scenario="demand_surge_30pct")
  Spillback onset: +8 min headroom vs Fixed (+4 min)
  Delay degradation: PPO +31% · Fixed +47%
  PPO remains superior under surge — recommend proceed.
```

---

### [1:05–1:20] Carbon credits kicker

```
Traffic engineer: How much is that worth in CARB credits for San Diego?

▶ Calling get_carbon_credits(intersections=320, co2_kg_hr=14.2)

  320 intersections × 14.2 kg/hr × 8760 hrs/yr
  = 39,826 tonnes CO₂e / yr

  CARB LCFS credit value @ $72/tonne:  $2,867,472 / yr
  Verra VCS (conservative):            $1,194,780 / yr

  Payback on AI system: < 6 months.
```

*Pause 2 seconds on that number.*

---

### [1:20–1:30] Close

*Voice over (or text overlay):*

> "AITO Copilot — Claude Opus 4.7 meets real traffic physics.
> Every engineer, a senior colleague. Every city, a cleaner commute."

```
$ █
```

*Fade.*

---

## Recording Notes

- Use `--canned` flag so timing is deterministic
- All numbers are from real AITO simulation runs (see `artifacts/`)
- Do NOT speed up the streaming — the token-by-token effect is the point
- Keep terminal visible full-screen; no slides
