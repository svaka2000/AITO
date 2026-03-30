"""DQN stability verification after hyperparameter fixes.

Trains a DQN for 2000 episodes with the corrected configuration, prints
a phase distribution breakdown to confirm all 4 phases are explored,
and verifies that average wait time beats fixed_timing.
"""
from __future__ import annotations

import collections

import numpy as np

from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv
from traffic_ai.rl_models.dqn import train_dqn
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.simulation_engine.types import IDX_TO_PHASE, PHASE_TO_IDX
from traffic_ai.controllers.base import BaseController
from traffic_ai.controllers.rl_controller import RLPolicyController
from traffic_ai.controllers.fixed_timing import FixedTimingController
from traffic_ai.metrics import simulation_result_to_summary_row

SEED = 42
EPISODES = 2000
SIM_STEPS = 500


# ---------------------------------------------------------------------------
# Instrument the training env to record per-step action choices
# ---------------------------------------------------------------------------

class InstrumentedEnv(SignalControlEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_counts: dict[int, int] = collections.defaultdict(int)

    def step(self, action: int):  # type: ignore[override]
        self.action_counts[int(action) % 4] += 1
        return super().step(action)


env = InstrumentedEnv(EnvConfig(seed=SEED, step_limit=SIM_STEPS))

print(
    f"Training DQN for {EPISODES} episodes "
    f"(lr=5e-4, gamma=0.99, warmup=1000, grad_clip=1.0, "
    f"reward_scale=3.0, switch_penalty=0.1) ..."
)
policy, reward_history, _ = train_dqn(env, episodes=EPISODES, seed=SEED)

# ---------------------------------------------------------------------------
# Phase distribution during TRAINING
# ---------------------------------------------------------------------------
total_steps = sum(env.action_counts.values())
print("\n--- Phase distribution during training ---")
for idx in range(4):
    name = IDX_TO_PHASE[idx]
    count = env.action_counts.get(idx, 0)
    pct = 100.0 * count / max(total_steps, 1)
    bar = "#" * int(pct / 2)
    print(f"  Phase {idx} ({name:12s}): {count:7d} steps  {pct:5.1f}%  {bar}")
print(f"  Total training steps: {total_steps:,}")

# Check all 4 phases were explored
phases_used = sum(1 for c in env.action_counts.values() if c > 0)
print(f"\n  Phases explored: {phases_used}/4  {'OK' if phases_used == 4 else 'WARNING: some phases never selected'}")

# ---------------------------------------------------------------------------
# Reward convergence
# ---------------------------------------------------------------------------
milestones = [0, EPISODES // 4, EPISODES // 2, 3 * EPISODES // 4, EPISODES - 1]
print("\n--- Reward trend ---")
print("  ep  " + "  ".join(f"ep{i}" for i in milestones))
print("  rwd " + "  ".join(f"{reward_history[i]:+.1f}" for i in milestones))

# ---------------------------------------------------------------------------
# Evaluation: compare DQN vs fixed_timing on training domain
# ---------------------------------------------------------------------------
sim_cfg = SimulatorConfig(steps=SIM_STEPS, intersections=1, seed=SEED)

simulator = TrafficNetworkSimulator(sim_cfg)
dqn_ctrl = RLPolicyController(policy=policy, name="rl_dqn", min_green=6)
result_dqn = simulator.run(dqn_ctrl, steps=SIM_STEPS)
dqn_wait = simulation_result_to_summary_row(result_dqn)["average_wait_time"]

simulator2 = TrafficNetworkSimulator(sim_cfg)
result_fixed = simulator2.run(FixedTimingController(), steps=SIM_STEPS)
fixed_wait = simulation_result_to_summary_row(result_fixed)["average_wait_time"]

# ---------------------------------------------------------------------------
# Phase distribution during EVALUATION
# ---------------------------------------------------------------------------
class PhaseCounting(RLPolicyController):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.eval_counts: dict[int, int] = collections.defaultdict(int)

    def compute_actions(self, observations, step):
        result = super().compute_actions(observations, step)
        for phase in result.values():
            self.eval_counts[PHASE_TO_IDX.get(phase, 0)] += 1
        return result

simulator3 = TrafficNetworkSimulator(sim_cfg)
counting_ctrl = PhaseCounting(policy=policy, name="rl_dqn", min_green=6)
res3 = simulator3.run(counting_ctrl, steps=SIM_STEPS)
eval_wait = simulation_result_to_summary_row(res3)["average_wait_time"]
eval_total = sum(counting_ctrl.eval_counts.values())

print("\n--- Phase distribution during evaluation (1-int 500 steps) ---")
for idx in range(4):
    name = IDX_TO_PHASE[idx]
    count = counting_ctrl.eval_counts.get(idx, 0)
    pct = 100.0 * count / max(eval_total, 1)
    bar = "#" * int(pct / 2)
    print(f"  Phase {idx} ({name:12s}): {count:5d} steps  {pct:5.1f}%  {bar}")

# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
print(f"\nfixed_timing  avg_wait = {fixed_wait:.2f}s")
print(f"DQN ({EPISODES} ep)  avg_wait = {dqn_wait:.2f}s")
if dqn_wait < fixed_wait:
    print(
        f"PASS: DQN beats fixed_timing by {fixed_wait - dqn_wait:.2f}s "
        f"({(fixed_wait - dqn_wait) / fixed_wait * 100:.1f}% reduction)"
    )
else:
    print(f"FAIL: DQN still worse than fixed_timing by {dqn_wait - fixed_wait:.2f}s")
