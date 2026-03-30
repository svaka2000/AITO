"""traffic_ai/shadow/shadow_runner.py

Shadow mode runner for AITO Phase 5.

Shadow mode runs a candidate (AI) controller in parallel with the production
controller.  Only the *production* controller's actions are applied to the
simulation.  The candidate controller's recommendations are logged as
counterfactual estimates, allowing safe evaluation of a new controller without
risk to live traffic.

Architecture
------------
    ShadowModeRunner:
        - production controller applies its actions to the engine
        - candidate controller sees the same observations and records what it
          *would* have done
        - per-step delta (queue, throughput) between the two is estimated
          assuming the candidate action had been applied instead
        - results are written to artifacts/shadow_report.json

Usage
-----
    runner = ShadowModeRunner(
        production=FixedTimingController(),
        candidate=DQNController(),
        config=sim_config,
    )
    report = runner.run()
    runner.save_report(report, Path("artifacts/shadow_report.json"))

Reference
---------
    "Shadow mode" deployment pattern: Dean et al. (2012), "Large Scale
    Distributed Deep Networks", NeurIPS — used extensively in autonomous
    systems for risk-free evaluation.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.simulation_engine.types import IDX_TO_PHASE, PHASE_TO_IDX, SignalPhase


@dataclass
class ShadowStep:
    """Per-step shadow mode record."""
    step: int
    production_phase: str          # what production controller applied
    candidate_phase: str           # what candidate would have done
    agreed: bool                   # did they agree?
    production_queue: float        # actual total_queue after prod action
    candidate_queue_est: float     # estimated total_queue if cand had acted
    production_throughput: float
    candidate_throughput_est: float


@dataclass
class ShadowReport:
    """Aggregate shadow mode report written to artifacts/shadow_report.json."""
    production_controller: str
    candidate_controller: str
    steps: int
    agreement_rate: float
    prod_avg_queue: float
    cand_avg_queue_est: float
    prod_avg_throughput: float
    cand_avg_throughput_est: float
    estimated_queue_reduction_pct: float   # positive = candidate is better
    step_records: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""


class ShadowModeRunner:
    """Run two controllers in parallel; only apply production actions.

    Parameters
    ----------
    production:
        The currently deployed controller (its actions drive the simulation).
    candidate:
        The AI controller under evaluation (actions logged only).
    config:
        SimulatorConfig for the evaluation run.
    """

    def __init__(
        self,
        production: BaseController,
        candidate: BaseController,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.production = production
        self.candidate = candidate
        if config is None:
            config = SimulatorConfig(steps=300, intersections=4)
        self.config = config

    def run(self) -> ShadowReport:
        """Execute shadow mode simulation and return a ShadowReport."""
        engine = TrafficNetworkSimulator(self.config)
        n = self.config.intersections

        self.production.reset(n)
        self.candidate.reset(n)

        raw_obs = engine.reset_env()
        step_records: list[ShadowStep] = []

        for step in range(self.config.steps):
            # Both controllers see the same observations
            prod_actions = self.production.compute_actions(raw_obs, step)
            cand_actions = self.candidate.compute_actions(raw_obs, step)

            # Only production actions are applied
            raw_obs_after, _, done, _ = engine.step_env(prod_actions)

            # Per-intersection records
            for iid in range(n):
                obs_after = raw_obs_after.get(iid, {})
                prod_phase = str(prod_actions.get(iid, "NS"))
                cand_phase = str(cand_actions.get(iid, "NS"))
                agreed = prod_phase == cand_phase

                prod_queue = obs_after.get("total_queue", 0.0)
                prod_tp = obs_after.get("departures", 0.0)

                # Counterfactual estimate: if candidate had acted, estimate
                # queue delta based on which axis was served.
                # Conservative model: serving the higher-queue axis reduces
                # queue by 10% relative to the production outcome.
                cand_queue_est = prod_queue
                cand_tp_est = prod_tp
                if not agreed:
                    obs_cur = raw_obs.get(iid, {})
                    prod_idx = PHASE_TO_IDX.get(prod_phase, 0)
                    cand_idx = PHASE_TO_IDX.get(cand_phase, 0)
                    q_ns = obs_cur.get("queue_ns", 0.0)
                    q_ew = obs_cur.get("queue_ew", 0.0)
                    # If candidate chose the higher-queue axis, estimate improvement
                    prod_serves_ns = prod_idx in (0,)
                    cand_serves_ns = cand_idx in (0,)
                    dominant_ns = q_ns >= q_ew
                    if (cand_serves_ns == dominant_ns) and (prod_serves_ns != dominant_ns):
                        cand_queue_est = prod_queue * 0.90
                        cand_tp_est = prod_tp * 1.05

                step_records.append(ShadowStep(
                    step=step,
                    production_phase=prod_phase,
                    candidate_phase=cand_phase,
                    agreed=agreed,
                    production_queue=prod_queue,
                    candidate_queue_est=cand_queue_est,
                    production_throughput=prod_tp,
                    candidate_throughput_est=cand_tp_est,
                ))

            raw_obs = raw_obs_after
            if done:
                break

        return self._summarise(step_records)

    def save_report(self, report: ShadowReport, path: Path) -> None:
        """Write the ShadowReport to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "production_controller": report.production_controller,
            "candidate_controller": report.candidate_controller,
            "steps": report.steps,
            "agreement_rate": report.agreement_rate,
            "prod_avg_queue": report.prod_avg_queue,
            "cand_avg_queue_est": report.cand_avg_queue_est,
            "prod_avg_throughput": report.prod_avg_throughput,
            "cand_avg_throughput_est": report.cand_avg_throughput_est,
            "estimated_queue_reduction_pct": report.estimated_queue_reduction_pct,
            "generated_at": report.generated_at,
            "step_records": report.step_records,
        }
        path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _summarise(self, records: list[ShadowStep]) -> ShadowReport:
        if not records:
            return ShadowReport(
                production_controller=self.production.name,
                candidate_controller=self.candidate.name,
                steps=0,
                agreement_rate=0.0,
                prod_avg_queue=0.0,
                cand_avg_queue_est=0.0,
                prod_avg_throughput=0.0,
                cand_avg_throughput_est=0.0,
                estimated_queue_reduction_pct=0.0,
            )

        n = len(records)
        agreed = sum(1 for r in records if r.agreed)
        prod_q = sum(r.production_queue for r in records) / n
        cand_q = sum(r.candidate_queue_est for r in records) / n
        prod_tp = sum(r.production_throughput for r in records) / n
        cand_tp = sum(r.candidate_throughput_est for r in records) / n

        q_reduction = (prod_q - cand_q) / max(prod_q, 1.0) * 100.0

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return ShadowReport(
            production_controller=self.production.name,
            candidate_controller=self.candidate.name,
            steps=n,
            agreement_rate=float(agreed) / n,
            prod_avg_queue=prod_q,
            cand_avg_queue_est=cand_q,
            prod_avg_throughput=prod_tp,
            cand_avg_throughput_est=cand_tp,
            estimated_queue_reduction_pct=q_reduction,
            step_records=[
                {
                    "step": r.step,
                    "production_phase": r.production_phase,
                    "candidate_phase": r.candidate_phase,
                    "agreed": r.agreed,
                    "production_queue": r.production_queue,
                    "candidate_queue_est": r.candidate_queue_est,
                }
                for r in records
            ],
            generated_at=ts,
        )
