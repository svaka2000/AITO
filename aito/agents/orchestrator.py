"""aito/agents/orchestrator.py

Intent classification + agent delegation for AITO.

Routes incoming queries to one of four specialist agents:
  - scenario_agent  : what-if, spillback, event demand, timing analysis
  - incident_agent  : incident detection, rerouting, advisor chain
  - negotiation_agent: cross-jurisdiction Caltrans / City coordination
  - carbon_agent    : emissions accounting, carbon credit portfolio

For queries that don't match a specialist, the orchestrator answers
directly using its own Claude call with prompt caching on the
module-doc system prompt.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator, Optional

from .base_agent import AgentResult, BaseAgent, _extract_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class Intent:
    SCENARIO    = "scenario"
    INCIDENT    = "incident"
    NEGOTIATION = "negotiation"
    CARBON      = "carbon"
    GENERAL     = "general"


_INTENT_KEYWORDS: dict[str, list[str]] = {
    Intent.SCENARIO: [
        "what if", "what happens", "simulate", "scenario", "spillback",
        "queue", "cycle", "green time", "offset", "timing plan", "event",
        "petco", "pechanga", "sdsu", "snapdragon", "concert", "game day",
        "demand", "closure", "detour", "retiming", "bandwidth", "delay",
    ],
    Intent.INCIDENT: [
        "incident", "crash", "accident", "disabled vehicle", "breakdown",
        "signal out", "signal failure", "power outage", "flooding", "fire", "emergency",
        "closure", "lane blocked", "pd on scene", "chp", "first responder",
        "reroute", "alternate route", "divert",
    ],
    Intent.NEGOTIATION: [
        "caltrans", "city of san diego", "sandag", "jurisdiction",
        "coordinate", "ntcip", "boundary", "district 11", "hand-off",
        "cross-agency", "mts", "nctd", "agency",
    ],
    Intent.CARBON: [
        "carbon", "co2", "emissions", "greenhouse", "climate",
        "verra", "vcs", "gold standard", "lcfs", "carb", "credit",
        "tonne", "ton", "epa", "moves", "fuel", "electric", "ev",
        "sustainability", "net zero",
    ],
}


def classify_intent(query: str) -> str:
    q = query.lower()
    scores: dict[str, int] = {k: 0 for k in _INTENT_KEYWORDS}
    for intent, kws in _INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else Intent.GENERAL


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OrchestratorAgent(BaseAgent):
    """Routes queries to specialist agents and merges their outputs."""

    AGENT_NAME = "orchestrator"

    SYSTEM_PROMPT = """You are the AITO Orchestrator — the primary interface between traffic engineers
and the AI Traffic Optimization platform deployed in San Diego, CA.

Your role:
1. Understand the engineer's request in the context of real San Diego corridors:
   • I-5 / I-15 freeway mainline (Caltrans District 11)
   • Rosecrans Street corridor (City of San Diego)
   • Mira Mesa Boulevard (City of San Diego)
   • Genesee Avenue (City of San Diego / UCSD area)
   • Mission Valley (cross-jurisdictional SANDAG/City)

2. Delegate to specialist agents when appropriate (scenario, incident, negotiation, carbon).

3. For general traffic engineering questions, answer directly citing:
   HCM 7th Edition, MUTCD 2023, FHWA Signal Timing Manual, NTCIP 1211.

4. Always distinguish between:
   • Confirmed findings from AITO optimization runs
   • Real-time probe data inferences
   • Engineering estimates

5. Format all numeric results with units: s/veh, kg CO₂/hr, veh/hr, m, %.

Tone: direct, technically precise, professional — as if briefing a Caltrans District 11
traffic operations engineer."""

    def __init__(
        self,
        corridor=None,
        optimization_result=None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(api_key=api_key)
        self.corridor = corridor
        self.optimization_result = optimization_result
        self._specialist_cache: dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Specialist agent factory
    # ------------------------------------------------------------------

    def _get_agent(self, intent: str) -> BaseAgent | None:
        if intent in self._specialist_cache:
            return self._specialist_cache[intent]

        agent: BaseAgent | None = None
        try:
            if intent == Intent.SCENARIO:
                from .scenario_agent import ScenarioAgent
                agent = ScenarioAgent(
                    corridor=self.corridor,
                    optimization_result=self.optimization_result,
                    api_key=self._api_key,
                )
            elif intent == Intent.INCIDENT:
                from .incident_agent import IncidentAgent
                agent = IncidentAgent(
                    corridor=self.corridor,
                    api_key=self._api_key,
                )
            elif intent == Intent.NEGOTIATION:
                from .negotiation_agent import NegotiationAgent
                agent = NegotiationAgent(
                    corridor=self.corridor,
                    api_key=self._api_key,
                )
            elif intent == Intent.CARBON:
                from .carbon_agent import CarbonAgent
                agent = CarbonAgent(
                    corridor=self.corridor,
                    api_key=self._api_key,
                )
        except ImportError as exc:
            logger.warning("Could not import specialist agent: %s", exc)

        if agent:
            self._specialist_cache[intent] = agent
        return agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> AgentResult:
        """Classify intent and run the appropriate specialist (or self)."""
        intent = classify_intent(query)
        context = self._build_corridor_context()

        specialist = self._get_agent(intent)
        if specialist and intent != Intent.GENERAL:
            result = specialist.run(query, context)
            result.agent_name = f"{intent}_agent (via orchestrator)"
            return result

        return self.run(query, context)

    def stream_route(self, query: str) -> Generator[dict, None, AgentResult]:
        """Streaming version of route() for the Streamlit UI."""
        intent = classify_intent(query)
        context = self._build_corridor_context()

        yield {"type": "routing", "intent": intent}

        specialist = self._get_agent(intent)
        if specialist and intent != Intent.GENERAL:
            yield from specialist.stream(query, context)
            return

        yield from self.stream(query, context)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_corridor_context(self) -> dict:
        ctx: dict[str, Any] = {}
        if self.corridor:
            ctx["corridor_name"] = self.corridor.name
            ctx["intersections"] = len(self.corridor.intersections)
            ctx["aadt"] = getattr(self.corridor, "aadt", None)
            ctx["speed_limits_mph"] = getattr(self.corridor, "speed_limits_mph", None)

        if self.optimization_result:
            rec = self.optimization_result.recommended_solution
            ctx["optimized_cycle_s"] = round(rec.plan.cycle_length, 1)
            ctx["delay_s_veh"] = round(rec.delay_score, 1)
            ctx["emissions_kg_hr"] = round(rec.emissions_score, 1)
            ctx["pareto_solutions"] = len(self.optimization_result.pareto_solutions)

        return ctx

    def set_corridor(self, corridor) -> None:
        self.corridor = corridor
        self._specialist_cache.clear()

    def set_optimization_result(self, result) -> None:
        self.optimization_result = result
        self._specialist_cache.clear()
