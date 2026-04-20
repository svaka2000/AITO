"""aito/interface/nl_engineer.py

GF10: Natural Language Interface for Traffic Engineers.

Entry point for all engineer queries. Routes through the multi-agent
orchestrator (aito/agents/orchestrator.py) which delegates to:
  - ScenarioAgent  : what-if / spillback / event demand / retiming
  - IncidentAgent  : 3-sub-agent incident response chain
  - NegotiationAgent: dual-Claude Caltrans ↔ City of San Diego coordination
  - CarbonAgent    : EPA MOVES + Verra VCS / CARB LCFS credit portfolio

Without ANTHROPIC_API_KEY: falls back to structured template responses (no billing).

Usage:
    session = NLEngineerSession(corridor=rosecrans, anthropic_api_key="...")
    response = session.ask("Why is PM peak cycle so long at Midway?")
    response2 = session.ask("What if we reduce cycle to 100s?")
    for event in session.stream("Negotiate Caltrans timing on Rosecrans"):
        print(event)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Generator, Optional


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class NLResponse:
    query: str
    answer: str
    agent_name: str = "nl_engineer"
    reasoning_trace: str = ""
    tool_calls: list = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    followup_suggestions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    used_claude_api: bool = False


# ---------------------------------------------------------------------------
# Query classifier (template-mode fallback — no API key needed)
# ---------------------------------------------------------------------------

class QueryType:
    EXPLAIN_TIMING   = "explain_timing"
    WHAT_IF          = "what_if"
    COMPARE          = "compare"
    OPTIMIZE         = "optimize"
    VALIDATE         = "validate"
    CARBON           = "carbon"
    GENERAL          = "general"


_QUERY_KEYWORDS: dict[str, list[str]] = {
    QueryType.EXPLAIN_TIMING: [
        "why", "explain", "what is", "what does", "cycle", "offset", "split",
        "green", "yellow", "red", "phase", "bandwidth"
    ],
    QueryType.WHAT_IF: [
        "what if", "what happens", "if i", "increase", "decrease", "reduce",
        "raise", "lower", "change", "adjust"
    ],
    QueryType.COMPARE: [
        "compare", "vs", "versus", "better than", "worse than", "insync",
        "scoot", "surtrac", "improvement", "difference"
    ],
    QueryType.CARBON: [
        "carbon", "co2", "emissions", "climate", "credits", "tonnes",
        "greenhouse", "reduce emissions"
    ],
    QueryType.VALIDATE: [
        "valid", "compliant", "mutcd", "hcm", "ite", "violation", "error",
        "issue", "problem", "wrong"
    ],
    QueryType.OPTIMIZE: [
        "optimize", "optimise", "best", "improve", "recommend", "suggest",
        "plan for", "timing for"
    ],
}


def classify_query(query: str) -> str:
    q = query.lower()
    scores: dict[str, int] = {qt: 0 for qt in _QUERY_KEYWORDS}
    for qt, keywords in _QUERY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[qt] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else QueryType.GENERAL


# ---------------------------------------------------------------------------
# Template responses (no API key)
# ---------------------------------------------------------------------------

def _explain_timing_plan_template(corridor_name: str, optimization_result) -> str:
    if optimization_result is None:
        return f"No optimization result available for {corridor_name} yet. Run an optimization first."
    rec = optimization_result.recommended_solution
    plan = rec.plan
    lines = [
        f"**{corridor_name} — Optimization Summary**\n",
        f"Common cycle: **{plan.cycle_length:.0f} seconds**",
        f"Outbound bandwidth: {plan.bandwidth_outbound:.1f}s ({plan.bandwidth_outbound / plan.cycle_length * 100:.1f}%)",
        f"Inbound bandwidth: {plan.bandwidth_inbound:.1f}s ({plan.bandwidth_inbound / plan.cycle_length * 100:.1f}%)",
        f"\nObjective scores:",
        f"  • Delay: {rec.delay_score:.1f} s/veh",
        f"  • Emissions: {rec.emissions_score:.1f} kg CO₂/hr",
        f"  • Stops: {rec.stops_score:.2f} stops/veh",
        f"\n{len(optimization_result.pareto_solutions)} Pareto-optimal solutions found.",
    ]
    return "\n".join(lines)


def _what_if_template(query: str, corridor) -> str:
    return (
        f"To evaluate '{query}', call `session.ask()` with ANTHROPIC_API_KEY set "
        f"for full scenario simulation, or call the ScenarioAgent directly."
    )


def _carbon_template(corridor, reduction_pct: float = 23.5) -> str:
    n = len(corridor.intersections)
    aadt = getattr(corridor, "aadt", 28000)
    est_tonnes = n * aadt * 14.0 * 1.38 * 365 / 1e9 * 1000
    return (
        f"**Carbon Impact — {corridor.name}**\n\n"
        f"Estimated CO₂ reduction: **~{reduction_pct:.0f}%** vs. fixed-time baseline\n"
        f"≈ {est_tonnes:.0f} tonnes CO₂/year across {n} intersections\n\n"
        f"At California LCFS market price (~$65/tonne):\n"
        f"  Estimated annual revenue: **${est_tonnes * 65 * 0.92:,.0f}**\n\n"
        f"*Based on EPA MOVES2014b idle factor (1.38 g CO₂/s) and {aadt:,} AADT.*"
    )


def _compare_template(competitor: str) -> str:
    comparisons = {
        "insync": (
            "**AITO vs. InSync (Rhythm)**\n\n"
            "| Capability | InSync | AITO |\n"
            "|---|---|---|\n"
            "| Loop detector dependency | Required | Optional (probe-data-first) |\n"
            "| Optimization algorithm | Neural Genetic | NSGA-III + MAXBAND |\n"
            "| Objectives | Delay only | Delay + Emissions + Stops + Safety + Equity |\n"
            "| Carbon accounting | None | EPA MOVES2014b certified |\n"
            "| Auto-retiming | Manual call-out | Continuous (GF7) |\n"
            "| Open source | No | Yes (MIT) |\n\n"
            "San Diego benchmark: InSync achieved 25% TT reduction, 53% stop reduction (2017)."
        ),
        "scoot": (
            "**AITO vs. SCOOT (TfL)**\n\n"
            "SCOOT uses real-time loop detector occupancy. "
            "AITO uses probe data (CV trajectories, INRIX) without detector infrastructure. "
            "SCOOT requires detectors at every approach; AITO works at 25%+ CV penetration."
        ),
    }
    for key, text in comparisons.items():
        if key in competitor.lower():
            return text
    return f"Comparison with '{competitor}' not in database. Supported: InSync, SCOOT, SURTRAC."


# ---------------------------------------------------------------------------
# NLEngineerSession
# ---------------------------------------------------------------------------

class NLEngineerSession:
    """Conversational interface routing through the AITO multi-agent orchestrator.

    Modes:
      1. Orchestrator mode (with ANTHROPIC_API_KEY): full Claude Opus 4.7 multi-agent system.
      2. Template mode (no API key): structured rule-based responses, no billing.
    """

    def __init__(
        self,
        corridor=None,
        optimization_result=None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self.corridor = corridor
        self.optimization_result = optimization_result
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._use_claude = self._api_key is not None
        self._orchestrator = None  # lazy init

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from aito.agents.orchestrator import OrchestratorAgent
            self._orchestrator = OrchestratorAgent(
                corridor=self.corridor,
                optimization_result=self.optimization_result,
                api_key=self._api_key,
            )
        return self._orchestrator

    def ask(self, query: str) -> NLResponse:
        """Ask a question and get an engineer-grade response."""
        if self._use_claude:
            return self._ask_orchestrator(query)
        return self._ask_template(query, classify_query(query))

    def stream(self, query: str) -> Generator[dict, None, None]:
        """Stream events for the Streamlit 4-pane UI.

        Yields dicts:
          {"type": "routing", "intent": ...}
          {"type": "thinking_delta", "text": ...}
          {"type": "text_delta", "text": ...}
          {"type": "tool_start", "name": ..., "inputs": ...}
          {"type": "tool_result", "name": ..., "output": ..., "duration_ms": ...}
          {"type": "done", "result": NLResponse}
        """
        if not self._use_claude:
            qt = classify_query(query)
            result = self._ask_template(query, qt)
            yield {"type": "text_delta", "text": result.answer}
            yield {"type": "done", "result": result}
            return

        orchestrator = self._get_orchestrator()
        for event in orchestrator.stream_route(query):
            if event.get("type") == "done":
                agent_result = event["result"]
                yield {
                    "type": "done",
                    "result": NLResponse(
                        query=query,
                        answer=agent_result.final_output,
                        agent_name=agent_result.agent_name,
                        reasoning_trace=agent_result.reasoning_trace,
                        tool_calls=agent_result.tool_calls,
                        citations=agent_result.citations_to_modules,
                        followup_suggestions=self._suggest_followups(classify_query(query)),
                        used_claude_api=True,
                    ),
                }
            else:
                yield event

    def _ask_orchestrator(self, query: str) -> NLResponse:
        try:
            orchestrator = self._get_orchestrator()
            agent_result = orchestrator.route(query)
            return NLResponse(
                query=query,
                answer=agent_result.final_output,
                agent_name=agent_result.agent_name,
                reasoning_trace=agent_result.reasoning_trace,
                tool_calls=agent_result.tool_calls,
                citations=agent_result.citations_to_modules,
                followup_suggestions=self._suggest_followups(classify_query(query)),
                used_claude_api=True,
            )
        except Exception as exc:
            result = self._ask_template(query, classify_query(query))
            result.answer = f"[API error: {exc}]\n\n" + result.answer
            return result

    def _ask_template(self, query: str, query_type: str) -> NLResponse:
        corridor_name = self.corridor.name if self.corridor else "Unknown corridor"
        if query_type == QueryType.EXPLAIN_TIMING:
            answer = _explain_timing_plan_template(corridor_name, self.optimization_result)
        elif query_type == QueryType.WHAT_IF:
            answer = _what_if_template(query, self.corridor)
        elif query_type == QueryType.CARBON:
            answer = _carbon_template(self.corridor) if self.corridor else "No corridor loaded."
        elif query_type == QueryType.COMPARE:
            answer = _compare_template(query)
        elif query_type == QueryType.VALIDATE:
            answer = (
                "Validation runs automatically during optimization. "
                "Check `optimization_result.pareto_solutions[i].plan` for MUTCD/HCM compliance."
            )
        else:
            answer = (
                f"I understand you're asking about: '{query}'\n\n"
                f"For full multi-agent analysis (scenario simulation, incident response, "
                f"carbon accounting, cross-jurisdiction negotiation), set ANTHROPIC_API_KEY.\n"
                f"Current corridor: {corridor_name}"
            )
        return NLResponse(
            query=query,
            answer=answer,
            used_claude_api=False,
            citations=["HCM 7th Edition Ch.19", "MUTCD 2023 §4E", "EPA MOVES2014b"],
            followup_suggestions=self._suggest_followups(query_type),
        )

    def _suggest_followups(self, query_type: str) -> list[str]:
        suggestions = {
            QueryType.EXPLAIN_TIMING: [
                "Why is the cycle length set to this value?",
                "What are the Pareto trade-offs for this corridor?",
                "How does this compare to the existing fixed-time plan?",
            ],
            QueryType.WHAT_IF: [
                "Show me the simulation results for this scenario.",
                "What's the CO₂ impact of this change?",
                "How many Pareto solutions does NSGA-III find?",
            ],
            QueryType.CARBON: [
                "Which carbon credit market pays the most per tonne?",
                "Generate the full Verra VCS MRV report.",
                "What's the CARB LCFS revenue for this corridor?",
            ],
            QueryType.COMPARE: [
                "Show the full Pareto front vs. InSync.",
                "How does AITO handle detector failures differently?",
                "What's the performance at 0% CV penetration?",
            ],
        }
        return suggestions.get(query_type, [
            "Run the NSGA-III optimization for Rosecrans corridor.",
            "Show me the carbon impact.",
            "Negotiate timing with Caltrans District 11.",
        ])

    def set_optimization_result(self, result) -> None:
        self.optimization_result = result
        self._orchestrator = None

    def set_corridor(self, corridor) -> None:
        self.corridor = corridor
        self._orchestrator = None

    @property
    def conversation_turns(self) -> int:
        if self._orchestrator:
            return getattr(self._orchestrator, "_turn_count", 0)
        return 0
