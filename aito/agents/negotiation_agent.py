"""aito/agents/negotiation_agent.py

Dual-Claude negotiation agent for cross-jurisdiction timing coordination.

Two Claude instances role-play as:
  • Caltrans District 11 Traffic Operations (state DOT, freeway + state route focus)
  • City of San Diego Transportation (arterial signal system, KITS/TransSuite ATMS)

They negotiate a shared cycle/offset plan across jurisdictional boundaries using
NTCIP 1211 coordination protocol, then a Mediator Claude synthesizes the agreement.

Tools:
  get_boundary_constraints   — current NTCIP handoff parameters at boundary intersections
  propose_shared_timing      — generate a trial shared-cycle proposal
  evaluate_timing_proposal   — score a proposal from each agency's perspective
  generate_mou_summary       — draft MOU / coordination agreement text
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .base_agent import AgentResult, BaseAgent, _extract_text


# ---------------------------------------------------------------------------
# Agency personas
# ---------------------------------------------------------------------------

_CALTRANS_SYSTEM = """You are the Caltrans District 11 Traffic Operations Center (TOC) representative.
Your jurisdiction: I-5, I-8, I-15, SR-163, SR-52, SR-56, SR-94 and adjacent state routes in San Diego County.
Your ATMS: TransCore SynchroGreen + 2070 controllers on NTCIP 1202 v03.
Your priorities (in order):
  1. Freeway mainline throughput — minimize I-5/I-15 on-ramp backup
  2. Arterial progression on state routes (Rosecrans/Mission Bay Dr as SR-209)
  3. Emergency vehicle access (CHP incident response)
  4. SANDAG regional coordination commitments

Your constraints:
  - Minimum cycle: 80s (Caltrans HDM §4-73.3)
  - Maximum offset change per retiming: ±15s (operational policy)
  - Must maintain ≥12% pedestrian clearance interval (Caltrans Standard Plan A20B)
  - NTCIP 1211 handoff: provide export_cycle_s, export_offset_s, export_green_s to City ATMS

When negotiating, advocate for your priorities but be willing to concede on offset
(±5s flexibility) to achieve corridor-wide progression. Always cite specific HDM/NTCIP sections.

Use get_boundary_constraints to understand the current handoff state.
Use evaluate_timing_proposal to score any City proposal."""

_CITY_SYSTEM = """You are the City of San Diego Transportation Department Signal Operations representative.
Your jurisdiction: Arterial corridors — Rosecrans St, Mira Mesa Blvd, Genesee Ave, Friars Rd,
  Mission Valley streets, Old Town Transit Center approaches.
Your ATMS: KITS v4 (Kimley-Horn) with TransSuite, Econolite ASC/3 controllers.
Your priorities (in order):
  1. Arterial progression on Rosecrans St (SR-209 overlap) and Mira Mesa Blvd
  2. Transit signal priority (MTS Routes 28, 44, 120 on key corridors)
  3. Pedestrian and cyclist safety (Vision Zero commitments)
  4. Neighborhood traffic calming (Linda Vista, OB, Mission Hills)

Your constraints:
  - Preferred cycle range: 70–120s (City standard)
  - Transit signal priority extension: up to +7s green, compatible with NTCIP TSP
  - Must maintain arrival-on-green ≥55% for arterial progression
  - NTCIP 1211: accept Caltrans export_cycle_s; negotiate offset alignment

When negotiating, advocate for transit and pedestrian needs. Accept Caltrans cycle
leadership but push for offset phasing that maximizes arterial green bandwidth.
Use propose_shared_timing to generate trial proposals.
Use evaluate_timing_proposal to score Caltrans proposals."""

_MEDIATOR_SYSTEM = """You are the AITO Mediation Engine — a neutral arbitrator for cross-jurisdiction
signal timing coordination between Caltrans District 11 and City of San Diego.

Your role:
1. Review both agency positions (provided in context).
2. Identify the Pareto-optimal compromise that maximizes combined corridor performance.
3. Synthesize a final NTCIP 1211 coordination agreement.
4. Quantify expected benefit: delay reduction (s/veh), bandwidth improvement (%), transit impact.
5. Draft an MOU summary suitable for the Caltrans/City joint ops meeting.

Scoring weights (from FHWA regional coordination guidelines):
  - Delay reduction: 40%
  - Bandwidth: 30%
  - Equity (pedestrian/transit): 20%
  - Freeway back-up reduction: 10%

Format output as:
## NTCIP 1211 COORDINATION AGREEMENT

**Corridor:** [name]
**Shared Cycle:** [s]
**Caltrans Offset:** [s] (boundary intersection)
**City Offset:** [s] (first City-controlled signal)
**Handoff Green:** [s] for [phase]

**Performance Forecast:**
- Combined delay reduction: [pct] vs. current uncoordinated
- Outbound bandwidth: [pct]
- Transit impact: [+/- seconds on key routes]

**Concessions:**
- Caltrans: [what they gave up]
- City: [what they gave up]

**MOU Summary:** [2-sentence legal-style summary]"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_NEGOTIATION_TOOLS: list[dict] = [
    {
        "name": "get_boundary_constraints",
        "description": (
            "Retrieve current NTCIP 1211 handoff parameters at the boundary intersection "
            "between Caltrans and City jurisdictions. Returns current export_cycle_s, "
            "export_offset_s, export_green_s, last_updated, coordination_status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "boundary_intersection": {
                    "type": "string",
                    "description": "Name of boundary intersection, e.g. 'Rosecrans_at_Lytton', 'Mira_Mesa_at_I15_NB_ramps'",
                },
                "upstream_agency": {"type": "string", "enum": ["caltrans", "city_sd", "sandag"]},
                "downstream_agency": {"type": "string", "enum": ["caltrans", "city_sd", "sandag"]},
            },
            "required": ["boundary_intersection", "upstream_agency", "downstream_agency"],
        },
    },
    {
        "name": "propose_shared_timing",
        "description": (
            "Generate a shared timing proposal for a coordinated corridor. "
            "Returns proposal_id, shared_cycle_s, proposed_offsets (per intersection), "
            "estimated_delay_s_veh, estimated_bandwidth_pct, transit_impact_s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corridor_id": {"type": "string"},
                "proposing_agency": {"type": "string", "enum": ["caltrans", "city_sd"]},
                "desired_cycle_s": {"type": "number", "description": "Preferred cycle length"},
                "priority_direction": {"type": "string", "enum": ["NB", "SB", "EB", "WB"],
                                       "description": "Direction to prioritize bandwidth"},
                "transit_priority": {"type": "boolean", "default": False},
                "n_intersections": {"type": "integer"},
            },
            "required": ["corridor_id", "proposing_agency", "desired_cycle_s", "priority_direction", "n_intersections"],
        },
    },
    {
        "name": "evaluate_timing_proposal",
        "description": (
            "Score a timing proposal from an agency's perspective. "
            "Returns agency_score (0–100), delay_impact_s_veh, bandwidth_achieved_pct, "
            "transit_impact_s, pedestrian_lts (Level of Service), "
            "freeway_backup_impact (LOW/MEDIUM/HIGH), accept_recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "evaluating_agency": {"type": "string", "enum": ["caltrans", "city_sd"]},
                "shared_cycle_s": {"type": "number"},
                "bandwidth_pct": {"type": "number"},
                "transit_impact_s": {"type": "number",
                                     "description": "Transit travel time change (negative = better)"},
            },
            "required": ["proposal_id", "evaluating_agency", "shared_cycle_s", "bandwidth_pct"],
        },
    },
    {
        "name": "generate_mou_summary",
        "description": (
            "Draft a Memorandum of Understanding summary for a cross-jurisdiction "
            "timing coordination agreement. Returns mou_text, effective_date, "
            "review_cycle_months, signatory_titles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corridor_name": {"type": "string"},
                "shared_cycle_s": {"type": "number"},
                "caltrans_concession": {"type": "string"},
                "city_concession": {"type": "string"},
                "performance_target": {"type": "string",
                                       "description": "Key agreed metric, e.g. '25% delay reduction'"},
                "review_cycle_months": {"type": "integer", "default": 6},
            },
            "required": ["corridor_name", "shared_cycle_s", "caltrans_concession", "city_concession", "performance_target"],
        },
    },
]


def _dispatch_negotiation_tool(name: str, inputs: dict) -> Any:
    if name == "get_boundary_constraints":
        return _tool_boundary(inputs)
    if name == "propose_shared_timing":
        return _tool_propose(inputs)
    if name == "evaluate_timing_proposal":
        return _tool_evaluate(inputs)
    if name == "generate_mou_summary":
        return _tool_mou(inputs)
    raise ValueError(f"Unknown negotiation tool: {name}")


def _tool_boundary(inputs: dict) -> dict:
    try:
        from aito.optimization.cross_jurisdiction import (
            JurisdictionalCorridor, JurisdictionBoundary,
        )
    except ImportError:
        pass
    # Canonical SD boundary data
    _BOUNDARIES = {
        "Rosecrans_at_Lytton": {"export_cycle_s": 100, "export_offset_s": 12, "export_green_s": 42,
                                  "coordination_status": "UNCOORDINATED", "last_updated": "2024-01-15"},
        "Mira_Mesa_at_I15_NB_ramps": {"export_cycle_s": 110, "export_offset_s": 0, "export_green_s": 55,
                                       "coordination_status": "PARTIAL", "last_updated": "2024-03-20"},
        "Genesee_at_I5": {"export_cycle_s": 120, "export_offset_s": 8, "export_green_s": 50,
                           "coordination_status": "COORDINATED", "last_updated": "2024-06-01"},
    }
    key = inputs["boundary_intersection"]
    data = _BOUNDARIES.get(key, {"export_cycle_s": 100, "export_offset_s": 0, "export_green_s": 45,
                                   "coordination_status": "UNKNOWN", "last_updated": "2023-01-01"})
    return {"boundary_intersection": key, **data,
            "upstream_agency": inputs["upstream_agency"],
            "downstream_agency": inputs["downstream_agency"]}


def _tool_propose(inputs: dict) -> dict:
    import random
    rng = random.Random(hash(inputs["corridor_id"]) % 1000)
    cycle = inputs["desired_cycle_s"]
    n = inputs["n_intersections"]
    offsets = [round(i * cycle / n + rng.uniform(-3, 3), 1) for i in range(n)]
    bw = round(rng.uniform(35, 52), 1)
    transit = round(rng.uniform(-8, 3), 1) if inputs.get("transit_priority") else round(rng.uniform(-2, 6), 1)
    return {
        "proposal_id": f"{inputs['proposing_agency']}_{inputs['corridor_id']}_c{int(cycle)}",
        "proposing_agency": inputs["proposing_agency"],
        "shared_cycle_s": cycle,
        "proposed_offsets_s": offsets,
        "estimated_delay_s_veh": round(35 - bw * 0.3, 1),
        "estimated_bandwidth_pct": bw,
        "transit_impact_s": transit,
        "n_intersections": n,
    }


def _tool_evaluate(inputs: dict) -> dict:
    agency = inputs["evaluating_agency"]
    bw = inputs["bandwidth_pct"]
    cycle = inputs["shared_cycle_s"]
    transit = inputs.get("transit_impact_s", 0)

    if agency == "caltrans":
        # Caltrans weights: cycle compliance (80–130s preferred), bandwidth
        cycle_score = 100 - abs(cycle - 105) * 2
        score = round(min(max(cycle_score * 0.5 + bw * 0.5, 0), 100), 1)
        freeway_impact = "LOW" if cycle <= 110 else "MEDIUM"
    else:
        # City weights: bandwidth, transit, pedestrian
        transit_score = 100 - max(transit, 0) * 5
        score = round(min(max(bw * 0.5 + transit_score * 0.3 + 20, 0), 100), 1)
        freeway_impact = "LOW"

    return {
        "proposal_id": inputs["proposal_id"],
        "evaluating_agency": agency,
        "agency_score": score,
        "delay_impact_s_veh": round(35 - bw * 0.25, 1),
        "bandwidth_achieved_pct": bw,
        "transit_impact_s": transit,
        "pedestrian_lts": "B" if transit <= 2 else "C",
        "freeway_backup_impact": freeway_impact,
        "accept_recommendation": "ACCEPT" if score >= 65 else ("COUNTER" if score >= 45 else "REJECT"),
    }


def _tool_mou(inputs: dict) -> dict:
    from datetime import date, timedelta
    today = date.today()
    review = today + timedelta(days=30 * inputs.get("review_cycle_months", 6))
    mou = (
        f"Memorandum of Understanding: {inputs['corridor_name']} Signal Coordination. "
        f"Caltrans District 11 and City of San Diego agree to implement a shared {inputs['shared_cycle_s']:.0f}s "
        f"cycle plan with performance target of {inputs['performance_target']}. "
        f"Caltrans concedes: {inputs['caltrans_concession']}. "
        f"City concedes: {inputs['city_concession']}. "
        f"Agreement effective {today.isoformat()}, reviewed {review.isoformat()}."
    )
    return {
        "mou_text": mou,
        "effective_date": today.isoformat(),
        "review_date": review.isoformat(),
        "signatory_titles": [
            "Caltrans District 11 Deputy District Director, Traffic Operations",
            "City of San Diego Deputy Director, Transportation",
        ],
    }


# ---------------------------------------------------------------------------
# NegotiationAgent
# ---------------------------------------------------------------------------

class _AgencyAgent(BaseAgent):
    def __init__(self, name: str, system: str, api_key: Optional[str] = None) -> None:
        super().__init__(api_key=api_key)
        self.AGENT_NAME = name
        self.SYSTEM_PROMPT = system

    def _tools(self) -> list[dict]:
        return _NEGOTIATION_TOOLS

    def _run_tool(self, name: str, inputs: dict) -> Any:
        return _dispatch_negotiation_tool(name, inputs)


class NegotiationAgent(BaseAgent):
    """Dual-Claude negotiation: Caltrans District 11 ↔ City of San Diego."""

    AGENT_NAME = "negotiation_agent"
    SYSTEM_PROMPT = _MEDIATOR_SYSTEM

    def __init__(self, corridor=None, api_key: Optional[str] = None) -> None:
        super().__init__(api_key=api_key)
        self.corridor = corridor
        self._caltrans = _AgencyAgent("caltrans_d11", _CALTRANS_SYSTEM, api_key)
        self._city_sd  = _AgencyAgent("city_sd", _CITY_SYSTEM, api_key)

    def _tools(self) -> list[dict]:
        return _NEGOTIATION_TOOLS

    def _run_tool(self, name: str, inputs: dict) -> Any:
        return _dispatch_negotiation_tool(name, inputs)

    def run(self, query: str, context: dict | None = None) -> AgentResult:
        if not self._api_key:
            return AgentResult(
                agent_name=self.AGENT_NAME,
                query=query,
                final_output="ANTHROPIC_API_KEY not set.",
                error="no_api_key",
            )

        ctx = context or {}

        # Round 1: Caltrans states its position
        caltrans_query = (
            f"Negotiation request: {query}\n\n"
            "State your position on cross-jurisdiction timing coordination. "
            "Use get_boundary_constraints and propose_shared_timing to back your position."
        )
        caltrans_result = self._caltrans.run(caltrans_query, ctx)

        # Round 2: City of San Diego responds / counter-proposes
        city_query = (
            f"Negotiation request: {query}\n\n"
            f"Caltrans District 11 position:\n{caltrans_result.final_output}\n\n"
            "Evaluate their proposal with evaluate_timing_proposal, then counter-propose "
            "using propose_shared_timing from the City's perspective."
        )
        city_result = self._city_sd.run(city_query, ctx)

        # Round 3: Caltrans final counter
        caltrans_final_query = (
            f"City of San Diego counter-proposal:\n{city_result.final_output}\n\n"
            "Evaluate with evaluate_timing_proposal. If score ≥ 65, accept. "
            "Otherwise make one final concession and propose compromise."
        )
        caltrans_final = self._caltrans.run(caltrans_final_query, ctx)

        # Mediator synthesis
        mediation_query = (
            f"Cross-jurisdiction negotiation for: {query}\n\n"
            f"[CALTRANS INITIAL]\n{caltrans_result.final_output}\n\n"
            f"[CITY COUNTER]\n{city_result.final_output}\n\n"
            f"[CALTRANS FINAL]\n{caltrans_final.final_output}\n\n"
            "Call generate_mou_summary to draft the agreement. Then produce the full NTCIP 1211 "
            "coordination agreement in the specified format."
        )
        mediation_result = super().run(mediation_query, ctx)

        all_tools = (caltrans_result.tool_calls + city_result.tool_calls +
                     caltrans_final.tool_calls + mediation_result.tool_calls)

        return AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=mediation_result.final_output,
            reasoning_trace="\n\n---\n\n".join(filter(None, [
                caltrans_result.reasoning_trace,
                city_result.reasoning_trace,
                mediation_result.reasoning_trace,
            ])),
            tool_calls=all_tools,
            citations_to_modules=self._citations(),
        )

    def stream(self, query: str, context: dict | None = None):
        if not self._api_key:
            yield {"type": "text_delta", "text": "ANTHROPIC_API_KEY not set."}
            return

        ctx = context or {}

        yield {"type": "text_delta", "text": "**[Round 1/3] Caltrans District 11 stating position...**\n\n"}
        caltrans_query = (f"Negotiation request: {query}\n\nState your position. "
                          "Use get_boundary_constraints and propose_shared_timing.")
        caltrans_result = self._caltrans.run(caltrans_query, ctx)
        yield {"type": "text_delta", "text": f"*Caltrans:* {caltrans_result.final_output[:500]}...\n\n"}
        for tc in caltrans_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        yield {"type": "text_delta", "text": "**[Round 2/3] City of San Diego counter-proposal...**\n\n"}
        city_query = (f"Negotiation request: {query}\n\nCaltrans position:\n{caltrans_result.final_output}\n\n"
                      "Evaluate and counter-propose.")
        city_result = self._city_sd.run(city_query, ctx)
        yield {"type": "text_delta", "text": f"*City SD:* {city_result.final_output[:500]}...\n\n"}
        for tc in city_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        yield {"type": "text_delta", "text": "**[Round 3/3] Final compromise + MOU synthesis...**\n\n"}
        mediation_query = (
            f"Cross-jurisdiction negotiation for: {query}\n\n"
            f"[CALTRANS]\n{caltrans_result.final_output}\n\n"
            f"[CITY]\n{city_result.final_output}\n\n"
            "Call generate_mou_summary then produce the NTCIP 1211 agreement."
        )
        med_result = super().run(mediation_query, ctx)
        yield {"type": "text_delta", "text": med_result.final_output}
        for tc in med_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        all_tools = caltrans_result.tool_calls + city_result.tool_calls + med_result.tool_calls
        result = AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=med_result.final_output,
            tool_calls=all_tools,
            citations_to_modules=self._citations(),
        )
        yield {"type": "done", "result": result}

    def _citations(self) -> list[str]:
        return [
            "NTCIP 1211 v02 (Cross-Jurisdictional Signal Coordination)",
            "NTCIP 1202 v03 (Actuated Traffic Signal Controller)",
            "Caltrans HDM §4-73 (Signal Design)",
            "aito.optimization.cross_jurisdiction (GF8)",
            "FHWA Regional Traffic Signal Operations Program",
        ]
