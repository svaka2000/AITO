"""aito/agents/incident_agent.py

Incident response agent using a three-sub-agent sequential chain:

  1. DetectorAgent  — classifies incident severity, identifies impacted approaches
  2. RerouterAgent  — generates alternate route recommendations + timing adjustments
  3. AdvisorAgent   — drafts the operator notification and post-incident retiming plan

Each sub-agent is a focused Claude call with its own tools and system prompt.
The top-level IncidentAgent orchestrates the chain and merges outputs.

Tools:
  assess_spillback_impact  — D/D/1 queue model for incident capacity reduction
  get_alternate_routes     — network resilience module alternate path lookup
  compute_diversion_timing — timing adjustments for diversion routes
  draft_operator_alert     — structured ATMS/511 operator notification
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .base_agent import AgentResult, BaseAgent, _extract_text


# ---------------------------------------------------------------------------
# Sub-agent system prompts
# ---------------------------------------------------------------------------

_DETECTOR_SYSTEM = """You are AITO's Incident Detector — a specialist in rapid incident impact assessment.

Your job:
1. Classify incident severity: MINOR (1 lane blocked), MODERATE (2 lanes blocked or signal failure),
   MAJOR (full closure or multi-intersection impact), CRITICAL (freeway/arterial crossover).
2. Identify all impacted intersection approaches (NB/SB/EB/WB at each intersection).
3. Estimate capacity reduction percentage using the AITO spillback predictor.
4. Determine if spillback will propagate upstream to adjacent intersections.
5. Calculate estimated queue propagation time (minutes until adjacent block fills).

Use assess_spillback_impact tool for each blocked approach.

Output format:
SEVERITY: [MINOR/MODERATE/MAJOR/CRITICAL]
LOCATION: [intersection name or segment]
IMPACTED_APPROACHES: [list]
CAPACITY_REDUCTION_PCT: [number]
QUEUE_PROPAGATION_TIME_MIN: [number]
SPILLBACK_TO_UPSTREAM: [yes/no]"""

_REROUTER_SYSTEM = """You are AITO's Rerouter — a specialist in alternate route planning during incidents.

Given an incident severity assessment, your job:
1. Call get_alternate_routes to identify viable detour paths.
2. For each alternate route, call compute_diversion_timing to recommend timing adjustments.
3. Prioritize routes by: capacity > travel time increase > signal coordination difficulty.
4. Specify exact timing changes: which intersections, which phases, green time increase (seconds).
5. Flag if cross-jurisdiction coordination is required (Caltrans ↔ City of San Diego).

Output format:
PRIMARY_DIVERSION: [route description]
SECONDARY_DIVERSION: [route description]
TIMING_ADJUSTMENTS: [list of intersection → phase → delta_green_s]
CROSS_JURISDICTION: [yes/no + agencies]
ESTIMATED_TRAVEL_TIME_INCREASE_MIN: [number]"""

_ADVISOR_SYSTEM = """You are AITO's Post-Incident Advisor — a specialist in operator notification and recovery planning.

Given incident assessment + diversion routes, your job:
1. Draft a 511/ATMS operator alert (≤ 280 characters for broadcast).
2. Write a post-incident retiming recommendation (when to restore normal plan).
3. Specify monitoring KPIs: queue clearance threshold (veh/hr), delay recovery target (s/veh).
4. Recommend whether a full NSGA-III retiming should be triggered after clearance.
5. Estimate time to normal operations (minutes).

Output must include:
OPERATOR_ALERT: [511-ready text ≤ 280 chars]
RECOVERY_TRIGGER: [condition to restore timing + ETA minutes]
POST_INCIDENT_RETIMING: [OFFSET_ONLY / CYCLE_ADJUST / NSGA_RETIMING]
MONITORING_KPIS: [delay target s/veh, queue clearance veh/hr]"""


# ---------------------------------------------------------------------------
# Shared tools for all three sub-agents
# ---------------------------------------------------------------------------

_INCIDENT_TOOLS: list[dict] = [
    {
        "name": "assess_spillback_impact",
        "description": (
            "Assess queue spillback impact at an approach during an incident "
            "with reduced capacity. Returns queue_length_m, capacity_remaining_veh_hr, "
            "spillback_risk, time_to_fill_min."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "approach": {"type": "string", "description": "Approach direction: NB/SB/EB/WB"},
                "volume_veh_hr": {"type": "number", "description": "Demand volume (veh/hr)"},
                "original_capacity_veh_hr": {"type": "number", "description": "Pre-incident capacity (veh/hr)"},
                "lanes_blocked": {"type": "integer", "description": "Number of lanes blocked by incident"},
                "total_lanes": {"type": "integer", "description": "Total approach lanes", "default": 2},
                "cycle_s": {"type": "number", "description": "Current cycle length (s)"},
                "green_s": {"type": "number", "description": "Current green time (s)"},
                "block_length_m": {"type": "number", "description": "Available storage (m)", "default": 150.0},
            },
            "required": ["approach", "volume_veh_hr", "original_capacity_veh_hr", "lanes_blocked", "cycle_s", "green_s"],
        },
    },
    {
        "name": "get_alternate_routes",
        "description": (
            "Look up alternate routes for a blocked corridor using the AITO "
            "network resilience module. Returns route_name, added_distance_km, "
            "added_time_min, signal_coordination_complexity (LOW/MEDIUM/HIGH), "
            "requires_caltrans_coordination."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "blocked_corridor": {
                    "type": "string",
                    "description": "Name of blocked corridor, e.g. 'rosecrans_st', 'mira_mesa_blvd', 'i5_nb'",
                },
                "severity": {
                    "type": "string",
                    "enum": ["MINOR", "MODERATE", "MAJOR", "CRITICAL"],
                },
                "primary_direction": {
                    "type": "string",
                    "enum": ["NB", "SB", "EB", "WB"],
                    "description": "Primary traffic flow direction impacted.",
                },
            },
            "required": ["blocked_corridor", "severity", "primary_direction"],
        },
    },
    {
        "name": "compute_diversion_timing",
        "description": (
            "Compute signal timing adjustments needed on a diversion route to "
            "accommodate increased volume. Returns list of timing changes per intersection: "
            "intersection_id, phase, delta_green_s, new_cycle_s, estimated_delay_increase_s_veh."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "diversion_route": {"type": "string", "description": "Route name from get_alternate_routes"},
                "diverted_volume_veh_hr": {"type": "number", "description": "Additional volume diverted to this route"},
                "current_volume_veh_hr": {"type": "number", "description": "Current volume on diversion route"},
                "n_signals": {"type": "integer", "description": "Number of signals on diversion route"},
            },
            "required": ["diversion_route", "diverted_volume_veh_hr", "current_volume_veh_hr", "n_signals"],
        },
    },
    {
        "name": "draft_operator_alert",
        "description": (
            "Draft a 511/ATMS operator notification. Returns alert_text (≤280 chars), "
            "priority (LOW/MEDIUM/HIGH/CRITICAL), estimated_clearance_min, affected_routes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "incident_type": {"type": "string", "description": "e.g. 'crash', 'disabled vehicle', 'signal failure'"},
                "location": {"type": "string", "description": "Intersection or segment name"},
                "severity": {"type": "string", "enum": ["MINOR", "MODERATE", "MAJOR", "CRITICAL"]},
                "diversion_route": {"type": "string", "description": "Primary diversion route name"},
                "estimated_clearance_min": {"type": "number", "description": "Estimated minutes to clear"},
            },
            "required": ["incident_type", "location", "severity", "diversion_route", "estimated_clearance_min"],
        },
    },
]


# ---------------------------------------------------------------------------
# Sub-agent class
# ---------------------------------------------------------------------------

class _SubAgent(BaseAgent):
    """Minimal sub-agent with a custom system prompt and the shared incident tool set."""

    def __init__(self, name: str, system_prompt: str, api_key: Optional[str] = None) -> None:
        super().__init__(api_key=api_key)
        self.AGENT_NAME = name
        self.SYSTEM_PROMPT = system_prompt

    def _tools(self) -> list[dict]:
        return _INCIDENT_TOOLS

    def _run_tool(self, name: str, inputs: dict) -> Any:
        return _dispatch_incident_tool(name, inputs)


# ---------------------------------------------------------------------------
# Shared tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_incident_tool(name: str, inputs: dict) -> Any:
    if name == "assess_spillback_impact":
        return _tool_spillback_impact(inputs)
    if name == "get_alternate_routes":
        return _tool_alternate_routes(inputs)
    if name == "compute_diversion_timing":
        return _tool_diversion_timing(inputs)
    if name == "draft_operator_alert":
        return _tool_operator_alert(inputs)
    raise ValueError(f"Unknown incident tool: {name}")


def _tool_spillback_impact(inputs: dict) -> dict:
    try:
        from aito.optimization.spillback_predictor import (
            estimate_queue_length_m, estimate_discharge_rate_veh_s,
        )
        total = inputs.get("total_lanes", 2)
        blocked = inputs["lanes_blocked"]
        remaining_lanes = max(total - blocked, 0)
        cap_reduction = blocked / total
        remaining_cap = inputs["original_capacity_veh_hr"] * (remaining_lanes / total)
        adjusted_green = inputs["green_s"] * (remaining_lanes / max(total, 1))
        queue_m = estimate_queue_length_m(
            volume_veh_hr=inputs["volume_veh_hr"],
            capacity_veh_hr=remaining_cap,
            cycle_s=inputs["cycle_s"],
            green_s=max(adjusted_green, 7.0),
            n_lanes=remaining_lanes or 1,
        )
        block_len = inputs.get("block_length_m", 150.0)
        veh_per_m = 0.125
        fill_min = (block_len - queue_m) / max(
            (inputs["volume_veh_hr"] - remaining_cap) / 3600 / veh_per_m, 0.001
        ) / 60 if inputs["volume_veh_hr"] > remaining_cap else 999
        return {
            "approach": inputs["approach"],
            "capacity_remaining_veh_hr": round(remaining_cap, 1),
            "capacity_reduction_pct": round(cap_reduction * 100, 1),
            "queue_length_m": round(queue_m, 1),
            "spillback_risk": (remaining_cap == 0) or (queue_m > block_len),
            "time_to_fill_min": round(min(fill_min, 999), 1),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_alternate_routes(inputs: dict) -> dict:
    _SD_ALTERNATES = {
        "rosecrans_st": [
            {"route_name": "Sports Arena Blvd", "added_distance_km": 0.8, "added_time_min": 3,
             "signal_coordination_complexity": "LOW", "requires_caltrans_coordination": False},
            {"route_name": "Midway Dr", "added_distance_km": 1.2, "added_time_min": 5,
             "signal_coordination_complexity": "MEDIUM", "requires_caltrans_coordination": False},
        ],
        "mira_mesa_blvd": [
            {"route_name": "Carroll Canyon Rd", "added_distance_km": 1.5, "added_time_min": 4,
             "signal_coordination_complexity": "LOW", "requires_caltrans_coordination": False},
            {"route_name": "Camino Ruiz", "added_distance_km": 2.1, "added_time_min": 7,
             "signal_coordination_complexity": "MEDIUM", "requires_caltrans_coordination": False},
        ],
        "i5_nb": [
            {"route_name": "I-805 NB via I-8", "added_distance_km": 4.2, "added_time_min": 8,
             "signal_coordination_complexity": "HIGH", "requires_caltrans_coordination": True},
            {"route_name": "Pacific Hwy arterial", "added_distance_km": 2.0, "added_time_min": 12,
             "signal_coordination_complexity": "HIGH", "requires_caltrans_coordination": True},
        ],
        "genesee_ave": [
            {"route_name": "Governor Dr", "added_distance_km": 0.9, "added_time_min": 3,
             "signal_coordination_complexity": "LOW", "requires_caltrans_coordination": False},
            {"route_name": "Regents Rd", "added_distance_km": 1.1, "added_time_min": 4,
             "signal_coordination_complexity": "MEDIUM", "requires_caltrans_coordination": False},
        ],
    }
    corridor = inputs["blocked_corridor"].lower().replace(" ", "_").replace("-", "_")
    routes = _SD_ALTERNATES.get(corridor, [
        {"route_name": f"Parallel arterial to {inputs['blocked_corridor']}",
         "added_distance_km": 1.5, "added_time_min": 5,
         "signal_coordination_complexity": "MEDIUM", "requires_caltrans_coordination": False},
    ])
    return {"blocked_corridor": inputs["blocked_corridor"], "severity": inputs["severity"], "alternates": routes}


def _tool_diversion_timing(inputs: dict) -> dict:
    n = inputs.get("n_signals", 4)
    added_vol = inputs["diverted_volume_veh_hr"]
    current = inputs["current_volume_veh_hr"]
    total = current + added_vol
    adjustments = []
    for i in range(n):
        delta = round(min(added_vol / n / 100 * 8, 15), 1)
        adjustments.append({
            "intersection_idx": i,
            "phase": "main_street",
            "delta_green_s": delta,
            "new_cycle_s": 100 + round(delta * 0.5),
            "estimated_delay_increase_s_veh": round(delta * 0.4, 1),
        })
    return {
        "diversion_route": inputs["diversion_route"],
        "total_volume_veh_hr": total,
        "timing_adjustments": adjustments,
    }


def _tool_operator_alert(inputs: dict) -> dict:
    sev = inputs["severity"]
    loc = inputs["location"]
    itype = inputs["incident_type"]
    div = inputs["diversion_route"]
    eta = inputs["estimated_clearance_min"]
    alert = f"TRAFFIC ALERT: {itype.upper()} at {loc}. {sev} impact. Use {div}. Clear in ~{eta} min. -AITO"
    if len(alert) > 280:
        alert = alert[:277] + "..."
    priority_map = {"MINOR": "LOW", "MODERATE": "MEDIUM", "MAJOR": "HIGH", "CRITICAL": "CRITICAL"}
    return {
        "alert_text": alert,
        "character_count": len(alert),
        "priority": priority_map.get(sev, "MEDIUM"),
        "estimated_clearance_min": eta,
    }


# ---------------------------------------------------------------------------
# IncidentAgent — chains three sub-agents
# ---------------------------------------------------------------------------

class IncidentAgent(BaseAgent):
    """Three-sub-agent incident response chain: Detector → Rerouter → Advisor."""

    AGENT_NAME = "incident_agent"

    SYSTEM_PROMPT = """You are AITO's Incident Response Coordinator. You have already received
assessments from three specialist sub-agents (Detector, Rerouter, Advisor).
Synthesize their outputs into a single coherent incident response brief.

Format:
## INCIDENT RESPONSE BRIEF

**Severity:** [level]
**Location:** [intersection/segment]
**Queue Impact:** [queue_length_m at primary approach]
**Capacity Reduction:** [pct]

**Diversion Plan:**
- Primary: [route] (+[time] min travel time)
- Timing adjustments: [key changes]

**Operator Alert (511-ready):**
[alert text]

**Recovery Plan:**
- Trigger: [condition]
- Post-clearance action: [retiming recommendation]
- ETA normal ops: [minutes]

Cite specific module outputs (queue model, resilience scorer) where possible."""

    def __init__(self, corridor=None, api_key: Optional[str] = None) -> None:
        super().__init__(api_key=api_key)
        self.corridor = corridor
        self._detector = _SubAgent("detector", _DETECTOR_SYSTEM, api_key)
        self._rerouter = _SubAgent("rerouter", _REROUTER_SYSTEM, api_key)
        self._advisor  = _SubAgent("advisor", _ADVISOR_SYSTEM, api_key)

    def run(self, query: str, context: dict | None = None) -> AgentResult:
        if not self._api_key:
            return AgentResult(
                agent_name=self.AGENT_NAME,
                query=query,
                final_output="ANTHROPIC_API_KEY not set.",
                error="no_api_key",
            )

        # Chain the three sub-agents sequentially
        ctx = context or {}

        # Stage 1: Detector
        detection_result = self._detector.run(query, ctx)

        # Stage 2: Rerouter — receives detector output as context
        reroute_ctx = {**ctx, "incident_assessment": detection_result.final_output}
        reroute_result = self._rerouter.run(query, reroute_ctx)

        # Stage 3: Advisor — receives both previous outputs
        advisor_ctx = {
            **ctx,
            "incident_assessment": detection_result.final_output,
            "diversion_plan": reroute_result.final_output,
        }
        advisor_result = self._advisor.run(query, advisor_ctx)

        # Final synthesis call
        synthesis_query = (
            f"Synthesize this incident response:\n\n"
            f"[DETECTION]\n{detection_result.final_output}\n\n"
            f"[DIVERSION]\n{reroute_result.final_output}\n\n"
            f"[ADVISORY]\n{advisor_result.final_output}"
        )
        final = self._run_synthesis(synthesis_query, ctx)

        all_tool_calls = (
            detection_result.tool_calls +
            reroute_result.tool_calls +
            advisor_result.tool_calls
        )
        all_reasoning = "\n\n---SUB-AGENT BOUNDARY---\n\n".join(filter(None, [
            detection_result.reasoning_trace,
            reroute_result.reasoning_trace,
            advisor_result.reasoning_trace,
        ]))

        return AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=final,
            reasoning_trace=all_reasoning,
            tool_calls=all_tool_calls,
            confidence=min(detection_result.confidence, reroute_result.confidence, advisor_result.confidence),
            citations_to_modules=self._citations(),
        )

    def _run_synthesis(self, synthesis_query: str, context: dict) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.MODEL,
            max_tokens=4096,
            system=[{"type": "text", "text": self.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "xhigh"},
            messages=[{"role": "user", "content": synthesis_query}],
        )
        return _extract_text(response.content)

    def stream(self, query: str, context: dict | None = None):
        if not self._api_key:
            yield {"type": "text_delta", "text": "ANTHROPIC_API_KEY not set."}
            return

        ctx = context or {}
        yield {"type": "text_delta", "text": "**[Stage 1/3] Incident Detector running...**\n"}
        detection_result = self._detector.run(query, ctx)
        yield {"type": "text_delta", "text": f"\n*Detection complete.*\n\n"}
        for tc in detection_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        yield {"type": "text_delta", "text": "**[Stage 2/3] Rerouter running...**\n"}
        reroute_ctx = {**ctx, "incident_assessment": detection_result.final_output}
        reroute_result = self._rerouter.run(query, reroute_ctx)
        yield {"type": "text_delta", "text": f"\n*Diversion plan computed.*\n\n"}
        for tc in reroute_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        yield {"type": "text_delta", "text": "**[Stage 3/3] Post-Incident Advisor running...**\n"}
        advisor_ctx = {**ctx, "incident_assessment": detection_result.final_output,
                       "diversion_plan": reroute_result.final_output}
        advisor_result = self._advisor.run(query, advisor_ctx)
        yield {"type": "text_delta", "text": f"\n*Advisory complete.*\n\n"}
        for tc in advisor_result.tool_calls:
            yield {"type": "tool_result", "name": tc.name, "output": tc.output, "duration_ms": tc.duration_ms}

        yield {"type": "text_delta", "text": "**Synthesizing response...**\n\n"}
        synthesis_query = (
            f"Synthesize this incident response:\n\n"
            f"[DETECTION]\n{detection_result.final_output}\n\n"
            f"[DIVERSION]\n{reroute_result.final_output}\n\n"
            f"[ADVISORY]\n{advisor_result.final_output}"
        )
        final = self._run_synthesis(synthesis_query, ctx)
        yield {"type": "text_delta", "text": final}

        all_tool_calls = detection_result.tool_calls + reroute_result.tool_calls + advisor_result.tool_calls
        result = AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=final,
            tool_calls=all_tool_calls,
            citations_to_modules=self._citations(),
        )
        yield {"type": "done", "result": result}

    def _citations(self) -> list[str]:
        return [
            "aito.optimization.spillback_predictor (GF12)",
            "aito.analytics.resilience_scorer (GF11)",
            "HCM 7th Edition Ch.19",
            "MUTCD 2023 §4D.06 (minimum green)",
            "FHWA Traffic Incident Management Handbook",
        ]
