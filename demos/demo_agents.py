"""demos/demo_agents.py

Hackathon demo: exercise all 5 AITO agents without requiring
a live corridor object or API key (synthetic data throughout).

Run:
    python demos/demo_agents.py                     # template mode (no API key)
    ANTHROPIC_API_KEY=sk-ant-... python demos/demo_agents.py  # full Claude mode

Expected output (~3s template mode, ~60s Claude mode):
    [ORCHESTRATOR] Intent classification for 5 sample queries
    [SCENARIO]     Spillback prediction + what-if delta
    [INCIDENT]     3-sub-agent chain header
    [NEGOTIATION]  Boundary constraints + proposal
    [CARBON]       EPA MOVES + CARB LCFS revenue
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aito.agents.orchestrator import OrchestratorAgent, classify_intent
from aito.agents.scenario_agent import ScenarioAgent
from aito.agents.incident_agent import IncidentAgent
from aito.agents.negotiation_agent import NegotiationAgent
from aito.agents.carbon_agent import CarbonAgent
from aito.interface.nl_engineer import NLEngineerSession, classify_query

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
USE_CLAUDE = bool(API_KEY)


def _hr(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def demo_intent_classification():
    _hr("1. ORCHESTRATOR — Intent Classification")
    queries = [
        "What happens if we reduce the cycle to 90s?",
        "Crash at Rosecrans and Midway, 2 lanes blocked",
        "Negotiate timing with Caltrans District 11",
        "What's the CARB LCFS revenue for Rosecrans?",
        "Explain why the PM peak cycle is 120s",
    ]
    for q in queries:
        intent = classify_intent(q)
        nl_type = classify_query(q)
        print(f"  [{intent:>12}]  {q}")


def demo_scenario_agent():
    _hr("2. SCENARIO AGENT — Spillback + What-If")
    agent = ScenarioAgent(api_key=API_KEY)

    # Direct tool call (no Claude API required)
    print("\n  [spillback prediction]")
    result = agent._tool_spillback({
        "volume_veh_hr": 1800,
        "capacity_veh_hr": 1500,
        "cycle_s": 100,
        "green_s": 42,
        "n_lanes": 2,
        "block_length_m": 150,
    })
    print(f"    queue_length_m = {result.get('queue_length_m')} m")
    print(f"    spillback_risk = {result.get('spillback_risk')}")
    print(f"    v/c ratio      = {result.get('v_c_ratio')}")

    print("\n  [event demand — Petco Park SPORTS_MAJOR]")
    event = agent._tool_event_demand({"venue": "petco_park", "event_type": "SPORTS_MAJOR"})
    if "error" not in event:
        print(f"    inbound_peak  = {event.get('inbound_peak_veh_hr')} veh/hr")
        print(f"    outbound_peak = {event.get('outbound_peak_veh_hr')} veh/hr")
        print(f"    pre_deploy    = {event.get('pre_deploy_min')} min before event")
    else:
        print(f"    (module not available: {event['error'][:60]})")

    print("\n  [retiming drift detection]")
    drift = agent._tool_retiming_drift({
        "baseline_delay_s_veh": 32.0,
        "current_delay_s_veh": 41.0,
        "baseline_split_failure_pct": 5.0,
        "current_split_failure_pct": 18.0,
        "baseline_volume_veh_hr": 1200,
        "current_volume_veh_hr": 1580,
        "drift_threshold_pct": 0.15,
    })
    print(f"    triggered      = {drift.get('triggered')}")
    print(f"    severity       = {drift.get('severity')}")
    print(f"    action         = {drift.get('recommended_action')}")
    print(f"    delay_delta    = {drift.get('delay_delta_pct', 'N/A')}%")

    if USE_CLAUDE:
        print("\n  [full Claude Opus 4.7 scenario run]")
        t0 = time.monotonic()
        result = agent.run(
            "Rosecrans at Midway: volume is 1800 veh/hr with 1500 capacity. "
            "Predict spillback, then simulate a 10% demand increase.",
            context={"corridor_id": "rosecrans_st", "n_intersections": 4, "aadt": 28000},
        )
        elapsed = time.monotonic() - t0
        print(f"    tool_calls     = {len(result.tool_calls)}")
        print(f"    elapsed        = {elapsed:.1f}s")
        print(f"    answer[:200]   = {result.final_output[:200]}…")


def demo_incident_agent():
    _hr("3. INCIDENT AGENT — 3-sub-agent chain")
    agent = IncidentAgent(api_key=API_KEY)

    print("\n  [spillback impact — 2 lanes blocked]")
    impact = agent._detector._run_tool("assess_spillback_impact", {
        "approach": "SB",
        "volume_veh_hr": 1600,
        "original_capacity_veh_hr": 1800,
        "lanes_blocked": 2,
        "total_lanes": 3,
        "cycle_s": 100,
        "green_s": 45,
        "block_length_m": 150,
    })
    print(f"    capacity_remaining  = {impact.get('capacity_remaining_veh_hr')} veh/hr")
    print(f"    queue_length_m      = {impact.get('queue_length_m')} m")
    print(f"    spillback_risk      = {impact.get('spillback_risk')}")
    print(f"    time_to_fill_min    = {impact.get('time_to_fill_min')} min")

    print("\n  [alternate routes — rosecrans_st MAJOR SB]")
    routes = agent._detector._run_tool("get_alternate_routes", {
        "blocked_corridor": "rosecrans_st",
        "severity": "MAJOR",
        "primary_direction": "SB",
    })
    for r in routes.get("alternates", [])[:2]:
        print(f"    {r['route_name']}: +{r['added_time_min']} min, complexity={r['signal_coordination_complexity']}")

    if USE_CLAUDE:
        print("\n  [full 3-sub-agent chain]")
        t0 = time.monotonic()
        result = agent.run(
            "Major crash at Rosecrans and Midway Dr, 2 of 3 SB lanes blocked, PM peak (17:15).",
        )
        elapsed = time.monotonic() - t0
        print(f"    sub-agent tool calls = {len(result.tool_calls)}")
        print(f"    elapsed              = {elapsed:.1f}s")
        print(f"    answer[:300]         = {result.final_output[:300]}…")


def demo_negotiation_agent():
    _hr("4. NEGOTIATION AGENT — Caltrans × City of SD")
    agent = NegotiationAgent(api_key=API_KEY)

    print("\n  [boundary constraints — Rosecrans at Lytton]")
    boundary = agent._run_tool("get_boundary_constraints", {
        "boundary_intersection": "Rosecrans_at_Lytton",
        "upstream_agency": "caltrans",
        "downstream_agency": "city_sd",
    })
    print(f"    export_cycle_s      = {boundary.get('export_cycle_s')} s")
    print(f"    export_offset_s     = {boundary.get('export_offset_s')} s")
    print(f"    coordination_status = {boundary.get('coordination_status')}")

    print("\n  [Caltrans proposal]")
    prop = agent._run_tool("propose_shared_timing", {
        "corridor_id": "rosecrans_st",
        "proposing_agency": "caltrans",
        "desired_cycle_s": 100,
        "priority_direction": "NB",
        "transit_priority": False,
        "n_intersections": 4,
    })
    print(f"    proposal_id         = {prop.get('proposal_id')}")
    print(f"    shared_cycle_s      = {prop.get('shared_cycle_s')} s")
    print(f"    bandwidth_pct       = {prop.get('estimated_bandwidth_pct')}%")
    print(f"    transit_impact_s    = {prop.get('transit_impact_s')} s")

    print("\n  [City evaluation]")
    eval_r = agent._run_tool("evaluate_timing_proposal", {
        "proposal_id": prop["proposal_id"],
        "evaluating_agency": "city_sd",
        "shared_cycle_s": prop["shared_cycle_s"],
        "bandwidth_pct": prop["estimated_bandwidth_pct"],
        "transit_impact_s": prop.get("transit_impact_s", 0),
    })
    print(f"    agency_score        = {eval_r.get('agency_score')}/100")
    print(f"    recommendation      = {eval_r.get('accept_recommendation')}")
    print(f"    pedestrian_lts      = {eval_r.get('pedestrian_lts')}")

    if USE_CLAUDE:
        print("\n  [full dual-Claude negotiation]")
        t0 = time.monotonic()
        result = agent.run(
            "Negotiate a shared cycle plan for Rosecrans Street between Caltrans District 11 "
            "and City of San Diego, prioritizing PM peak progression and MTS bus priority."
        )
        elapsed = time.monotonic() - t0
        print(f"    total tool calls    = {len(result.tool_calls)}")
        print(f"    elapsed             = {elapsed:.1f}s")
        print(f"    answer[:300]        = {result.final_output[:300]}…")


def demo_carbon_agent():
    _hr("5. CARBON AGENT — EPA MOVES + CARB LCFS")
    agent = CarbonAgent(api_key=API_KEY)

    print("\n  [emissions — rosecrans baseline]")
    emissions = agent._tool_emissions({
        "corridor_id": "rosecrans_st",
        "volume_veh_hr": 1620,
        "avg_delay_s_veh": 48.2,
        "n_intersections": 4,
    })
    print(f"    co2_kg_hr           = {emissions.get('co2_kg_hr')} kg/hr")
    print(f"    nox_g_hr            = {emissions.get('nox_g_hr')} g/hr")
    print(f"    fuel_l_hr           = {emissions.get('fuel_l_hr')} L/hr")

    print("\n  [emission reduction — AITO vs. fixed-time]")
    reduction = agent._tool_reduction({
        "baseline_delay_s_veh": 48.2,
        "optimized_delay_s_veh": 33.3,
        "volume_veh_hr": 1620,
        "n_intersections": 4,
    })
    print(f"    co2_reduction_kg_hr  = {reduction.get('co2_reduction_kg_hr')} kg/hr")
    print(f"    co2_reduction_t/yr   = {reduction.get('co2_reduction_tonnes_year')} t/yr")
    print(f"    reduction_pct        = {reduction.get('reduction_pct')}%")
    print(f"    equivalent_cars      = {reduction.get('equivalent_cars_removed')} cars removed")

    print("\n  [carbon credit scoring]")
    tonnes = reduction["co2_reduction_tonnes_year"]
    credits = agent._tool_credits({
        "co2_reduction_tonnes_year": tonnes,
        "additionality_level": "HIGH",
    })
    print(f"    best_market          = {credits.get('best_market')}")
    print(f"    best_revenue_usd     = ${credits.get('best_market_revenue_usd'):,.0f}/yr")
    print(f"    portfolio_total      = ${credits.get('total_portfolio_revenue_usd'):,.0f}/yr")

    print("\n  [resilience check]")
    resilience = agent._tool_resilience({
        "corridor_id": "rosecrans_st",
        "n_intersections": 4,
        "has_probe_data": True,
        "has_adaptive_control": True,
        "transit_routes": 2,
    })
    print(f"    overall_score        = {resilience.get('overall_score')}/100")
    print(f"    grade                = {resilience.get('grade')}")

    print("\n  [MRV report generation]")
    report = agent._tool_report({
        "corridor_id": "rosecrans_st",
        "corridor_name": "Rosecrans Street",
        "co2_reduction_tonnes_year": tonnes,
        "target_market": "CARB_LCFS",
        "monitoring_period_years": 5,
    })
    print(f"    exec_summary[:160]   = {report['executive_summary'][:160]}…")
    print(f"    certification_months = {report['registry_recommendations']['certification_timeline_months']}")

    if USE_CLAUDE:
        print("\n  [full Claude Opus 4.7 carbon analysis]")
        t0 = time.monotonic()
        result = agent.run(
            "Generate the full CARB LCFS carbon credit portfolio for Rosecrans Street: "
            "4 intersections, 28,000 AADT. Include MRV report and resilience score.",
        )
        elapsed = time.monotonic() - t0
        print(f"    tool_calls          = {len(result.tool_calls)}")
        print(f"    elapsed             = {elapsed:.1f}s")
        print(f"    answer[:400]        = {result.final_output[:400]}…")


def demo_nl_session():
    _hr("6. NL ENGINEER SESSION — End-to-end routing")
    session = NLEngineerSession(anthropic_api_key=API_KEY)

    query = "What happens to Rosecrans delays if we reduce cycle to 90s?"
    print(f"\n  Query: {query}")

    response = session.ask(query)
    print(f"  Agent:   {response.agent_name}")
    print(f"  Claude:  {response.used_claude_api}")
    print(f"  Answer[:200]: {response.answer[:200]}…")
    print(f"  Citations: {response.citations[:3]}")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  AITO Multi-Agent Demo")
    mode = "Claude Opus 4.7" if USE_CLAUDE else "Template (no API key)"
    print(f"  Mode: {mode}")
    print("═" * 60)

    demo_intent_classification()
    demo_scenario_agent()
    demo_incident_agent()
    demo_negotiation_agent()
    demo_carbon_agent()
    demo_nl_session()

    print("\n" + "═" * 60)
    print("  Demo complete.")
    if not USE_CLAUDE:
        print("  Set ANTHROPIC_API_KEY to run full Claude Opus 4.7 agents.")
    print("═" * 60 + "\n")
