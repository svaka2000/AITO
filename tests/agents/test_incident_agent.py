"""tests/agents/test_incident_agent.py — IncidentAgent tool tests."""
import pytest
from aito.agents.incident_agent import (
    IncidentAgent,
    _tool_spillback_impact,
    _tool_alternate_routes,
    _tool_diversion_timing,
    _tool_operator_alert,
)


class TestSpillbackImpact:
    def test_two_lanes_blocked_of_three(self):
        result = _tool_spillback_impact({
            "approach": "SB",
            "volume_veh_hr": 1600,
            "original_capacity_veh_hr": 1800,
            "lanes_blocked": 2,
            "total_lanes": 3,
            "cycle_s": 100,
            "green_s": 45,
            "block_length_m": 150,
        })
        assert result["capacity_reduction_pct"] == pytest.approx(66.7, abs=0.5)
        assert result["capacity_remaining_veh_hr"] == pytest.approx(600, abs=5)
        assert result["spillback_risk"] is True
        assert result["queue_length_m"] >= 150

    def test_minor_blockage(self):
        result = _tool_spillback_impact({
            "approach": "NB",
            "volume_veh_hr": 800,
            "original_capacity_veh_hr": 1800,
            "lanes_blocked": 1,
            "total_lanes": 3,
            "cycle_s": 100,
            "green_s": 45,
            "block_length_m": 200,
        })
        assert result["capacity_remaining_veh_hr"] == pytest.approx(1200, abs=10)
        assert result["spillback_risk"] is False

    def test_full_closure(self):
        result = _tool_spillback_impact({
            "approach": "EB",
            "volume_veh_hr": 900,
            "original_capacity_veh_hr": 1200,
            "lanes_blocked": 2,
            "total_lanes": 2,
            "cycle_s": 100,
            "green_s": 40,
        })
        assert result["capacity_remaining_veh_hr"] == 0.0
        assert result["spillback_risk"] is True


class TestAlternateRoutes:
    def test_rosecrans_routes(self):
        result = _tool_alternate_routes({
            "blocked_corridor": "rosecrans_st",
            "severity": "MAJOR",
            "primary_direction": "SB",
        })
        assert "alternates" in result
        assert len(result["alternates"]) >= 1
        for alt in result["alternates"]:
            assert "route_name" in alt
            assert "added_time_min" in alt
            assert "signal_coordination_complexity" in alt

    def test_mira_mesa_routes(self):
        result = _tool_alternate_routes({
            "blocked_corridor": "mira_mesa_blvd",
            "severity": "MODERATE",
            "primary_direction": "EB",
        })
        assert len(result["alternates"]) >= 1

    def test_unknown_corridor_fallback(self):
        result = _tool_alternate_routes({
            "blocked_corridor": "unknown_street",
            "severity": "MINOR",
            "primary_direction": "NB",
        })
        assert "alternates" in result
        assert len(result["alternates"]) >= 1

    def test_i5_requires_caltrans(self):
        result = _tool_alternate_routes({
            "blocked_corridor": "i5_nb",
            "severity": "CRITICAL",
            "primary_direction": "NB",
        })
        caltrans_routes = [a for a in result["alternates"] if a["requires_caltrans_coordination"]]
        assert len(caltrans_routes) >= 1


class TestDiversionTiming:
    def test_basic_timing_adjustments(self):
        result = _tool_diversion_timing({
            "diversion_route": "Sports Arena Blvd",
            "diverted_volume_veh_hr": 600,
            "current_volume_veh_hr": 800,
            "n_signals": 4,
        })
        assert "timing_adjustments" in result
        assert len(result["timing_adjustments"]) == 4
        for adj in result["timing_adjustments"]:
            assert adj["delta_green_s"] >= 0
            assert adj["new_cycle_s"] > 80
            assert adj["estimated_delay_increase_s_veh"] >= 0

    def test_total_volume_computed(self):
        result = _tool_diversion_timing({
            "diversion_route": "Midway Dr",
            "diverted_volume_veh_hr": 400,
            "current_volume_veh_hr": 700,
            "n_signals": 3,
        })
        assert result["total_volume_veh_hr"] == 1100


class TestOperatorAlert:
    def test_alert_length(self):
        result = _tool_operator_alert({
            "incident_type": "crash",
            "location": "Rosecrans at Midway Dr",
            "severity": "MAJOR",
            "diversion_route": "Sports Arena Blvd",
            "estimated_clearance_min": 45,
        })
        assert len(result["alert_text"]) <= 280
        assert result["priority"] == "HIGH"
        assert result["estimated_clearance_min"] == 45

    def test_critical_priority(self):
        result = _tool_operator_alert({
            "incident_type": "signal failure",
            "location": "I-5 at Rosecrans",
            "severity": "CRITICAL",
            "diversion_route": "Pacific Hwy",
            "estimated_clearance_min": 90,
        })
        assert result["priority"] == "CRITICAL"

    def test_minor_priority(self):
        result = _tool_operator_alert({
            "incident_type": "disabled vehicle",
            "location": "Genesee Ave",
            "severity": "MINOR",
            "diversion_route": "Governor Dr",
            "estimated_clearance_min": 15,
        })
        assert result["priority"] == "LOW"

    def test_alert_contains_location(self):
        result = _tool_operator_alert({
            "incident_type": "crash",
            "location": "Mira Mesa Blvd",
            "severity": "MODERATE",
            "diversion_route": "Carroll Canyon Rd",
            "estimated_clearance_min": 30,
        })
        assert "Mira Mesa Blvd" in result["alert_text"]


class TestIncidentAgent:
    def test_agent_name(self):
        agent = IncidentAgent()
        assert agent.AGENT_NAME == "incident_agent"

    def test_no_api_key_returns_error(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            agent = IncidentAgent(api_key=None)
            result = agent.run("test incident query")
            assert result.error == "no_api_key"
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_sub_agents_initialized(self):
        agent = IncidentAgent(api_key="test")
        assert agent._detector is not None
        assert agent._rerouter is not None
        assert agent._advisor is not None

    def test_sub_agents_have_incident_tools(self):
        agent = IncidentAgent(api_key="test")
        names = {t["name"] for t in agent._detector._tools()}
        assert "assess_spillback_impact" in names
        assert "get_alternate_routes" in names
        assert "compute_diversion_timing" in names
        assert "draft_operator_alert" in names
