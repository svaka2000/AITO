"""tests/agents/test_negotiation_agent.py — NegotiationAgent tool tests."""
import pytest
from aito.agents.negotiation_agent import (
    NegotiationAgent,
    _tool_boundary,
    _tool_propose,
    _tool_evaluate,
    _tool_mou,
)


class TestBoundaryConstraints:
    def test_rosecrans_at_lytton(self):
        result = _tool_boundary({
            "boundary_intersection": "Rosecrans_at_Lytton",
            "upstream_agency": "caltrans",
            "downstream_agency": "city_sd",
        })
        assert result["export_cycle_s"] > 0
        assert result["export_offset_s"] >= 0
        assert result["coordination_status"] in (
            "COORDINATED", "PARTIAL", "UNCOORDINATED", "UNKNOWN"
        )

    def test_mira_mesa_boundary(self):
        result = _tool_boundary({
            "boundary_intersection": "Mira_Mesa_at_I15_NB_ramps",
            "upstream_agency": "caltrans",
            "downstream_agency": "city_sd",
        })
        assert result["export_cycle_s"] > 0

    def test_unknown_boundary_fallback(self):
        result = _tool_boundary({
            "boundary_intersection": "Unknown_Intersection",
            "upstream_agency": "caltrans",
            "downstream_agency": "city_sd",
        })
        assert result["export_cycle_s"] > 0
        assert result["coordination_status"] == "UNKNOWN"


class TestProposeSharedTiming:
    def test_caltrans_proposal(self):
        result = _tool_propose({
            "corridor_id": "rosecrans_st",
            "proposing_agency": "caltrans",
            "desired_cycle_s": 100,
            "priority_direction": "NB",
            "transit_priority": False,
            "n_intersections": 4,
        })
        assert result["shared_cycle_s"] == 100
        assert result["proposing_agency"] == "caltrans"
        assert len(result["proposed_offsets_s"]) == 4
        assert 20 <= result["estimated_bandwidth_pct"] <= 65
        assert "proposal_id" in result

    def test_city_proposal_with_transit(self):
        result = _tool_propose({
            "corridor_id": "mira_mesa_blvd",
            "proposing_agency": "city_sd",
            "desired_cycle_s": 90,
            "priority_direction": "EB",
            "transit_priority": True,
            "n_intersections": 6,
        })
        assert result["shared_cycle_s"] == 90
        assert len(result["proposed_offsets_s"]) == 6

    def test_proposal_id_format(self):
        result = _tool_propose({
            "corridor_id": "genesee_ave",
            "proposing_agency": "caltrans",
            "desired_cycle_s": 110,
            "priority_direction": "SB",
            "n_intersections": 5,
        })
        assert "caltrans" in result["proposal_id"]
        assert "genesee_ave" in result["proposal_id"]


class TestEvaluateProposal:
    def test_caltrans_evaluates_proposal(self):
        result = _tool_evaluate({
            "proposal_id": "test_proposal",
            "evaluating_agency": "caltrans",
            "shared_cycle_s": 100,
            "bandwidth_pct": 45.0,
        })
        assert 0 <= result["agency_score"] <= 100
        assert result["accept_recommendation"] in ("ACCEPT", "COUNTER", "REJECT")
        assert result["freeway_backup_impact"] in ("LOW", "MEDIUM", "HIGH")

    def test_city_evaluates_proposal(self):
        result = _tool_evaluate({
            "proposal_id": "test_proposal",
            "evaluating_agency": "city_sd",
            "shared_cycle_s": 90,
            "bandwidth_pct": 50.0,
            "transit_impact_s": -3.0,
        })
        assert 0 <= result["agency_score"] <= 100
        assert result["pedestrian_lts"] in ("A", "B", "C", "D", "E", "F")

    def test_high_bandwidth_improves_score(self):
        low_bw = _tool_evaluate({"proposal_id": "p", "evaluating_agency": "caltrans",
                                  "shared_cycle_s": 100, "bandwidth_pct": 25.0})
        high_bw = _tool_evaluate({"proposal_id": "p", "evaluating_agency": "caltrans",
                                   "shared_cycle_s": 100, "bandwidth_pct": 50.0})
        assert high_bw["agency_score"] >= low_bw["agency_score"]


class TestGenerateMOU:
    def test_mou_contains_corridor(self):
        result = _tool_mou({
            "corridor_name": "Rosecrans Street",
            "shared_cycle_s": 100,
            "caltrans_concession": "Accept 7s offset flexibility",
            "city_concession": "Accept 100s cycle (above preferred 90s)",
            "performance_target": "25% delay reduction",
        })
        assert "Rosecrans Street" in result["mou_text"]
        assert "100" in result["mou_text"]
        assert result["effective_date"] is not None
        assert len(result["signatory_titles"]) == 2

    def test_mou_review_date_after_effective(self):
        from datetime import date
        result = _tool_mou({
            "corridor_name": "Mira Mesa Blvd",
            "shared_cycle_s": 90,
            "caltrans_concession": "offset",
            "city_concession": "cycle",
            "performance_target": "20% reduction",
            "review_cycle_months": 6,
        })
        eff = date.fromisoformat(result["effective_date"])
        rev = date.fromisoformat(result["review_date"])
        assert rev > eff


class TestNegotiationAgent:
    def test_agent_name(self):
        agent = NegotiationAgent()
        assert agent.AGENT_NAME == "negotiation_agent"

    def test_no_api_key_returns_error(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            agent = NegotiationAgent(api_key=None)
            result = agent.run("negotiate timing")
            assert result.error == "no_api_key"
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_agency_agents_initialized(self):
        agent = NegotiationAgent(api_key="test")
        assert agent._caltrans is not None
        assert agent._city_sd is not None

    def test_citations_include_ntcip(self):
        agent = NegotiationAgent()
        cites = agent._citations()
        ntcip_refs = [c for c in cites if "NTCIP" in c]
        assert len(ntcip_refs) >= 1
