"""tests/agents/test_carbon_agent.py — CarbonAgent tool tests."""
import pytest
from aito.agents.carbon_agent import CarbonAgent


@pytest.fixture
def agent():
    return CarbonAgent(api_key=None)


class TestEmissionsCalculation:
    def test_basic_emissions(self, agent):
        result = agent._tool_emissions({
            "corridor_id": "rosecrans_st",
            "volume_veh_hr": 1620,
            "avg_delay_s_veh": 48.2,
            "n_intersections": 4,
        })
        assert "co2_kg_hr" in result
        assert result["co2_kg_hr"] > 0
        assert result["nox_g_hr"] > 0
        assert result["fuel_l_hr"] > 0

    def test_higher_delay_more_emissions(self, agent):
        low = agent._tool_emissions({"corridor_id": "c", "volume_veh_hr": 1000,
                                      "avg_delay_s_veh": 20, "n_intersections": 3})
        high = agent._tool_emissions({"corridor_id": "c", "volume_veh_hr": 1000,
                                       "avg_delay_s_veh": 60, "n_intersections": 3})
        assert high["co2_kg_hr"] > low["co2_kg_hr"]

    def test_more_intersections_more_emissions(self, agent):
        few = agent._tool_emissions({"corridor_id": "c", "volume_veh_hr": 1000,
                                      "avg_delay_s_veh": 40, "n_intersections": 2})
        many = agent._tool_emissions({"corridor_id": "c", "volume_veh_hr": 1000,
                                       "avg_delay_s_veh": 40, "n_intersections": 6})
        assert many["co2_kg_hr"] > few["co2_kg_hr"]


class TestEmissionReduction:
    def test_reduction_positive(self, agent):
        result = agent._tool_reduction({
            "baseline_delay_s_veh": 48.2,
            "optimized_delay_s_veh": 33.3,
            "volume_veh_hr": 1620,
            "n_intersections": 4,
        })
        assert result["co2_reduction_kg_hr"] > 0
        assert result["co2_reduction_tonnes_year"] > 0
        assert result["reduction_pct"] > 0
        assert result["equivalent_cars_removed"] > 0

    def test_reduction_pct_matches_delay_delta(self, agent):
        result = agent._tool_reduction({
            "baseline_delay_s_veh": 40.0,
            "optimized_delay_s_veh": 30.0,
            "volume_veh_hr": 1000,
            "n_intersections": 4,
        })
        assert result["reduction_pct"] == pytest.approx(25.0, abs=1.0)

    def test_zero_improvement(self, agent):
        result = agent._tool_reduction({
            "baseline_delay_s_veh": 35.0,
            "optimized_delay_s_veh": 35.0,
            "volume_veh_hr": 1000,
            "n_intersections": 4,
        })
        assert result["co2_reduction_kg_hr"] == pytest.approx(0.0, abs=0.01)
        assert result["reduction_pct"] == pytest.approx(0.0, abs=0.1)

    def test_annual_reduction_uses_hours(self, agent):
        r_default = agent._tool_reduction({
            "baseline_delay_s_veh": 48.0, "optimized_delay_s_veh": 33.0,
            "volume_veh_hr": 1000, "n_intersections": 3,
        })
        r_custom = agent._tool_reduction({
            "baseline_delay_s_veh": 48.0, "optimized_delay_s_veh": 33.0,
            "volume_veh_hr": 1000, "n_intersections": 3,
            "hours_per_year": 4380,
        })
        ratio = r_default["co2_reduction_tonnes_year"] / r_custom["co2_reduction_tonnes_year"]
        assert ratio == pytest.approx(6570 / 4380, abs=0.01)


class TestCarbonCredits:
    def test_lcfs_eligible(self, agent):
        result = agent._tool_credits({
            "co2_reduction_tonnes_year": 100.0,
            "additionality_level": "HIGH",
        })
        assert "best_market" in result
        assert result["best_market_revenue_usd"] > 0
        assert result["total_portfolio_revenue_usd"] > 0

    def test_lcfs_best_market_for_california(self, agent):
        result = agent._tool_credits({
            "co2_reduction_tonnes_year": 500.0,
            "additionality_level": "HIGH",
        })
        assert result["best_market"] == "CARB_LCFS"

    def test_low_additionality_reduces_revenue(self, agent):
        high = agent._tool_credits({"co2_reduction_tonnes_year": 200, "additionality_level": "HIGH"})
        low = agent._tool_credits({"co2_reduction_tonnes_year": 200, "additionality_level": "LOW"})
        assert high["best_market_revenue_usd"] >= low["best_market_revenue_usd"]

    def test_vcs_minimum_threshold(self, agent):
        result = agent._tool_credits({
            "co2_reduction_tonnes_year": 50.0,  # below VCS minimum of 100
            "additionality_level": "HIGH",
        })
        markets = result.get("markets", {})
        if "VERRA_VCS" in markets:
            assert not markets["VERRA_VCS"].get("eligible", True)


class TestResilienceCheck:
    def test_returns_score_and_grade(self, agent):
        result = agent._tool_resilience({
            "corridor_id": "rosecrans_st",
            "n_intersections": 4,
            "has_probe_data": True,
            "has_adaptive_control": True,
        })
        assert 0 <= result["overall_score"] <= 100
        assert result["grade"] in ("A", "B", "C", "D", "E", "F")

    def test_adaptive_control_improves_score(self, agent):
        without = agent._tool_resilience({"corridor_id": "c", "n_intersections": 4,
                                           "has_adaptive_control": False, "has_probe_data": False})
        with_it = agent._tool_resilience({"corridor_id": "c", "n_intersections": 4,
                                           "has_adaptive_control": True, "has_probe_data": True})
        assert with_it["overall_score"] >= without["overall_score"]


class TestMRVReport:
    def test_report_structure(self, agent):
        result = agent._tool_report({
            "corridor_id": "rosecrans_st",
            "corridor_name": "Rosecrans Street",
            "co2_reduction_tonnes_year": 542.8,
            "target_market": "CARB_LCFS",
            "monitoring_period_years": 5,
        })
        assert "executive_summary" in result
        assert "Rosecrans Street" in result["executive_summary"]
        assert "registry_recommendations" in result
        reg = result["registry_recommendations"]
        assert reg["annual_revenue_usd"] > 0
        assert reg["certification_timeline_months"] > 0
        assert len(reg["next_steps"]) >= 2

    def test_longer_period_higher_total(self, agent):
        r5 = agent._tool_report({"corridor_id": "c", "corridor_name": "C",
                                   "co2_reduction_tonnes_year": 100, "monitoring_period_years": 5})
        r10 = agent._tool_report({"corridor_id": "c", "corridor_name": "C",
                                    "co2_reduction_tonnes_year": 100, "monitoring_period_years": 10})
        assert r10["registry_recommendations"]["total_revenue_5yr_usd"] > r5["registry_recommendations"]["total_revenue_5yr_usd"]


class TestCarbonAgentMeta:
    def test_tools_count(self, agent):
        tools = agent._tools()
        assert len(tools) == 5

    def test_tool_names(self, agent):
        names = {t["name"] for t in agent._tools()}
        assert "calculate_corridor_emissions" in names
        assert "compute_emission_reduction" in names
        assert "score_carbon_credits" in names
        assert "run_resilience_check" in names
        assert "generate_carbon_report" in names

    def test_citations_cover_epa_and_carb(self, agent):
        cites = agent._citations()
        assert any("EPA MOVES" in c for c in cites)
        assert any("CARB" in c or "LCFS" in c for c in cites)
        assert any("Verra" in c for c in cites)
