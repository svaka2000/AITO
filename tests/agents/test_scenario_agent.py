"""tests/agents/test_scenario_agent.py — ScenarioAgent tool unit tests."""
import pytest
from aito.agents.scenario_agent import ScenarioAgent, _synthetic_what_if


class TestScenarioAgentTools:
    @pytest.fixture
    def agent(self):
        return ScenarioAgent(api_key=None)

    def test_spillback_overcapacity(self, agent):
        result = agent._tool_spillback({
            "volume_veh_hr": 1800,
            "capacity_veh_hr": 1500,
            "cycle_s": 100,
            "green_s": 42,
            "n_lanes": 2,
            "block_length_m": 150,
        })
        assert "queue_length_m" in result
        assert result["spillback_risk"] is True
        assert result["v_c_ratio"] > 1.0
        assert result["queue_length_m"] <= 300.0  # cap at 300m

    def test_spillback_undercapacity(self, agent):
        result = agent._tool_spillback({
            "volume_veh_hr": 800,
            "capacity_veh_hr": 1800,
            "cycle_s": 100,
            "green_s": 55,
            "n_lanes": 2,
            "block_length_m": 200,
        })
        assert result["v_c_ratio"] < 1.0
        assert result["discharge_rate_veh_s"] > 0

    def test_spillback_default_lanes(self, agent):
        result = agent._tool_spillback({
            "volume_veh_hr": 1000,
            "capacity_veh_hr": 1200,
            "cycle_s": 100,
            "green_s": 40,
        })
        assert "queue_length_m" in result
        assert "error" not in result

    def test_event_demand_sports_major(self, agent):
        result = agent._tool_event_demand({
            "venue": "petco_park",
            "event_type": "SPORTS_MAJOR",
        })
        assert "error" not in result
        assert result["inbound_peak_veh_hr"] > 0
        assert result["outbound_peak_veh_hr"] > 0
        assert result["pre_deploy_min"] > 0

    def test_event_demand_concert(self, agent):
        result = agent._tool_event_demand({
            "venue": "pechanga_arena",
            "event_type": "CONCERT_LARGE",
        })
        assert "inbound_peak_veh_hr" in result

    def test_retiming_drift_triggered(self, agent):
        result = agent._tool_retiming_drift({
            "baseline_delay_s_veh": 30.0,
            "current_delay_s_veh": 45.0,
            "baseline_split_failure_pct": 5.0,
            "current_split_failure_pct": 20.0,
            "baseline_volume_veh_hr": 1200,
            "current_volume_veh_hr": 1600,
        })
        assert result["triggered"] is True
        assert result["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result["recommended_action"] in (
            "NO_ACTION", "OFFSET_ONLY", "CYCLE_ADJUST", "FULL_RETIMING",
            "NSGA_RETIMING", "offset_only", "cycle_adjust", "full_retiming",
        )

    def test_retiming_no_drift(self, agent):
        result = agent._tool_retiming_drift({
            "baseline_delay_s_veh": 35.0,
            "current_delay_s_veh": 36.0,
            "baseline_split_failure_pct": 5.0,
            "current_split_failure_pct": 6.0,
            "baseline_volume_veh_hr": 1200,
            "current_volume_veh_hr": 1210,
        })
        assert result["triggered"] is False
        assert result["recommended_action"] == "NO_ACTION"

    def test_tools_list(self, agent):
        tools = agent._tools()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert "predict_spillback" in names
        assert "get_event_demand" in names
        assert "detect_retiming_drift" in names
        assert "run_what_if_scenario" in names
        assert "run_multi_objective_opt" in names

    def test_tools_have_required_schema(self, agent):
        for tool in agent._tools():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_citations_cover_golden_features(self, agent):
        cites = agent._citations()
        gf_refs = [c for c in cites if "GF" in c or "aito." in c]
        assert len(gf_refs) >= 4


class TestSyntheticWhatIf:
    def test_returns_all_fields(self):
        result = _synthetic_what_if(
            {"scenario_type": "DEMAND_SHIFT", "name": "test", "demand_scale_factor": 1.2},
            "test error",
        )
        assert "baseline_delay_s_veh" in result
        assert "scenario_delay_s_veh" in result
        assert "delay_change_pct" in result
        assert "throughput_change_pct" in result
        assert "_note" in result

    def test_demand_scale_increases_delay(self):
        base = _synthetic_what_if({"demand_scale_factor": 1.0, "name": "base"}, "e")
        high = _synthetic_what_if({"demand_scale_factor": 1.5, "name": "high"}, "e")
        assert high["scenario_delay_s_veh"] > base["scenario_delay_s_veh"]
