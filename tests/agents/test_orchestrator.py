"""tests/agents/test_orchestrator.py — Orchestrator intent classification tests."""
import pytest
from aito.agents.orchestrator import OrchestratorAgent, classify_intent, Intent


class TestClassifyIntent:
    def test_scenario_keywords(self):
        assert classify_intent("what if we reduce the cycle to 90s?") == Intent.SCENARIO
        assert classify_intent("simulate a Petco Park event on Harbor Drive") == Intent.SCENARIO
        assert classify_intent("predict spillback at Midway intersection") == Intent.SCENARIO

    def test_incident_keywords(self):
        assert classify_intent("there's a crash at Rosecrans and Midway") == Intent.INCIDENT
        assert classify_intent("accident blocking 2 lanes, need reroute") == Intent.INCIDENT
        assert classify_intent("signal failure at the intersection") == Intent.INCIDENT
        assert classify_intent("disabled vehicle on Genesee NB approach") == Intent.INCIDENT

    def test_negotiation_keywords(self):
        assert classify_intent("negotiate timing with Caltrans District 11") == Intent.NEGOTIATION
        assert classify_intent("NTCIP 1211 coordination at boundary intersection") == Intent.NEGOTIATION
        assert classify_intent("SANDAG regional coordination needed") == Intent.NEGOTIATION

    def test_carbon_keywords(self):
        assert classify_intent("What's the CARB LCFS revenue for Rosecrans?") == Intent.CARBON
        assert classify_intent("calculate CO2 emissions from signal optimization") == Intent.CARBON
        assert classify_intent("Verra VCS carbon credit portfolio") == Intent.CARBON
        assert classify_intent("how many tonnes of CO2 do we save?") == Intent.CARBON

    def test_general_fallback(self):
        result = classify_intent("hello there")
        assert result == Intent.GENERAL

    def test_case_insensitive(self):
        assert classify_intent("CRASH ON ROSECRANS") == Intent.INCIDENT
        assert classify_intent("CARBON CREDITS") == Intent.CARBON


class TestOrchestratorAgent:
    def test_instantiation(self):
        o = OrchestratorAgent()
        assert o.AGENT_NAME == "orchestrator"
        assert o.corridor is None
        assert o.optimization_result is None

    def test_no_api_key_returns_error(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            o = OrchestratorAgent(api_key=None)
            result = o.route("test query")
            assert result.error == "no_api_key"
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_set_corridor_clears_cache(self):
        o = OrchestratorAgent(api_key="test")
        o._specialist_cache["scenario"] = "stub"
        o.set_corridor(None)
        assert o._specialist_cache == {}

    def test_set_optimization_result_clears_cache(self):
        o = OrchestratorAgent(api_key="test")
        o._specialist_cache["carbon"] = "stub"
        o.set_optimization_result(None)
        assert o._specialist_cache == {}

    def test_build_corridor_context_empty(self):
        o = OrchestratorAgent()
        ctx = o._build_corridor_context()
        assert isinstance(ctx, dict)
        assert len(ctx) == 0
