"""tests/agents/test_nl_engineer.py — NLEngineerSession and query classifier tests."""
import pytest
from aito.interface.nl_engineer import (
    NLEngineerSession, NLResponse, classify_query, QueryType,
)


class TestClassifyQuery:
    def test_timing_keywords(self):
        assert classify_query("why is the cycle so long?") == QueryType.EXPLAIN_TIMING
        assert classify_query("explain the green split") == QueryType.EXPLAIN_TIMING
        assert classify_query("what does the offset do?") == QueryType.EXPLAIN_TIMING

    def test_what_if_keywords(self):
        # "what if" alone (no timing keyword) → what_if
        assert classify_query("what if demand increases by 20%?") == QueryType.WHAT_IF
        assert classify_query("what happens if I increase green time?") == QueryType.WHAT_IF

    def test_compare_keywords(self):
        assert classify_query("compare AITO vs insync") == QueryType.COMPARE
        assert classify_query("is AITO better than scoot?") == QueryType.COMPARE

    def test_carbon_keywords(self):
        assert classify_query("what are the carbon emissions?") == QueryType.CARBON
        assert classify_query("how much CO2 do we save?") == QueryType.CARBON

    def test_optimize_keywords(self):
        assert classify_query("optimize the timing plan for PM peak") == QueryType.OPTIMIZE

    def test_general_fallback(self):
        assert classify_query("hello world") == QueryType.GENERAL


class TestNLResponse:
    def test_defaults(self):
        r = NLResponse(query="q", answer="a")
        assert r.agent_name == "nl_engineer"
        assert r.reasoning_trace == ""
        assert r.tool_calls == []
        assert r.citations == []
        assert r.used_claude_api is False
        assert r.confidence == 1.0


class TestNLEngineerSession:
    def test_template_mode_no_api_key(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            session = NLEngineerSession(anthropic_api_key=None)
            assert not session._use_claude
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_template_response_timing(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            session = NLEngineerSession(anthropic_api_key=None)
            resp = session.ask("explain the cycle length")
            assert isinstance(resp, NLResponse)
            assert resp.used_claude_api is False
            assert len(resp.answer) > 10
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_template_carbon_response(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from unittest.mock import MagicMock
            mock_corridor = MagicMock()
            mock_corridor.name = "Rosecrans St"
            mock_corridor.intersections = [1, 2, 3, 4]
            mock_corridor.aadt = 28000
            session = NLEngineerSession(corridor=mock_corridor, anthropic_api_key=None)
            resp = session.ask("what are the carbon emissions?")
            assert "Carbon" in resp.answer or "CO₂" in resp.answer
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_template_compare_insync(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            session = NLEngineerSession(anthropic_api_key=None)
            resp = session.ask("compare AITO vs insync")
            assert "InSync" in resp.answer or "AITO" in resp.answer
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_followup_suggestions_not_empty(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            session = NLEngineerSession(anthropic_api_key=None)
            resp = session.ask("explain the timing plan")
            assert len(resp.followup_suggestions) > 0
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_set_corridor_resets_orchestrator(self):
        session = NLEngineerSession(anthropic_api_key="test")
        session._orchestrator = "stub"
        session.set_corridor(None)
        assert session._orchestrator is None

    def test_set_optimization_result_resets_orchestrator(self):
        session = NLEngineerSession(anthropic_api_key="test")
        session._orchestrator = "stub"
        session.set_optimization_result(None)
        assert session._orchestrator is None

    def test_stream_template_mode_yields_events(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            session = NLEngineerSession(anthropic_api_key=None)
            events = list(session.stream("explain the green split"))
            types = {e.get("type") for e in events}
            assert "text_delta" in types
            assert "done" in types
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_api_key_from_env(self):
        import os
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        session = NLEngineerSession()
        assert session._use_claude
        del os.environ["ANTHROPIC_API_KEY"]
