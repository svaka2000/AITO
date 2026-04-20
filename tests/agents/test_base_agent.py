"""tests/agents/test_base_agent.py — Unit tests for BaseAgent and AgentResult."""
import pytest
from aito.agents.base_agent import AgentResult, BaseAgent, ToolCall, _extract_text


class _NoOpAgent(BaseAgent):
    AGENT_NAME = "noop"
    SYSTEM_PROMPT = "Test agent."

    def _tools(self):
        return []

    def _run_tool(self, name, inputs):
        raise NotImplementedError(name)


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(agent_name="test", query="q", final_output="out")
        assert r.reasoning_trace == ""
        assert r.tool_calls == []
        assert r.confidence == 1.0
        assert r.citations_to_modules == []
        assert r.error is None

    def test_with_tool_calls(self):
        tc = ToolCall(name="my_tool", inputs={"x": 1}, output={"y": 2}, duration_ms=12.3)
        r = AgentResult(agent_name="a", query="q", final_output="f", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "my_tool"
        assert r.tool_calls[0].duration_ms == 12.3


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(name="tool", inputs={"a": 1}, output={"b": 2}, duration_ms=5.0)
        assert tc.name == "tool"
        assert tc.inputs["a"] == 1
        assert tc.output["b"] == 2
        assert tc.duration_ms == 5.0


class TestBaseAgent:
    def test_no_api_key_returns_error_result(self):
        agent = _NoOpAgent(api_key=None)
        # Patch env to ensure no key
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = agent.run("test query")
            assert result.error == "no_api_key"
            assert "ANTHROPIC_API_KEY" in result.final_output
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_model_constant(self):
        assert BaseAgent.MODEL == "claude-opus-4-7"

    def test_max_tool_rounds(self):
        assert BaseAgent.MAX_TOOL_ROUNDS >= 8

    def test_build_initial_messages_no_context(self):
        agent = _NoOpAgent(api_key="test")
        msgs = agent._build_initial_messages("hello", None)
        assert msgs[0]["role"] == "user"
        assert "hello" in msgs[0]["content"]

    def test_build_initial_messages_with_context(self):
        agent = _NoOpAgent(api_key="test")
        msgs = agent._build_initial_messages("hello", {"corridor": "rosecrans"})
        content = msgs[0]["content"]
        assert "rosecrans" in content

    def test_citations_default(self):
        agent = _NoOpAgent()
        cites = agent._citations()
        assert any("HCM" in c for c in cites)
        assert any("MUTCD" in c for c in cites)


class TestExtractText:
    def test_extracts_text_blocks(self):
        class _Block:
            def __init__(self, btype, text):
                self.type = btype
                self.text = text

        blocks = [
            _Block("thinking", "internal reasoning"),
            _Block("text", "hello world"),
            _Block("text", "more text"),
        ]
        result = _extract_text(blocks)
        assert "hello world" in result
        assert "more text" in result
        assert "internal reasoning" not in result

    def test_empty_blocks(self):
        assert _extract_text([]) == ""
