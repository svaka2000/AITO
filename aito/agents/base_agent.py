"""aito/agents/base_agent.py

Shared foundation for all AITO specialist agents.

Contract:
  Every agent subclass implements _tools() -> list[dict] and _run_tool(name, inputs).
  Call .run(query, context) to get an AgentResult.
  Call .stream(query, context) to get a streaming generator for the Streamlit UI.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    inputs: dict
    output: Any
    duration_ms: float


@dataclass
class AgentResult:
    """Standardised return contract shared by all agents."""
    agent_name: str
    query: str
    final_output: str
    reasoning_trace: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    confidence: float = 1.0
    citations_to_modules: list[str] = field(default_factory=list)
    error: Optional[str] = None


class BaseAgent:
    """Base class — wraps Anthropic SDK with Opus 4.7, adaptive thinking, prompt caching."""

    MODEL = "claude-opus-4-7"
    MAX_TOOL_ROUNDS = 12

    # Subclasses override these
    AGENT_NAME: str = "base"
    SYSTEM_PROMPT: str = "You are an AITO traffic engineering agent."

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazy init

    # ------------------------------------------------------------------
    # Client setup
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError as exc:
                raise RuntimeError("pip install anthropic") from exc
        return self._client

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _tools(self) -> list[dict]:
        """Return list of Claude tool definitions (JSON schema format)."""
        return []

    def _run_tool(self, name: str, inputs: dict) -> Any:
        """Dispatch a tool call to the underlying aito module function."""
        raise NotImplementedError(f"No tool handler for '{name}'")

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    def run(self, query: str, context: dict | None = None) -> AgentResult:
        """Run the agentic loop and return a complete AgentResult."""
        if not self._api_key:
            return AgentResult(
                agent_name=self.AGENT_NAME,
                query=query,
                final_output="ANTHROPIC_API_KEY not set — cannot call Claude.",
                error="no_api_key",
            )

        client = self._get_client()
        messages = self._build_initial_messages(query, context)
        tools = self._tools()
        tool_calls_log: list[ToolCall] = []
        reasoning_blocks: list[str] = []

        for _round in range(self.MAX_TOOL_ROUNDS):
            kwargs = dict(
                model=self.MODEL,
                max_tokens=16000,
                system=[
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "xhigh"},
                messages=messages,
            )
            if tools:
                kwargs["tools"] = tools

            response = client.messages.create(**kwargs)

            # Collect reasoning traces
            for block in response.content:
                if block.type == "thinking" and getattr(block, "summary", None):
                    reasoning_blocks.append(block.summary)

            if response.stop_reason == "end_turn":
                final_text = _extract_text(response.content)
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    query=query,
                    final_output=final_text,
                    reasoning_trace="\n\n---\n\n".join(reasoning_blocks),
                    tool_calls=tool_calls_log,
                    citations_to_modules=self._citations(),
                )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        t0 = time.monotonic()
                        try:
                            result = self._run_tool(block.name, block.input)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        elapsed = (time.monotonic() - t0) * 1000
                        tool_calls_log.append(
                            ToolCall(
                                name=block.name,
                                inputs=block.input,
                                output=result,
                                duration_ms=round(elapsed, 1),
                            )
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            break

        final_text = _extract_text(response.content) if "response" in dir() else "Agent loop exhausted."
        return AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=final_text,
            reasoning_trace="\n\n---\n\n".join(reasoning_blocks),
            tool_calls=tool_calls_log,
            citations_to_modules=self._citations(),
        )

    def stream(self, query: str, context: dict | None = None) -> Generator[dict, None, AgentResult]:
        """
        Streaming generator yielding events for the Streamlit UI.

        Yields dicts:
          {"type": "thinking_delta", "text": ...}
          {"type": "text_delta", "text": ...}
          {"type": "tool_start", "name": ..., "inputs": ...}
          {"type": "tool_result", "name": ..., "output": ..., "duration_ms": ...}
          {"type": "done", "result": AgentResult}
        """
        if not self._api_key:
            yield {"type": "text_delta", "text": "ANTHROPIC_API_KEY not set."}
            return AgentResult(
                agent_name=self.AGENT_NAME,
                query=query,
                final_output="ANTHROPIC_API_KEY not set.",
                error="no_api_key",
            )

        client = self._get_client()
        messages = self._build_initial_messages(query, context)
        tools = self._tools()
        tool_calls_log: list[ToolCall] = []
        reasoning_blocks: list[str] = []
        final_text_parts: list[str] = []

        for _round in range(self.MAX_TOOL_ROUNDS):
            kwargs = dict(
                model=self.MODEL,
                max_tokens=16000,
                system=[
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "xhigh"},
                messages=messages,
            )
            if tools:
                kwargs["tools"] = tools

            import anthropic
            stop_reason = None
            response_content = []

            with client.messages.stream(**kwargs) as stream:
                current_thinking: list[str] = []

                for event in stream:
                    etype = type(event).__name__

                    if etype == "RawContentBlockStartEvent":
                        block = event.content_block
                        if getattr(block, "type", None) == "thinking":
                            current_thinking = []

                    elif etype == "RawContentBlockDeltaEvent":
                        delta = event.delta
                        dtype = getattr(delta, "type", None)
                        if dtype == "thinking_delta":
                            current_thinking.append(delta.thinking)
                            yield {"type": "thinking_delta", "text": delta.thinking}
                        elif dtype == "text_delta":
                            final_text_parts.append(delta.text)
                            yield {"type": "text_delta", "text": delta.text}

                    elif etype == "RawMessageStopEvent":
                        pass

                msg = stream.get_final_message()
                stop_reason = msg.stop_reason
                response_content = msg.content

                # Collect thinking summaries
                for block in response_content:
                    if getattr(block, "type", None) == "thinking" and getattr(block, "summary", None):
                        reasoning_blocks.append(block.summary)

            if stop_reason == "end_turn":
                result = AgentResult(
                    agent_name=self.AGENT_NAME,
                    query=query,
                    final_output="".join(final_text_parts),
                    reasoning_trace="\n\n---\n\n".join(reasoning_blocks),
                    tool_calls=tool_calls_log,
                    citations_to_modules=self._citations(),
                )
                yield {"type": "done", "result": result}
                return result

            if stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response_content})
                tool_results = []
                for block in response_content:
                    if getattr(block, "type", None) == "tool_use":
                        yield {"type": "tool_start", "name": block.name, "inputs": block.input}
                        t0 = time.monotonic()
                        try:
                            result = self._run_tool(block.name, block.input)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        elapsed = (time.monotonic() - t0) * 1000
                        tool_calls_log.append(
                            ToolCall(name=block.name, inputs=block.input, output=result, duration_ms=round(elapsed, 1))
                        )
                        yield {"type": "tool_result", "name": block.name, "output": result, "duration_ms": round(elapsed, 1)}
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
                continue

            break

        final = "".join(final_text_parts) or "Agent loop exhausted."
        result = AgentResult(
            agent_name=self.AGENT_NAME,
            query=query,
            final_output=final,
            reasoning_trace="\n\n---\n\n".join(reasoning_blocks),
            tool_calls=tool_calls_log,
            citations_to_modules=self._citations(),
        )
        yield {"type": "done", "result": result}
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_messages(self, query: str, context: dict | None) -> list[dict]:
        ctx_text = ""
        if context:
            ctx_text = "\n\n[Context]\n" + json.dumps(context, indent=2, default=str)
        return [{"role": "user", "content": query + ctx_text}]

    def _citations(self) -> list[str]:
        return ["HCM 7th Edition", "MUTCD 2023", "EPA MOVES2014b", "FHWA Signal Timing Manual"]


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _extract_text(content_blocks) -> str:
    parts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)
