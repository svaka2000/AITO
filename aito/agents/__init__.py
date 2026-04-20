"""aito/agents/ — Multi-agent Claude Opus 4.7 integration for AITO.

Five agents:
  orchestrator    — intent classification + delegation
  scenario_agent  — what-if / spillback / event-demand analysis
  incident_agent  — three-sub-agent incident response loop
  negotiation_agent — dual-Claude Caltrans / City of San Diego coordination
  carbon_agent    — EPA MOVES + Verra VCS / CARB LCFS report generation
"""
from .base_agent import AgentResult, BaseAgent

__all__ = ["AgentResult", "BaseAgent"]
