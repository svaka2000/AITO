"""aito/interface/agent_dashboard.py

AITO Multi-Agent Intelligence Dashboard — Claude Opus 4.7 powered.

4-pane layout:
  LEFT COL  (40%): Chat input + conversation history
  RIGHT COL (60%):
    top-left:  Agent Reasoning (thinking traces)
    top-right: Tool Call Log (name, inputs, outputs, timing)
    bottom:    Agent Result + Citations

Sidebar: corridor selector, API key input, agent graph visualization.

Run: streamlit run aito/interface/agent_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

st.set_page_config(
    page_title="AITO Intelligence — Multi-Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

AITO_AGENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep:    #07111F;
    --bg-app:     #0A1628;
    --bg-card:    #0D1E35;
    --bg-card-hi: #112545;
    --border-subtle: rgba(0, 194, 203, 0.16);
    --border-medium: rgba(0, 194, 203, 0.30);
    --teal:        #00C2CB;
    --teal-dim:    #008E95;
    --teal-glow:   rgba(0, 194, 203, 0.18);
    --gold:        #F59E0B;
    --purple:      #8B5CF6;
    --success:      #10B981;
    --danger:       #EF4444;
    --warning:      #F59E0B;
    --text-primary:   #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted:     #64748B;
    --font-ui:   'Fira Sans', sans-serif;
    --font-mono: 'Fira Code', monospace;
}

.stApp { background: linear-gradient(165deg, #07111F 0%, #0A1628 60%, #081424 100%); color: var(--text-primary); font-family: var(--font-ui); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #070F1C 0%, #0A1628 100%); border-right: 1px solid var(--border-subtle); }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
h1, h2, h3 { color: var(--teal) !important; font-weight: 700 !important; }
h4, h5, h6 { color: var(--gold) !important; }
p { color: var(--text-secondary); }

.agent-badge {
    display: inline-block;
    background: var(--teal-glow);
    border: 1px solid var(--border-medium);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--teal);
    font-weight: 600;
}

.tool-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.tool-name {
    font-family: var(--font-mono);
    color: var(--gold);
    font-size: 13px;
    font-weight: 600;
}

.tool-timing {
    font-family: var(--font-mono);
    color: var(--text-muted);
    font-size: 11px;
    float: right;
}

.thinking-block {
    background: rgba(139, 92, 246, 0.08);
    border-left: 3px solid var(--purple);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    font-style: italic;
}

.chat-user {
    background: rgba(0, 194, 203, 0.08);
    border: 1px solid var(--border-subtle);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 14px;
    color: var(--text-primary);
}

.chat-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px 12px 12px 4px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-size: 14px;
    color: var(--text-primary);
}

.citation-pill {
    display: inline-block;
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    color: var(--gold);
    margin: 2px 3px;
    font-family: var(--font-mono);
}

.intent-tag {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: var(--font-mono);
    font-weight: 600;
    margin-left: 8px;
}
.intent-scenario  { background: rgba(99,102,241,0.15); color: #818CF8; }
.intent-incident  { background: rgba(239,68,68,0.15); color: #F87171; }
.intent-negotiation { background: rgba(16,185,129,0.15); color: #34D399; }
.intent-carbon    { background: rgba(245,158,11,0.15); color: #FCD34D; }
.intent-general   { background: rgba(100,116,139,0.15); color: #94A3B8; }

.agent-graph {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-secondary);
}

.pane-header {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--teal);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
}
</style>
"""

st.markdown(AITO_AGENT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# San Diego demo corridors
# ---------------------------------------------------------------------------

SD_CORRIDORS = {
    "Rosecrans Street (4 intersections)": {
        "id": "rosecrans_st", "n_intersections": 4,
        "aadt": 28000, "speed_mph": 35,
        "description": "Rosecrans St between Sports Arena Blvd and Midway Dr",
        "key_cross_streets": ["Sports Arena Blvd", "Evergreen St", "Lytton St", "Midway Dr"],
        "jurisdiction": "City of San Diego",
        "baseline_delay_s_veh": 48.2,
        "optimized_delay_s_veh": 33.3,
    },
    "Mira Mesa Blvd (6 intersections)": {
        "id": "mira_mesa_blvd", "n_intersections": 6,
        "aadt": 35000, "speed_mph": 45,
        "description": "Mira Mesa Blvd from I-15 to Camino Ruiz",
        "key_cross_streets": ["I-15 NB ramps", "Camino Santa Fe", "Westview Pkwy", "Black Mountain Rd", "Camino Ruiz", "Mira Mesa Mall Loop"],
        "jurisdiction": "City of San Diego",
        "baseline_delay_s_veh": 42.8,
        "optimized_delay_s_veh": 29.5,
    },
    "Genesee Avenue (5 intersections)": {
        "id": "genesee_ave", "n_intersections": 5,
        "aadt": 22000, "speed_mph": 35,
        "description": "Genesee Ave from Gilman Dr to Governor Dr",
        "key_cross_streets": ["Gilman Dr", "Nobel Dr", "La Jolla Village Dr", "Regents Rd", "Governor Dr"],
        "jurisdiction": "City of San Diego / UCSD",
        "baseline_delay_s_veh": 38.5,
        "optimized_delay_s_veh": 26.9,
    },
}

SAMPLE_QUERIES = {
    "scenario": [
        "What happens to Rosecrans delays if we reduce cycle from 120s to 90s during AM peak?",
        "Simulate a Petco Park SPORTS_MAJOR event impact on Harbor Drive approach",
        "If we close the SB approach at Lytton due to construction, how far does the queue back up?",
    ],
    "incident": [
        "There's a major crash blocking 2 of 3 lanes on Rosecrans at Midway Dr. What's the response plan?",
        "Signal failure at the Mira Mesa/I-15 ramp intersection. All-red flash condition.",
        "Disabled vehicle on Genesee Ave NB approach at La Jolla Village Dr during PM peak.",
    ],
    "negotiation": [
        "Negotiate a shared cycle plan for Rosecrans St with Caltrans District 11",
        "Coordinate Mira Mesa Blvd timing at the I-15 NB ramp intersection with Caltrans",
        "What NTCIP 1211 handoff parameters should we use at the Genesee/I-5 boundary?",
    ],
    "carbon": [
        "What's the full CARB LCFS carbon credit portfolio for the Rosecrans corridor?",
        "Generate a Verra VCS MRV report for Mira Mesa Blvd (35,000 AADT, 6 intersections)",
        "Compare CO₂ reduction between Rosecrans and Genesee Ave — which generates more credits?",
    ],
}

INTENT_LABELS = {
    "scenario": ("Scenario", "intent-scenario"),
    "incident": ("Incident", "intent-incident"),
    "negotiation": ("Negotiation", "intent-negotiation"),
    "carbon": ("Carbon", "intent-carbon"),
    "general": ("General", "intent-general"),
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "messages": [],           # list of {"role", "content", "meta"}
        "tool_calls_log": [],     # list of ToolCall objects from last run
        "reasoning_traces": [],   # list of thinking block strings
        "last_citations": [],
        "last_agent": None,
        "last_intent": None,
        "is_streaming": False,
        "api_key_valid": False,
        "selected_corridor": "Rosecrans Street (4 intersections)",
        "session": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 12px;">
      <div style="font-family:'Fira Code',monospace; font-size:22px; font-weight:700; letter-spacing:6px; color:#00C2CB;">AITO</div>
      <div style="font-size:11px; color:#64748B; letter-spacing:2px; margin-top:2px;">AI TRAFFIC OPTIMIZER</div>
      <div style="font-size:10px; color:#475569; margin-top:4px;">Multi-Agent Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Corridor")
    corridor_choice = st.selectbox(
        "Active Corridor",
        list(SD_CORRIDORS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    if corridor_choice != st.session_state.get("selected_corridor"):
        st.session_state.selected_corridor = corridor_choice
        st.session_state.session = None
        st.session_state.messages = []
        st.session_state.tool_calls_log = []
        st.session_state.reasoning_traces = []

    c = SD_CORRIDORS[corridor_choice]
    st.markdown(f"""
    <div style="background:#0D1E35; border:1px solid rgba(0,194,203,0.16); border-radius:8px; padding:10px 12px; font-size:12px; margin-top:6px;">
      <div style="color:#00C2CB; font-weight:600;">{c['id']}</div>
      <div style="color:#94A3B8; margin-top:4px;">{c['description']}</div>
      <div style="margin-top:6px; display:flex; gap:12px;">
        <span style="color:#F59E0B;">{c['n_intersections']} signals</span>
        <span style="color:#94A3B8;">{c['aadt']:,} AADT</span>
        <span style="color:#94A3B8;">{c['speed_mph']} mph</span>
      </div>
      <div style="margin-top:4px; color:#64748B; font-size:11px;">{c['jurisdiction']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### API Key")
    api_key_input = st.text_input(
        "ANTHROPIC_API_KEY",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        label_visibility="collapsed",
        placeholder="sk-ant-...",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        st.session_state.api_key_valid = True
        st.markdown('<div style="color:#10B981; font-size:12px;">✓ API key set — Claude Opus 4.7 active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#64748B; font-size:12px;">No API key — template mode</div>', unsafe_allow_html=True)

    st.markdown("#### Agent Graph")
    st.markdown("""
    <div class="agent-graph">
      <div style="color:#00C2CB; font-weight:600; margin-bottom:12px;">ORCHESTRATOR</div>
      <div style="display:flex; justify-content:space-around; margin-top:8px;">
        <div style="text-align:center;">
          <div style="background:rgba(99,102,241,0.2); border:1px solid #6366F1; border-radius:6px; padding:4px 8px; color:#818CF8; font-size:10px;">SCENARIO</div>
          <div style="color:#475569; font-size:9px; margin-top:2px;">GF3·6·7·12·15</div>
        </div>
        <div style="text-align:center;">
          <div style="background:rgba(239,68,68,0.2); border:1px solid #EF4444; border-radius:6px; padding:4px 8px; color:#F87171; font-size:10px;">INCIDENT</div>
          <div style="color:#475569; font-size:9px; margin-top:2px;">3-sub-agent</div>
        </div>
        <div style="text-align:center;">
          <div style="background:rgba(16,185,129,0.2); border:1px solid #10B981; border-radius:6px; padding:4px 8px; color:#34D399; font-size:10px;">NEGOTIATE</div>
          <div style="color:#475569; font-size:9px; margin-top:2px;">GF8·NTCIP</div>
        </div>
        <div style="text-align:center;">
          <div style="background:rgba(245,158,11,0.2); border:1px solid #F59E0B; border-radius:6px; padding:4px 8px; color:#FCD34D; font-size:10px;">CARBON</div>
          <div style="color:#475569; font-size:9px; margin-top:2px;">GF2·9·11</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Sample Queries")
    tab_s, tab_i, tab_n, tab_c = st.tabs(["Scenario", "Incident", "Negotiate", "Carbon"])
    with tab_s:
        for q in SAMPLE_QUERIES["scenario"]:
            if st.button(q[:55] + "…", key=f"sq_{hash(q)}", use_container_width=True):
                st.session_state["_prefill_query"] = q
    with tab_i:
        for q in SAMPLE_QUERIES["incident"]:
            if st.button(q[:55] + "…", key=f"iq_{hash(q)}", use_container_width=True):
                st.session_state["_prefill_query"] = q
    with tab_n:
        for q in SAMPLE_QUERIES["negotiation"]:
            if st.button(q[:55] + "…", key=f"nq_{hash(q)}", use_container_width=True):
                st.session_state["_prefill_query"] = q
    with tab_c:
        for q in SAMPLE_QUERIES["carbon"]:
            if st.button(q[:55] + "…", key=f"cq_{hash(q)}", use_container_width=True):
                st.session_state["_prefill_query"] = q


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

@st.cache_resource
def _make_session(corridor_id: str, api_key: str):
    from aito.interface.nl_engineer import NLEngineerSession
    return NLEngineerSession(anthropic_api_key=api_key or None)


def get_session():
    if st.session_state.session is None:
        api = os.environ.get("ANTHROPIC_API_KEY", "")
        st.session_state.session = _make_session(
            st.session_state.selected_corridor, api
        )
    return st.session_state.session


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.markdown("## 🧠 AITO Multi-Agent Intelligence")
st.markdown(
    '<div style="color:#64748B; font-size:13px; margin-bottom:20px;">'
    'Claude Opus 4.7 · Adaptive Thinking · 5 Specialist Agents · 15 Golden Features'
    '</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([4, 6], gap="large")


# ---------------------------------------------------------------------------
# LEFT: Chat pane
# ---------------------------------------------------------------------------

with left_col:
    st.markdown('<div class="pane-header">ENGINEER CONSOLE</div>', unsafe_allow_html=True)

    chat_container = st.container(height=480, border=False)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                intent = msg.get("meta", {}).get("intent", "")
                badge_html = ""
                if intent in INTENT_LABELS:
                    label, cls = INTENT_LABELS[intent]
                    badge_html = f'<span class="intent-tag {cls}">{label}</span>'
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}{badge_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                agent = msg.get("meta", {}).get("agent", "")
                badge = f'<span class="agent-badge">{agent}</span> ' if agent else ""
                st.markdown(
                    f'<div class="chat-assistant">{badge}{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    prefill = st.session_state.pop("_prefill_query", "")
    query = st.chat_input(
        "Ask AITO anything about your corridor…",
        key="chat_input",
    ) or prefill

    if st.button("Clear conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.tool_calls_log = []
        st.session_state.reasoning_traces = []
        st.session_state.last_citations = []
        st.session_state.last_agent = None
        st.rerun()


# ---------------------------------------------------------------------------
# RIGHT: 3-pane analysis area
# ---------------------------------------------------------------------------

with right_col:
    r_top_left, r_top_right = st.columns(2, gap="medium")

    with r_top_left:
        st.markdown('<div class="pane-header">AGENT REASONING</div>', unsafe_allow_html=True)
        reasoning_container = st.container(height=230, border=False)
        with reasoning_container:
            if st.session_state.reasoning_traces:
                for i, trace in enumerate(st.session_state.reasoning_traces):
                    if trace.strip():
                        st.markdown(
                            f'<div class="thinking-block">{trace[:600]}{"…" if len(trace) > 600 else ""}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    '<div style="color:#475569; font-size:12px; font-style:italic;">Thinking traces will appear here during agent reasoning…</div>',
                    unsafe_allow_html=True,
                )

    with r_top_right:
        st.markdown('<div class="pane-header">TOOL CALLS</div>', unsafe_allow_html=True)
        tools_container = st.container(height=230, border=False)
        with tools_container:
            if st.session_state.tool_calls_log:
                for tc in st.session_state.tool_calls_log[-8:]:
                    name = getattr(tc, "name", tc.get("name", "?") if isinstance(tc, dict) else "?")
                    duration = getattr(tc, "duration_ms", tc.get("duration_ms", 0) if isinstance(tc, dict) else 0)
                    inputs = getattr(tc, "inputs", tc.get("inputs", {}) if isinstance(tc, dict) else {})
                    output = getattr(tc, "output", tc.get("output", {}) if isinstance(tc, dict) else {})
                    key_inputs = json.dumps({k: v for k, v in (inputs or {}).items() if k not in ("description",)}, default=str)[:120]
                    key_output = json.dumps(output, default=str)[:120] if output else ""
                    st.markdown(
                        f'<div class="tool-card">'
                        f'<span class="tool-name">{name}</span>'
                        f'<span class="tool-timing">{duration:.0f}ms</span>'
                        f'<div style="font-size:11px; color:#64748B; margin-top:4px; font-family:var(--font-mono);">{key_inputs}</div>'
                        f'<div style="font-size:11px; color:#94A3B8; margin-top:2px;">{key_output}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:#475569; font-size:12px; font-style:italic;">Tool calls will appear here as agents work…</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="pane-header" style="margin-top:16px;">AGENT RESULT & CITATIONS</div>', unsafe_allow_html=True)
    result_container = st.container(height=240, border=False)
    with result_container:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last = st.session_state.messages[-1]
            agent = last.get("meta", {}).get("agent", "")
            if agent:
                st.markdown(
                    f'<div style="margin-bottom:8px;"><span class="agent-badge">{agent}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(last["content"])
            if st.session_state.last_citations:
                cites_html = "".join(
                    f'<span class="citation-pill">{c}</span>'
                    for c in st.session_state.last_citations
                )
                st.markdown(
                    f'<div style="margin-top:10px;">{cites_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#475569; font-size:12px; font-style:italic;">Agent response will appear here…</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

if query:
    from aito.agents.orchestrator import classify_intent

    intent = classify_intent(query)
    st.session_state.messages.append({"role": "user", "content": query, "meta": {"intent": intent}})
    st.session_state.tool_calls_log = []
    st.session_state.reasoning_traces = []
    st.session_state.last_citations = []

    session = get_session()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        # Streaming mode — show live token output
        with right_col:
            stream_placeholder = st.empty()
        partial_text = []
        tool_calls_interim = []
        reasoning_interim = []
        agent_name = f"{intent}_agent"

        with st.spinner(f"AITO [{intent}_agent] thinking…"):
            try:
                for event in session.stream(query):
                    etype = event.get("type")
                    if etype == "routing":
                        agent_name = f"{event.get('intent','?')}_agent (via orchestrator)"
                    elif etype == "thinking_delta":
                        reasoning_interim.append(event.get("text", ""))
                    elif etype == "text_delta":
                        partial_text.append(event.get("text", ""))
                    elif etype == "tool_result":
                        tool_calls_interim.append(event)
                    elif etype == "done":
                        result = event.get("result")
                        if result:
                            agent_name = getattr(result, "agent_name", agent_name)
                            tc_list = getattr(result, "tool_calls", [])
                            tool_calls_interim = list(tc_list)
                            reasoning_interim = [getattr(result, "reasoning_trace", "")]
                            st.session_state.last_citations = getattr(result, "citations", [])
            except Exception as exc:
                partial_text.append(f"\n\n*[Agent error: {exc}]*")

        final_answer = "".join(partial_text) or "No response."
        st.session_state.tool_calls_log = tool_calls_interim
        st.session_state.reasoning_traces = [t for t in reasoning_interim if t and t.strip()]
        st.session_state.last_agent = agent_name
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "meta": {"agent": agent_name, "intent": intent},
        })
    else:
        # Template mode
        from aito.interface.nl_engineer import classify_query
        response = session.ask(query)
        st.session_state.last_citations = response.citations
        st.session_state.last_agent = "template_mode"
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "meta": {"agent": "template_mode", "intent": intent},
        })

    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin-top:32px; padding-top:16px; border-top:1px solid rgba(0,194,203,0.10);
     text-align:center; color:#334155; font-size:11px; font-family:'Fira Code',monospace;">
  AITO v2.0 · Claude Opus 4.7 · 15 Golden Features · Caltrans District 11 / City of San Diego
</div>
""", unsafe_allow_html=True)
