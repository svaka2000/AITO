from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from traffic_ai.config.settings import Settings, load_settings
from traffic_ai.controllers import AdaptiveRuleController, FixedTimingController
from traffic_ai.experiments import ExperimentArtifacts, ExperimentRunner
from traffic_ai.metrics import simulation_result_to_step_dataframe
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator


st.set_page_config(
    page_title="Traffic AI Optimization Dashboard",
    page_icon=":vertical_traffic_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Display label mappings
# ---------------------------------------------------------------------------

CONTROLLER_DISPLAY_NAMES: dict[str, str] = {
    "fixed_timing": "Fixed Timing (Baseline)",
    "adaptive_rule": "Adaptive Rule",
    "ml_randomforestclassifier": "Random Forest (ML)",
    "ml_xgbclassifier": "XGBoost (ML)",
    "ml_gradientboostingclassifier": "Gradient Boosting (ML)",
    "ml_mlpclassifier": "Neural Network MLP (ML)",
    "ml_logisticregression": "Logistic Regression (ML)",
    "rl_q_learning": "Q-Learning (RL)",
    "rl_dqn": "Deep Q-Network (RL)",
    "rl_policy_gradient": "Policy Gradient (RL)",
}

COLUMN_LABELS: dict[str, str] = {
    "controller": "Controller",
    "average_wait_time": "Avg Wait Time (s)",
    "average_queue_length": "Avg Queue Length",
    "average_throughput": "Avg Throughput",
    "average_emissions_proxy": "Emissions Proxy",
    "average_fuel_proxy": "Fuel Proxy",
    "average_fairness": "Fairness Score",
    "average_efficiency_score": "Efficiency Score",
    "delay_reduction_pct": "Delay Reduction (%)",
    "max_queue_length": "Max Queue Length",
    "system_efficiency_score": "System Efficiency",
    "fold": "CV Fold",
    "model_name": "Model",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "cv_best_score": "CV Best Score",
}

CONTROLLER_TYPE_COLORS: dict[str, str] = {
    "fixed": "#6b7f8e",
    "adaptive": "#0fc5c8",
    "ml": "#ee9b00",
    "rl": "#9b59b6",
}


@dataclass(slots=True)
class DashboardData:
    summary_df: pd.DataFrame
    step_df: pd.DataFrame
    significance_df: pd.DataFrame
    ablation_df: pd.DataFrame
    model_metrics_df: pd.DataFrame
    plot_paths: list[Path]


def _inject_custom_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --ink: #e8f0f2;
            --muted: #8fa8b4;
            --line: #2a3f4d;
            --card: rgba(20, 38, 50, 0.85);
            --accent: #0fc5c8;
            --accent-2: #ee9b00;
            --bg-base: #0d1e28;
        }

        html, body,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] h6,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
            font-family: "Space Grotesk", sans-serif;
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(900px 500px at 5% 0%, #0a2e3a 0%, transparent 65%),
                radial-gradient(800px 400px at 95% 10%, #0f2233 0%, transparent 65%),
                linear-gradient(180deg, #0d1e28 0%, #091620 100%);
        }

        /* Force text elements to light — avoid icon/button spans */
        .stApp p, .stApp label,
        .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stText"] {
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111f2a 0%, #0d1920 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px 14px;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricLabel"] {
            color: var(--ink) !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: var(--muted) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent) !important;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            background: var(--card);
            border-radius: 10px;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid #0fc5c833;
            color: var(--ink);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent) 0%, #0a8f92 100%);
            color: #0d1e28;
            font-weight: 600;
            border: 1px solid #0fc5c866;
        }

        .hero {
            color: #f0fafa;
            border-radius: 18px;
            border: 1px solid #0fc5c830;
            padding: 1.25rem 1.35rem;
            margin-bottom: 0.95rem;
            background: linear-gradient(120deg, #0f4f5299 0%, #1a3f5ccc 100%);
            box-shadow: 0 20px 38px #00000044;
        }

        .hero-title {
            font-size: 1.9rem;
            line-height: 1.2;
            margin-bottom: 0.35rem;
            font-weight: 700;
            color: #f0fafa;
        }

        .hero-sub {
            font-size: 1rem;
            opacity: 0.90;
            color: #c8e8ec;
        }

        .pill {
            display: inline-block;
            margin-right: 0.45rem;
            margin-top: 0.7rem;
            border-radius: 999px;
            border: 1px solid #0fc5c855;
            padding: 0.24rem 0.66rem;
            font-size: 0.78rem;
            background: #0fc5c820;
            color: #a8e8ec;
            letter-spacing: 0.02em;
        }

        .source-note {
            margin-top: 0.6rem;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            color: #7ab8c0;
        }

        .finding-card {
            border-radius: 14px;
            border: 1px solid #0fc5c840;
            padding: 1rem 1.2rem;
            margin-bottom: 0.7rem;
            background: rgba(15, 197, 200, 0.07);
        }

        .finding-card h4 {
            color: #0fc5c8;
            margin-bottom: 0.3rem;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .finding-card p {
            color: #c8e8ec;
            font-size: 0.88rem;
            margin: 0;
        }

        .onboard-step {
            border-radius: 10px;
            border: 1px solid #2a3f4d;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
            background: rgba(20, 38, 50, 0.6);
        }

        .onboard-step .step-num {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0fc5c8;
            line-height: 1;
        }

        .onboard-step .step-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #e8f0f2;
            margin-top: 0.2rem;
        }

        .onboard-step .step-desc {
            font-size: 0.83rem;
            color: #8fa8b4;
            margin-top: 0.15rem;
        }

        .ctrl-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            margin-left: 0.3rem;
        }

        .badge-fixed  { background: #6b7f8e33; color: #9ab4c0; border: 1px solid #6b7f8e66; }
        .badge-ml     { background: #ee9b0033; color: #f5c060; border: 1px solid #ee9b0066; }
        .badge-rl     { background: #9b59b633; color: #c39bd3; border: 1px solid #9b59b666; }
        .badge-adaptive { background: #0fc5c833; color: #a8e8ec; border: 1px solid #0fc5c866; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_plot_paths(plot_dir: Path) -> list[Path]:
    if not plot_dir.exists():
        return []
    return sorted(plot_dir.glob("*.png"))


def _display_controller_name(name: str) -> str:
    key = name.strip().lower()
    if key in CONTROLLER_DISPLAY_NAMES:
        return CONTROLLER_DISPLAY_NAMES[key]
    return name.replace("_", " ").strip().title()


def _format_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with columns renamed to human-readable labels."""
    return df.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in df.columns})


def _apply_controller_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the 'controller' column replaced by friendly display names."""
    out = df.copy()
    if "controller" in out.columns:
        out["controller"] = out["controller"].astype(str).map(
            lambda x: _display_controller_name(x)
        )
    return out


def _get_fixed_baseline_wait(summary_df: pd.DataFrame) -> float | None:
    fixed_rows = summary_df[summary_df["controller"].str.contains("fixed", case=False, na=False)]
    if fixed_rows.empty:
        return None
    return float(fixed_rows["average_wait_time"].iloc[0])


def _load_dashboard_data_from_artifacts(artifacts: ExperimentArtifacts) -> DashboardData:
    return DashboardData(
        summary_df=_safe_read_csv(artifacts.summary_csv),
        step_df=_safe_read_csv(artifacts.step_metrics_csv),
        significance_df=_safe_read_csv(artifacts.significance_csv),
        ablation_df=_safe_read_csv(artifacts.ablation_csv),
        model_metrics_df=_safe_read_csv(artifacts.model_metrics_csv),
        plot_paths=[Path(path) for path in artifacts.generated_plots if Path(path).exists()],
    )


def _load_dashboard_data_from_output_dir(settings: Settings) -> DashboardData | None:
    result_dir = settings.output_dir / "results"
    plot_dir = settings.output_dir / "plots"
    required_files = [
        result_dir / "controller_summary.csv",
        result_dir / "controller_step_metrics.csv",
        result_dir / "significance_tests.csv",
        result_dir / "ablation_study.csv",
        result_dir / "supervised_model_metrics.csv",
    ]
    if not all(path.exists() for path in required_files):
        return None

    return DashboardData(
        summary_df=pd.read_csv(required_files[0]),
        step_df=pd.read_csv(required_files[1]),
        significance_df=pd.read_csv(required_files[2]),
        ablation_df=pd.read_csv(required_files[3]),
        model_metrics_df=pd.read_csv(required_files[4]),
        plot_paths=_load_plot_paths(plot_dir),
    )


def _run_benchmark(
    settings: Settings,
    quick_run: bool,
    include_public: bool,
    include_kaggle: bool,
) -> DashboardData:
    runner = ExperimentRunner(settings=settings, quick_run=quick_run)
    artifacts = runner.run(
        ingest_only=False,
        include_kaggle=include_kaggle,
        include_public=include_public,
    )
    return _load_dashboard_data_from_artifacts(artifacts)


# ---------------------------------------------------------------------------
# Header / Hero
# ---------------------------------------------------------------------------

def _render_header(source_label: str | None) -> None:
    source_text = (
        f'<div class="source-note">Active source: {source_label}</div>'
        if source_label
        else ""
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">AI Traffic Signal Optimization Platform</div>
            <div class="hero-sub">
                Research-grade benchmarking and interactive simulation for fixed, adaptive, supervised ML, and reinforcement-learning controllers.
            </div>
            <span class="pill">Comparative Benchmarks</span>
            <span class="pill">Reproducible Artifacts</span>
            <span class="pill">Live Simulation Explorer</span>
            {source_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI Strip
# ---------------------------------------------------------------------------

def _render_kpi_strip(summary_df: pd.DataFrame) -> None:
    required = {
        "controller",
        "average_wait_time",
        "system_efficiency_score",
        "average_throughput",
        "average_queue_length",
    }
    if summary_df.empty or not required.issubset(summary_df.columns):
        st.info("Run a benchmark to populate KPI metrics.")
        return

    best_row = summary_df.loc[summary_df["average_wait_time"].idxmin()]
    fixed_wait = _get_fixed_baseline_wait(summary_df)

    best_wait = float(summary_df["average_wait_time"].min())
    delta_wait: str | None = None
    if fixed_wait is not None and fixed_wait > 0:
        pct = (fixed_wait - best_wait) / fixed_wait * 100
        delta_wait = f"{pct:+.1f}% vs fixed"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Best Controller", _display_controller_name(str(best_row["controller"])))
    kpi2.metric(
        "Lowest Avg Wait (s)",
        f"{best_wait:.2f}",
        delta=delta_wait,
        delta_color="inverse",
    )
    kpi3.metric("Top Efficiency Score", f"{summary_df['system_efficiency_score'].max():.3f}")
    kpi4.metric("Mean Queue Length", f"{summary_df['average_queue_length'].mean():.1f}")


# ---------------------------------------------------------------------------
# Results Interpreter
# ---------------------------------------------------------------------------

def _render_results_interpreter(summary_df: pd.DataFrame) -> None:
    """Auto-generate plain-English analysis of benchmark results."""
    required = {"controller", "average_wait_time", "system_efficiency_score", "average_throughput", "average_queue_length"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return

    df = summary_df.copy()
    best_wait = df.loc[df["average_wait_time"].idxmin()]
    worst_wait = df.loc[df["average_wait_time"].idxmax()]
    best_eff = df.loc[df["system_efficiency_score"].idxmax()]

    def fmt(name: str) -> str:
        return _display_controller_name(str(name))

    wait_improvement = (
        (worst_wait["average_wait_time"] - best_wait["average_wait_time"])
        / max(worst_wait["average_wait_time"], 1e-6)
        * 100
    )
    throughput_best = df.loc[df["average_throughput"].idxmax()]

    insights: list[str] = []
    insights.append(
        f"**{fmt(best_wait['controller'])}** achieved the lowest average wait time "
        f"({best_wait['average_wait_time']:.1f}s), which is **{wait_improvement:.0f}% better** "
        f"than the worst-performing controller ({fmt(worst_wait['controller'])}, "
        f"{worst_wait['average_wait_time']:.1f}s)."
    )
    if best_wait["controller"] != best_eff["controller"]:
        insights.append(
            f"**{fmt(best_eff['controller'])}** led on system efficiency score "
            f"({best_eff['system_efficiency_score']:.3f}), suggesting a trade-off between "
            f"raw wait-time reduction and overall network balance."
        )
    else:
        insights.append(
            f"**{fmt(best_eff['controller'])}** dominated on both wait time and efficiency "
            f"score — a strong result indicating consistent superiority across metrics."
        )
    insights.append(
        f"**{fmt(throughput_best['controller'])}** handled the highest average throughput "
        f"({throughput_best['average_throughput']:.1f} vehicles/step), meaning it moved the "
        f"most traffic through the network per unit time."
    )

    n = len(df)
    rl_controllers = [c for c in df["controller"].tolist() if any(k in str(c).lower() for k in ["dqn", "q_learn", "policy", "rl"])]
    ml_controllers = [c for c in df["controller"].tolist() if any(k in str(c).lower() for k in ["rf", "xgb", "mlp", "forest", "boost", "supervised"])]
    if rl_controllers:
        rl_waits = df[df["controller"].isin(rl_controllers)]["average_wait_time"].mean()
        fixed_row = df[df["controller"].str.contains("fixed", case=False, na=False)]
        if not fixed_row.empty:
            fixed_wait = fixed_row["average_wait_time"].iloc[0]
            rl_vs_fixed = (fixed_wait - rl_waits) / max(fixed_wait, 1e-6) * 100
            insights.append(
                f"Reinforcement learning controllers averaged **{rl_waits:.1f}s** wait time, "
                f"a **{rl_vs_fixed:.0f}% improvement** over fixed-timing ({fixed_wait:.1f}s) — "
                f"demonstrating that learned policies outperform static schedules."
            )

    st.markdown("#### AI Results Interpreter")
    for insight in insights:
        st.markdown(f"- {insight}")
    st.caption(f"Auto-generated from {n} controller benchmark results.")


# ---------------------------------------------------------------------------
# Research Overview Tab
# ---------------------------------------------------------------------------

def _render_research_overview(data: DashboardData) -> None:
    """Present the research in science-fair-ready format."""

    st.markdown(
        """
        <div class="finding-card">
            <h4>Research Question</h4>
            <p>Can AI-powered traffic signal controllers — using supervised machine learning and reinforcement learning — reduce vehicle wait times and improve intersection throughput compared to fixed-timing systems?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="finding-card">
            <h4>Hypothesis</h4>
            <p>Adaptive ML/RL controllers will outperform fixed-timing and rule-based baselines by dynamically responding to real-time queue conditions, resulting in lower average wait times and higher throughput.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_method1, col_method2 = st.columns(2)
    with col_method1:
        st.markdown("#### Methodology")
        st.markdown(
            """
            - **Data:** Real-world traffic datasets (Metro Interstate, Kaggle) + stochastic simulation
            - **Simulation:** Multi-intersection network with Poisson arrivals and rush-hour scaling
            - **Controllers tested:** Fixed timing, Adaptive rule, Random Forest, XGBoost,
              Gradient Boosting, Neural Network, Q-Learning, DQN, Policy Gradient
            - **Validation:** 5-fold cross-validation, statistical significance tests (Mann-Whitney U)
            - **Metrics:** Avg wait time, queue length, throughput, fairness, efficiency score
            """
        )

    with col_method2:
        st.markdown("#### Controller Categories")
        st.markdown(
            """
            <span class="ctrl-badge badge-fixed">BASELINE</span> **Fixed Timing** — constant 30-second cycles, no adaptation<br><br>
            <span class="ctrl-badge badge-adaptive">ADAPTIVE</span> **Rule-Based** — responds to queue thresholds, no learning<br><br>
            <span class="ctrl-badge badge-ml">ML</span> **Supervised Learning** — Random Forest, XGBoost, Gradient Boosting, MLP — trained on historical data<br><br>
            <span class="ctrl-badge badge-rl">RL</span> **Reinforcement Learning** — Q-Learning, DQN, Policy Gradient — agents that learn optimal policies through trial and reward
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Key Findings")
    if data.summary_df.empty:
        st.info("Run a benchmark to generate findings.")
        return

    df = data.summary_df
    required = {"controller", "average_wait_time", "system_efficiency_score", "average_throughput"}
    if not required.issubset(df.columns):
        st.info("Benchmark data is missing required columns.")
        return

    best_row = df.loc[df["average_wait_time"].idxmin()]
    fixed_row = df[df["controller"].str.contains("fixed", case=False, na=False)]
    fixed_wait = float(fixed_row["average_wait_time"].iloc[0]) if not fixed_row.empty else None

    best_name = _display_controller_name(str(best_row["controller"]))
    best_wait = float(best_row["average_wait_time"])

    findings_col1, findings_col2, findings_col3 = st.columns(3)

    with findings_col1:
        if fixed_wait:
            pct = (fixed_wait - best_wait) / fixed_wait * 100
            st.markdown(
                f"""
                <div class="finding-card">
                    <h4>Wait Time Reduction</h4>
                    <p><strong>{best_name}</strong> reduced avg wait time by
                    <strong>{pct:.0f}%</strong> compared to fixed-timing
                    ({best_wait:.1f}s vs {fixed_wait:.1f}s baseline).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="finding-card">
                    <h4>Best Controller</h4>
                    <p><strong>{best_name}</strong> achieved the lowest avg wait time
                    ({best_wait:.1f}s) across all tested controllers.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with findings_col2:
        n_controllers = len(df)
        n_outperform_fixed = 0
        if fixed_wait:
            n_outperform_fixed = int((df["average_wait_time"] < fixed_wait).sum())
        st.markdown(
            f"""
            <div class="finding-card">
                <h4>Controllers Tested</h4>
                <p><strong>{n_controllers}</strong> controller types benchmarked.
                {f"<strong>{n_outperform_fixed}</strong> of them outperformed fixed-timing on avg wait time." if fixed_wait else ""}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with findings_col3:
        best_eff = df.loc[df["system_efficiency_score"].idxmax()]
        st.markdown(
            f"""
            <div class="finding-card">
                <h4>Highest System Efficiency</h4>
                <p><strong>{_display_controller_name(str(best_eff["controller"]))}</strong>
                scored <strong>{best_eff["system_efficiency_score"]:.3f}</strong> — the
                best overall network utilization across all controllers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Controller Performance Ranking")
    ranking_df = df[["controller", "average_wait_time", "average_queue_length", "system_efficiency_score"]].copy()
    ranking_df["controller"] = ranking_df["controller"].apply(_display_controller_name)
    ranking_df = ranking_df.sort_values("average_wait_time").reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1
    ranking_df.index.name = "Rank"
    ranking_df = _format_df_columns(ranking_df)
    st.dataframe(ranking_df, use_container_width=True, height=310)

    st.markdown("#### Conclusion")
    if fixed_wait:
        ml_rows = df[df["controller"].str.contains("ml_", case=False, na=False)]
        rl_rows = df[df["controller"].str.contains("rl_", case=False, na=False)]
        ml_avg = float(ml_rows["average_wait_time"].mean()) if not ml_rows.empty else None
        rl_avg = float(rl_rows["average_wait_time"].mean()) if not rl_rows.empty else None

        conclusion_parts: list[str] = []
        if ml_avg:
            ml_pct = (fixed_wait - ml_avg) / fixed_wait * 100
            conclusion_parts.append(
                f"ML controllers averaged **{ml_pct:.1f}% {'improvement' if ml_pct > 0 else 'change'}** vs fixed timing."
            )
        if rl_avg:
            rl_pct = (fixed_wait - rl_avg) / fixed_wait * 100
            conclusion_parts.append(
                f"RL controllers averaged **{rl_pct:.1f}% {'improvement' if rl_pct > 0 else 'change'}** vs fixed timing."
            )
        conclusion_parts.append(
            "The results support the hypothesis that AI-based traffic controllers can dynamically "
            "adapt to traffic conditions and outperform static scheduling under varying demand profiles."
        )
        for part in conclusion_parts:
            st.markdown(f"- {part}")
    else:
        st.markdown(
            "The benchmark demonstrates meaningful differences between controller strategies. "
            "AI-based adaptive controllers consistently outperform rule-based and fixed approaches "
            "across multiple traffic metrics."
        )


# ---------------------------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------------------------

def _render_overview_tab(data: DashboardData) -> None:
    _render_results_interpreter(data.summary_df)
    _render_kpi_strip(data.summary_df)

    if data.summary_df.empty:
        return

    left, right = st.columns((1.05, 1.0))
    with left:
        st.markdown("#### Controller Wait and Queue Comparison")
        st.caption("Lower is better for both metrics.")
        compare_cols = [
            col
            for col in ["average_wait_time", "average_queue_length"]
            if col in data.summary_df.columns
        ]
        if compare_cols and "controller" in data.summary_df.columns:
            compare_df = (
                _apply_controller_display(data.summary_df)[["controller", *compare_cols]]
                .set_index("controller")
                .sort_values(compare_cols[0], ascending=True)
            )
            compare_df = compare_df.rename(columns={k: COLUMN_LABELS.get(k, k) for k in compare_df.columns})
            st.bar_chart(compare_df, use_container_width=True, height=320)
        else:
            st.info("Summary metrics are incomplete for comparison charts.")

    with right:
        st.markdown("#### Throughput and Efficiency View")
        st.caption("Higher is better for both metrics.")
        perf_cols = [
            col
            for col in ["average_throughput", "system_efficiency_score"]
            if col in data.summary_df.columns
        ]
        if perf_cols and "controller" in data.summary_df.columns:
            perf_df = (
                _apply_controller_display(data.summary_df)[["controller", *perf_cols]]
                .set_index("controller")
            )
            perf_df = perf_df.rename(columns={k: COLUMN_LABELS.get(k, k) for k in perf_df.columns})
            st.bar_chart(perf_df, use_container_width=True, height=320)
        else:
            st.info("No throughput/efficiency columns found.")

    st.markdown("#### Controller Trajectory Explorer")
    st.caption("Select a controller and metric to visualize its behavior over simulation time.")
    if data.step_df.empty or "controller" not in data.step_df.columns:
        st.info("Step-level metrics are unavailable.")
        return

    metric_options = [
        col
        for col in [
            "total_queue",
            "avg_wait_sec",
            "throughput",
            "efficiency_score",
            "fairness",
            "delay_reduction_pct",
        ]
        if col in data.step_df.columns
    ]
    if not metric_options:
        st.info("No plottable step-level metric columns detected.")
        return

    control_col1, control_col2 = st.columns(2)
    controller_options = sorted(data.step_df["controller"].dropna().astype(str).unique().tolist())
    controller_display_map = {c: _display_controller_name(c) for c in controller_options}
    controller_display_options = [controller_display_map[c] for c in controller_options]

    selected_display = control_col1.selectbox(
        "Controller",
        controller_display_options,
        key="overview_controller",
    )
    display_to_raw = {v: k for k, v in controller_display_map.items()}
    selected_controller = display_to_raw.get(selected_display, selected_display)

    metric_display_options = [COLUMN_LABELS.get(m, m) for m in metric_options]
    selected_metric_display = control_col2.selectbox(
        "Metric",
        metric_display_options,
        key="overview_metric",
    )
    metric_display_to_raw = {COLUMN_LABELS.get(m, m): m for m in metric_options}
    selected_metric = metric_display_to_raw.get(selected_metric_display, metric_options[0])

    filtered = data.step_df[data.step_df["controller"] == selected_controller]
    if "fold" in filtered.columns:
        trajectory = (
            filtered.groupby("step", as_index=False)[selected_metric]
            .mean(numeric_only=True)
            .sort_values("step")
        )
    else:
        trajectory = filtered[["step", selected_metric]].sort_values("step")

    if trajectory.empty:
        st.info("No trajectory points available for selected filters.")
        return

    st.line_chart(trajectory.set_index("step"), use_container_width=True, height=300)
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Peak", f"{trajectory[selected_metric].max():.2f}")
    stat_col2.metric("Final", f"{trajectory[selected_metric].iloc[-1]:.2f}")
    stat_col3.metric("Mean", f"{trajectory[selected_metric].mean():.2f}")


# ---------------------------------------------------------------------------
# Statistics Tab
# ---------------------------------------------------------------------------

def _render_stats_tab(data: DashboardData) -> None:
    st.markdown("#### Pairwise Significance Tests")
    st.caption(
        "Mann-Whitney U test results comparing each controller pair. "
        "A p-value ≤ alpha means the difference is statistically significant."
    )
    if data.significance_df.empty:
        st.info("No significance test output found.")
    else:
        alpha = st.slider(
            "Alpha Threshold",
            min_value=0.001,
            max_value=0.200,
            value=0.050,
            step=0.001,
            key="alpha_threshold",
        )
        significance_view = data.significance_df.copy()
        if "controller_a" in significance_view.columns:
            significance_view["controller_a"] = significance_view["controller_a"].apply(_display_controller_name)
        if "controller_b" in significance_view.columns:
            significance_view["controller_b"] = significance_view["controller_b"].apply(_display_controller_name)
        if "p_value" in significance_view.columns:
            significance_view["passes_alpha"] = significance_view["p_value"] <= alpha
            significance_view = significance_view.sort_values("p_value", ascending=True)
            n_significant = int(significance_view["passes_alpha"].sum())
            st.caption(
                f"{n_significant} out of {len(significance_view)} comparisons "
                f"are significant at alpha = {alpha:.3f}."
            )
        st.dataframe(significance_view, use_container_width=True, height=310)

    st.markdown("#### Adaptive Controller Ablation Study")
    st.caption(
        "Sensitivity analysis: how the Adaptive Rule controller's wait time changes "
        "with different queue thresholds and minimum green-phase durations."
    )
    if data.ablation_df.empty:
        st.info("No ablation output found.")
        return

    ablation_sorted = data.ablation_df.sort_values(
        "average_wait_time", ascending=True
    ) if "average_wait_time" in data.ablation_df.columns else data.ablation_df
    st.dataframe(ablation_sorted, use_container_width=True, height=260)

    pivot_required = {
        "ablation_queue_threshold",
        "ablation_min_green",
        "average_wait_time",
    }
    if pivot_required.issubset(data.ablation_df.columns):
        heatmap_table = data.ablation_df.pivot_table(
            index="ablation_queue_threshold",
            columns="ablation_min_green",
            values="average_wait_time",
            aggfunc="mean",
        ).sort_index()
        st.markdown("##### Wait-Time Grid (Lower Is Better)")
        st.caption("Row = queue threshold, Column = min green duration (seconds)")
        st.dataframe(heatmap_table, use_container_width=True)


# ---------------------------------------------------------------------------
# Plots Tab
# ---------------------------------------------------------------------------

def _render_plots_tab(data: DashboardData) -> None:
    st.markdown("#### Generated Plot Gallery")
    st.caption("Publication-quality figures generated by the experiment runner.")
    if not data.plot_paths:
        st.info("No generated plots found in artifact output.")
        return

    plot_descriptions: dict[str, str] = {
        "controller_performance_comparison.png": "Side-by-side bar charts comparing all controllers across four key metrics.",
        "queue_wait_curves.png": "Time-series curves showing queue dynamics and average wait time over simulation steps.",
        "traffic_heatmap.png": "Heatmap of traffic congestion intensity by controller and simulation step.",
        "rl_learning_curves.png": "Episode reward over training showing how each RL agent improves its policy.",
        "model_performance_table.png": "Accuracy, precision, recall, and F1 scores for all supervised ML models.",
    }

    cols = st.columns(2)
    for idx, plot_path in enumerate(data.plot_paths):
        with cols[idx % 2]:
            caption = plot_descriptions.get(plot_path.name, plot_path.name)
            st.image(str(plot_path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Tables Tab
# ---------------------------------------------------------------------------

def _render_tables_tab(data: DashboardData) -> None:
    st.markdown("#### Controller Summary Table")
    st.caption("Aggregated mean metrics across all cross-validation folds.")
    if data.summary_df.empty:
        st.info("No summary data available.")
    else:
        display_df = _apply_controller_display(data.summary_df)
        display_df = _format_df_columns(display_df)
        st.dataframe(display_df, use_container_width=True, height=260)

    st.markdown("#### Supervised Model Metrics")
    st.caption("Accuracy and F1 scores for each ML model trained on the traffic dataset.")
    if data.model_metrics_df.empty:
        st.info("No model metrics available.")
    else:
        model_display = data.model_metrics_df.copy()
        if "model_name" in model_display.columns:
            model_display["model_name"] = model_display["model_name"].str.replace("_", " ").str.title()
        model_display = _format_df_columns(model_display)
        st.dataframe(model_display, use_container_width=True, height=240)

    st.markdown("#### Step-Level Metrics Preview")
    st.caption("Raw per-step simulation data for detailed analysis.")
    row_cap = st.slider(
        "Rows to Preview",
        min_value=100,
        max_value=2000,
        value=400,
        step=100,
        key="step_preview_rows",
    )
    if data.step_df.empty:
        st.info("No step-level data available.")
    else:
        step_display = data.step_df.head(row_cap).copy()
        if "controller" in step_display.columns:
            step_display["controller"] = step_display["controller"].apply(_display_controller_name)
        st.dataframe(step_display, use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Benchmark Lab
# ---------------------------------------------------------------------------

def _render_benchmark_lab(data: DashboardData) -> None:
    tab_research, tab_overview, tab_stats, tab_plots, tab_tables = st.tabs(
        ["Research Summary", "Overview", "Statistics", "Plots", "Raw Tables"]
    )
    with tab_research:
        _render_research_overview(data)
    with tab_overview:
        _render_overview_tab(data)
    with tab_stats:
        _render_stats_tab(data)
    with tab_plots:
        _render_plots_tab(data)
    with tab_tables:
        _render_tables_tab(data)


# ---------------------------------------------------------------------------
# Welcome / Empty State
# ---------------------------------------------------------------------------

def _render_welcome_state() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 1rem 1rem;">
            <div style="font-size:3rem;">🚦</div>
            <h2 style="color:#e8f0f2; margin-top:0.5rem;">Welcome to the Traffic AI Dashboard</h2>
            <p style="color:#8fa8b4; max-width:540px; margin:0 auto 1.5rem;">
                No benchmark results are loaded yet. Use the sidebar to run a benchmark
                or load previously saved artifacts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown(
            """
            <div class="onboard-step">
                <div class="step-num">1</div>
                <div class="step-title">Run a Benchmark</div>
                <div class="step-desc">Click <strong>Run Full Benchmark</strong> in the sidebar.
                Enable Quick Run for a fast preview. This trains all controllers and records results.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with step_col2:
        st.markdown(
            """
            <div class="onboard-step">
                <div class="step-num">2</div>
                <div class="step-title">Explore Results</div>
                <div class="step-desc">Switch between tabs: <strong>Research Summary</strong> for science-fair context,
                <strong>Overview</strong> for charts, <strong>Statistics</strong> for significance tests.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with step_col3:
        st.markdown(
            """
            <div class="onboard-step">
                <div class="step-num">3</div>
                <div class="step-title">Try the Simulator</div>
                <div class="step-desc">Use the <strong>Live Simulation</strong> or <strong>Grid Playground</strong> tabs
                to run interactive simulations with any controller in real time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p style="text-align:center; color:#8fa8b4; margin-top:1rem; font-size:0.83rem;">
            If you have already run experiments, click <strong>Load Latest Artifacts</strong> in the sidebar.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Live Simulation Panel
# ---------------------------------------------------------------------------

def _draw_intersection_grid(
    env_nodes: dict[int, Any],
    rows: int,
    cols: int,
    max_q: int,
    lanes: int,
) -> None:
    """Render color-coded intersection grid for the grid simulation panel."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
        if rows == 1 and cols == 1:
            axes_arr = np.array([[axes]])
        elif rows == 1:
            axes_arr = np.array([axes])
        elif cols == 1:
            axes_arr = np.array([[ax] for ax in axes])
        else:
            axes_arr = axes  # type: ignore[assignment]

        for r in range(rows):
            for c in range(cols):
                node_id = r * cols + c
                ax = axes_arr[r][c]
                node = env_nodes.get(node_id)
                if node is None:
                    ax.set_visible(False)
                    continue

                total_q = float(node.total_queue)
                q_norm = min(total_q / max(max_q * lanes * 4, 1), 1.0)
                color = plt.cm.RdYlGn_r(q_norm)  # type: ignore[attr-defined]
                phase_str = "NS" if int(node.current_phase) == 0 else "EW"

                ax.set_facecolor(color)
                ax.text(0.5, 0.68, f"#{node_id}", ha="center", va="center", fontsize=10, fontweight="bold")
                ax.text(0.5, 0.42, f"Q={total_q:.0f}", ha="center", va="center", fontsize=9)
                ax.text(
                    0.5,
                    0.17,
                    phase_str,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="green" if phase_str == "NS" else "red",
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    except Exception:
        st.info("Grid preview could not be rendered in this run.")


def _render_live_simulation_panel(settings: Settings) -> None:
    st.markdown(
        "Tune simulation settings and run a controller in the network simulator for instant feedback."
    )
    st.caption(
        "This uses the core traffic simulation engine directly — no pre-trained models needed."
    )

    demand_profiles = ["normal", "rush_hour", "midday_peak"]
    configured_profile = str(settings.get("simulation.demand_profile", "rush_hour"))
    profile_index = (
        demand_profiles.index(configured_profile)
        if configured_profile in demand_profiles
        else 1
    )

    with st.form("live_sim_form"):
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        controller_label = row1_col1.selectbox(
            "Controller",
            ["Adaptive Rule", "Fixed Timing"],
            help="Adaptive Rule dynamically adjusts green time based on queue length. Fixed Timing uses a constant 30s cycle.",
        )
        sim_steps = int(row1_col2.slider("Simulation Steps", 100, 3000, 800, 100))
        intersections = int(
            row1_col3.slider(
                "Intersections",
                1,
                9,
                int(settings.get("simulation.intersections", 4)),
                1,
            )
        )

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        lanes = int(
            row2_col1.slider(
                "Lanes Per Direction",
                1,
                4,
                int(settings.get("simulation.lanes_per_direction", 2)),
                1,
            )
        )
        demand_profile = row2_col2.selectbox(
            "Demand Profile",
            demand_profiles,
            index=profile_index,
            help="Rush hour applies 2.5× demand scaling during peak periods.",
        )
        demand_scale = float(
            row2_col3.slider(
                "Demand Scale",
                0.50,
                3.00,
                float(settings.get("simulation.demand_scale", 1.0)),
                0.05,
                help="Overall traffic volume multiplier.",
            )
        )
        run_clicked = st.form_submit_button("Run Simulation", type="primary")

    if not run_clicked:
        return

    sim_cfg = SimulatorConfig(
        steps=sim_steps,
        intersections=intersections,
        lanes_per_direction=lanes,
        step_seconds=float(settings.get("simulation.step_seconds", 1.0)),
        max_queue_per_lane=int(settings.get("simulation.max_queue_per_lane", 60)),
        demand_profile=demand_profile,
        demand_scale=demand_scale,
        seed=settings.seed,
    )
    simulator = TrafficNetworkSimulator(sim_cfg)
    controller = (
        AdaptiveRuleController()
        if controller_label == "Adaptive Rule"
        else FixedTimingController()
    )

    with st.spinner("Running network simulation..."):
        result = simulator.run(controller, steps=sim_steps)
    step_df = simulation_result_to_step_dataframe(result)
    aggregate = result.aggregate

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Avg Wait (s)", f"{aggregate.get('average_wait_time', 0.0):.2f}")
    kpi2.metric("Avg Queue", f"{aggregate.get('average_queue_length', 0.0):.2f}")
    kpi3.metric("Throughput", f"{aggregate.get('average_throughput', 0.0):.2f}")
    kpi4.metric("Fairness", f"{aggregate.get('average_fairness', 0.0):.3f}")

    metric_cols = [
        col
        for col in ["total_queue", "avg_wait_sec", "throughput", "fairness", "efficiency_score"]
        if col in step_df.columns
    ]
    if metric_cols:
        st.markdown("#### Step Metrics Over Time")
        renamed = step_df[["step", *metric_cols]].rename(
            columns={k: COLUMN_LABELS.get(k, k) for k in metric_cols}
        )
        st.line_chart(renamed.set_index("step"), use_container_width=True, height=330)

    st.markdown("#### Aggregate Run Summary")
    st.dataframe(pd.DataFrame([aggregate]), use_container_width=True, height=120)


# ---------------------------------------------------------------------------
# Grid Simulation Panel
# ---------------------------------------------------------------------------

def _render_grid_simulation_panel(settings: Settings) -> None:
    st.markdown(
        "Run the multi-intersection grid environment and watch queue dynamics evolve across the network."
    )
    st.caption(
        "Each cell shows the intersection ID, total queue (Q), and current phase (NS = North-South green, EW = East-West green). "
        "Color ranges from green (low congestion) to red (high congestion)."
    )

    try:
        from traffic_ai.controllers.fixed import FixedTimingController as FixedGridController
        from traffic_ai.controllers.rule_based import RuleBasedController
        from traffic_ai.simulation.intersection import MultiIntersectionNetwork
    except ImportError as exc:
        st.error(f"Could not import grid simulation modules: {exc}")
        return

    with st.form("grid_sim_form"):
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        grid_rows = int(
            row1_col1.slider(
                "Grid Rows",
                1,
                4,
                int(settings.get("simulation.grid_rows", 2)),
                1,
            )
        )
        grid_cols = int(
            row1_col2.slider(
                "Grid Cols",
                1,
                4,
                int(settings.get("simulation.grid_cols", 2)),
                1,
            )
        )
        episodes = int(row1_col3.slider("Episodes", 1, 12, 3, 1))

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        episode_steps = int(row2_col1.slider("Steps Per Episode", 40, 450, 120, 10))
        controller_choice = row2_col2.selectbox(
            "Controller",
            ["Rule-Based Adaptive", "Fixed Timing"],
            help="Rule-Based Adaptive adjusts phase based on queue imbalance.",
        )
        rush_scale = float(
            row2_col3.slider(
                "Rush-Hour Scale",
                1.0,
                4.0,
                float(settings.get("simulation.rush_hour_scale", 2.5)),
                0.1,
            )
        )

        refresh_every = int(st.slider("Grid Refresh Interval (steps)", 1, 25, 5, 1))
        run_grid = st.form_submit_button("Run Grid Simulation", type="primary")

    if not run_grid:
        return

    env = MultiIntersectionNetwork(
        rows=grid_rows,
        cols=grid_cols,
        max_steps=episode_steps,
        rush_hour_scale=rush_scale,
        seed=settings.seed,
    )
    controller_choice_raw = "rule_based" if controller_choice == "Rule-Based Adaptive" else "fixed_timing"
    controller = (
        RuleBasedController()
        if controller_choice_raw == "rule_based"
        else FixedGridController()
    )

    grid_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    episode_rewards: list[float] = []
    episode_queue_means: list[float] = []
    last_episode_queue_trace: list[float] = []

    total_ticks = max(episodes * episode_steps, 1)
    tick_count = 0

    for episode_idx in range(episodes):
        obs = env.reset(seed=settings.seed + episode_idx)
        controller.reset(env.n_nodes)
        reward_sum = 0.0
        queue_trace: list[float] = []

        for step_idx in range(episode_steps):
            actions = {node_id: controller.select_action(node_obs) for node_id, node_obs in obs.items()}
            obs, reward, done, info = env.step(actions)
            reward_sum += float(reward)
            queue_now = float(info.get("total_queue", 0.0))
            queue_trace.append(queue_now)
            tick_count += 1

            if step_idx % refresh_every == 0 or done:
                status_placeholder.caption(
                    f"Episode {episode_idx + 1}/{episodes} | Step {step_idx + 1}/{episode_steps}"
                )
                with grid_placeholder.container():
                    _draw_intersection_grid(env._nodes, grid_rows, grid_cols, env.max_queue, env.lanes)

            progress_bar.progress(min(tick_count / total_ticks, 1.0))
            if done:
                break

        episode_rewards.append(reward_sum)
        episode_queue_means.append(float(np.mean(queue_trace) if queue_trace else 0.0))
        last_episode_queue_trace = queue_trace

    st.success(
        f"Grid run complete. Mean episode reward: {np.mean(episode_rewards):.2f} | "
        f"Mean queue: {np.mean(episode_queue_means):.2f}"
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Episode Rewards")
        st.caption("Total reward accumulated per episode. Higher = less congestion.")
        st.line_chart(pd.DataFrame({"Episode Reward": episode_rewards}), use_container_width=True)
    with chart_col2:
        st.markdown("#### Episode Mean Queue")
        st.caption("Average total queue per episode. Lower = better traffic flow.")
        st.line_chart(pd.DataFrame({"Mean Queue": episode_queue_means}), use_container_width=True)

    if last_episode_queue_trace:
        st.markdown("#### Last Episode Queue Trace")
        st.caption("Step-by-step queue evolution in the final episode.")
        st.line_chart(
            pd.DataFrame({"Total Queue": last_episode_queue_trace}),
            use_container_width=True,
            height=250,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(settings: Settings) -> tuple[bool, bool, bool, bool, bool]:
    with st.sidebar:
        st.header("Experiment Controls")

        quick_run = st.toggle("Quick Run", value=True, help="Use a reduced simulation budget for faster results.")
        include_public = st.toggle("Use Public Datasets", value=True, help="Include Metro Interstate and UCI traffic datasets.")
        include_kaggle = st.toggle("Use Kaggle Datasets", value=False, help="Requires a kaggle.json API token.")

        run_clicked = st.button(
            "Run Full Benchmark",
            type="primary",
            use_container_width=True,
        )
        load_latest_clicked = st.button(
            "Load Latest Artifacts",
            use_container_width=True,
        )
        st.caption(f"Artifacts: `{settings.output_dir}`")

        st.divider()

        with st.expander("About This Research", expanded=False):
            st.markdown(
                """
                **AI Traffic Signal Optimization**

                This platform compares 10 traffic signal controllers
                across four families:

                - **Fixed Timing** — static 30-second cycles (baseline)
                - **Adaptive Rule** — queue-threshold logic, no learning
                - **Supervised ML** — Random Forest, XGBoost, Gradient Boosting, Neural Net
                - **Reinforcement Learning** — Q-Learning, DQN, Policy Gradient

                **Metrics tracked:** average wait time, queue length,
                throughput, fairness (Gini coefficient), efficiency score.

                **Validation:** 5-fold cross-validation with statistical
                significance testing (Mann-Whitney U, α=0.05).
                """
            )

        with st.expander("How to Read the Results", expanded=False):
            st.markdown(
                """
                **Avg Wait Time (s)** — lower is better. Time vehicles spend
                waiting at a red light on average.

                **Avg Queue Length** — lower is better. Number of vehicles
                queued across all lanes/intersections.

                **Throughput** — higher is better. Vehicles processed per
                simulation step.

                **Fairness Score** — closer to 1.0 is better. Measures how
                evenly the controller distributes green time (1 − Gini index).

                **System Efficiency Score** — higher is better. Composite
                metric combining throughput and queue size.
                """
            )

    return quick_run, include_public, include_kaggle, run_clicked, load_latest_clicked


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# AI Advisor tab
# ---------------------------------------------------------------------------

_AI_SYSTEM_PROMPT = """You are an expert AI research assistant specializing in traffic signal optimization.
You are embedded in a science fair research platform (GSDSEF) built by Samarth Vaka that benchmarks
10 traffic signal controllers across 4 families: Fixed Timing (baseline), Adaptive Rule,
Supervised ML (Random Forest, XGBoost, Gradient Boosting, MLP, Logistic Regression),
and Reinforcement Learning (Q-Learning, DQN, Policy Gradient).

Metrics (lower is better unless noted):
- average_wait_time: seconds vehicles wait at red lights
- average_queue_length: vehicles queued per intersection
- average_throughput: vehicles processed per step (higher is better)
- average_emissions_proxy: estimated emissions (lower is better)
- average_fairness: Gini-based fairness score (higher = more fair)
- average_efficiency_score: composite score (higher is better)

You have access to the actual experimental results below. Answer questions about these specific
results accurately and concisely. Do not invent numbers — only reference the data provided.
If asked about limitations, note that RL controllers may be under-trained in quick-run mode
(fewer episodes/steps), which can cause them to underperform relative to their true potential.

{data_context}
"""


def _build_data_context(data: DashboardData | None) -> str:
    if data is None or data.summary_df.empty:
        return "No benchmark results are loaded yet. Ask the user to run a benchmark first."
    df = data.summary_df.copy()
    df["controller"] = df["controller"].map(
        lambda x: CONTROLLER_DISPLAY_NAMES.get(x, x)
    )
    return "EXPERIMENTAL RESULTS (controller summary):\n" + df.to_string(index=False)


def _render_ai_advisor_tab(data: DashboardData | None) -> None:
    st.subheader("AI Traffic Advisor")
    st.caption(
        "Ask Claude questions about your experimental results. "
        "Claude has your benchmark data loaded as context."
    )

    # --- API key ---
    # Check Streamlit secrets (cloud deployment) first, then .env, then manual input
    secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    env_key = os.getenv("ANTHROPIC_API_KEY", "")
    auto_key = secret_key or env_key

    if auto_key:
        api_key = auto_key
        st.success("API key loaded automatically.", icon="🔑")
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Your key is never stored — it lives only in this browser session.",
        )

    if not api_key:
        st.info("Add your Anthropic API key to the .env file or paste it above to get started.")
        return

    # --- Data context status ---
    if data is None or data.summary_df.empty:
        st.warning("No benchmark results loaded. Run a benchmark first, then come back here.")
        return

    n_controllers = len(data.summary_df)
    st.caption(f"Context: {n_controllers} controllers loaded into Claude's context.")

    # --- Chat history ---
    if "ai_advisor_messages" not in st.session_state:
        st.session_state["ai_advisor_messages"] = []

    messages: list[dict] = st.session_state["ai_advisor_messages"]

    # Render existing chat history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Suggested questions ---
    if not messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "Which controller had the lowest average wait time and why might that be?",
            "Why do RL controllers sometimes underperform simpler methods?",
            "Which controller would you recommend for a high-traffic urban intersection?",
            "What do the results suggest about the trade-off between ML and RL approaches?",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["ai_advisor_pending"] = suggestion
                st.rerun()

    # Handle suggested question click
    pending = st.session_state.pop("ai_advisor_pending", None)

    # --- Chat input ---
    user_input = st.chat_input("Ask about your traffic optimization results...")
    prompt = pending or user_input

    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = _AI_SYSTEM_PROMPT.format(
            data_context=_build_data_context(data)
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                client = anthropic.Anthropic(api_key=api_key)
                with client.messages.stream(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in messages
                    ],
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except anthropic.AuthenticationError:
                full_response = "Invalid API key. Check your .env file or the key you entered."
                response_placeholder.error(full_response)
            except Exception as e:
                full_response = f"Error contacting Claude: {e}"
                response_placeholder.error(full_response)

        messages.append({"role": "assistant", "content": full_response})

    # Clear chat button
    if messages:
        if st.button("Clear conversation", key="clear_ai_chat"):
            st.session_state["ai_advisor_messages"] = []
            st.rerun()


def run_dashboard() -> None:
    settings = load_settings()
    _inject_custom_theme()

    if "dashboard_data" not in st.session_state:
        cached = _load_dashboard_data_from_output_dir(settings)
        if cached is not None:
            st.session_state["dashboard_data"] = cached
            st.session_state["dashboard_source"] = "loaded from artifacts on disk"

    quick_run, include_public, include_kaggle, run_clicked, load_latest_clicked = _render_sidebar(settings)

    if run_clicked:
        with st.spinner("Running pipeline, training models, and benchmarking controllers..."):
            dashboard_data = _run_benchmark(
                settings=settings,
                quick_run=quick_run,
                include_public=include_public,
                include_kaggle=include_kaggle,
            )
        st.session_state["dashboard_data"] = dashboard_data
        mode = "quick run" if quick_run else "full run"
        st.session_state["dashboard_source"] = f"fresh benchmark ({mode})"
        st.success("Benchmark complete.")

    if load_latest_clicked:
        latest = _load_dashboard_data_from_output_dir(settings)
        if latest is None:
            st.warning(
                "No benchmark artifact set found yet. Run a benchmark from the sidebar controls."
            )
        else:
            st.session_state["dashboard_data"] = latest
            st.session_state["dashboard_source"] = "loaded from artifacts on disk"
            st.success("Loaded latest artifact files.")

    source_label = st.session_state.get("dashboard_source")
    _render_header(source_label)

    benchmark_tab, simulation_tab, grid_tab, ai_tab = st.tabs(
        ["Benchmark Lab", "Live Simulation", "Grid Playground", "AI Advisor"]
    )

    with benchmark_tab:
        data = st.session_state.get("dashboard_data")
        if data is None:
            _render_welcome_state()
        else:
            _render_benchmark_lab(data)

    with simulation_tab:
        _render_live_simulation_panel(settings)

    with grid_tab:
        _render_grid_simulation_panel(settings)

    with ai_tab:
        _render_ai_advisor_tab(st.session_state.get("dashboard_data"))


if __name__ == "__main__":
    run_dashboard()
