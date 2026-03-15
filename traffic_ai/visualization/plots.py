from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="deep")

# Consistent color coding: gray=fixed, teal=adaptive, gold=ML, purple=RL
_CONTROLLER_COLORS: dict[str, str] = {
    "fixed": "#6b7f8e",
    "rule": "#0fc5c8",
    "adaptive": "#0fc5c8",
    "random_forest": "#ee9b00",
    "xgb": "#ee9b00",
    "gradient": "#ee9b00",
    "mlp": "#ee9b00",
    "ml_": "#ee9b00",
    "q_learn": "#9b59b6",
    "dqn": "#9b59b6",
    "policy": "#9b59b6",
    "rl_": "#9b59b6",
    "ppo": "#9b59b6",
}


def _controller_color(name: str) -> str:
    n = str(name).lower()
    for key, color in _CONTROLLER_COLORS.items():
        if key in n:
            return color
    return "#888888"


def plot_controller_performance(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("average_wait_time", "Average Wait Time (s) — Lower is Better"),
        ("average_queue_length", "Average Queue Length (vehicles) — Lower is Better"),
        ("average_throughput", "Average Throughput (vehicles/step) — Higher is Better"),
        ("average_efficiency_score", "System Efficiency Score — Higher is Better"),
    ]
    controllers = summary_df["controller"].tolist() if "controller" in summary_df.columns else []
    palette = [_controller_color(c) for c in controllers]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        if metric not in summary_df.columns:
            ax.set_visible(False)
            continue
        sns.barplot(data=summary_df, x="controller", y=metric, palette=palette, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("")
    fig.suptitle(
        "AI Traffic Signal Controller Benchmark — 10 Controllers Across 4 Families",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    target = output_dir / "controller_performance_comparison.png"
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_queue_and_wait_curves(step_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = (
        step_df.groupby(["step", "controller"], as_index=False)[["total_queue", "avg_wait_sec"]]
        .mean()
        .sort_values("step")
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for controller, frame in aggregated.groupby("controller"):
        axes[0].plot(frame["step"], frame["total_queue"], linewidth=1.6, label=controller)
    axes[0].set_title("Queue Dynamics Over Time")
    axes[0].legend(fontsize=8, ncol=2)
    for controller, frame in aggregated.groupby("controller"):
        axes[1].plot(frame["step"], frame["avg_wait_sec"], linewidth=1.6, label=controller)
    axes[1].set_title("Average Wait Time Over Time")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    target = output_dir / "queue_wait_curves.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_traffic_heatmap(step_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = step_df.pivot_table(
        index="controller", columns="step", values="total_queue", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap="magma", cbar_kws={"label": "Total Queue"}, ax=ax)
    ax.set_title("Traffic Congestion Heatmap")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Controller")
    fig.tight_layout()
    target = output_dir / "traffic_heatmap.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_learning_curves(rl_history_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=rl_history_df, x="episode", y="reward", hue="algorithm", ax=ax)
    ax.set_title("RL Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    fig.tight_layout()
    target = output_dir / "rl_learning_curves.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_model_metrics_table(model_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(model_df))))
    ax.axis("off")
    table = ax.table(
        cellText=model_df.round(4).values,
        colLabels=model_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.tight_layout()
    target = output_dir / "model_performance_table.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


# ---------------------------------------------------------------------------
# New plot functions (Phase 6)
# ---------------------------------------------------------------------------

def plot_controller_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    wait_col: str = "avg_wait_time",
    co2_col: str = "estimated_co2_grams",
) -> Path:
    """Grouped bar chart of avg_wait_time and estimated_co2_grams with std-dev error bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (col, label) in zip(
        axes,
        [
            (wait_col, "Avg Wait Time (s)"),
            (co2_col, "Estimated CO₂ (g)"),
        ],
    ):
        if col not in summary_df.columns:
            ax.set_visible(False)
            continue
        means = summary_df.groupby("controller")[col].mean()
        stds = summary_df.groupby("controller")[col].std().fillna(0)
        controllers = means.index.tolist()
        ax.bar(
            controllers,
            means.values,
            yerr=stds.values,
            capsize=5,
            color=sns.color_palette("deep", len(controllers)),
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Controller")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Controller Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    target = output_dir / "controller_comparison.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_feature_importance(
    importances: dict[str, "np.ndarray"],
    feature_names: list[str],
    output_dir: Path,
) -> Path:
    """Horizontal bar chart of feature importances for RF and XGBoost."""
    import numpy as np
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(importances)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(7 * max(n, 1), 5))
    if n == 1:
        axes = [axes]  # type: ignore[list-item]
    for ax, (model_name, imp) in zip(axes, importances.items()):
        sorted_idx = np.argsort(imp)
        ax.barh([feature_names[i] for i in sorted_idx], imp[sorted_idx], color="steelblue")
        ax.set_title(f"{model_name} Feature Importance")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    target = output_dir / "feature_importance.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_ablation(
    ablation_df: pd.DataFrame,
    output_dir: Path,
    metric_col: str = "avg_wait_time",
) -> Path:
    """Heatmap of metric degradation when each feature group is removed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if ablation_df.empty or metric_col not in ablation_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No ablation data available", ha="center", va="center")
        target = output_dir / "ablation_heatmap.png"
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    pivot_cols = [c for c in ablation_df.columns if c not in (metric_col, "controller")]
    if not pivot_cols:
        pivot_cols = [ablation_df.columns[0]]
    row_col = pivot_cols[0]
    col_col = pivot_cols[1] if len(pivot_cols) > 1 else row_col

    try:
        pivot = ablation_df.pivot_table(index=row_col, columns=col_col, values=metric_col, aggfunc="mean")
    except Exception:
        pivot = ablation_df[[metric_col]].T

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Ablation Study: {metric_col}")
    fig.tight_layout()
    target = output_dir / "ablation_heatmap.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_queue_over_time(
    step_df: pd.DataFrame,
    output_dir: Path,
    queue_col: str = "total_queue",
    controller_col: str = "controller",
) -> Path:
    """Line chart of queue length over simulation steps for all controllers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    if "step" in step_df.columns and queue_col in step_df.columns:
        aggregated = (
            step_df.groupby(["step", controller_col], as_index=False)[queue_col]
            .mean()
            .sort_values("step")
        )
        for ctrl, frame in aggregated.groupby(controller_col):
            ax.plot(frame["step"], frame[queue_col], linewidth=1.5, label=ctrl)
    ax.set_title("Queue Length Over Time")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Total Queue Length")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    target = output_dir / "queue_over_time.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target
