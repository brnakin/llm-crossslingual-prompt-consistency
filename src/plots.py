"""
Visualization module for cross-lingual consistency analysis.

This module generates heatmaps, distribution plots, and summary tables
for the evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .similarity import load_metrics, load_stability, aggregate_by_language_pair, aggregate_by_task_type
from .task_checks import load_task_metrics, aggregate_task_metrics_by_task_type


# Default paths
DEFAULT_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUTS_DIR / "plots"
DEFAULT_REPORTS_DIR = DEFAULT_OUTPUTS_DIR / "reports"


# Plot style configuration
PLOT_STYLE = {
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_theme(style="whitegrid")


def create_similarity_heatmap(
    metrics_df: pd.DataFrame,
    model_id: str,
    run_id: Optional[int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create a heatmap of cross-lingual similarities by prompt.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Model to visualize
        run_id: Optional run filter (if None, averages across runs)
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    df = metrics_df[metrics_df["model_id"] == model_id].copy()
    
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    
    # Pivot to create heatmap data
    pivot_df = df.pivot_table(
        index="prompt_id",
        columns="pair",
        values="cosine_similarity",
        aggfunc="mean"
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Cosine Similarity"}
    )
    
    run_str = f" (Run {run_id})" if run_id else " (Averaged)"
    ax.set_title(f"Cross-Lingual Similarity Heatmap\n{model_id}{run_str}")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("Prompt ID")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_similarity_distribution(
    metrics_df: pd.DataFrame,
    model_id: str,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create distribution plots of similarities by language pair.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Model to visualize
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    df = metrics_df[metrics_df["model_id"] == model_id].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    ax1 = axes[0]
    sns.boxplot(data=df, x="pair", y="cosine_similarity", ax=ax1, palette="Set2")
    ax1.set_title(f"Similarity Distribution by Language Pair\n{model_id}")
    ax1.set_xlabel("Language Pair")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_ylim(0, 1)
    
    # Add mean markers
    means = df.groupby("pair")["cosine_similarity"].mean()
    for i, pair in enumerate(means.index):
        ax1.scatter(i, means[pair], color="red", s=100, zorder=5, marker="D", label="Mean" if i == 0 else "")
    ax1.legend(loc="lower right")
    
    # KDE plot
    ax2 = axes[1]
    for pair in df["pair"].unique():
        pair_data = df[df["pair"] == pair]["cosine_similarity"]
        sns.kdeplot(data=pair_data, ax=ax2, label=pair, fill=True, alpha=0.3)
    
    ax2.set_title(f"Similarity Density by Language Pair\n{model_id}")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 1)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_task_type_comparison(
    metrics_df: pd.DataFrame,
    model_id: str,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create comparison plot of similarities by task type.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Model to visualize
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    df = metrics_df[metrics_df["model_id"] == model_id].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df, x="task_type", y="cosine_similarity", hue="pair", ax=ax, palette="Set2")
    
    ax.set_title(f"Similarity by Task Type and Language Pair\n{model_id}")
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)
    ax.legend(title="Language Pair", loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_stability_comparison(
    stability_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_id: str,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create comparison of cross-lingual vs intra-language stability.
    
    Args:
        stability_df: Stability DataFrame
        metrics_df: Metrics DataFrame
        model_id: Model to visualize
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    # Filter data
    stab_df = stability_df[
        (stability_df["model_id"] == model_id) & 
        (stability_df["stability_type"] == "open_text_cosine")
    ].copy()
    
    met_df = metrics_df[metrics_df["model_id"] == model_id].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Intra-language stability
    ax1 = axes[0]
    if len(stab_df) > 0:
        sns.boxplot(data=stab_df, x="language", y="stability_value", ax=ax1, palette="Set3")
        ax1.set_title(f"Intra-Language Stability (Run1 vs Run2)\n{model_id}")
        ax1.set_xlabel("Language")
        ax1.set_ylabel("Cosine Similarity")
        ax1.set_ylim(0, 1)
    else:
        ax1.text(0.5, 0.5, "No stability data available", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(f"Intra-Language Stability\n{model_id}")
    
    # Cross-lingual consistency
    ax2 = axes[1]
    if len(met_df) > 0:
        sns.boxplot(data=met_df, x="pair", y="cosine_similarity", ax=ax2, palette="Set2")
        ax2.set_title(f"Cross-Lingual Consistency\n{model_id}")
        ax2.set_xlabel("Language Pair")
        ax2.set_ylabel("Cosine Similarity")
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, "No metrics data available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title(f"Cross-Lingual Consistency\n{model_id}")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_discrete_task_summary(
    task_metrics_df: pd.DataFrame,
    model_id: str,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create summary visualization for discrete-answer task agreement.
    
    Args:
        task_metrics_df: Task metrics DataFrame
        model_id: Model to visualize
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    df = task_metrics_df[task_metrics_df["model_id"] == model_id].copy()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No discrete task data available", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Match rates by task type
    ax1 = axes[0]
    task_summary = df.groupby("task_type")["result"].value_counts().unstack(fill_value=0)
    task_summary_pct = task_summary.div(task_summary.sum(axis=1), axis=0) * 100
    
    task_summary_pct.plot(kind="bar", stacked=True, ax=ax1, color=["#2ecc71", "#e74c3c", "#95a5a6"])
    ax1.set_title(f"Cross-Lingual Agreement by Task Type\n{model_id}")
    ax1.set_xlabel("Task Type")
    ax1.set_ylabel("Percentage")
    ax1.legend(title="Result", loc="upper right")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Match rates by prompt
    ax2 = axes[1]
    prompt_summary = df.groupby("prompt_id")["result"].apply(
        lambda x: (x == "match").sum() / len(x) * 100
    )
    
    colors = ["#2ecc71" if v >= 75 else "#f39c12" if v >= 50 else "#e74c3c" for v in prompt_summary.values]
    prompt_summary.plot(kind="bar", ax=ax2, color=colors)
    ax2.set_title(f"Cross-Lingual Match Rate by Prompt\n{model_id}")
    ax2.set_xlabel("Prompt ID")
    ax2.set_ylabel("Match Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=75, color="green", linestyle="--", alpha=0.7, label="75%")
    ax2.axhline(y=50, color="orange", linestyle="--", alpha=0.7, label="50%")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_model_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create comparison plot across models.
    
    Args:
        metrics_df: Metrics DataFrame
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=metrics_df, x="pair", y="cosine_similarity", hue="model_id", ax=ax, palette="Set1")
    
    ax.set_title("Cross-Lingual Similarity Comparison Across Models")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def generate_summary_table(
    metrics_df: pd.DataFrame,
    task_metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    model_id: str
) -> pd.DataFrame:
    """Generate a comprehensive summary table.
    
    Args:
        metrics_df: Metrics DataFrame (open-text)
        task_metrics_df: Task metrics DataFrame (discrete)
        stability_df: Stability DataFrame
        model_id: Model to summarize
        
    Returns:
        Summary DataFrame
    """
    rows = []
    
    # Open-text metrics by language pair
    met_df = metrics_df[metrics_df["model_id"] == model_id]
    if len(met_df) > 0:
        for pair in met_df["pair"].unique():
            pair_data = met_df[met_df["pair"] == pair]["cosine_similarity"]
            rows.append({
                "Category": "Cross-Lingual (Open-Text)",
                "Metric": f"{pair} Similarity",
                "Mean": pair_data.mean(),
                "Std": pair_data.std(),
                "Count": len(pair_data)
            })
    
    # Discrete task metrics
    task_df = task_metrics_df[task_metrics_df["model_id"] == model_id]
    if len(task_df) > 0:
        for task_type in task_df["task_type"].unique():
            type_data = task_df[task_df["task_type"] == task_type]
            match_rate = (type_data["result"] == "match").sum() / len(type_data)
            rows.append({
                "Category": "Cross-Lingual (Discrete)",
                "Metric": f"{task_type.capitalize()} Match Rate",
                "Mean": match_rate,
                "Std": None,
                "Count": len(type_data)
            })
    
    # Stability metrics
    stab_df = stability_df[stability_df["model_id"] == model_id]
    if len(stab_df) > 0:
        for stab_type in stab_df["stability_type"].unique():
            for lang in stab_df["language"].unique():
                lang_data = stab_df[(stab_df["stability_type"] == stab_type) & (stab_df["language"] == lang)]
                if len(lang_data) > 0:
                    rows.append({
                        "Category": f"Stability ({stab_type})",
                        "Metric": f"{lang} Run1-vs-Run2",
                        "Mean": lang_data["stability_value"].mean(),
                        "Std": lang_data["stability_value"].std(),
                        "Count": len(lang_data)
                    })
    
    return pd.DataFrame(rows)


def save_summary_table(
    summary_df: pd.DataFrame,
    model_id: str,
    reports_dir: Optional[Path] = None
) -> Path:
    """Save summary table to CSV.
    
    Args:
        summary_df: Summary DataFrame
        model_id: Model ID for filename
        reports_dir: Directory for reports
        
    Returns:
        Path to saved file
    """
    if reports_dir is None:
        reports_dir = DEFAULT_REPORTS_DIR
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean model_id for filename
    clean_model_id = model_id.replace(":", "_").replace("/", "_")
    filepath = reports_dir / f"summary_{clean_model_id}.csv"
    
    summary_df.to_csv(filepath, index=False)
    return filepath


def generate_all_plots(
    metrics_df: Optional[pd.DataFrame] = None,
    task_metrics_df: Optional[pd.DataFrame] = None,
    stability_df: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    show: bool = False
) -> Dict[str, List[Path]]:
    """Generate all plots and save them.
    
    Args:
        metrics_df: Metrics DataFrame (loads if None)
        task_metrics_df: Task metrics DataFrame (loads if None)
        stability_df: Stability DataFrame (loads if None)
        plots_dir: Directory for plots
        reports_dir: Directory for reports
        show: Whether to display plots
        
    Returns:
        Dictionary mapping plot type to list of saved paths
    """
    if plots_dir is None:
        plots_dir = DEFAULT_PLOTS_DIR
    if reports_dir is None:
        reports_dir = DEFAULT_REPORTS_DIR
    
    # Load data if not provided
    if metrics_df is None:
        metrics_df = load_metrics()
    if task_metrics_df is None:
        task_metrics_df = load_task_metrics()
    if stability_df is None:
        stability_df = load_stability()
    
    saved_paths = {"heatmaps": [], "distributions": [], "comparisons": [], "reports": []}
    
    # Get unique models
    models = set()
    if len(metrics_df) > 0:
        models.update(metrics_df["model_id"].unique())
    if len(task_metrics_df) > 0:
        models.update(task_metrics_df["model_id"].unique())
    
    for model_id in models:
        clean_id = model_id.replace(":", "_").replace("/", "_")
        
        # Heatmap
        if len(metrics_df[metrics_df["model_id"] == model_id]) > 0:
            path = plots_dir / f"heatmap_{clean_id}.png"
            create_similarity_heatmap(metrics_df, model_id, save_path=path, show=show)
            saved_paths["heatmaps"].append(path)
            
            # Distribution
            path = plots_dir / f"distribution_{clean_id}.png"
            create_similarity_distribution(metrics_df, model_id, save_path=path, show=show)
            saved_paths["distributions"].append(path)
            
            # Task type comparison
            path = plots_dir / f"task_comparison_{clean_id}.png"
            create_task_type_comparison(metrics_df, model_id, save_path=path, show=show)
            saved_paths["comparisons"].append(path)
            
            # Stability comparison
            path = plots_dir / f"stability_{clean_id}.png"
            create_stability_comparison(stability_df, metrics_df, model_id, save_path=path, show=show)
            saved_paths["comparisons"].append(path)
        
        # Discrete task summary
        if len(task_metrics_df[task_metrics_df["model_id"] == model_id]) > 0:
            path = plots_dir / f"discrete_summary_{clean_id}.png"
            create_discrete_task_summary(task_metrics_df, model_id, save_path=path, show=show)
            saved_paths["comparisons"].append(path)
        
        # Summary table
        summary = generate_summary_table(metrics_df, task_metrics_df, stability_df, model_id)
        if len(summary) > 0:
            path = save_summary_table(summary, model_id, reports_dir)
            saved_paths["reports"].append(path)
    
    # Model comparison (if multiple models)
    if len(models) > 1 and len(metrics_df) > 0:
        path = plots_dir / "model_comparison.png"
        create_model_comparison(metrics_df, save_path=path, show=show)
        saved_paths["comparisons"].append(path)
    
    return saved_paths


if __name__ == "__main__":
    print("Plots module loaded. Run generate_all_plots() to create visualizations.")
