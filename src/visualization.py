"""
visualization.py
----------------
Reusable plotting functions built on Matplotlib and Seaborn.
All functions return the active Figure so callers can further customise or
save it (see :func:`save_figure`).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    """Save *fig* to ``reports/figures/<filename>``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        E.g. ``"correlation_heatmap.png"``.
    dpi : int
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------
# Exploratory plots
# ---------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame, title: str = "Missing Values") -> plt.Figure:
    """Bar chart of missing-value counts per column.

    Only columns with at least one missing value are shown.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
    else:
        missing.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("Missing Count")
        ax.set_xlabel("Column")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, columns: list, rows: int = 1, bins: int = 30, kde: bool = True) -> plt.Figure:
    """Histogram (with optional KDE) for a single numeric column."""
    num_cols = math.ceil(len(columns) / rows)
    fig, axes = plt.subplots(nrows=rows, ncols=num_cols, figsize=(5 * num_cols, 5 * rows))
    for i, ax in enumerate(axes.flatten()):
        try:
            col = columns[i]
            sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax, color="steelblue")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        except IndexError:
            ax.axis('off')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, title: str = "Correlation Heatmap"
) -> plt.Figure:
    """Heatmap of Pearson correlations for all numeric columns."""
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, columns: list, hue: str | None = None) -> sns.PairGrid:
    """Seaborn pairplot for a selection of numeric columns."""
    grid = sns.pairplot(df[columns + ([hue] if hue else [])], hue=hue, diag_kind="kde")
    grid.figure.suptitle("Pair Plot", y=1.02)
    return grid


def plot_boxplots(df: pd.DataFrame, columns: list, rows: int = 1) -> plt.Figure:
    """Side-by-side box plots for multiple numeric columns."""
    num_cols = math.ceil(len(columns) / rows)
    fig, axes = plt.subplots(nrows=rows, ncols=num_cols, figsize=(5 * num_cols, 5 * rows))
    for i, ax in enumerate(axes.flatten()):
        try:
            col = columns[i]
            sns.boxplot(y=df[col], ax=ax, color="steelblue")
            ax.set_title(col)
        except IndexError:
            ax.axis('off')
    plt.tight_layout()
    return fig


def plot_count(df: pd.DataFrame, columns: str, rows: int = 1, title: str | None = None) -> plt.Figure:
    """Bar chart of value counts for a categorical column."""
    num_cols = math.ceil(len(columns) / rows)
    fig, axes = plt.subplots(nrows=rows, ncols=num_cols, figsize=(5 * num_cols, 5 * rows))
    
    for i, ax in enumerate(axes.flatten()):
        try:
            column = columns[i]
            order = df[column].value_counts().index
            sns.countplot(data=df, x=column, order=order, ax=ax, legend=False)
            ax.set_title(title or f"Count of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
        except IndexError:
            ax.axis('off')
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, columns: list, rows: int = 1, title: str | None = None) -> plt.Figure:
    """Bar chart of value counts for a categorical column."""
    target_column = 'readmitted_binary'
    
    num_cols = math.ceil(len(columns) / rows)
    fig, axes = plt.subplots(nrows=rows, ncols=num_cols, figsize=(5 * num_cols, 5 * rows))
    
    for i, ax in enumerate(axes.flatten()):
        try:
            column = columns[i]
            ax.scatter(df[column], df[target_column], alpha=0.4)
            ax.set_xlabel(column)
            ax.set_ylabel(target_column)
            ax.set_title(f'{column} vs {target_column}')
        except IndexError:
            ax.axis('off')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Display a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Bar chart of built-in feature importances (tree-based models).

    For models that expose ``.feature_importances_`` (e.g. Random Forest,
    Gradient Boosting).  Use :func:`plot_permutation_importance` for
    model-agnostic importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(names[::-1], values[::-1], color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test,
    top_n: int = 20,
    random_state: int = 42,
    title: str = "Permutation Importance",
) -> plt.Figure:
    """Model-agnostic permutation importance chart."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=random_state
    )
    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
    feature_names = list(X_test.columns)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx],
    )
    ax.set_title(title)
    ax.set_xlabel("Importance decrease")
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str = "Residual Plot") -> plt.Figure:
    """Scatter plot of residuals vs. predicted values (regression)."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="steelblue", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Association-rule plots
# ---------------------------------------------------------------------------

def plot_association_rules(
    rules: pd.DataFrame,
    x: str = "support",
    y: str = "confidence",
    color: str = "lift",
    top_n: int | None = None,
    title: str = "Association Rules (support vs confidence)",
) -> plt.Figure:
    """Scatter plot of association rules coloured by a metric.

    Parameters
    ----------
    rules : pd.DataFrame
        Output of ``mlxtend.frequent_patterns.association_rules``.
    x, y : str
        Columns mapped to the x- and y-axes (default support / confidence).
    color : str
        Column mapped to marker colour (default lift).
    top_n : int or None
        If given, only the *top_n* rules by *color* are plotted.
    """
    df = rules.nlargest(top_n, color) if top_n else rules
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df[x], df[y], c=df[color], cmap="coolwarm", alpha=0.7, edgecolors="grey",
        linewidths=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color.capitalize())
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(y.capitalize())
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------

def plot_elbow_curve(inertia_results: list, title: str = "Elbow Curve") -> plt.Figure:
    """Plot inertia vs. k for the elbow method.

    Parameters
    ----------
    inertia_results : list of (k, inertia) tuples
        Output of :func:`~modeling.find_optimal_k`.
    """
    ks, inertias = zip(*inertia_results)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, marker="o", color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(ks)
    plt.tight_layout()
    return fig


def plot_cluster_scatter(
    X: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str,
    title: str = "Cluster Scatter",
) -> plt.Figure:
    """2-D scatter coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[x_col], X[y_col], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROC and PR curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    experiment_results: dict,
    y_true: np.ndarray,
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot ROC curves for all models from an experiment results dict.

    Each model entry must have an ``oof_probabilities`` array.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    y_true = np.asarray(y_true)

    for name, res in experiment_results.items():
        y_prob = res["oof_probabilities"]
        if np.all(np.isnan(y_prob)):
            continue
        valid = ~np.isnan(y_prob)
        fpr, tpr, _ = roc_curve(y_true[valid], y_prob[valid])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_pr_curves(
    experiment_results: dict,
    y_true: np.ndarray,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    y_true = np.asarray(y_true)
    prevalence = y_true.mean()

    for name, res in experiment_results.items():
        y_prob = res["oof_probabilities"]
        if np.all(np.isnan(y_prob)):
            continue
        valid = ~np.isnan(y_prob)
        prec, rec, _ = precision_recall_curve(y_true[valid], y_prob[valid])
        ap = average_precision_score(y_true[valid], y_prob[valid])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    ax.axhline(prevalence, color="k", linestyle="--", alpha=0.4, label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration curves
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    experiment_results: dict,
    y_true: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curves",
) -> plt.Figure:
    """Plot reliability diagrams (calibration curves) for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    y_true = np.asarray(y_true)

    for name, res in experiment_results.items():
        y_prob = res["oof_probabilities"]
        if np.all(np.isnan(y_prob)):
            continue
        valid = ~np.isnan(y_prob)
        fraction_pos, mean_pred = calibration_curve(
            y_true[valid], y_prob[valid], n_bins=n_bins, strategy="uniform",
        )
        ax.plot(mean_pred, fraction_pos, "s-", label=name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    summary_table: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Model Comparison (mean ± 95% CI)",
) -> plt.Figure:
    """Grouped bar chart comparing models across metrics with CI error bars.

    *summary_table* is the output of
    :func:`~modeling.build_metrics_summary_table`.
    """
    if metrics is None:
        metrics = ["f1", "auc_roc", "recall", "kappa"]

    n_models = len(summary_table)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), 6))
    for i, m in enumerate(metrics):
        mean_col = f"{m}_mean"
        ci_lo = f"{m}_ci_lower"
        ci_hi = f"{m}_ci_upper"
        if mean_col not in summary_table.columns:
            continue
        means = summary_table[mean_col].values
        errs_lo = means - summary_table[ci_lo].values
        errs_hi = summary_table[ci_hi].values - means
        ax.bar(
            x + i * width, means, width,
            yerr=[errs_lo, errs_hi], capsize=3,
            label=m, alpha=0.85,
        )

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(summary_table.index, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confidence_intervals(
    summary_table: pd.DataFrame,
    metric: str = "f1",
    title: str | None = None,
) -> plt.Figure:
    """Forest plot showing mean and 95% CI for a single metric."""
    mean_col = f"{metric}_mean"
    ci_lo = f"{metric}_ci_lower"
    ci_hi = f"{metric}_ci_upper"

    df = summary_table.sort_values(mean_col, ascending=True)
    means = df[mean_col].values
    lo = df[ci_lo].values
    hi = df[ci_hi].values
    names = df.index.tolist()

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=[means - lo, hi - means],
            capsize=4, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} – Mean with 95% CI")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cost / threshold sweep
# ---------------------------------------------------------------------------

def plot_cost_threshold_sweep(
    sweep_df: pd.DataFrame,
    title: str = "Expected Cost & Recall vs. Threshold",
) -> plt.Figure:
    """Dual-axis plot of expected cost (left) and recall (right) vs. threshold."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        sweep_df["threshold"], sweep_df["avg_cost"],
        "b-o", markersize=4, label="Avg cost / patient",
    )
    ax2.plot(
        sweep_df["threshold"], sweep_df["recall"],
        "r-s", markersize=4, label="Recall",
    )

    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Average Cost per Patient ($)", color="b")
    ax2.set_ylabel("Recall", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Partial Dependence Plots
# ---------------------------------------------------------------------------

def plot_pdp(
    model,
    X: pd.DataFrame,
    features: list[int] | list[str],
    title: str = "Partial Dependence Plots",
) -> plt.Figure:
    """Plot Partial Dependence for specified features using sklearn.

    *model* must be a fitted estimator or Pipeline.
    *features* is a list of feature indices or names.
    """
    n = len(features)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
    )
    PartialDependenceDisplay.from_estimator(
        model, X, features, ax=axes,
        kind="average",
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP summary wrapper
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    title: str = "SHAP Feature Importance",
    max_display: int = 20,
) -> plt.Figure:
    """Wrapper around shap.summary_plot that returns a Matplotlib figure.

    Requires ``shap`` to be installed.
    """
    try:
        import shap as _shap
    except ImportError:
        raise ImportError("shap is required. Install via: pip install shap")

    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.35)))
    plt.sca(ax)
    _shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display, show=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig
