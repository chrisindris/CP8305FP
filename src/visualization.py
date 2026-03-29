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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


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


def plot_distribution(
    df: pd.DataFrame, column: str, bins: int = 30, kde: bool = True
) -> plt.Figure:
    """Histogram (with optional KDE) for a single numeric column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
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


def plot_pairplot(
    df: pd.DataFrame, columns: list, hue: str | None = None
) -> sns.PairGrid:
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


def plot_count(
    df: pd.DataFrame, column: str, title: str | None = None
) -> plt.Figure:
    """Bar chart of value counts for a categorical column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order, ax=ax, palette="Blues_d")
    ax.set_title(title or f"Count of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
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
