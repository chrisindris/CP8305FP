"""
feature_engineering.py
-----------------------
Functions for creating new features from the processed dataset.
Replace the placeholder transformations with domain-specific logic.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Model-specific numerical preprocessing rules
# ---------------------------------------------------------------------------

PREPROCESSING_RULES: dict[str, dict[str, str]] = {
    "time_in_hospital": {
        "knn": "standardize",
        "decision_tree": "none",
        "logistic_regression": "standardize",
        "gaussian_nb": "optional_log1p",
    },
    "num_lab_procedures": {
        "knn": "standardize",
        "decision_tree": "none",
        "logistic_regression": "standardize",
        "gaussian_nb": "none",
    },
    "num_procedures": {
        "knn": "log1p_then_standardize",
        "decision_tree": "none",
        "logistic_regression": "log1p_then_standardize",
        "gaussian_nb": "log1p",
    },
    "num_medications": {
        "knn": "standardize",
        "decision_tree": "none",
        "logistic_regression": "standardize",
        "gaussian_nb": "none",
    },
    "number_outpatient": {
        "knn": "log1p_then_standardize",
        "decision_tree": "none",
        "logistic_regression": "log1p_then_standardize",
        "gaussian_nb": "log1p",
    },
    "number_emergency": {
        "knn": "log1p_then_standardize",
        "decision_tree": "none",
        "logistic_regression": "log1p_then_standardize",
        "gaussian_nb": "log1p",
    },
    "number_inpatient": {
        "knn": "log1p_then_standardize",
        "decision_tree": "none",
        "logistic_regression": "log1p_then_standardize",
        "gaussian_nb": "log1p",
    },
    "number_diagnoses": {
        "knn": "standardize",
        "decision_tree": "none",
        "logistic_regression": "standardize",
        "gaussian_nb": "none",
    },
}

SKEWNESS_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Column-level transforms
# ---------------------------------------------------------------------------

def apply_column_transform(
    series: pd.Series,
    transform: str,
    skewness_threshold: float = SKEWNESS_THRESHOLD,
) -> tuple[pd.Series, object]:
    """Apply a named transform to a single numeric column.

    Parameters
    ----------
    series : pd.Series
        The column values to transform.
    transform : str
        One of ``"none"``, ``"standardize"``, ``"log1p"``,
        ``"log1p_then_standardize"``, or ``"optional_log1p"``.
    skewness_threshold : float
        Absolute skewness above which ``"optional_log1p"`` applies log1p.

    Returns
    -------
    tuple[pd.Series, object]
        The transformed series and fitted metadata.  Metadata is ``None``
        for ``"none"`` and ``"log1p"``, a :class:`StandardScaler` for
        ``"standardize"``, a ``("log1p", StandardScaler)`` tuple for
        ``"log1p_then_standardize"``, or ``"log1p"`` / ``None`` for
        ``"optional_log1p"`` depending on whether the transform fired.
    """
    if transform == "none":
        return series, None

    if transform == "standardize":
        scaler = StandardScaler()
        vals = scaler.fit_transform(series.values.reshape(-1, 1)).ravel()
        return pd.Series(vals, index=series.index, name=series.name), scaler

    if transform == "log1p":
        return pd.Series(
            np.log1p(series.values), index=series.index, name=series.name,
        ), None

    if transform == "log1p_then_standardize":
        logged = np.log1p(series.values)
        scaler = StandardScaler()
        vals = scaler.fit_transform(logged.reshape(-1, 1)).ravel()
        return pd.Series(vals, index=series.index, name=series.name), (
            "log1p",
            scaler,
        )

    if transform == "optional_log1p":
        if abs(series.skew()) > skewness_threshold:
            return pd.Series(
                np.log1p(series.values), index=series.index, name=series.name,
            ), "log1p"
        return series, None

    raise ValueError(
        f"Unknown transform: '{transform}'. "
        "Use 'none', 'standardize', 'log1p', "
        "'log1p_then_standardize', or 'optional_log1p'."
    )


def preprocess_numericals_for_model(
    df: pd.DataFrame,
    model_type: str,
    rules: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply model-specific preprocessing to numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
    model_type : str
        Key into each column's rule dict (e.g. ``"knn"``).
    rules : dict, optional
        Mapping of ``{column: {model_type: transform}}``.
        Defaults to :data:`PREPROCESSING_RULES`.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A copy of *df* with transformed columns and a dict mapping
        column names to fitted metadata (scalers / transform labels).
    """
    if rules is None:
        rules = PREPROCESSING_RULES

    df = df.copy()
    fitted: dict[str, object] = {}

    for col, model_rules in rules.items():
        if col not in df.columns:
            continue
        transform = model_rules.get(model_type, "none")
        df[col], fitted[col] = apply_column_transform(df[col], transform)

    return df, fitted


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    columns: list,
    method: str = "standard",
) -> tuple[pd.DataFrame, object]:
    """Scale numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns to scale.
    method : str
        ``"standard"`` (zero mean, unit variance) or ``"minmax"`` (0–1 range).

    Returns
    -------
    tuple[pd.DataFrame, scaler]
        Updated DataFrame and the fitted scaler (useful for inverse transforms
        and for scaling the test set with the same parameters).
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'standard' or 'minmax'.")

    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def label_encode(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, dict]:
    """Apply label encoding to categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Updated DataFrame and a dictionary mapping column names to fitted
        :class:`~sklearn.preprocessing.LabelEncoder` instances.
    """
    df = df.copy()
    encoders: dict = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def add_interaction_term(
    df: pd.DataFrame, col_a: str, col_b: str, name: str | None = None
) -> pd.DataFrame:
    """Add a multiplicative interaction term between two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    col_a, col_b : str
        Columns to multiply.
    name : str, optional
        Name of the new column. Defaults to ``"<col_a>_x_<col_b>"``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    col_name = name or f"{col_a}_x_{col_b}"
    df[col_name] = df[col_a] * df[col_b]
    return df


def add_log_transform(
    df: pd.DataFrame, columns: list, offset: float = 1.0
) -> pd.DataFrame:
    """Add log-transformed versions of numeric columns.

    A small *offset* (default 1) is added before taking the log to handle
    zero values.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
    offset : float

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        df[f"log_{col}"] = np.log(df[col] + offset)
    return df


def add_binned_column(
    df: pd.DataFrame, column: str, bins: int = 4, labels: list | None = None
) -> pd.DataFrame:
    """Bin a numeric column into equal-frequency quartiles (or custom bins).

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    bins : int
        Number of quantile bins.
    labels : list, optional
        Labels for the bins. Must have length equal to *bins*.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df[f"{column}_bin"] = pd.qcut(df[column], q=bins, labels=labels, duplicates="drop")
    return df


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def get_feature_target_split(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
