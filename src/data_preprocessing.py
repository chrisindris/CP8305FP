"""
data_preprocessing.py
---------------------
Utility functions for loading, cleaning, and preprocessing raw data.
Replace the placeholder logic with your own dataset-specific steps.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(filename: str) -> pd.DataFrame:
    """Load a CSV (or Excel) file from the raw data directory.

    Parameters
    ----------
    filename : str
        Name of the file to load (e.g. ``"dataset.csv"``).

    Returns
    -------
    pd.DataFrame
    """
    filepath = RAW_DATA_DIR / filename
    if filepath.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(filepath)
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    categorical_fill: str = "Unknown",
) -> pd.DataFrame:
    """Impute missing values.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        Strategy for numeric columns – ``"median"`` (default) or ``"mean"``.
    categorical_fill : str
        Value used to fill missing categorical columns.

    Returns
    -------
    pd.DataFrame
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'median' or 'mean'.")

    df[categorical_cols] = df[categorical_cols].fillna(categorical_fill)
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove rows where any of *columns* falls outside the IQR fence.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Numeric columns to check for outliers.

    Returns
    -------
    pd.DataFrame
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    before = len(df)
    df = df[mask]
    print(f"Removed {before - len(df)} outlier rows.")
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode the specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns to encode.

    Returns
    -------
    pd.DataFrame
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save the processed DataFrame to the processed data directory.

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
        Output filename (e.g. ``"processed_dataset.csv"``).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    raw_filename: str,
    processed_filename: str,
    outlier_columns: list | None = None,
    categorical_columns: list | None = None,
) -> pd.DataFrame:
    """End-to-end preprocessing pipeline.

    Parameters
    ----------
    raw_filename : str
        Input file name in ``data/raw/``.
    processed_filename : str
        Output file name saved to ``data/processed/``.
    outlier_columns : list of str, optional
        Numeric columns to apply IQR outlier removal.
    categorical_columns : list of str, optional
        Columns to one-hot encode.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame.
    """
    df = load_raw_data(raw_filename)
    print(f"Loaded {len(df)} rows from '{raw_filename}'.")

    df = drop_duplicate_rows(df)
    df = handle_missing_values(df)

    if outlier_columns:
        df = remove_outliers_iqr(df, outlier_columns)

    if categorical_columns:
        df = encode_categoricals(df, categorical_columns)

    save_processed_data(df, processed_filename)
    return df


if __name__ == "__main__":
    # TODO: Update filenames and column lists to match your dataset.
    run_preprocessing_pipeline(
        raw_filename="dataset.csv",
        processed_filename="processed_dataset.csv",
        outlier_columns=[],       # e.g. ["price", "quantity"]
        categorical_columns=[],   # e.g. ["region", "category"]
    )
