"""
data_preprocessing.py
---------------------
Utility functions for loading, cleaning, and preprocessing raw data.

This module contains reusable helpers used by the Diabetes 130-US
preprocessing notebook and modeling workflow.
"""

import pandas as pd
import numpy as np
import io
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DIABETES_RAW_DATA_DIR = PROJECT_ROOT / "data" / "diabetes130" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


# Clinical outcome IDs that make 30-day readmission impossible.
TERMINAL_OR_HOSPICE_DISCHARGE_IDS = {"11", "13", "14", "19", "20", "21"}


def _resolve_raw_filepath(filename: str | Path) -> Path:
    """Resolve *filename* across supported raw-data locations.

    Resolution order:
    1. Absolute path (if provided)
    2. ``data/raw``
    3. ``data/diabetes130/raw``
    """
    filepath = Path(filename)
    if filepath.is_absolute():
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        return filepath

    candidates = [
        RAW_DATA_DIR / filepath,
        DIABETES_RAW_DATA_DIR / filepath,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate raw data file. Checked:\n"
        f"  - {joined}"
    )


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
    filepath = _resolve_raw_filepath(filename)
    if filepath.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(filepath)
    return pd.read_csv(filepath)


def read_multiple_tables_csv(filepath: str | Path, table_delimiter: str = ",\n") -> list[pd.DataFrame]:
    """Read a CSV that concatenates multiple tables separated by `,\n`.

    The Diabetes 130-US ``IDS_mapping.csv`` file is published in this format.
    """
    content = Path(filepath).read_text(encoding="utf-8")
    tables = [block for block in content.split(table_delimiter) if block.strip()]
    return [pd.read_csv(io.StringIO(block)) for block in tables]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")
    return df


def keep_first_encounter_per_patient(
    df: pd.DataFrame,
    patient_column: str = "patient_nbr",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """Keep the first encounter per patient after deterministic sorting.

    This is a leakage-safe fallback when group-aware CV is not used later.
    """
    if patient_column not in df.columns or encounter_column not in df.columns:
        return df

    before = len(df)
    deduped = (
        df.sort_values(encounter_column, kind="stable")
        .drop_duplicates(subset=[patient_column], keep="first")
        .copy()
    )
    print(
        "Kept first encounter per patient: "
        f"removed {before - len(deduped)} duplicate encounters."
    )
    return deduped


def filter_terminal_hospice_discharges(
    df: pd.DataFrame,
    discharge_column: str = "discharge_disposition_id",
    excluded_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Drop rows with discharge outcomes that cannot be readmitted."""
    if discharge_column not in df.columns:
        return df

    if excluded_ids is None:
        excluded_ids = TERMINAL_OR_HOSPICE_DISCHARGE_IDS

    before = len(df)
    mask = ~df[discharge_column].astype(str).str.strip().isin(excluded_ids)
    filtered = df.loc[mask].copy()
    print(
        "Removed terminal/hospice discharges: "
        f"{before - len(filtered)} rows filtered."
    )
    return filtered


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


def decode_categorical_ids(
    df: pd.DataFrame,
    mapping_file: str | Path,
) -> pd.DataFrame:
    """Decode clinical ID columns into readable description columns.

    Adds:
    - ``admission_type_desc``
    - ``discharge_disposition_desc``
    - ``admission_source_desc``
    """
    tables = read_multiple_tables_csv(mapping_file)
    if len(tables) < 3:
        raise ValueError(
            "Expected three mapping tables in IDS_mapping.csv "
            "(admission_type, discharge_disposition, admission_source)."
        )

    mappings: list[tuple[str, str, pd.DataFrame]] = [
        ("admission_type_id", "admission_type_desc", tables[0]),
        ("discharge_disposition_id", "discharge_disposition_desc", tables[1]),
        ("admission_source_id", "admission_source_desc", tables[2]),
    ]

    def _normalize_code(value: object) -> str:
        text = str(value).strip()
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    decoded = df.copy()
    for id_col, desc_col, table in mappings:
        if id_col not in decoded.columns:
            continue

        mapping_df = table[[id_col, "description"]].dropna(subset=[id_col]).copy()
        mapping_df[id_col] = mapping_df[id_col].apply(_normalize_code)
        mapping_df["description"] = mapping_df["description"].astype(str).str.strip()
        lookup = dict(zip(mapping_df[id_col], mapping_df["description"]))

        decoded[desc_col] = (
            decoded[id_col].apply(_normalize_code).map(lookup).fillna("Unknown")
        )

    return decoded


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

def encode_categoricals(df: pd.DataFrame, columns: list, drop_first: bool = True) -> pd.DataFrame:
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
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


def drop_identifier_columns(df: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
    """Drop leakage-prone identifier columns."""
    if id_columns is None:
        id_columns = ["encounter_id", "patient_nbr"]
    cols_to_drop = [c for c in id_columns if c in df.columns]
    if cols_to_drop:
        print(f"Dropped identifier columns: {cols_to_drop}")
    return df.drop(columns=cols_to_drop, errors="ignore")


def encode_categoricals_wrapper(
    df: pd.DataFrame,
    target_column: str,
    drop_first: bool = True,
    drop_id_columns: bool = False,
) -> pd.DataFrame:
    """One-hot encode categorical columns except target and identifiers.
    
    Parameters
    ----------
    df : pd.DataFrame
    target_column : str
    drop_first : bool, optional
    """
    # Ensure that we do not encode ID columns or the target column.
    id_columns = ['encounter_id', 'patient_nbr']
    columns_to_exclude = id_columns + [target_column]

    # One-hot encode all categorical columns except the target column and the id columns
    categorical_columns = [
        c for c in df.select_dtypes(include=['object', 'category']).columns
        if c not in columns_to_exclude
    ]

    if categorical_columns:
        df = encode_categoricals(df, categorical_columns, drop_first=drop_first)

    if drop_id_columns:
        df = drop_identifier_columns(df, id_columns=id_columns)

    print(f'Encoded {len(categorical_columns)} categorical columns.')
    print(f'Shape after encoding: {df.shape}')
    return df


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
    drop_id_columns: bool = False,
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
    drop_id_columns : bool, default False
        If True, drop identifier columns prior to saving.

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

    if drop_id_columns:
        df = drop_identifier_columns(df)

    save_processed_data(df, processed_filename)
    return df


if __name__ == "__main__":
    # Use a realistic default for this repository and provide a useful fallback
    # message when the raw file is not present locally.
    try:
        run_preprocessing_pipeline(
            raw_filename="diabetic_data.csv",
            processed_filename="processed_dataset.csv",
            outlier_columns=[],
            categorical_columns=[],
            drop_id_columns=False,
        )
    except FileNotFoundError as exc:
        print(exc)
        print(
            "Run with an explicit filename/path, for example:\n"
            "run_preprocessing_pipeline(raw_filename='diabetes130/raw/diabetic_data.csv', ...)"
        )
