"""
modeling.py
-----------
Functions for training, evaluating, and persisting SKLearn models.
Supports both classification and regression pipelines.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split into training and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest",
    scale: bool = True,
) -> Pipeline:
    """Train a classification pipeline.

    Parameters
    ----------
    X_train, y_train : array-like
    model_name : str
        Key from :data:`CLASSIFICATION_MODELS`.
    scale : bool
        If ``True``, prepend a :class:`~sklearn.preprocessing.StandardScaler`.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(CLASSIFICATION_MODELS)}"
        )
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", CLASSIFICATION_MODELS[model_name]))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a classification pipeline and print a summary.

    Returns
    -------
    dict with accuracy, report, confusion_matrix
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"\nClassification Report:\n{report}")
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=0.1, max_iter=10000),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest",
    scale: bool = True,
) -> Pipeline:
    """Train a regression pipeline.

    Parameters
    ----------
    X_train, y_train : array-like
    model_name : str
        Key from :data:`REGRESSION_MODELS`.
    scale : bool
        If ``True``, prepend a :class:`~sklearn.preprocessing.StandardScaler`.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if model_name not in REGRESSION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(REGRESSION_MODELS)}"
        )
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", REGRESSION_MODELS[model_name]))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_regressor(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a regression pipeline and print a summary.

    Returns
    -------
    dict with rmse, mae, r2
    """
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple:
    """Fit KMeans and return the model together with cluster labels.

    Returns
    -------
    tuple[KMeans, np.ndarray]
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"KMeans (k={n_clusters}) – Silhouette Score: {score:.4f}")
    return model, labels


def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 11)) -> list:
    """Compute inertia for a range of *k* values (elbow method).

    Returns
    -------
    list of (k, inertia) tuples
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        results.append((k, km.inertia_))
        print(f"  k={k}  inertia={km.inertia_:.2f}")
    return results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """Run k-fold cross-validation and return mean ± std scores.

    Returns
    -------
    dict with scores array, mean, std
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(
        f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f} "
        f"(across {cv} folds)"
    )
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# ---------------------------------------------------------------------------
# Hyper-parameter tuning
# ---------------------------------------------------------------------------

def grid_search(
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> GridSearchCV:
    """Run GridSearchCV and print the best parameters.

    Returns
    -------
    GridSearchCV (fitted)
    """
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Best params : {gs.best_params_}")
    print(f"Best score  : {gs.best_score_:.4f}")
    return gs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, filename: str) -> None:
    """Serialize *model* to ``reports/<filename>`` using :mod:`joblib`.

    Parameters
    ----------
    model : fitted estimator or pipeline
    filename : str
        E.g. ``"random_forest_classifier.pkl"``.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(filename: str):
    """Load a previously saved model from ``reports/<filename>``.

    Returns
    -------
    Fitted estimator or pipeline.
    """
    path = REPORTS_DIR / filename
    return joblib.load(path)
