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
from scipy.special import expit

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

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
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.utils.multiclass import type_of_target

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


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
    stratify: bool = True,
) -> tuple:
    """Split into training and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def get_stratified_cv(
    n_splits: int = 10,
    random_state: int = 42,
    groups: pd.Series | None = None,
):
    """Return stratified CV splitter, optionally group-aware.

    If *groups* is provided, stratification and group isolation are both
    enforced via :class:`StratifiedGroupKFold`.
    """
    if groups is not None:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


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
    class_weight: str | dict | None = None,
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
    estimator = clone(CLASSIFICATION_MODELS[model_name])
    # Apply class weighting when the estimator supports it.
    if class_weight is not None and hasattr(estimator, "class_weight"):
        estimator.set_params(class_weight=class_weight)

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
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
    groups: pd.Series | None = None,
    random_state: int = 42,
) -> dict:
    """Run k-fold cross-validation and return mean ± std scores.

    Returns
    -------
    dict with scores array, mean, std
    """
    target_type = type_of_target(y)
    is_classification_target = target_type in {"binary", "multiclass"}

    if is_classification_target:
        splitter = get_stratified_cv(
            n_splits=cv,
            random_state=random_state,
            groups=groups,
        )
        scores = cross_val_score(
            model,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            groups=groups,
        )
    else:
        # Regression targets cannot use stratified splitters.
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
    print(
        f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f} "
        f"(across {cv} folds)"
    )
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "logistic_regression",
    cv: int = 5,
    random_state: int = 42,
    groups: pd.Series | None = None,
    scoring: str = "f1",
) -> dict[str, dict[str, float]]:
    """Compare baseline, class-weighted, and SMOTE-based training strategies.

    All estimates are computed with stratified CV on the training set only.
    """
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(CLASSIFICATION_MODELS)}"
        )

    estimator = clone(CLASSIFICATION_MODELS[model_name])
    use_scaler = model_name in {"logistic_regression", "svm", "knn"}
    splitter = get_stratified_cv(n_splits=cv, random_state=random_state, groups=groups)

    results: dict[str, dict[str, float]] = {}

    baseline_steps = []
    if use_scaler:
        baseline_steps.append(("scaler", StandardScaler()))
    baseline_steps.append(("classifier", clone(estimator)))
    baseline_pipeline = Pipeline(baseline_steps)
    baseline_scores = cross_val_score(
        baseline_pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring=scoring,
        groups=groups,
    )
    results["baseline"] = {
        "mean": float(np.mean(baseline_scores)),
        "std": float(np.std(baseline_scores)),
    }

    if hasattr(estimator, "class_weight"):
        weighted_estimator = clone(estimator)
        weighted_estimator.set_params(class_weight="balanced")
        weighted_steps = []
        if use_scaler:
            weighted_steps.append(("scaler", StandardScaler()))
        weighted_steps.append(("classifier", weighted_estimator))
        weighted_pipeline = Pipeline(weighted_steps)
        weighted_scores = cross_val_score(
            weighted_pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            groups=groups,
        )
        results["class_weight_balanced"] = {
            "mean": float(np.mean(weighted_scores)),
            "std": float(np.std(weighted_scores)),
        }

    if IMBLEARN_AVAILABLE:
        smote_steps = [("smote", SMOTE(random_state=random_state))]
        if use_scaler:
            smote_steps.append(("scaler", StandardScaler()))
        smote_steps.append(("classifier", clone(estimator)))
        smote_pipeline = ImbPipeline(smote_steps)
        smote_scores = cross_val_score(
            smote_pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            groups=groups,
        )
        results["smote"] = {
            "mean": float(np.mean(smote_scores)),
            "std": float(np.std(smote_scores)),
        }
    else:
        results["smote"] = {
            "mean": np.nan,
            "std": np.nan,
        }

    print("Imbalance strategy comparison completed.")
    return results


def predict_positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier pipeline.

    Falls back to calibrated sigmoid of decision scores when ``predict_proba``
    is unavailable.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return expit(scores)

    raise AttributeError(
        "Pipeline does not expose predict_proba or decision_function."
    )


def tune_decision_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> dict:
    """Tune binary classification threshold on validation predictions.

    Returns the best threshold for the selected metric and a full score table.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_true_arr = np.asarray(y_true)
    rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true_arr, y_pred, zero_division=0)
        recall = recall_score(y_true_arr, y_pred, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true_arr, y_pred)

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_acc),
        }
        rows.append(row)

        current_metric = row.get(metric)
        if current_metric is None:
            raise ValueError(
                "Unknown metric. Use one of: "
                "'precision', 'recall', 'f1', 'balanced_accuracy'."
            )
        if current_metric > best_score:
            best_score = current_metric
            best_threshold = float(threshold)

    table = pd.DataFrame(rows).sort_values(metric, ascending=False)
    print(
        f"Best threshold by {metric}: {best_threshold:.2f} "
        f"(score={best_score:.4f})"
    )
    return {
        "best_threshold": best_threshold,
        "best_score": float(best_score),
        "metric": metric,
        "table": table,
    }


def select_features_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], LogisticRegression]:
    """Select features with L1-regularized logistic regression.

    Returns selected column names and the fitted selector estimator.
    """
    selector_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, prefit=True, threshold="median")
    selected_mask = selector.get_support()
    selected_columns = X_train.columns[selected_mask].tolist()

    # Ensure downstream code still runs if regularization is too aggressive.
    if not selected_columns:
        selected_columns = X_train.columns.tolist()

    print(
        f"L1 feature selection kept {len(selected_columns)} of {X_train.shape[1]} features."
    )
    return selected_columns, selector_model


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
