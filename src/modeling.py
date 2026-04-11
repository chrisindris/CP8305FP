"""
modeling.py
-----------
Functions for training, evaluating, and persisting sklearn classification models.

Covers leakage-safe fold-local pipelines, unified cross-validation experiments,
comprehensive metrics (AUC-ROC, AUPRC, Cohen's Kappa, top-K), 95 % confidence
intervals, statistical significance testing, calibration diagnostics,
cost-sensitive evaluation, SHAP explainability, and decision-rule export.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
from scipy.stats import t as t_dist, wilcoxon

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
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# Classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    auc as sklearn_auc,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    import lightgbm.compat
    import lightgbm.sklearn

    for mod in (lightgbm.compat, lightgbm.sklearn):
        if hasattr(mod, '_LGBMCheckXY') and getattr(mod, '_LGBMCheckXY') is not None:
            _orig_check_xy = getattr(mod, '_LGBMCheckXY')
            def _patched_check_xy(*args, _orig=_orig_check_xy, **kwargs):
                if "force_all_finite" in kwargs:
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return _orig(*args, **kwargs)
            setattr(mod, '_LGBMCheckXY', _patched_check_xy)

        if hasattr(mod, '_LGBMCheckArray') and getattr(mod, '_LGBMCheckArray') is not None:
            _orig_check_array = getattr(mod, '_LGBMCheckArray')
            def _patched_check_array(*args, _orig=_orig_check_array, **kwargs):
                if "force_all_finite" in kwargs:
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return _orig(*args, **kwargs)
            setattr(mod, '_LGBMCheckArray', _patched_check_array)

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    lime = None
    LIME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
CLASSIFICATION_MODELS: dict[str, object] = {
    # Strict baselines
    "zero_r": DummyClassifier(strategy="most_frequent"), # fast
    "one_r": DecisionTreeClassifier(max_depth=1, random_state=42), # fast
    # Interpretable models
    "naive_bayes": GaussianNB(), # fast
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42), # fast
    "decision_tree": DecisionTreeClassifier(random_state=42), # fast
    "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1), # fast
    "sgd_classifier": SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=5000,
        random_state=42,
        tol=1e-3,
        class_weight="balanced",
    ),
    # Ensemble models
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), # pretty fast
    "gradient_boosting": GradientBoostingClassifier( # pretty fast
        n_estimators=100, random_state=42
    ),
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["xgboost"] = XGBClassifier( # fast
        n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss", n_jobs=-1,
    )
if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODELS["lightgbm"] = LGBMClassifier( # very fast
        n_estimators=100, random_state=42, verbose=-1, n_jobs=-1,
    )
if CATBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["catboost"] = CatBoostClassifier(
        iterations=100, random_seed=42, verbose=0,
    )

# Models that benefit from feature scaling.
SCALE_SENSITIVE_MODELS = {
    "logistic_regression",
    "sgd_classifier",
    "knn",
}

# Alias of the full registry (kept for notebooks that import FAST explicitly).
CLASSIFICATION_MODELS_FAST = dict(CLASSIFICATION_MODELS)

CLASSIFICATION_MODELS_ADVANCED = {k: v for k, v in CLASSIFICATION_MODELS.items() if k in ["random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"]}

CLASSIFICATION_MODELS_SGD = {k: v for k, v in CLASSIFICATION_MODELS.items() if k in ["zero_r", "one_r", "sgd_classifier"]}

# ---------------------------------------------------------------------------
# Hyperparameter search spaces (used by the nested-CV tuning harness)
# ---------------------------------------------------------------------------
import sklearn
import re
_sk_match = re.match(r"^(\d+)\.(\d+)", sklearn.__version__)
_sk_major_minor = tuple(map(int, _sk_match.groups())) if _sk_match else (0, 0)

_lr_penalty_grid = (
    {"classifier__l1_ratio": [1.0, 0.0]}
    if _sk_major_minor >= (1, 8)
    else {"classifier__penalty": ["l1", "l2"]}
)

HYPERPARAM_GRIDS: dict[str, dict] = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        **_lr_penalty_grid,
        "classifier__solver": ["liblinear"],
    },
    "decision_tree": {
        "classifier__max_depth": [3, 5, 10, 20, None],
        "classifier__min_samples_leaf": [1, 5, 10, 20],
        "classifier__criterion": ["gini", "entropy"],
    },
    "knn": {
        "classifier__n_neighbors": [3, 5, 7, 11, 15],
        "classifier__weights": ["uniform", "distance"],
    },
    "sgd_classifier": {
        "classifier__alpha": [1e-5, 1e-4, 1e-3],
        "classifier__learning_rate": ["optimal", "adaptive"],
    },
    "random_forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 20, None],
        "classifier__min_samples_leaf": [1, 5, 10],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
    },
}

if XGBOOST_AVAILABLE:
    HYPERPARAM_GRIDS["xgboost"] = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
    }
if LIGHTGBM_AVAILABLE:
    HYPERPARAM_GRIDS["lightgbm"] = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7, -1],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__num_leaves": [31, 63, 127],
    }


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
# Fold-local pipeline builder
# ---------------------------------------------------------------------------

def make_reduction_step(
    reduction: tuple[str, dict],
    *,
    random_state: int = 42,
) -> tuple[str, object]:
    """Build a sklearn feature-reduction step from a spec (unfitted).

    Parameters
    ----------
    reduction : tuple[str, dict]
        - ``("pca", kwargs)`` → :class:`~sklearn.decomposition.PCA` (dense
          input). ``random_state`` defaults to *random_state* if omitted.
        - ``("select_kbest", kwargs)`` → :class:`~sklearn.feature_selection.SelectKBest`.
          If ``score_func`` is missing, :func:`~sklearn.feature_selection.f_classif`
          is used.
        - ``("select_from_model", kwargs)`` → :class:`~sklearn.feature_selection.SelectFromModel`.
          ``kwargs`` must include ``estimator`` (cloned unfitted).

    random_state : int
        Default PCA random state.

    Returns
    -------
    tuple[str, object]
        ``("reducer", transformer)`` for use in a :class:`~sklearn.pipeline.Pipeline`.
    """
    kind, kw = reduction
    kw = dict(kw)
    if kind == "pca":
        kw.setdefault("random_state", random_state)
        return ("reducer", PCA(**kw))
    if kind == "select_kbest":
        kw.setdefault("score_func", f_classif)
        return ("reducer", SelectKBest(**kw))
    if kind == "select_from_model":
        if "estimator" not in kw:
            raise ValueError(
                'select_from_model reduction requires kwargs["estimator"].'
            )
        estimator = clone(kw.pop("estimator"))
        return ("reducer", SelectFromModel(estimator=estimator, **kw))
    raise ValueError(
        f"Unknown reduction kind {kind!r}; use 'pca', 'select_kbest', or "
        "'select_from_model'."
    )


def _apply_class_weighting(est, y_train: np.ndarray | None = None) -> bool:
    """Set class-weight parameters on *est* in-place.

    Returns ``True`` if the estimator was configured (caller should NOT pass
    ``sample_weight`` during fit); ``False`` if the estimator does not support
    any built-in class weighting (caller should fall back to ``sample_weight``).
    """
    if hasattr(est, "class_weight"):
        est.set_params(class_weight="balanced")
        return True

    if XGBOOST_AVAILABLE and isinstance(est, XGBClassifier):
        if y_train is not None:
            n_neg = int((y_train == 0).sum())
            n_pos = max(int((y_train == 1).sum()), 1)
            est.set_params(scale_pos_weight=n_neg / n_pos)
        return True

    if LIGHTGBM_AVAILABLE and isinstance(est, LGBMClassifier):
        est.set_params(is_unbalance=True)
        return True

    if CATBOOST_AVAILABLE and isinstance(est, CatBoostClassifier):
        est.set_params(auto_class_weights="Balanced")
        return True

    return False


def build_fold_pipeline(
    estimator,
    scale: bool = True,
    imbalance_strategy: str = "none",
    random_state: int = 42,
    reduction: tuple[str, dict] | None = None,
    y_train: np.ndarray | None = None,
) -> Pipeline:
    """Build a leakage-safe fold-local pipeline.

    Ordering: StandardScaler → optional dimensionality reduction → optional
    SMOTE → classifier. The reducer is always fit on the training fold only
    (same as the scaler). The pipeline is returned *unfitted*.

    When both *reduction* and SMOTE are used, order is
    **scaler → reducer → SMOTE → classifier**. SMOTE after PCA can distort
    geometry in the reduced space; for linear classifiers on many features,
    prefer ``imbalance_strategy="class_weight"`` (or ``"none"``) instead of
    SMOTE, or use ``select_kbest`` without PCA.

    Parameters
    ----------
    estimator : unfitted sklearn estimator
    scale : bool
        Prepend a StandardScaler step.
    imbalance_strategy : {"none", "class_weight", "smote", "class_weight+smote"}
        ``"class_weight"`` sets ``class_weight='balanced'`` on the estimator
        (or equivalent for XGBoost/LightGBM/CatBoost).
        ``"smote"`` inserts a SMOTE resampling step.
        ``"class_weight+smote"`` applies both strategies.
    random_state : int
    reduction : tuple[str, dict] or None
        See :func:`make_reduction_step`.
    y_train : ndarray, optional
        Training-fold labels, needed to compute ``scale_pos_weight`` for
        XGBoost when using a class-weight strategy.
    """
    est = clone(estimator)

    use_class_weight = imbalance_strategy in ("class_weight", "class_weight+smote")
    if use_class_weight:
        _apply_class_weighting(est, y_train)

    steps: list[tuple] = []
    if scale:
        steps.append(("scaler", StandardScaler()))

    if reduction is not None:
        steps.append(make_reduction_step(reduction, random_state=random_state))

    use_smote = (
        imbalance_strategy in ("smote", "class_weight+smote")
        and IMBLEARN_AVAILABLE
    )
    if use_smote:
        steps.append(("sampler", SMOTE(random_state=random_state)))

    steps.append(("classifier", est))

    if use_smote:
        return ImbPipeline(steps)
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full required metrics suite for one fold or holdout set.

    Hard-prediction metrics: accuracy, balanced accuracy, precision, recall,
    F1, weighted F1, Cohen's Kappa.
    Probability metrics (when *y_prob* is provided): AUC-ROC, AUPRC,
    PR-AUC (trapezoidal), Brier score.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = float(sklearn_auc(rec_vals, prec_vals))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def compute_topk_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_fractions: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Precision@K and Recall@K for given population fractions.

    Simulates operational capacity constraints: "If we can intervene on the
    top-K% highest-risk patients, how many true readmissions do we capture?"
    """
    if k_fractions is None:
        k_fractions = [0.05, 0.10, 0.15, 0.20]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    total_pos = y_true.sum()
    sorted_idx = np.argsort(y_prob)[::-1]

    results = {}
    for frac in k_fractions:
        k = max(1, int(n * frac))
        tp_at_k = y_true[sorted_idx[:k]].sum()

        label = f"top_{int(frac * 100)}pct"
        results[label] = {
            "precision_at_k": float(tp_at_k / k) if k > 0 else 0.0,
            "recall_at_k": float(tp_at_k / total_pos) if total_pos > 0 else 0.0,
            "k": k,
        }
    return results


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def compute_confidence_interval(
    fold_scores: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute mean and CI from fold-level scores using the t-distribution.

    Appropriate for the small-sample (k = 10) distribution of CV fold scores.
    """
    fold_scores = np.asarray(fold_scores, dtype=float)
    valid = fold_scores[~np.isnan(fold_scores)]
    n = len(valid)
    mean = float(np.mean(valid)) if n > 0 else float("nan")
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci_lower": mean, "ci_upper": mean}

    std = float(np.std(valid, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(t_dist.ppf((1 + confidence) / 2, df=n - 1))
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - t_crit * se,
        "ci_upper": mean + t_crit * se,
    }


# ---------------------------------------------------------------------------
# Unified CV experiment runner
# ---------------------------------------------------------------------------

def run_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_configs: dict | None = None,
    n_splits: int = 10,
    groups: pd.Series | None = None,
    scale: bool = True,
    imbalance_strategy: str = "none",
    threshold: float = 0.5,
    random_state: int = 42,
    reduction: tuple[str, dict] | None = None,
    n_jobs: int = -1,
) -> dict:
    """Run fold-local cross-validation for all candidate models.

    Every fold builds an independent preprocessing pipeline
    (scaler -> optional SMOTE -> estimator) fitted strictly on the training
    fold.  Out-of-fold (OOF) predictions and probabilities are collected for
    downstream calibration, cost analysis, and threshold tuning.

    Parameters
    ----------
    X : DataFrame
        Features (no leakage columns).
    y : Series
        Binary target.
    model_configs : dict, optional
        Mapping of model name -> unfitted estimator.  Defaults to
        :data:`CLASSIFICATION_MODELS`.
    n_splits : int
        Number of CV folds (default 10).
    groups : Series, optional
        Group identifiers for group-aware CV.
    scale : bool
        Prepend a StandardScaler to every fold pipeline.
    imbalance_strategy : {"none", "class_weight", "smote", "class_weight+smote"}
    threshold : float
        Decision threshold for converting probabilities to hard predictions.
        When not 0.5, probabilities are obtained first and thresholded;
        falls back to ``predict()`` if no probability output is available.
    random_state : int
    reduction : tuple[str, dict] or None
        Optional fold-local reduction; see :func:`make_reduction_step`.

    Returns
    -------
    dict keyed by model name, each containing:
        fold_metrics      – list of per-fold metric dicts
        oof_predictions   – ndarray of out-of-fold hard predictions
        oof_probabilities – ndarray of out-of-fold positive-class probabilities
        aggregate         – dict of ``{metric: {mean, std, ci_lower, ci_upper}}``
    """
    if model_configs is None:
        model_configs = CLASSIFICATION_MODELS

    cv = get_stratified_cv(n_splits=n_splits, random_state=random_state, groups=groups)
    splits = list(cv.split(X, y, groups))

    results = {}
    for name, estimator in model_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  Model: {name}")
        print(f"{'=' * 60}")

        fold_metrics_list: list[dict] = []
        oof_preds = np.full(len(y), fill_value=-1, dtype=int)
        oof_probs = np.full(len(y), fill_value=np.nan)

        def _evaluate_fold(fold_idx, train_idx, val_idx):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                use_cw = imbalance_strategy in ("class_weight", "class_weight+smote")
                pipeline = build_fold_pipeline(
                    estimator,
                    scale=scale,
                    imbalance_strategy=imbalance_strategy,
                    random_state=random_state,
                    reduction=reduction,
                    y_train=y_tr.values if use_cw else None,
                )

                uses_smote = imbalance_strategy in ("smote", "class_weight+smote")
                needs_sample_weight = (
                    use_cw
                    and not uses_smote
                    and not _apply_class_weighting(clone(estimator), y_tr.values)
                )
                if needs_sample_weight:
                    sw = compute_sample_weight("balanced", y_tr)
                    pipeline.fit(X_tr, y_tr, classifier__sample_weight=sw)
                else:
                    pipeline.fit(X_tr, y_tr)

                y_prob = None
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_val)[:, 1]
                elif hasattr(pipeline, "decision_function"):
                    y_prob = expit(pipeline.decision_function(X_val))

                if y_prob is not None and threshold != 0.5:
                    y_pred = (y_prob >= threshold).astype(int)
                else:
                    y_pred = pipeline.predict(X_val)

                fold_m = compute_classification_metrics(
                    y_val.values, y_pred, y_prob,
                )
                return fold_idx, val_idx, y_pred, y_prob, fold_m, None
            except Exception as exc:
                return fold_idx, val_idx, None, None, None, str(exc)

        # Run folds in parallel
        results_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_evaluate_fold)(fold_idx, tr_idx, val_idx)
            for fold_idx, (tr_idx, val_idx) in enumerate(splits)
        )

        for fold_idx, val_idx, y_pred, y_prob, fold_m, exc_msg in results_list:
            if exc_msg is not None:
                print(f"  Fold {fold_idx + 1:2d}: FAILED – {exc_msg}")
                fold_m = {
                    k: float("nan")
                    for k in [
                        "accuracy", "balanced_accuracy", "precision", "recall",
                        "f1", "f1_weighted", "kappa", "auc_roc", "auprc",
                        "pr_auc", "brier_score",
                    ]
                }
            else:
                oof_preds[val_idx] = y_pred
                if y_prob is not None:
                    oof_probs[val_idx] = y_prob
                
                auc_str = f"{fold_m.get('auc_roc', float('nan')):.4f}"
                print(
                    f"  Fold {fold_idx + 1:2d}: "
                    f"F1={fold_m['f1']:.4f}  AUC={auc_str}  "
                    f"Kappa={fold_m['kappa']:.4f}"
                )
            fold_metrics_list.append(fold_m)

        # Aggregate fold metrics with 95 % CIs.
        aggregate: dict[str, dict] = {}
        for metric in fold_metrics_list[0].keys():
            scores = np.array([fm[metric] for fm in fold_metrics_list])
            aggregate[metric] = compute_confidence_interval(scores)

        results[name] = {
            "fold_metrics": fold_metrics_list,
            "oof_predictions": oof_preds,
            "oof_probabilities": oof_probs,
            "aggregate": aggregate,
        }

        for metric in ["f1", "auc_roc", "kappa"]:
            if metric in aggregate:
                a = aggregate[metric]
                print(
                    f"  {metric:>12s}: {a['mean']:.4f} "
                    f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]"
                )

    return results


# ---------------------------------------------------------------------------
# Metrics summary table
# ---------------------------------------------------------------------------

def build_metrics_summary_table(
    experiment_results: dict,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of mean ± 95 % CI for each metric across all models.

    Rows are models (sorted by F1 descending); columns include mean, std,
    ci_lower, and ci_upper for every requested metric.
    """
    if metrics is None:
        metrics = [
            "f1", "auc_roc", "auprc", "pr_auc", "recall", "precision",
            "kappa", "brier_score", "balanced_accuracy",
        ]

    rows = []
    for model_name, res in experiment_results.items():
        row: dict[str, object] = {"model": model_name}
        for m in metrics:
            agg = res["aggregate"].get(m)
            if agg:
                row[f"{m}_mean"] = agg["mean"]
                row[f"{m}_std"] = agg["std"]
                row[f"{m}_ci_lower"] = agg["ci_lower"]
                row[f"{m}_ci_upper"] = agg["ci_upper"]
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_ci_lower"] = np.nan
                row[f"{m}_ci_upper"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    if "f1_mean" in df.columns:
        df = df.sort_values("f1_mean", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def run_significance_tests(
    experiment_results: dict,
    baseline_model: str | None = None,
    metric: str = "f1",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired significance tests between a baseline and all other models.

    Uses a corrected resampled paired t-test with a Wilcoxon signed-rank
    non-parametric fallback.  Reports both p-values and a significance flag
    at the given *alpha* level.

    Parameters
    ----------
    experiment_results : dict
        Output of :func:`run_cv_experiment`.
    baseline_model : str, optional
        Reference model name.  Defaults to ``"zero_r"`` if present.
    metric : str
        Metric to compare (must appear in fold_metrics dicts).
    alpha : float
        Significance threshold.
    """
    if baseline_model is None:
        baseline_model = (
            "zero_r"
            if "zero_r" in experiment_results
            else next(iter(experiment_results))
        )

    baseline_scores = np.array(
        [fm[metric] for fm in experiment_results[baseline_model]["fold_metrics"]]
    )

    rows = []
    for name, res in experiment_results.items():
        if name == baseline_model:
            continue

        cand_scores = np.array([fm[metric] for fm in res["fold_metrics"]])
        diff = cand_scores - baseline_scores
        mean_diff = float(np.mean(diff))

        # Corrected resampled paired t-test.
        std_diff = float(np.std(diff, ddof=1))
        n = len(diff)
        if std_diff > 0:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            p_t = float(2 * t_dist.sf(abs(t_stat), df=n - 1))
        else:
            t_stat, p_t = 0.0, 1.0

        # Wilcoxon signed-rank (non-parametric fallback).
        try:
            _, p_w = wilcoxon(diff, alternative="two-sided")
            p_w = float(p_w)
        except ValueError:
            p_w = 1.0

        rows.append({
            "model": name,
            "baseline": baseline_model,
            "metric": metric,
            "mean_diff": mean_diff,
            "t_stat": float(t_stat),
            "p_value_t": p_t,
            "p_value_wilcoxon": p_w,
            "significant_t": p_t < alpha,
            "significant_w": p_w < alpha,
        })

    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute Brier score and calibration curve data.

    Returns Brier score and the (fraction_of_positives, mean_predicted_value)
    arrays needed for plotting a reliability diagram.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform",
    )
    return {
        "brier_score": brier,
        "fraction_of_positives": fraction_pos,
        "mean_predicted_value": mean_pred,
    }


def calibrate_model(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Apply post-hoc calibration (Platt scaling or isotonic regression).

    Parameters
    ----------
    method : {"sigmoid", "isotonic"}
        ``"sigmoid"`` for Platt scaling, ``"isotonic"`` for isotonic regression.
    """
    calibrated = CalibratedClassifierCV(
        estimator=clone(estimator), method=method, cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Cost-sensitive evaluation
# ---------------------------------------------------------------------------

# Default cost matrix aligned to HRRP readmission penalties.
# FN is heavily penalised: a missed readmission incurs ~$15 k in HRRP penalties.
# FP is lightly penalised: a false alarm triggers low-cost preventative care.
DEFAULT_COST_MATRIX = {
    "TP": -2_000.0,
    "TN": 0.0,
    "FP": -200.0,
    "FN": -15_000.0,
}


def compute_expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict | None = None,
) -> dict[str, float]:
    """Compute expected cost from a confusion-matrix-aligned cost matrix.

    Returns total cost and per-patient average cost, plus the confusion
    matrix counts.
    """
    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total_cost = (
        tp * cost_matrix["TP"]
        + tn * cost_matrix["TN"]
        + fp * cost_matrix["FP"]
        + fn * cost_matrix["FN"]
    )
    n = len(y_true)
    return {
        "total_cost": float(total_cost),
        "avg_cost_per_patient": float(total_cost / n) if n > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def sweep_thresholds_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: dict | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep decision thresholds and report expected cost + recall at each.

    Used to select an operating point that balances clinical safety (recall)
    with financial viability (expected cost).
    """
    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    y_true = np.asarray(y_true)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cost = compute_expected_cost(y_true, y_pred, cost_matrix)
        rows.append({
            "threshold": float(thr),
            "total_cost": cost["total_cost"],
            "avg_cost": cost["avg_cost_per_patient"],
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": cost["tp"], "fp": cost["fp"],
            "fn": cost["fn"], "tn": cost["tn"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

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
        prec = precision_score(y_true_arr, y_pred, zero_division=0)
        rec = recall_score(y_true_arr, y_pred, zero_division=0)
        f1_val = f1_score(y_true_arr, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true_arr, y_pred)

        row = {
            "threshold": float(threshold),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
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


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap_values(
    pipeline_or_model,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> tuple:
    """Compute SHAP values for a fitted model or pipeline.

    Handles sklearn Pipelines by extracting the final estimator and
    pre-transforming X through any preprocessing steps.  Uses TreeExplainer
    for tree-based models and KernelExplainer otherwise.

    Returns
    -------
    (shap_values, explainer, X_sample)
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for SHAP explanations. "
            "Install via: pip install shap"
        )

    # Extract the estimator and pre-transform X when given a Pipeline.
    is_pipeline = isinstance(pipeline_or_model, Pipeline) or (
        ImbPipeline is not None and isinstance(pipeline_or_model, ImbPipeline)
    )
    if is_pipeline:
        estimator = pipeline_or_model.steps[-1][1]
        X_transformed = X.copy()
        for _, step in pipeline_or_model.steps[:-1]:
            if hasattr(step, "transform"):
                X_transformed = step.transform(X_transformed)
        cols = feature_names or (list(X.columns) if hasattr(X, "columns") else None)
        if isinstance(X_transformed, np.ndarray) and cols:
            X_transformed = pd.DataFrame(X_transformed, columns=cols)
    else:
        estimator = pipeline_or_model
        X_transformed = X

    # Subsample for speed on large datasets.
    if len(X_transformed) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_transformed), max_samples, replace=False)
        X_sample = (
            X_transformed.iloc[idx]
            if hasattr(X_transformed, "iloc")
            else X_transformed[idx]
        )
    else:
        X_sample = X_transformed

    # Choose explainer by estimator type.
    tree_types = {
        "RandomForestClassifier", "GradientBoostingClassifier",
        "DecisionTreeClassifier", "XGBClassifier", "LGBMClassifier",
        "CatBoostClassifier",
    }
    model_type = type(estimator).__name__

    if model_type in tree_types:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        background = shap.kmeans(X_sample, min(10, len(X_sample)))
        explainer = shap.KernelExplainer(estimator.predict_proba, background)
        sv = explainer.shap_values(X_sample, nsamples=100)
        if isinstance(sv, list):
            sv = sv[1]

    return sv, explainer, X_sample


# ---------------------------------------------------------------------------
# LIME explainability
# ---------------------------------------------------------------------------

def compute_lime_explanation(
    pipeline_or_model,
    X_train: pd.DataFrame,
    instance: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
    num_features: int = 10,
):
    """Generate a LIME explanation for a single instance.

    Returns a :class:`lime.explanation.Explanation` object whose
    ``.as_pyplot_figure()`` method produces a Matplotlib figure.
    """
    if not LIME_AVAILABLE:
        raise ImportError(
            "lime is required for LIME explanations. "
            "Install via: pip install lime"
        )

    names = feature_names or (
        list(X_train.columns) if hasattr(X_train, "columns") else None
    )
    train_data = X_train.values if hasattr(X_train, "values") else X_train

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_data,
        feature_names=names,
        class_names=["Not Readmitted", "Readmitted"],
        mode="classification",
    )

    predict_fn = (
        pipeline_or_model.predict_proba
        if hasattr(pipeline_or_model, "predict_proba")
        else pipeline_or_model.predict
    )
    instance_arr = instance.values if hasattr(instance, "values") else instance

    return explainer.explain_instance(
        instance_arr, predict_fn, num_features=num_features,
    )


# ---------------------------------------------------------------------------
# Rule export
# ---------------------------------------------------------------------------

def export_tree_rules(
    tree_model: DecisionTreeClassifier,
    feature_names: list[str],
    max_depth: int | None = None,
) -> str:
    """Export human-readable decision rules from a fitted decision tree.

    Returns a string representation suitable for clinical checklists.
    """
    return export_text(
        tree_model,
        feature_names=feature_names,
        max_depth=max_depth,
        show_weights=True,
    )


def export_forest_top_rules(
    forest_model: RandomForestClassifier,
    feature_names: list[str],
    n_trees: int = 3,
    max_depth: int = 3,
) -> list[str]:
    """Export rules from the top-N trees in a random forest.

    Trees are taken in order of the ensemble; pass a fitted model.
    """
    rules = []
    for i, tree in enumerate(forest_model.estimators_[:n_trees]):
        rule_text = export_text(
            tree, feature_names=feature_names,
            max_depth=max_depth, show_weights=True,
        )
        rules.append(f"--- Tree {i + 1} ---\n{rule_text}")
    return rules


# ---------------------------------------------------------------------------
# Nested CV (unbiased tuning + evaluation)
# ---------------------------------------------------------------------------

def run_tuned_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_configs: dict | None = None,
    param_grids: dict | None = None,
    n_outer_splits: int = 10,
    n_inner_splits: int = 5,
    groups: pd.Series | None = None,
    scale: bool = True,
    scoring: str = "f1",
    random_state: int = 42,
    reduction: tuple[str, dict] | None = None,
    n_jobs: int = 1,
) -> dict:
    """Nested CV: inner loop tunes hyperparameters, outer loop evaluates.

    This two-level protocol prevents optimistic bias from hyperparameter
    selection: the outer folds never see the tuning decisions.

    Only models that have a corresponding entry in *param_grids* are tuned;
    others are silently skipped.

    Parameters
    ----------
    reduction : tuple[str, dict] or None
        Optional fold-local reduction; see :func:`make_reduction_step`.
        Grid-search parameters for the reducer use the ``reducer__`` prefix
        (e.g. ``reducer__n_components``).
    n_jobs : int
        Parallel jobs for inner :class:`~sklearn.model_selection.GridSearchCV`.
        Use ``1`` (default) to limit RAM and CPU load: ``-1`` uses all cores
        and joblib typically copies *X_tr* per worker, which spikes memory on
        large matrices. Pass ``-1`` only if you have headroom.
    """
    if model_configs is None:
        model_configs = CLASSIFICATION_MODELS
    if param_grids is None:
        param_grids = HYPERPARAM_GRIDS

    outer_cv = get_stratified_cv(n_outer_splits, random_state, groups)
    outer_splits = list(outer_cv.split(X, y, groups))

    results = {}
    for name, estimator in model_configs.items():
        grid = param_grids.get(name)
        if grid is None:
            continue

        print(f"\n{'=' * 60}")
        print(f"  Nested CV: {name}")
        print(f"{'=' * 60}")

        fold_metrics_list: list[dict] = []
        best_params_per_fold: list[dict] = []
        oof_preds = np.full(len(y), fill_value=-1, dtype=int)
        oof_probs = np.full(len(y), fill_value=np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = build_fold_pipeline(
                estimator,
                scale=scale,
                random_state=random_state,
                reduction=reduction,
            )

            inner_groups = (
                groups.iloc[train_idx] if groups is not None else None
            )
            inner_cv = get_stratified_cv(
                n_inner_splits, random_state + fold_idx, inner_groups,
            )

            gs = GridSearchCV(
                pipeline, grid, cv=inner_cv, scoring=scoring,
                n_jobs=n_jobs, refit=True,
            )
            gs.fit(X_tr, y_tr, groups=inner_groups)
            best_params = gs.best_params_
            best_params_per_fold.append(best_params)

            best_pipeline = gs.best_estimator_
            y_pred = best_pipeline.predict(X_val)
            oof_preds[val_idx] = y_pred

            y_prob = None
            if hasattr(best_pipeline, "predict_proba"):
                y_prob = best_pipeline.predict_proba(X_val)[:, 1]
                oof_probs[val_idx] = y_prob

            fold_m = compute_classification_metrics(y_val.values, y_pred, y_prob)
            fold_metrics_list.append(fold_m)
            print(
                f"  Fold {fold_idx + 1:2d}: "
                f"F1={fold_m['f1']:.4f}  "
                f"AUC={fold_m.get('auc_roc', float('nan')):.4f}  "
                f"Best={best_params}"
            )
            del gs

        aggregate: dict[str, dict] = {}
        for metric in fold_metrics_list[0].keys():
            scores = np.array([fm[metric] for fm in fold_metrics_list])
            aggregate[metric] = compute_confidence_interval(scores)

        results[name] = {
            "fold_metrics": fold_metrics_list,
            "oof_predictions": oof_preds,
            "oof_probabilities": oof_probs,
            "aggregate": aggregate,
            "best_params_per_fold": best_params_per_fold,
        }

    return results


# ---------------------------------------------------------------------------
# Imbalance strategy comparison
# ---------------------------------------------------------------------------

def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "logistic_regression",
    cv: int = 5,
    random_state: int = 42,
    groups: pd.Series | None = None,
    scoring: str = "f1",
    reduction: tuple[str, dict] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare baseline, class-weighted, and SMOTE-based training strategies.

    All estimates are computed with stratified CV on the training set only.
    Pipelines use the leakage-safe fold-local builder so scaling and SMOTE
    are fit strictly on training folds.

    Parameters
    ----------
    reduction : tuple[str, dict] or None
        Optional fold-local reduction; see :func:`make_reduction_step`.
    """
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(CLASSIFICATION_MODELS)}"
        )

    estimator = clone(CLASSIFICATION_MODELS[model_name])
    use_scaler = model_name in SCALE_SENSITIVE_MODELS
    splitter = get_stratified_cv(n_splits=cv, random_state=random_state, groups=groups)

    results: dict[str, dict[str, float]] = {}

    # Baseline (no imbalance handling).
    baseline_pipe = build_fold_pipeline(
        estimator,
        scale=use_scaler,
        imbalance_strategy="none",
        random_state=random_state,
        reduction=reduction,
    )
    baseline_scores = cross_val_score(
        baseline_pipe, X_train, y_train,
        cv=splitter, scoring=scoring, groups=groups,
    )
    results["baseline"] = {
        "mean": float(np.mean(baseline_scores)),
        "std": float(np.std(baseline_scores)),
    }

    # Class-weight balanced.
    if hasattr(estimator, "class_weight"):
        cw_pipe = build_fold_pipeline(
            estimator,
            scale=use_scaler,
            imbalance_strategy="class_weight",
            random_state=random_state,
            reduction=reduction,
        )
        cw_scores = cross_val_score(
            cw_pipe, X_train, y_train,
            cv=splitter, scoring=scoring, groups=groups,
        )
        results["class_weight_balanced"] = {
            "mean": float(np.mean(cw_scores)),
            "std": float(np.std(cw_scores)),
        }

    # SMOTE (scaler -> SMOTE -> classifier, all fold-local).
    if IMBLEARN_AVAILABLE:
        smote_pipe = build_fold_pipeline(
            estimator,
            scale=use_scaler,
            imbalance_strategy="smote",
            random_state=random_state,
            reduction=reduction,
        )
        smote_scores = cross_val_score(
            smote_pipe, X_train, y_train,
            cv=splitter, scoring=scoring, groups=groups,
        )
        results["smote"] = {
            "mean": float(np.mean(smote_scores)),
            "std": float(np.std(smote_scores)),
        }
    else:
        results["smote"] = {"mean": np.nan, "std": np.nan}

    print("Imbalance strategy comparison completed.")
    return results


# ---------------------------------------------------------------------------
# Probability helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], LogisticRegression]:
    """Select features with L1-regularized logistic regression.

    Returns selected column names and the fitted selector estimator.
    """
    import sklearn
    import re
    _sk_match = re.match(r"^(\d+)\.(\d+)", sklearn.__version__)
    _sk_major_minor = tuple(map(int, _sk_match.groups())) if _sk_match else (0, 0)

    kwargs = {
        "solver": "liblinear",
        "C": C,
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": random_state,
    }
    if _sk_major_minor >= (1, 8):
        kwargs["l1_ratio"] = 1.0
    else:
        kwargs["penalty"] = "l1"

    selector_model = LogisticRegression(**kwargs)
    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, prefit=True, threshold="median")
    selected_mask = selector.get_support()
    selected_columns = X_train.columns[selected_mask].tolist()

    if not selected_columns:
        selected_columns = X_train.columns.tolist()

    print(
        f"L1 feature selection kept {len(selected_columns)} "
        f"of {X_train.shape[1]} features."
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
# Classification (backward compatibility)
# ---------------------------------------------------------------------------

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
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
}


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest",
    scale: bool = True,
) -> Pipeline:
    """Train a regression pipeline."""
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
    """Evaluate a regression pipeline and print a summary."""
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
    """Fit KMeans and return the model together with cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"KMeans (k={n_clusters}) – Silhouette Score: {score:.4f}")
    return model, labels


def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 11)) -> list:
    """Compute inertia for a range of *k* values (elbow method)."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        results.append((k, km.inertia_))
        print(f"  k={k}  inertia={km.inertia_:.2f}")
    return results


# ---------------------------------------------------------------------------
# Cross-validation (backward compatibility)
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
    """Run k-fold cross-validation and return mean ± std scores."""
    target_type = type_of_target(y)
    is_classification_target = target_type in {"binary", "multiclass"}

    if is_classification_target:
        splitter = get_stratified_cv(
            n_splits=cv, random_state=random_state, groups=groups,
        )
        scores = cross_val_score(
            model, X, y, cv=splitter, scoring=scoring, groups=groups,
        )
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
    print(
        f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f} "
        f"(across {cv} folds)"
    )
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, filename: str) -> None:
    """Serialize *model* to ``reports/<filename>`` using :mod:`joblib`."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(filename: str):
    """Load a previously saved model from ``reports/<filename>``."""
    path = REPORTS_DIR / filename
    return joblib.load(path)
