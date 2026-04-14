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
    total_cpus: int | None = None,
) -> dict:
    """Run fold-local cross-validation for all candidate models **in parallel**.

    Models are evaluated concurrently using the loky (process) backend with
    automatic CPU-budget allocation.  Within each model worker, folds run
    in parallel via threading.  The budget is sourced from *total_cpus*,
    or auto-detected from ``SLURM_CPUS_PER_TASK`` / ``OMP_NUM_THREADS``.

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
    n_jobs : int
        Legacy parameter (kept for API compat).  Effective parallelism is
        now derived from *total_cpus*.
    total_cpus : int, optional
        Total CPU cores available.  Auto-detected from environment when
        not provided.

    Returns
    -------
    dict keyed by model name, each containing:
        fold_metrics      - list of per-fold metric dicts
        oof_predictions   - ndarray of out-of-fold hard predictions
        oof_probabilities - ndarray of out-of-fold positive-class probabilities
        aggregate         - dict of ``{metric: {mean, std, ci_lower, ci_upper}}``
    """
    import re
    if hasattr(X, "columns"):
        X_clean = X.copy()
        X_clean.columns = [re.sub(r'[\[\]{}<>:,"]+', '_', str(col)) for col in X_clean.columns]
        X = X_clean

    if model_configs is None:
        model_configs = CLASSIFICATION_MODELS

    cv = get_stratified_cv(n_splits=n_splits, random_state=random_state, groups=groups)
    splits = list(cv.split(X, y, groups))

    # ---- CPU budget allocation ----
    cpu_budget = _get_cpu_budget(total_cpus)
    n_models = len(model_configs)
    n_parallel_models = min(n_models, cpu_budget)
    cpus_per_model = max(1, cpu_budget // max(1, n_parallel_models))
    n_jobs_folds = min(n_splits, cpus_per_model)
    n_jobs_estimator = max(1, cpus_per_model // max(1, n_jobs_folds))

    print(
        f"[Parallel] {n_models} models x {n_splits} folds | "
        f"CPU budget: {cpu_budget} | "
        f"{n_parallel_models} parallel models, "
        f"{n_jobs_folds} fold workers/model, "
        f"{n_jobs_estimator} CPUs/estimator"
    )

    if n_models == 1:
        # Single model - skip loky process overhead.
        name, estimator = next(iter(model_configs.items()))
        n_jobs_folds_1 = min(n_splits, cpu_budget)
        n_jobs_est_1 = max(1, cpu_budget // max(1, n_jobs_folds_1))
        name, result, output = _evaluate_single_model(
            name, estimator, X, y, splits, scale, imbalance_strategy,
            random_state, reduction, threshold,
            n_jobs_folds_1, n_jobs_est_1,
        )
        print(output, end="")
        return {name: result}

    # Multiple models - train concurrently with loky (process) backend.
    parallel_results = joblib.Parallel(
        n_jobs=n_parallel_models, backend="loky",
    )(
        joblib.delayed(_evaluate_single_model)(
            name, estimator, X, y, splits, scale, imbalance_strategy,
            random_state, reduction, threshold,
            n_jobs_folds, n_jobs_estimator,
        )
        for name, estimator in model_configs.items()
    )

    # Collect and display results in model-config order.
    results = {}
    for name, result, output in parallel_results:
        print(output, end="")
        results[name] = result

    return results
