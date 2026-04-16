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
    total_cpus: int | None = None,
) -> dict:
    """Nested CV: inner loop tunes hyperparameters, outer loop evaluates.

    Models and outer folds run **in parallel**.  This two-level protocol
    prevents optimistic bias from hyperparameter selection: the outer
    folds never see the tuning decisions.

    Only models that have a corresponding entry in *param_grids* are tuned;
    others are silently skipped.

    Parameters
    ----------
    reduction : tuple[str, dict] or None
        Optional fold-local reduction; see :func:`make_reduction_step`.
        Grid-search parameters for the reducer use the ``reducer__`` prefix
        (e.g. ``reducer__n_components``).
    n_jobs : int
        Legacy parameter (kept for API compat).
    total_cpus : int, optional
        Total CPU cores available.  Auto-detected from environment when
        not provided.
    """
    import re
    if hasattr(X, "columns"):
        X_clean = X.copy()
        X_clean.columns = [re.sub(r'[\[\]{}<>:,"]+', '_', str(col)) for col in X_clean.columns]
        X = X_clean

    if model_configs is None:
        model_configs = CLASSIFICATION_MODELS
    if param_grids is None:
        param_grids = HYPERPARAM_GRIDS

    outer_cv = get_stratified_cv(n_outer_splits, random_state, groups)
    outer_splits = list(outer_cv.split(X, y, groups))

    # Filter to models that have an associated hyper-parameter grid.
    tunable = {n: e for n, e in model_configs.items() if n in param_grids}
    if not tunable:
        return {}

    # ---- CPU budget ----
    cpu_budget = _get_cpu_budget(total_cpus)
    n_models = len(tunable)
    n_parallel_models = min(n_models, max(1, cpu_budget // n_outer_splits))
    cpus_per_model = max(1, cpu_budget // max(1, n_parallel_models))
    n_jobs_inner = max(1, cpus_per_model // 2)

    print(
        f"[Parallel Nested CV] {n_models} models x {n_outer_splits} outer folds | "
        f"CPU budget: {cpu_budget} | "
        f"{n_parallel_models} parallel models, "
        f"{n_jobs_inner} GridSearchCV workers/fold"
    )

    if n_models == 1:
        name = next(iter(tunable))
        grid = param_grids[name]
        n_jobs_inner_1 = max(1, cpu_budget // 2)
        name, result, output = _evaluate_tuned_model(
            name, tunable[name], grid, X, y, groups, outer_splits,
            scale, scoring, random_state, reduction,
            n_inner_splits, n_jobs_inner_1,
        )
        print(output, end="")
        return {name: result}

    parallel_results = joblib.Parallel(
        n_jobs=n_parallel_models, backend="loky",
    )(
        joblib.delayed(_evaluate_tuned_model)(
            name, tunable[name], param_grids[name], X, y, groups, outer_splits,
            scale, scoring, random_state, reduction,
            n_inner_splits, n_jobs_inner,
        )
        for name in tunable
    )

    results = {}
    for name, result, output in parallel_results:
        print(output, end="")
        results[name] = result

    return results
