def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "logistic_regression",
    cv: int = 5,
    random_state: int = 42,
    groups: pd.Series | None = None,
    scoring: str = "f1",
    reduction: tuple[str, dict] | None = None,
    total_cpus: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compare baseline, class-weighted, and SMOTE-based training strategies.

    All estimates are computed with stratified CV on the training set only.
    Pipelines use the leakage-safe fold-local builder so scaling and SMOTE
    are fit strictly on training folds.

    Strategies are evaluated **in parallel** via
    :class:`concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    reduction : tuple[str, dict] or None
        Optional fold-local reduction; see :func:`make_reduction_step`.
    total_cpus : int, optional
        Total CPU cores available.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor

    if hasattr(X_train, "columns"):
        X_clean = X_train.copy()
        X_clean.columns = [re.sub(r'[\[\]{}<>:,"]+', '_', str(col)) for col in X_clean.columns]
        X_train = X_clean

    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(CLASSIFICATION_MODELS)}"
        )

    estimator = clone(CLASSIFICATION_MODELS[model_name])
    use_scaler = model_name in SCALE_SENSITIVE_MODELS
    splitter = get_stratified_cv(n_splits=cv, random_state=random_state, groups=groups)

    def _eval_strategy(name, pipe):
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=splitter, scoring=scoring, groups=groups,
        )
        return name, {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    # Build all strategy pipelines.
    tasks = []
    baseline_pipe = build_fold_pipeline(
        estimator, scale=use_scaler, imbalance_strategy="none",
        random_state=random_state, reduction=reduction,
    )
    tasks.append(("baseline", baseline_pipe))

    if hasattr(estimator, "class_weight"):
        cw_pipe = build_fold_pipeline(
            estimator, scale=use_scaler, imbalance_strategy="class_weight",
            random_state=random_state, reduction=reduction,
        )
        tasks.append(("class_weight_balanced", cw_pipe))

    if IMBLEARN_AVAILABLE:
        smote_pipe = build_fold_pipeline(
            estimator, scale=use_scaler, imbalance_strategy="smote",
            random_state=random_state, reduction=reduction,
        )
        tasks.append(("smote", smote_pipe))

    # Evaluate strategies in parallel.
    results: dict[str, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(_eval_strategy, n, p) for n, p in tasks]
        for future in futures:
            name, res = future.result()
            results[name] = res

    if not IMBLEARN_AVAILABLE and "smote" not in results:
        results["smote"] = {"mean": np.nan, "std": np.nan}

    print("Imbalance strategy comparison completed.")
    return results
