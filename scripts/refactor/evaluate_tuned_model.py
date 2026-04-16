def _evaluate_tuned_model(
    name, estimator, grid, X, y, groups, outer_splits,
    scale, scoring, random_state, reduction,
    n_inner_splits, n_jobs_inner,
):
    """Run nested CV for a single model across all outer folds.

    Top-level function for loky pickling.  Uses
    :class:`concurrent.futures.ThreadPoolExecutor` for outer-fold
    parallelism so that the inner :class:`GridSearchCV` can use joblib
    without nesting issues.

    Returns
    -------
    tuple of (name, result_dict, output_text)
    """
    import io
    from concurrent.futures import ThreadPoolExecutor
    _suppress_fold_warnings()

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  Nested CV: {name}\n")
    buf.write(f"{'=' * 60}\n")

    n_outer = len(outer_splits)
    fold_metrics_list = [None] * n_outer
    best_params_per_fold = [None] * n_outer
    oof_preds = np.full(len(y), fill_value=-1, dtype=int)
    oof_probs = np.full(len(y), fill_value=np.nan)

    def _run_outer_fold(fold_idx, train_idx, val_idx):
        _suppress_fold_warnings()
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        est = clone(estimator)
        _set_estimator_njobs(est, 1)  # single-threaded fits; GridSearchCV parallelises

        pipeline = build_fold_pipeline(
            est, scale=scale, random_state=random_state, reduction=reduction,
        )

        inner_groups = groups.iloc[train_idx] if groups is not None else None
        inner_cv = get_stratified_cv(
            n_inner_splits, random_state + fold_idx, inner_groups,
        )

        gs = GridSearchCV(
            pipeline, grid, cv=inner_cv, scoring=scoring,
            n_jobs=n_jobs_inner, refit=True,
        )
        with joblib.parallel_config(backend="threading"):
            gs.fit(X_tr, y_tr, groups=inner_groups)

        best_params = gs.best_params_
        best_pipeline = gs.best_estimator_
        y_pred = best_pipeline.predict(X_val)

        y_prob = None
        if hasattr(best_pipeline, "predict_proba"):
            y_prob = best_pipeline.predict_proba(X_val)[:, 1]

        fold_m = compute_classification_metrics(y_val.values, y_pred, y_prob)
        del gs
        return fold_idx, val_idx, y_pred, y_prob, fold_m, best_params

    # Outer-fold parallelism via ThreadPoolExecutor (avoids joblib nesting
    # issues with the inner GridSearchCV).
    n_fold_workers = min(n_outer, max(1, n_jobs_inner))
    with ThreadPoolExecutor(max_workers=n_fold_workers) as executor:
        futures = [
            executor.submit(_run_outer_fold, fold_idx, tr_idx, val_idx)
            for fold_idx, (tr_idx, val_idx) in enumerate(outer_splits)
        ]
        for future in futures:
            fold_idx, val_idx, y_pred, y_prob, fold_m, best_params = future.result()
            fold_metrics_list[fold_idx] = fold_m
            best_params_per_fold[fold_idx] = best_params
            oof_preds[val_idx] = y_pred
            if y_prob is not None:
                oof_probs[val_idx] = y_prob
            buf.write(
                f"  Fold {fold_idx + 1:2d}: "
                f"F1={fold_m['f1']:.4f}  "
                f"AUC={fold_m.get('auc_roc', float('nan')):.4f}  "
                f"Best={best_params}\n"
            )

    aggregate = {}
    for metric in fold_metrics_list[0].keys():
        scores = np.array([fm[metric] for fm in fold_metrics_list])
        aggregate[metric] = compute_confidence_interval(scores)

    result = {
        "fold_metrics": fold_metrics_list,
        "oof_predictions": oof_preds,
        "oof_probabilities": oof_probs,
        "aggregate": aggregate,
        "best_params_per_fold": best_params_per_fold,
    }

    return name, result, buf.getvalue()
