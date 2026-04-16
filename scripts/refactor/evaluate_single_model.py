def _evaluate_single_model(
    name, estimator, X, y, splits, scale, imbalance_strategy,
    random_state, reduction, threshold, n_jobs_folds, n_jobs_estimator,
):
    """Evaluate a single model across all CV folds.

    Top-level function required by the loky (process-based) joblib backend
    so that it can be pickled and sent to worker processes.

    Returns
    -------
    tuple of (name, result_dict, output_text)
    """
    import io
    _suppress_fold_warnings()
    est = clone(estimator)
    _set_estimator_njobs(est, n_jobs_estimator)

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  Model: {name}\n")
    buf.write(f"{'=' * 60}\n")

    fold_metrics_list = []
    oof_preds = np.full(len(y), fill_value=-1, dtype=int)
    oof_probs = np.full(len(y), fill_value=np.nan)

    def _evaluate_fold(fold_idx, train_idx, val_idx):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        try:
            use_cw = imbalance_strategy in ("class_weight", "class_weight+smote")
            pipeline = build_fold_pipeline(
                est,
                scale=scale,
                imbalance_strategy=imbalance_strategy,
                random_state=random_state,
                reduction=reduction,
                y_train=y_tr.values if use_cw else None,
            )
            # Ensure the pipeline's classifier inherits the correct n_jobs.
            if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
                _set_estimator_njobs(
                    pipeline.named_steps["classifier"], n_jobs_estimator,
                )

            uses_smote = imbalance_strategy in ("smote", "class_weight+smote")
            needs_sample_weight = (
                use_cw
                and not uses_smote
                and not _apply_class_weighting(clone(est), y_tr.values)
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

    # Run folds in parallel.  Use threading backend so the LightGBM
    # monkey-patch (force_all_finite -> ensure_all_finite) is visible
    # to all workers in the same process.
    with joblib.parallel_config(backend="threading"):
        results_list = joblib.Parallel(n_jobs=n_jobs_folds)(
            joblib.delayed(_evaluate_fold)(fold_idx, tr_idx, val_idx)
            for fold_idx, (tr_idx, val_idx) in enumerate(splits)
        )

    for fold_idx, val_idx, y_pred, y_prob, fold_m, exc_msg in results_list:
        if exc_msg is not None:
            buf.write(f"  Fold {fold_idx + 1:2d}: FAILED - {exc_msg}\n")
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
            buf.write(
                f"  Fold {fold_idx + 1:2d}: "
                f"F1={fold_m['f1']:.4f}  AUC={auc_str}  "
                f"Kappa={fold_m['kappa']:.4f}\n"
            )
        fold_metrics_list.append(fold_m)

    # Aggregate fold metrics with 95 % CIs.
    aggregate = {}
    for metric in fold_metrics_list[0].keys():
        scores = np.array([fm[metric] for fm in fold_metrics_list])
        aggregate[metric] = compute_confidence_interval(scores)

    for metric in ["f1", "auc_roc", "kappa"]:
        if metric in aggregate:
            a = aggregate[metric]
            buf.write(
                f"  {metric:>12s}: {a['mean']:.4f} "
                f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]\n"
            )

    result = {
        "fold_metrics": fold_metrics_list,
        "oof_predictions": oof_preds,
        "oof_probabilities": oof_probs,
        "aggregate": aggregate,
    }

    return name, result, buf.getvalue()
