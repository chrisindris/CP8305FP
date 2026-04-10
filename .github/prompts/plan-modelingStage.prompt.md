## Plan: Proofread and Align Modeling and Evaluation (Diabetes 130-US)

Use this plan to reconcile the current modeling workflow with the full checklist while preserving leakage-safe validation, class-imbalance robustness, and clinically meaningful reporting.

Make sure to comment your code.

Status legend:
- [x] Implemented and acceptable
- [~] Partially implemented, needs correction or hardening
- [ ] Missing and must be implemented

## Phase-by-Phase Checklist Coverage

### Phase 1: Cross-Validation and Anti-Leakage Setup
- [x] Stratified 10-fold CV exists, with group-aware variant support.
- [x] Anti-leakage fold-local pipeline standardized via `build_fold_pipeline` (scaler → optional reduction → optional SMOTE → classifier) across `run_cv_experiment`, `compare_imbalance_strategies`, and `run_tuned_cv_experiment`.
- [x] Imbalance handling unified under the same fold-local pattern (`imbalance_strategy`: none / class_weight / SMOTE) and comparable via `compare_imbalance_strategies` and `run_cv_experiment`.

### Phase 2: Model Initialization
- [x] Baseline and interpretable set: ZeroR, OneR, Naive Bayes, Logistic Regression, Decision Tree, kNN, SGDClassifier (no kernel SVM in registry; SGD used on full or fold-locally reduced features in notebook §4.5).
- [x] Advanced ensemble coverage: Random Forest, Gradient Boosting, XGBoost, LightGBM (with optional dependency guards). CatBoost registers when installed; not pinned in `requirements.txt` (optional).
- [ ] Rule-induction track is missing (PRISM and Apriori/FP-Growth)—defer to optional unsupervised / association-rule notebook or future work.
- [ ] Optional deep learning comparison is missing (LSTM/DNN).

### Phase 3: Model Training and Tuning
- [x] CV training loop unified for all registered models under `run_cv_experiment` (single experiment contract, OOF predictions/probabilities).
- [x] Unbiased tuning: nested CV via `run_tuned_cv_experiment` (inner search, outer evaluation) with `HYPERPARAM_GRIDS`.

### Phase 4: Evaluation Metrics and Statistical Testing
- [x] Full metric suite per fold and holdout: AUC-ROC, AUPRC, precision, recall, F1, weighted F1, top-K (Precision@K / Recall@K), balanced accuracy, Brier score where applicable.
- [x] Cohen's Kappa (`compute_classification_metrics`).
- [x] 95% confidence intervals from fold-level distributions (`compute_confidence_interval`, `build_metrics_summary_table`).
- [x] Statistical significance testing (`run_significance_tests`: paired t-test and Wilcoxon vs. chosen baseline).
- [x] Calibration diagnostics (Brier, `calibration_curve` via `compute_calibration_diagnostics` / plots) and post-hoc calibration API (`calibrate_model` with Platt / isotonic).

### Phase 5: Cost-Sensitive Business Evaluation
- [x] Custom FN-heavy cost-benefit matrix (`DEFAULT_COST_MATRIX`, HRRP-aligned defaults).
- [x] Threshold tuning aligned to business cost: `sweep_thresholds_cost`, `plot_cost_threshold_sweep`, and comparison to default threshold in the modeling notebook.

### Phase 6: Explainability and Knowledge Discovery
- [x] SHAP pipeline (`compute_shap_values`, `plot_shap_summary`; TreeExplainer / KernelExplainer fallback).
- [x] Secondary XAI: permutation importance, PDP (`plot_pdp`), LIME helper (`compute_lime_explanation` in `src/modeling.py`).
- [x] Explicit rule export from tree models (`export_tree_rules`, `export_forest_top_rules`; notebook demonstrates decision tree + random forest rules).

## Implementation Workplan

1. Checklist integration and scope lock
- [x] 1.1. Define mandatory versus optional modeling tracks from the checklist (core supervised path in repo; rule induction & DL remain optional).
- [x] 1.2. Freeze experiment contract: target, grouping identifier usage, split policy, and reporting schema (documented in `03_modeling.ipynb` and `modeling.py`).
- [x] 1.3. Define final artifact schema for fold metrics, tuned parameters, threshold decisions, and XAI outputs (tables, `reports/figures`, saved model).

2. Pipeline and leakage hardening (depends on Step 1)
- [x] 2.1. Standardize one fold-local pipeline pattern where preprocessing, encoding, scaling, and resampling are fit only on training folds.
- [x] 2.2. Enforce strict fit-on-train and transform-on-validation/test behavior everywhere (CV and nested CV paths).
- [x] 2.3. Add unbiased model-selection flow (nested CV or equivalent separation of tuning and final evaluation).

3. Model catalog completion (parallel with late Step 2 work once pipeline contract is fixed)
- [x] 3.1. Add strict baselines and missing interpretable models.
- [x] 3.2. Add advanced gradient boosting libraries with deterministic defaults and optional dependency guards.
- [ ] 3.3. Add optional parallel tracks for rule induction and deep learning so they do not block core supervised delivery (still open).

4. Unified training and tuning harness (depends on Steps 2 and 3)
- [x] 4.1. Build per-model search spaces under one tuning runner (`HYPERPARAM_GRIDS`, `run_tuned_cv_experiment`).
- [x] 4.2. Compare imbalance strategies under identical folds and scorers (`compare_imbalance_strategies`).
- [x] 4.3. Persist out-of-fold predictions and probabilities for downstream statistics, calibration, and threshold analysis (`run_cv_experiment` outputs).

5. Evaluation and statistical evidence (depends on Step 4)
- [x] 5.1. Expand metrics to include AUC-ROC, AUPRC, weighted F1, recall, precision, top-K metrics, and Kappa.
- [x] 5.2. Compute 95% confidence intervals from fold-level distributions.
- [x] 5.3. Run paired significance tests against defined baselines and report assumptions and limitations.
- [x] 5.4. Add calibration diagnostics (Brier score and calibration curves) and post-hoc calibration when needed.

6. Cost-sensitive operating point selection (depends on Step 5)
- [x] 6.1. Define FN/FP business cost matrix aligned to HRRP and intervention costs.
- [x] 6.2. Run threshold sweeps and choose operating points via expected cost under recall constraints.
- [x] 6.3. Report default threshold versus tuned threshold side-by-side for operational decisions.

7. Explainability and rule export (depends on Step 4, can run in parallel with Steps 5 and 6)
- [x] 7.1. Generate SHAP global and local explanations for top models (global summary in notebook; local via SHAP/LIME APIs).
- [x] 7.2. Add LIME/PDP/permutation triangulation.
- [x] 7.3. Export human-readable rules from tree/rule models for administrator-facing artifacts.

8. Notebook and reporting handoff (depends on Steps 5, 6, and 7)
- [x] 8.1. Align notebooks to the finalized pipeline and deterministic reruns (`03_modeling.ipynb`).
- [ ] 8.2. Update checklist evidence in notes/03_modeling_checklist.md (still to sync markdown evidence with code).
- [x] 8.3. Save final figures/tables for analysis write-up in reporting outputs (`save_figure` → `reports/figures`).

## Files in Scope
- notes/03_modeling_checklist.md - requirements source and evidence checklist.
- notebooks/03_modeling.ipynb - supervised modeling orchestration.
- notebooks/03_unsupervised_modelling.ipynb - optional unsupervised/rule-discovery extension.
- src/modeling.py - model registry, CV, tuning, imbalance, thresholding, metrics.
- src/visualization.py - ROC/PR, calibration, explainability, model comparison visuals.
- src/data_preprocessing.py - leakage identifier and grouping contract.
- data/processed/processed_dataset.csv - canonical processed input.
- requirements.txt - modeling/XAI dependency alignment.

## Verification Gates

1. Leakage gate
- [x] No leakage identifiers in model feature matrices (notebook drops `encounter_id`, `patient_nbr` from *X*).
- [x] All transforms/samplers are fit only on training folds (fold-local pipelines in `modeling.py`).

2. Methodology gate
- [x] Stratified 10-fold CV confirmed (`run_cv_experiment`, `get_stratified_cv`).
- [x] Group-aware CV confirmed when repeated patient encounters are retained (`patient_nbr` → `StratifiedGroupKFold` when provided).

3. Metrics gate
- [x] Required metrics table produced at fold level with mean and 95% CI (`build_metrics_summary_table`).

4. Statistical gate
- [x] Baseline-vs-candidate significance testing completed and interpreted (`run_significance_tests` in notebook).

5. Calibration gate
- [x] Brier and calibration curves generated (`plot_calibration_curves`, holdout Brier in notebook).
- [~] Recalibration evidence if applied (`calibrate_model` available; notebook demonstrates diagnostics; optional explicit recalibration run can be added for full gate closure).

6. Business gate
- [x] Cost matrix documented and threshold selection justified by expected cost and recall (`DEFAULT_COST_MATRIX`, cost sweep vs. default threshold).

7. Explainability gate
- [x] SHAP global/local outputs plus secondary XAI artifacts produced (SHAP summary, permutation importance, PDP in notebook; LIME in library).
- [x] At least one deployable human-readable rule artifact exported (decision tree + random forest rule text in notebook).
