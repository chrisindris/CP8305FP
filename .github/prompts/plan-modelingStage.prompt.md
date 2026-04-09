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
- [~] Anti-leakage fold-local pipeline is only partially standardized across all workflows.
- [~] Imbalance handling exists but is not fully unified in one leakage-safe comparison harness across all model families.

### Phase 2: Model Initialization
- [~] Baseline and interpretable set is partial (core classifiers present; missing strict baseline set and some interpretable variants).
- [~] Advanced ensemble coverage is partial (Random Forest present; missing full XGBoost/LightGBM/CatBoost integration).
- [ ] Rule-induction track is missing (PRISM and Apriori/FP-Growth).
- [ ] Optional deep learning comparison is missing.

### Phase 3: Model Training and Tuning
- [~] CV training loop exists but is not yet fully unified for all candidate models under one experiment contract.
- [~] Hyperparameter tuning exists, but unbiased evaluation flow must be hardened (nested CV or equivalent two-level protocol).

### Phase 4: Evaluation Metrics and Statistical Testing
- [~] Some metrics exist (precision/recall/F1), but full required suite is incomplete (AUC-ROC, AUPRC, top-K metrics, etc.).
- [ ] Cohen Kappa is missing.
- [ ] 95% confidence intervals are missing.
- [ ] Statistical significance testing is missing.
- [ ] Calibration diagnostics and correction flow are missing.

### Phase 5: Cost-Sensitive Business Evaluation
- [ ] Custom FN-heavy cost-benefit matrix is missing.
- [x] Threshold tuning exists, but it needs business-cost alignment and governance.

### Phase 6: Explainability and Knowledge Discovery
- [ ] SHAP pipeline is missing.
- [~] Secondary XAI is partial (permutation importance exists; LIME and PDP are missing).
- [ ] Explicit rule export is missing.

## Implementation Workplan

1. Checklist integration and scope lock
1.1. Define mandatory versus optional modeling tracks from the checklist.
1.2. Freeze experiment contract: target, grouping identifier usage, split policy, and reporting schema.
1.3. Define final artifact schema for fold metrics, tuned parameters, threshold decisions, and XAI outputs.

2. Pipeline and leakage hardening (depends on Step 1)
2.1. Standardize one fold-local pipeline pattern where preprocessing, encoding, scaling, and resampling are fit only on training folds.
2.2. Enforce strict fit-on-train and transform-on-validation/test behavior everywhere.
2.3. Add unbiased model-selection flow (nested CV or equivalent separation of tuning and final evaluation).

3. Model catalog completion (parallel with late Step 2 work once pipeline contract is fixed)
3.1. Add strict baselines and missing interpretable models.
3.2. Add advanced gradient boosting libraries with deterministic defaults and optional dependency guards.
3.3. Add optional parallel tracks for rule induction and deep learning so they do not block core supervised delivery.

4. Unified training and tuning harness (depends on Steps 2 and 3)
4.1. Build per-model search spaces under one tuning runner.
4.2. Compare imbalance strategies under identical folds and scorers.
4.3. Persist out-of-fold predictions and probabilities for downstream statistics, calibration, and threshold analysis.

5. Evaluation and statistical evidence (depends on Step 4)
5.1. Expand metrics to include AUC-ROC, AUPRC, weighted F1, recall, precision, top-K metrics, and Kappa.
5.2. Compute 95% confidence intervals from fold-level distributions.
5.3. Run paired significance tests against defined baselines and report assumptions and limitations.
5.4. Add calibration diagnostics (Brier score and calibration curves) and post-hoc calibration when needed.

6. Cost-sensitive operating point selection (depends on Step 5)
6.1. Define FN/FP business cost matrix aligned to HRRP and intervention costs.
6.2. Run threshold sweeps and choose operating points via expected cost under recall constraints.
6.3. Report default threshold versus tuned threshold side-by-side for operational decisions.

7. Explainability and rule export (depends on Step 4, can run in parallel with Steps 5 and 6)
7.1. Generate SHAP global and local explanations for top models.
7.2. Add LIME/PDP/permutation triangulation.
7.3. Export human-readable rules from tree/rule models for administrator-facing artifacts.

8. Notebook and reporting handoff (depends on Steps 5, 6, and 7)
8.1. Align notebooks to the finalized pipeline and deterministic reruns.
8.2. Update checklist evidence in notes/03_modeling_checklist.md.
8.3. Save final figures/tables for analysis write-up in reporting outputs.

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
- No leakage identifiers in model feature matrices.
- All transforms/samplers are fit only on training folds.

2. Methodology gate
- Stratified 10-fold CV confirmed.
- Group-aware CV confirmed when repeated patient encounters are retained.

3. Metrics gate
- Required metrics table produced at fold level with mean and 95% CI.

4. Statistical gate
- Baseline-vs-candidate significance testing completed and interpreted.

5. Calibration gate
- Brier and calibration curves generated, with recalibration evidence if applied.

6. Business gate
- Cost matrix documented and threshold selection justified by expected cost and recall.

7. Explainability gate
- SHAP global/local outputs plus secondary XAI artifacts produced.
- At least one deployable human-readable rule artifact exported.
