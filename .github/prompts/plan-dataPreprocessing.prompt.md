## Plan: Proofread and Align Data Preprocessing (Diabetes 130-US)

Use this plan to reconcile current preprocessing with the full checklist. Keep binary target modeling, enforce outlier treatment/removal, and close leakage pathways.

Make sure to comment your code!

Status legend:
- `[x]` Implemented and acceptable
- `[~]` Partially implemented, needs correction or hardening
- `[ ]` Missing and must be implemented

## Phase-by-Phase Checklist Coverage

### Phase 1: Row Filtering and Leakage Prevention
- [x] Decode categorical clinical IDs with `IDS_mapping.csv` and preserve lookup metadata in preprocessing outputs.
- [x] Remove terminal/hospice discharges where readmission is impossible (`11`, `13`, `14`, `19`, `20`, `21`) before modeling.
- [x] Address multiple encounters for the same patient with one of two explicit strategies:
	- Preferred: keep all encounters and use stratified group-aware CV by `patient_nbr`.
	- Fallback: keep only first encounter per patient after deterministic ordering.
- [x] Drop `encounter_id` and `patient_nbr` from modeling features (currently not consistently enforced before export).

### Phase 2: Target Variable Formulation
- [x] Binarize `readmitted` as `<30 -> 1`, `>30/NO -> 0`.

### Phase 3: Feature Dropping and Type Casting
- [x] Drop high-missingness columns: `weight`, `medical_specialty`, `payer_code`.
- [x] Handle `race` missing values explicitly.
- [x] Cast clinical IDs (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) as categorical.

### Phase 4: Advanced Feature Engineering
- [x] Keep `None` as meaningful category in `max_glu_serum` and `A1Cresult`.
- [x] Encode age bracket as explicit ordinal or one-hot policy; verify not treated as continuous midpoint.
- [x] ICD-9 reduction exists, but align to publication-style 9-group Strack mapping instead of current broader grouping.
- [x] Add comorbidity acuity feature: count of unique diagnosis system groups across `diag_1/2/3` mapped categories.
- [x] Medication dynamics exist, but add `total_drug_changes` that counts `Up`/`Down` across all medication columns.
- [x] Utilization composite exists; align naming and definition to `total_prior_visits` convention.
- [x] Fix abnormal lab engineered feature bug (remove `value_counts` misuse; compute row-wise binary flags then `both_abnormal`).

### Phase 5: Anti-Leakage Split and Cross-Validation
- [~] Perform train/test split before fitting any imputers, encoders, or scalers.
- [x] Use stratified 10-fold CV for model development.
- [x] If retaining repeated patients, use stratified group-aware folds keyed by `patient_nbr`.
- [~] Enforce fit-on-train, transform-on-validation/test for every preprocessing operator.

### Phase 6: Scaling, Encoding, and Imbalance Handling (Train Only)
- [x] Scale the 8 primary continuous integer variables on train folds only.
- [~] One-hot encode collapsed nominal features and cast IDs after split within train-only fitting flow.
- [x] Compare imbalance mitigation strategies on training data only:
	- SMOTE resampling.
	- Class weighting / cost-sensitive training.
	- Probability threshold tuning below 0.5 for recall-focused operating points.

### Phase 7: Pre-Modeling Feature Selection
- [x] Add algorithmic feature selection step (e.g., L1/Lasso, stepwise, AIC/BIC proxy flow, or equivalent robust method).

## Implementation Workplan

1. Checklist integration and scope lock
1.1. Rebuild `notes/02_data_preprocessing_checklist.md` with this phase layout and pass/fail placeholders.
1.2. Record locked decisions: binary target, enforced outlier treatment, leakage-safe split policy, train-only transforms.
1.3. Add evidence fields for row counts, class prevalence, and dropped/engineered columns.

2. Notebook 2 preprocessing corrections
2.1. Apply terminal/hospice filtering before target engineering.
2.2. Correct abnormal lab flags and verify row-level outputs.
2.3. Add/standardize advanced engineered features: 9-group ICD mapping, comorbidity count, total drug changes, total prior visits.
2.4. Enforce outlier removal on selected numeric non-ID columns and log delta rows.
2.5. Ensure identifier columns are removed before final modeling dataset save.

3. Utility and pipeline alignment
3.1. Align `src/data_preprocessing.py` paths/defaults with repository data layout and real filenames.
3.2. Keep utility functions reusable but align behavior with notebook decisions (missing policy, outlier policy, leakage-safe exports).
3.3. Update modeling flow so preprocessing fits occur on training folds only.
3.4. Implement stratified 10-fold CV strategy, with group-aware variant when repeated patient encounters are retained.

4. Imbalance, thresholding, and feature selection
4.1. Add experimental branch for SMOTE vs class-weighted learners.
4.2. Add threshold sweep on validation folds to choose operating point by recall/F1/balanced metrics.
4.3. Add pre-modeling feature selection and compare against full-feature baseline.

5. Validation and handoff
5.1. Re-run preprocessing end-to-end and verify missing totals, row deltas, and final shape.
5.2. Confirm processed dataset excludes leakage identifiers and includes corrected engineered features.
5.3. Smoke-test modeling notebook on the refreshed processed dataset.
5.4. Mark each checklist item as pass/fail with evidence.

## Files in Scope
- `notes/02_data_preprocessing_checklist.md`
- `notebooks/02_data_preprocessing.ipynb`
- `src/data_preprocessing.py`
- `src/feature_engineering.py`
- `src/modeling.py`
- `notebooks/03_modeling.ipynb`

## Verification Gates

1. Leakage gate
- No terminal/hospice discharge rows in modeling dataset.
- No `encounter_id` or `patient_nbr` in training feature matrix.

2. Feature correctness gate
- Abnormal lab flags are row-wise binary and null-safe.
- ICD mappings follow declared 9-group taxonomy.
- `comorbidity_count`, `total_drug_changes`, and `total_prior_visits` present and validated.

3. Methodology gate
- All transformers fitted on train folds only.
- Stratified 10-fold (and group-aware where required) confirmed by code path.

4. Performance-evaluation gate
- Compare imbalance strategies under identical CV protocol.
- Report threshold-tuned and default-threshold metrics side by side.

5. Reproducibility gate
- Fixed random seeds and deterministic split/CV settings documented.
- Checklist artifact updated with final evidence and decisions.