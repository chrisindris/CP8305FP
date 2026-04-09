### Data Preprocessing Checklist for Diabetes 130-US Hospitals Dataset

Based on the provided sources, I have expanded the checklist to cover all missing bases—such as utilizing the mapping lookup table, handling missing demographic values, explicit cross-validation strategies, threshold tuning, and pre-modeling feature selection. 

#### Phase 1: Row Filtering & Data Leakage Prevention
- [ ] **Decode Categorical IDs:** Use the `IDS_mapping.csv` lookup table to translate numeric ID codes (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) into human-readable descriptions before filtering.
- [ ] **Remove Terminal/Hospice Discharges:** Filter out records where the patient cannot mathematically be readmitted, which would create target leakage. Drop all rows where `discharge_disposition_id` is `11`, `13`, `14`, `19`, `20`, or `21` (which correspond to "Expired" or "Hospice" outcomes).
- [ ] **Address Multiple Encounters:** Prevent data leakage by addressing duplicate patient visits. Either sort chronologically (by `encounter_id`) and keep only the first encounter using `patient_nbr`, or explicitly implement Group K-Fold cross-validation later in the pipeline. 
- [ ] **Drop Identifier Columns:** Completely remove the `encounter_id` and `patient_nbr` columns before feeding the data to your algorithm.

#### Phase 2: Target Variable Formulation
- [ ] **Binarize the Target Variable:** Convert the `readmitted` column into a binary classification target to align with hospital readmission penalty frameworks.
  - Assign `1` (Positive/High-Risk Class) to `<30` days.
  - Assign `0` (Negative Class) to `>30` days and `NO`.

#### Phase 3: Feature Dropping & Type Casting
- [ ] **Drop High-Missingness Columns (The "Big Three"):** Remove the following columns due to severe missing data that cannot be safely imputed:
  - `weight` (~97% missing).
  - `medical_specialty` (~47-53% missing).
  - `payer_code` (~40-52% missing).
- [ ] **Handle Missing Demographics:** Impute or drop the ~2% of missing values in the `race` column.
- [ ] **Cast Clinical IDs to Categorical:** Prevent the algorithm from treating nominal identifiers as continuous math. Explicitly cast the following columns to categorical (string/object) types:
  - `admission_type_id`.
  - `discharge_disposition_id`.
  - `admission_source_id`.

#### Phase 4: Advanced Feature Engineering (Publication-Level)
- [ ] **Handle Lab Results as Distinct Categories:** For `max_glu_serum` and `A1Cresult`, do *not* impute or drop the `None` values. Treat `None` as a valid, distinct categorical feature, as the clinical decision *not* to order a test is highly predictive.
- [ ] **Ordinal Encoding for Age:** Convert the 10-year `age` brackets (e.g., `[10-20)`) into ordinal integers (e.g., 1 to 10) or one-hot encode them. Do not treat them as continuous midpoints.
- [ ] **ICD-9 Cardinality Reduction (Strack Mapping):** Map the hundreds of unique ICD-9 codes in `diag_1`, `diag_2`, and `diag_3` into 9 high-level clinical categories to prevent the curse of dimensionality. The categories are:
  - `Circulatory` (390–459, 785).
  - `Respiratory` (460–519, 786).
  - `Digestive` (520–579, 787).
  - `Diabetes` (250.xx).
  - `Injury` (800–999).
  - `Musculoskeletal` (710–739).
  - `Genitourinary` (580–629, 788).
  - `Neoplasms` (140–239).
  - `Other` (Catch-all for remaining codes, including V/E codes).
- [ ] **Synthesize "Comorbidity Count" (Acuity Feature):** Create a new integer feature that counts the number of *unique* clinical systems failing across the mapped `diag_1`, `diag_2`, and `diag_3` columns.
- [ ] **Synthesize "Medication Change Velocity":** Create a new feature (e.g., `total_drug_changes`) that iterates through the 24 specific medication columns and counts the total number of times a dosage is listed as `Up` or `Down`.
- [ ] **Synthesize "Healthcare Utilization Index":** Combine the `number_inpatient`, `number_emergency`, and `number_outpatient` columns into a single `total_prior_visits` feature to identify frequent hospital users.

#### Phase 5: Anti-Leakage Data Splitting & Cross Validation
- [ ] **Apply Stratified Splitting *Before* Transformations:** Due to the severe ~11% minority class imbalance, split the data *before* applying any preprocessing techniques to prevent data leakage. 
- [ ] **Implement Stratified K-Fold:** Use Stratified 10-Fold Cross-Validation (instead of regular K-fold or random splitting) to guarantee that each split preserves the class distribution without leaving any fold entirely lacking the minority class.
- [ ] **Enforce the Preprocessing Rule:** Fit standardizers, scalers, and imputers *only* on the training data, then use `.transform()` on the test/validation sets.

#### Phase 6: Scaling, Encoding, & Imbalance Handling (Applied to Train Set Only)
- [ ] **Scale Numeric Features:** Apply Standard Scaling (or a robust equivalent) strictly to the 8 continuous integer variables (e.g., `time_in_hospital`, `num_lab_procedures`, `num_medications`).
- [ ] **One-Hot Encoding:** One-hot encode the collapsed nominal features, including `race`, `gender`, the 9-group `diag` columns, and the newly cast clinical IDs.
- [ ] **Address Class Imbalance Systematically:** Apply a mitigation strategy exclusively to the training data. Compare the following methods rather than just picking one:
  - **Data-Level Resampling:** Use SMOTE (Synthetic Minority Over-sampling Technique) to synthetically generate plausible `<30` readmission examples.
  - **Algorithm/Model-Level:** Apply cost-sensitive learning strategies (e.g., `scale_pos_weight` in XGBoost, or `class_weight='balanced'` in scikit-learn).
  - **Threshold Tuning:** Lower the probability decision threshold below 0.5 to prioritize high-risk patient recall.

#### Phase 7: Pre-Modeling Feature Selection
- [ ] **Execute Algorithmic Feature Selection:** Instead of keeping all processed features blindly, apply statistical or swarm intelligence techniques (like AIC/BIC, Lasso Regression, Stepwise Elimination, or Grey Wolf Optimizer) to strip out redundant columns and optimize predictive performance.