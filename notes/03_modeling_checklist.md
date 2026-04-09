### Modeling & Evaluation Checklist for Diabetes 130-US Hospitals Dataset

This markdown document serves as a strict technical checklist for a code-generation model to implement the training, validation, and evaluation pipeline. It incorporates Stratified 10-Fold Cross-Validation, specific performance metrics, statistical evaluations (including Confidence Intervals and the Kappa statistic), and cost-sensitive business alignment.

#### Phase 1: Cross-Validation & Anti-Leakage Setup
- [ ] **Implement Stratified 10-Fold Cross-Validation:** Initialize a `StratifiedKFold` object (with `n_splits=10`, `shuffle=True`) to ensure the ~11% minority class distribution (readmitted < 30 days) is preserved across all folds.
- [ ] **Enforce the Anti-Leakage Pipeline:** Build a pipeline object (e.g., using `imblearn.pipeline.Pipeline`) for the CV loop to guarantee that preprocessing (scaling, imputation, encoding) and imbalance mitigation (e.g., SMOTE) are applied **strictly to the training folds** and then used to `.transform()` the validation fold. 
- [ ] **Integrate Imbalance Handling:** Within the pipeline, include the chosen class imbalance strategies for comparison (e.g., SMOTE for synthetic oversampling, or algorithm-level cost-sensitive weights like `class_weight='balanced'` and `scale_pos_weight`).

#### Phase 2: Model Initialization
- [ ] **Initialize Baseline & Interpretable Models:** Instantiate 0R, 1R, Naïve Bayes, Logistic Regression, Decision Trees (e.g., C4.5/CART), k-Nearest Neighbors (kNN), and Support Vector Machines (SVM with Linear and RBF kernels).
- [ ] **Initialize Advanced Ensemble Models:** Instantiate Random Forest, XGBoost, LightGBM, and CatBoost. These gradient boosting and bagging algorithms are the state-of-the-art benchmarks for tabular data.
- [ ] **Initialize Rule-Induction Models:** Implement PRISM (Covering Rules) and Association Rule Mining algorithms (e.g., Apriori or FP-Growth) designed to extract actionable, human-readable "If-Then" rules.
- [ ] **Initialize Deep Learning (Optional/Comparison):** Instantiate an LSTM or DNN model to capture sequential/temporal patient data (if applicable to the sequence format) to compare against the tree-based models.

#### Phase 3: Model Training & Tuning
- [ ] **Execute the CV Loop:** Train each initialized model across the 10 stratified folds.
- [ ] **Perform Hyperparameter Tuning:** Integrate Grid Search or Randomized Search within the cross-validation loop to optimize model parameters (e.g., learning rate, tree depth, number of estimators, penalty parameters).

#### Phase 4: Evaluation Metrics & Statistical Testing
- [ ] **Calculate Core Predictive Metrics:** For each fold, and averaged across all folds, calculate:
  - AUC-ROC (Area Under the Receiver Operating Characteristic curve).
  - AUPRC / Average Precision (Area Under the Precision-Recall Curve), which is highly informative for imbalanced datasets.
  - Recall / Sensitivity (critical for clinical safety to minimize false negatives).
  - Weighted F1-Score (to balance precision and recall).
  - Precision@K and Recall@K (e.g., Top-5% highest risk) to map to operational hospital staffing capacity.
- [ ] **Compute the Kappa Statistic:** Calculate Cohen's Kappa score to measure the inter-rater reliability/agreement between the model's predictions and the true labels, adjusting for the probability of chance agreement.
- [ ] **Generate Confidence Intervals:** Calculate 95% Confidence Intervals for all primary metrics (F1, Accuracy, Precision, Recall, AUC-ROC) using the variance/distribution of scores obtained across the 10 cross-validation folds.
- [ ] **Conduct Statistical Significance Tests:** Implement Paired T-Tests or McNemar's Test to statistically prove whether performance improvements over baseline models (e.g., XGBoost vs. Logistic Regression) are significant ($p < 0.05$).
- [ ] **Evaluate Model Calibration:** Calculate the Brier Score and generate Calibration Curves to ensure that a predicted "20% risk" accurately corresponds to a true 20% clinical likelihood. Apply Platt Scaling if necessary.

#### Phase 5: Cost-Sensitive Business Evaluation
- [ ] **Implement a Custom Cost-Benefit Matrix:** Evaluate models based on financial viability by assigning specific dollar costs. Heavily penalize False Negatives (missed readmissions resulting in HRRP penalties) compared to False Positives (false alarms resulting in minor preventative care costs).
- [ ] **Execute Threshold Tuning:** Evaluate metric performance when the classification decision threshold is lowered below the default 0.5 (e.g., to 0.3 or 0.4) to purposely increase Recall and catch more high-risk patients.

#### Phase 6: Explainability (XAI) & Knowledge Discovery
- [ ] **Extract SHAP Values:** Run SHAP (SHapley Additive exPlanations) on the best-performing ensemble models (like XGBoost and Random Forest) to generate global feature importance plots and local, patient-specific explanations.
- [ ] **Implement Secondary XAI Tools:** Incorporate LIME (Local Interpretable Model-agnostic Explanations), Permutation Importance, and Partial Dependence Plots (PDP) to further map out the influence of features like `num_medications` and `discharge_disposition_id`.
- [ ] **Export Explicit Rules:** Extract the human-readable decision rules generated by the Random Forest, Decision Tree, PRISM, or Apriori algorithms so they can be provided to hospital administrators as clinical checklists.