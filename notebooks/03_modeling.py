#!/usr/bin/env python
# coding: utf-8

# # CP8305 Final Project
# ## Notebook 3 – Modeling (Diabetes 130-US Hospitals)
# 
# ### Objectives
# - Load the processed Diabetes 130-US dataset.
# - Train and compare a comprehensive set of classifiers via leakage-safe k-fold stratified CV (fold count: `K_FOLDS` in Setup).
# - Evaluate with AUC-ROC, AUPRC, F1, Cohen's Kappa, 95% CIs, and statistical significance tests.
# - Conduct cost-sensitive threshold tuning aligned to HRRP readmission penalties.
# - Produce SHAP, permutation importance, and PDP explainability artifacts.
# - Export human-readable rules from tree-based models.

# #### The methods we are using
# * group-k-fold cross validation with confidence interval
# -- we need group-k-fold to avoid data leakage since we have multiple encounters for the same patient.
# * Kappa Statistic

# #### Models
# 
# **Baselines:** ZeroR (majority class), OneR (single-split decision stump)
# 
# **Interpretable:** Gaussian Naive Bayes, Logistic Regression, SGDClassifier (log loss), Decision Tree (CART), kNN
# 
# **Ensemble:** Random Forest, Gradient Boosting, XGBoost, LightGBM
# 
# **Explainability:** SHAP, LIME, Permutation Importance, Partial Dependence Plots, Decision Rule Export
# 

# #### Metrics
# * Accuracy & Balanced Accuracy
# * Precision, Recall, F1, Weighted F1
# * AUC-ROC, AUPRC (critical for imbalanced data)
# * Cohen's Kappa (chance-corrected agreement)
# * Precision@K and Recall@K (top-5%, top-10% operational capacity)
# * Brier Score (calibration quality)
# * 95% Confidence Intervals from fold distributions
# * Paired t-test and Wilcoxon signed-rank significance tests

# #### Evaluation Approach
# All experiments use **stratified group k-fold cross-validation** (`K_FOLDS` folds, set in Section 1) with group-aware splitting (via `patient_nbr`) to prevent data leakage from repeated encounters. Preprocessing (scaling, SMOTE) is applied **strictly within each training fold**. Hyperparameter tuning uses **nested CV** (inner 5-fold) to produce unbiased performance estimates.

# ## 1. Setup

# In[1]:


get_ipython().run_line_magic('pip', 'install -r /scratch/indrisch/CP8305FP/requirements.txt')


# In[2]:


# Cross-validation: number of folds for all outer / standard stratified group CV in this notebook.
# Nested hyperparameter tuning keeps an inner 5-fold loop (run_tuned_cv_experiment(..., n_inner_splits=5)).
K_FOLDS = 10


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..').resolve()))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.modeling import (
    split_data,
    get_stratified_cv,
    CLASSIFICATION_MODELS,
    CLASSIFICATION_MODELS_FAST,
    CLASSIFICATION_MODELS_ADVANCED,
    CLASSIFICATION_MODELS_SGD,
    HYPERPARAM_GRIDS,
    SCALE_SENSITIVE_MODELS,
    build_fold_pipeline,
    compute_classification_metrics,
    compute_topk_metrics,
    compute_confidence_interval,
    run_cv_experiment,
    build_metrics_summary_table,
    run_significance_tests,
    compute_calibration_diagnostics,
    calibrate_model,
    compute_expected_cost,
    sweep_thresholds_cost,
    DEFAULT_COST_MATRIX,
    compute_shap_values,
    export_tree_rules,
    export_forest_top_rules,
    run_tuned_cv_experiment,
    compare_imbalance_strategies,
    predict_positive_probability,
    tune_decision_threshold,
    select_features_l1,
    train_classifier,
    evaluate_classifier,
    save_model,
    SHAP_AVAILABLE,
    LIME_AVAILABLE,
    REGRESSION_MODELS,
    train_regressor,
    evaluate_regressor,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curves,
    plot_model_comparison,
    plot_confidence_intervals,
    plot_cost_threshold_sweep,
    plot_shap_summary,
    plot_pdp,
    save_figure,
)

np.random.seed(42)
sns.set_theme(style='whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Load Processed Data

# In[4]:


DATA_FILE = Path('..') / 'data' / 'diabetes130' / 'processed' / 'processed_dataset.csv'
df = pd.read_csv(DATA_FILE)
print(f'Shape: {df.shape}')
df.head()


# ## 3. Feature / Target Split

# In[5]:


# Target column used in preprocessing notebook.
TARGET = 'readmitted_binary'

# Keep patient_nbr only for optional group-aware CV, never as a model feature.
group_ids = df['patient_nbr'].copy() if 'patient_nbr' in df.columns else None

drop_columns = [TARGET]
for leakage_col in ['encounter_id', 'patient_nbr']:
    if leakage_col in df.columns:
        drop_columns.append(leakage_col)

X = df.drop(columns=drop_columns)
y = df[TARGET]

print(f'Features: {X.shape[1]}  |  Target: {y.name}  |  Classes/Values: {y.nunique()}')
if group_ids is not None:
    print('Group-aware CV is enabled via patient_nbr (excluded from X).')
else:
    print('Group-aware CV unavailable (patient_nbr not present in processed data).')


# In[6]:


# Split once up-front with stratification so class imbalance is preserved.
X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=True,
 )

# If group IDs are available, keep the train-split groups for CV-only use.
groups_train = group_ids.loc[X_train.index] if group_ids is not None else None

print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')
print(f'Positive class ratio (train): {y_train.mean():.4f}')
print(f'Positive class ratio (test) : {y_test.mean():.4f}')


# In[7]:


# Run L1-based feature selection on training data only to avoid leakage.
selected_columns, l1_selector_model = select_features_l1(X_train, y_train, C=0.1)

X_train_fs = X_train[selected_columns].copy()
X_test_fs = X_test[selected_columns].copy()

print(f'Selected features for optional reduced model: {len(selected_columns)}')


# ---
# ## 4. Classification – Unified CV Experiment
# All candidate models are evaluated under identical fold-local pipelines (scaler → classifier) to ensure fair, leakage-free comparison across `K_FOLDS` stratified folds.

# In[8]:


# Run k-fold stratified CV with fold-local pipelines for all registered models.
# Preprocessing (scaling) is fit strictly on training folds to prevent leakage.
# Class weighting + fold-local SMOTE address the ~9% minority class imbalance;
# a lower threshold (0.1) favours recall of the minority class.
experiment_results = run_cv_experiment(
    X_train, y_train,
    model_configs=CLASSIFICATION_MODELS,
    n_splits=K_FOLDS,
    groups=groups_train,
    scale=True,
    imbalance_strategy='class_weight+smote',
    threshold=0.1,
    random_state=42,
)


# In[9]:


# Build the metrics summary table with 95% CIs from fold distributions.
summary_table = build_metrics_summary_table(experiment_results)
print('=== Model Comparison (sorted by F1, with 95% CIs) ===')
display(summary_table)

# Identify best model.
BEST_CLF = summary_table.index[0]
print(f'\nBest model by CV F1: {BEST_CLF}')

# Forest plot of F1 confidence intervals.
fig = plot_confidence_intervals(summary_table, metric='f1')
save_figure(fig, '03_f1_confidence_intervals.png')
plt.show()

# Grouped bar chart across key metrics.
fig = plot_model_comparison(summary_table, metrics=['f1', 'pr_auc', 'recall', 'kappa'])
save_figure(fig, '03_model_comparison.png')
plt.show()


# ---
# ## 4.5 SGDClassifier on reduced features (fold-local)
# 
# Training **SGDClassifier** on the full ~2.5k one-hot columns is possible but slow; here we compare three **fold-local** reductions first (fit on the training fold only, transform the validation fold—same leakage posture as scaling), then fit SGD on the reduced matrix.
# 
# **RFECV** at full dimension would require enormous numbers of model refits per fold. **SelectFromModel** with an ensemble is the practical substitute at this scale; if you need RFECV for a write-up, restrict it to a low-dimensional input (for example after PCA to 50 components) and use a large `step` to limit refits.
# 
# The next cell uses **lightweight defaults** for faster runs: **`K_FOLDS`-fold** CV (same as Section 4), **PCA** with 32 components, **SelectKBest** with `k=20`, and **SelectFromModel** capped with `max_features=20`, plus a smaller ensemble for the selector. Increase PCA / k / `max_features` if you need richer features.

# In[10]:


from sklearn.base import clone

# Tree/boosting model for SelectFromModel: best mean F1 among advanced models in this CV run.
_advanced_in_run = [
    k for k in CLASSIFICATION_MODELS
    if k in experiment_results and k in CLASSIFICATION_MODELS
]
if _advanced_in_run:
    ref_tree_name = max(
        _advanced_in_run,
        key=lambda k: experiment_results[k]["aggregate"]["f1"]["mean"],
    )
else:
    ref_tree_name = next(
        k for k in ("lightgbm", "xgboost", "random_forest") if k in CLASSIFICATION_MODELS
    )

selector_base = clone(CLASSIFICATION_MODELS[ref_tree_name])
if hasattr(selector_base, "set_params"):
    if ref_tree_name == "lightgbm":
        selector_base.set_params(n_estimators=20, num_leaves=31, max_depth=5, verbose=-1)
    elif ref_tree_name == "xgboost":
        selector_base.set_params(n_estimators=20, max_depth=4, verbosity=0)
    elif ref_tree_name == "catboost":
        selector_base.set_params(iterations=20, depth=5, verbose=0)
    elif ref_tree_name in ("random_forest", "gradient_boosting"):
        selector_base.set_params(n_estimators=25)

print(
    f"SelectFromModel base estimator: {ref_tree_name} "
    "(lighter settings than full CV models for speed)"
)

SGD_REDUCED_MODELS = {"sgd_classifier": CLASSIFICATION_MODELS["sgd_classifier"]}


def _prefix_experiment_results(results: dict, prefix: str) -> dict:
    return {f"{prefix}__{name}": out for name, out in results.items()}


sgd_reduced_results = {}
sgd_reduced_results.update(
    _prefix_experiment_results(
        run_cv_experiment(
            X_train,
            y_train,
            model_configs=SGD_REDUCED_MODELS,
            n_splits=K_FOLDS,
            groups=groups_train,
            scale=True,
            imbalance_strategy="class_weight",
            random_state=42,
            reduction=("pca", {"n_components": 32}),
        ),
        "pca32",
    )
)
sgd_reduced_results.update(
    _prefix_experiment_results(
        run_cv_experiment(
            X_train,
            y_train,
            model_configs=SGD_REDUCED_MODELS,
            n_splits=K_FOLDS,
            groups=groups_train,
            scale=True,
            imbalance_strategy="class_weight",
            random_state=42,
            reduction=("select_kbest", {"k": 20}),
        ),
        "kbest20",
    )
)
sgd_reduced_results.update(
    _prefix_experiment_results(
        run_cv_experiment(
            X_train,
            y_train,
            model_configs=SGD_REDUCED_MODELS,
            n_splits=K_FOLDS,
            groups=groups_train,
            scale=True,
            imbalance_strategy="class_weight",
            random_state=42,
            reduction=(
                "select_from_model",
                {
                    "estimator": selector_base,
                    "threshold": "median",
                    "max_features": 20,
                },
            ),
        ),
        "sfm_median",
    )
)

sgd_reduced_summary = build_metrics_summary_table(sgd_reduced_results)
print(f"=== SGDClassifier with fold-local dimensionality reduction ({K_FOLDS}-fold CV, light feature settings) ===")
display(sgd_reduced_summary)


# In[11]:


# Statistical significance: paired t-test and Wilcoxon signed-rank vs. baseline.
significance_df = run_significance_tests(
    experiment_results, baseline_model='zero_r', metric='f1', alpha=0.05,
)
print('=== Statistical Significance Tests (F1 vs. ZeroR baseline) ===')
display(significance_df)

# Also test against logistic regression as a stronger baseline.
significance_lr = run_significance_tests(
    experiment_results, baseline_model='one_r', metric='f1', alpha=0.05,
)
print('\n=== Statistical Significance Tests (F1 vs. Logistic Regression) ===')
display(significance_lr)


# In[12]:


# Compare class-imbalance strategies on train data only (fold-local pipelines).
print(f'Imbalance comparison for best model: {BEST_CLF}')
imbalance_results = compare_imbalance_strategies(
    X_train, y_train,
    model_name=BEST_CLF,
    cv=5,
    groups=groups_train,
    scoring='f1',
)
imbalance_df = pd.DataFrame(imbalance_results).T.sort_values('mean', ascending=False)
print('Imbalance strategy comparison (CV F1):')
display(imbalance_df)


# In[13]:


# hi


# In[ ]:


# Nested CV: inner 5-fold loop tunes hyperparameters, outer K_FOLDS loop evaluates.
# Only models with defined search spaces in HYPERPARAM_GRIDS are tuned.
top_models = {
    k: CLASSIFICATION_MODELS[k]
    for k in list(summary_table.index[:5])
    if k in HYPERPARAM_GRIDS
}
print(f'Tuning top models: {list(top_models.keys())}')

if top_models:
    # Inner GridSearchCV used n_jobs=-1 before: all CPU cores × joblib workers each
    # hold a copy of the outer-fold training data → heavy RAM and fan noise.
    tuned_results = run_tuned_cv_experiment(
        X_train, y_train,
        model_configs=top_models,
        param_grids=HYPERPARAM_GRIDS,
        n_outer_splits=K_FOLDS,
        n_inner_splits=5,
        groups=groups_train,
        scale=True,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
    )
    tuned_summary = build_metrics_summary_table(tuned_results)
    print('\n=== Tuned Model Comparison (Nested CV, unbiased) ===')
    display(tuned_summary)
else:
    print('No models have defined hyperparameter grids for tuning.')
    tuned_results = {}


# In[ ]:


# Train the final best model on the full training set for holdout evaluation.
best_clf_pipeline = train_classifier(X_train, y_train, model_name=BEST_CLF, scale=True)
y_pred_test = best_clf_pipeline.predict(X_test)
y_prob_test = predict_positive_probability(best_clf_pipeline, X_test)

# Holdout set metrics (comprehensive suite).
test_metrics = compute_classification_metrics(y_test.values, y_pred_test, y_prob_test)
print(f'=== Holdout Test Metrics ({BEST_CLF}) ===')
for k, v in test_metrics.items():
    print(f'  {k:>20s}: {v:.4f}')

# Top-K metrics (operational capacity simulation).
topk = compute_topk_metrics(y_test.values, y_prob_test)
print('\n=== Top-K Metrics (capacity simulation) ===')
for label, vals in topk.items():
    print(f'  {label}: Precision@K={vals["precision_at_k"]:.4f}, Recall@K={vals["recall_at_k"]:.4f} (k={vals["k"]})')


# In[ ]:


# ROC and Precision-Recall curves from out-of-fold probabilities.
fig = plot_roc_curves(experiment_results, y_train)
save_figure(fig, '03_roc_curves.png')
plt.show()

fig = plot_pr_curves(experiment_results, y_train)
save_figure(fig, '03_pr_curves.png')
plt.show()

# Confusion matrices: default threshold vs. recall-tuned threshold.
fig = plot_confusion_matrix(y_test, y_pred_test, title=f'{BEST_CLF} (threshold=0.50)')
save_figure(fig, '03_confusion_matrix.png')
plt.show()

threshold_results = tune_decision_threshold(y_test, y_prob_test, metric='recall')
best_threshold = threshold_results['best_threshold']
y_pred_tuned = (y_prob_test >= best_threshold).astype(int)

fig = plot_confusion_matrix(
    y_test, y_pred_tuned,
    title=f'{BEST_CLF} (threshold={best_threshold:.2f})',
)
save_figure(fig, '03_confusion_matrix_threshold_tuned.png')
plt.show()


# ---
# ## 5. Cost-Sensitive Business Evaluation
# Evaluate models against an FN-heavy cost matrix aligned to Hospital Readmissions Reduction Program (HRRP) penalties. False negatives (missed readmissions) are far more costly than false positives (unnecessary interventions).

# In[ ]:


# Cost matrix aligned to HRRP readmission penalties.
print('Cost matrix (per prediction outcome):')
for k, v in DEFAULT_COST_MATRIX.items():
    print(f'  {k}: ${v:,.0f}')

# Threshold sweep: expected cost and recall at each threshold.
cost_sweep = sweep_thresholds_cost(y_test.values, y_prob_test)

fig = plot_cost_threshold_sweep(cost_sweep)
save_figure(fig, '03_cost_threshold_sweep.png')
plt.show()

# Compare default vs. cost-optimal operating point.
best_cost_row = cost_sweep.sort_values('total_cost').iloc[0]
default_cost = compute_expected_cost(y_test.values, y_pred_test)

print(f'Cost-optimal threshold: {best_cost_row["threshold"]:.2f}')
print(f'  Total cost:  ${best_cost_row["total_cost"]:,.0f}')
print(f'  Recall:      {best_cost_row["recall"]:.4f}')
print(f'\nDefault threshold (0.50):')
print(f'  Total cost:  ${default_cost["total_cost"]:,.0f}')
print(f'  FN={default_cost["fn"]}, FP={default_cost["fp"]}')


# In[ ]:


# Calibration diagnostics: do predicted probabilities match observed frequencies?
fig = plot_calibration_curves(experiment_results, y_train, n_bins=10)
save_figure(fig, '03_calibration_curves.png')
plt.show()

# Brier score for the best model on the holdout set.
cal_diag = compute_calibration_diagnostics(y_test.values, y_prob_test)
print(f'Brier score ({BEST_CLF}): {cal_diag["brier_score"]:.4f}')
print('(Lower is better; 0 = perfect calibration)')


# ---
# ## 6. Explainability (XAI) & Knowledge Discovery
# SHAP global/local explanations, permutation importance, partial dependence plots, and human-readable rule export for clinical deployment.

# In[ ]:


# SHAP global feature importance for the best model.
if SHAP_AVAILABLE:
    shap_values, shap_explainer, X_shap_sample = compute_shap_values(
        best_clf_pipeline, X_test, feature_names=list(X_train.columns),
    )
    fig = plot_shap_summary(shap_values, X_shap_sample, max_display=20)
    save_figure(fig, '03_shap_summary.png')
    plt.show()
else:
    print('SHAP not installed. Install via: pip install shap')


# In[ ]:


# Permutation importance on holdout set (model-agnostic).
fig = plot_permutation_importance(best_clf_pipeline, X_test, y_test, top_n=20)
save_figure(fig, '03_permutation_importance.png')
plt.show()

# Partial Dependence Plots for top features (tree-based models only).
try:
    inner_model = best_clf_pipeline.named_steps.get('classifier')
    if hasattr(inner_model, 'feature_importances_'):
        top_idx = np.argsort(inner_model.feature_importances_)[::-1][:6]
        top_feature_names = [X_train.columns[i] for i in top_idx]
        fig = plot_pdp(best_clf_pipeline, X_test, top_feature_names)
        save_figure(fig, '03_partial_dependence.png')
        plt.show()
    else:
        print(f'PDP: {BEST_CLF} does not expose feature_importances_; skipping PDP.')
except Exception as e:
    print(f'PDP not available for this model type: {e}')


# In[ ]:


# Extract human-readable decision rules for clinical deployment.
# Decision tree rules (full model).
dt_pipeline = train_classifier(X_train, y_train, model_name='decision_tree', scale=False)
dt_model = dt_pipeline.named_steps['classifier']
dt_rules = export_tree_rules(dt_model, feature_names=list(X_train.columns), max_depth=5)
print('=== Decision Tree Rules (max depth 5) ===')
print(dt_rules)

# Random forest top-tree rules (representative subset of the ensemble).
rf_pipeline = train_classifier(X_train, y_train, model_name='random_forest', scale=False)
rf_model = rf_pipeline.named_steps['classifier']
rf_rules = export_forest_top_rules(
    rf_model, feature_names=list(X_train.columns), n_trees=3, max_depth=3,
)
print('\n=== Random Forest – Top 3 Tree Rules (max depth 3) ===')
for rule in rf_rules:
    print(rule)


# ---
# ## Optional: Regression Models
# _These sections are preserved for reference but are not part of the readmission classification analysis._

# In[ ]:


reg_results = {}
for name in REGRESSION_MODELS:
    print(f'\n--- {name} ---')
    pipeline = train_regressor(X_train, y_train, model_name=name, scale=True)
    metrics = evaluate_regressor(pipeline, X_test, y_test)
    reg_results[name] = metrics

print('\n=== RMSE Summary ===')
for name, m in sorted(reg_results.items(), key=lambda x: x[1]['rmse']):
    print(f'{name:<25} RMSE={m["rmse"]:.4f}  R²={m["r2"]:.4f}')


# In[ ]:


BEST_REG = min(reg_results, key=lambda k: reg_results[k]['rmse'])
best_reg_pipeline = train_regressor(X_train, y_train, model_name=BEST_REG, scale=True)
y_pred_reg = best_reg_pipeline.predict(X_test)

fig = plot_residuals(y_test, y_pred_reg, title=f'Residual Plot – {BEST_REG}')
save_figure(fig, '03_residuals.png')
plt.show()


# ---
# ## Optional: Clustering
# _These sections are preserved for reference but are not part of the readmission classification analysis._

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Scale features before clustering
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Elbow method
elbow_data = find_optimal_k(X_scaled, k_range=range(2, 11))
fig = plot_elbow_curve(elbow_data)
save_figure(fig, '03_elbow_curve.png')
plt.show()


# In[ ]:


# TODO: Set k based on the elbow plot above.
OPTIMAL_K = 3

kmeans_model, cluster_labels = train_kmeans(X_scaled, n_clusters=OPTIMAL_K)

# TODO: Replace 'feature_x' and 'feature_y' with two informative feature names.
# fig = plot_cluster_scatter(X_scaled, cluster_labels, 'feature_x', 'feature_y')
# save_figure(fig, '03_cluster_scatter.png')
# plt.show()


# ## 5. Feature Importance

# In[ ]:


# Works for tree-based classifiers/regressors. Adapt model variable as needed.
# Assumes the last step in the pipeline is named 'classifier' or 'regressor'.
try:
    inner_model = best_clf_pipeline.named_steps.get('classifier') \
                  or best_clf_pipeline.named_steps.get('regressor')
    if hasattr(inner_model, 'feature_importances_'):
        fig = plot_feature_importance(inner_model, list(X.columns), top_n=20)
        save_figure(fig, '03_feature_importance.png')
        plt.show()
except Exception as e:
    print(f'Feature importance not available: {e}')


# ## 6. Save Best Model

# In[ ]:


save_model(best_clf_pipeline, 'best_classifier.pkl')
print('Modeling complete. Proceed to Notebook 4 – Analysis & Findings.')


# In[ ]:




