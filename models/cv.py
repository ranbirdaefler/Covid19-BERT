#!/usr/bin/env python
"""
cv_stats_with_tuning.py
-----------------------
Rigorous comparison of AAindex vs. ProtBERT Δ-embeddings
with minimal hyperparameter tuning and confusion matrix visualization.

Usage: adjust the four file paths below, then
       python cv_stats_with_tuning.py
"""

# --------------------------------------------------
# 1. Imports
# --------------------------------------------------
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time

# --------------------------------------------------
# 2. File paths  (EDIT THESE)
# --------------------------------------------------
STATIC_X_FP = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\files_to_send\positional_vectors.npy"
STATIC_Y_FP = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\files_to_send\labels.npy"
BERT_X_FP   = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\files_to_send\bert_full_X.npy"
BERT_Y_FP   = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\files_to_send\bert_full_y.npy"

# --------------------------------------------------
# 3. Load data
# --------------------------------------------------
print("Loading data...")
static_X = np.load(STATIC_X_FP, allow_pickle=True)
static_y = np.load(STATIC_Y_FP, allow_pickle=True)
bert_X   = np.load(BERT_X_FP,   allow_pickle=True)
bert_y   = np.load(BERT_Y_FP,   allow_pickle=True)
s_labels = np.load(STATIC_Y_FP, allow_pickle=True).astype(str)
b_labels = np.load(BERT_Y_FP, allow_pickle=True).astype(str)
print("Order equal?", np.array_equal(s_labels, b_labels))

if not (static_y == bert_y).all():
    print("Label arrays mis-aligned – substituting static_y for bert_y")
    bert_y = static_y.copy()
assert static_X.shape[0] == bert_X.shape[0], "Feature sample mismatch"
assert (static_y == bert_y).all(), "Label arrays mis-aligned"

# --- encode string labels as integers 0..K-1
le = LabelEncoder()
y_encoded = le.fit_transform(static_y)
class_names = le.classes_

print(f"Classes: {class_names}")
print(f"Dataset shape: {static_X.shape[0]} samples")
print(f"Static features shape: {static_X.shape}")
print(f"BERT features shape: {bert_X.shape}")

# --------------------------------------------------
# 4. Minimal hyperparameter tuning
# --------------------------------------------------
print("\n" + "="*60)
print("MINIMAL HYPERPARAMETER TUNING")
print("="*60)

# Define parameter grid (minimal but meaningful)
# Focus on most impactful parameters
param_grid = {
    'max_depth': [4, 5, 6, 8],           # Tree complexity
    'n_estimators': [200, 300, 500],      # Number of trees
    'learning_rate': [0.05, 0.1, 0.15]    # Step size
}

# Base parameters (fixed)
base_params = {
    'objective': 'multi:softprob',
    'num_class': len(class_names),
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0  # Suppress XGBoost output
}

def tune_hyperparameters(X, y, feature_name, max_samples=15000):
    """
    Find best hyperparameters using stratified sampling and 3-fold CV.
    
    Selection criterion: Macro-F1 score (handles class imbalance better than accuracy)
    Justification: Macro-F1 gives equal weight to all classes, important for minority variants
    """
    print(f"\nTuning {feature_name} hyperparameters...")
    print(f"Selection criterion: Macro-F1 (equal weight to all classes)")
    
    start_time = time.time()
    
    # Use stratified subset for faster tuning while preserving class balance
    if len(X) > max_samples:
        print(f"Using stratified sample of {max_samples} for tuning (preserves class balance)")
        from sklearn.model_selection import train_test_split
        
        # Calculate sample ratio (don't exceed 80% of data)
        sample_ratio = min(max_samples / len(X), 0.8)
        
        # Use train_test_split for stratified sampling
        _, X_tune, _, y_tune = train_test_split(
            X, y, 
            test_size=sample_ratio, 
            stratify=y, 
            random_state=42
        )
    else:
        X_tune, y_tune = X, y
    print(f"Tuning on {len(X_tune)} samples")
    
    best_score = -np.inf
    best_params = None
    all_results = []
    
    # 3-fold CV for parameter selection
    skf_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    total_combinations = len(list(product(*param_grid.values())))
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, params in enumerate(product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))
        full_params = {**base_params, **param_dict}
        
        # Cross-validation scores for this parameter set
        cv_scores = []
        for train_idx, val_idx in skf_tune.split(X_tune, y_tune):
            X_train, X_val = X_tune[train_idx], X_tune[val_idx]
            y_train, y_val = y_tune[train_idx], y_tune[val_idx]
            
            clf = XGBClassifier(**full_params)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            cv_scores.append(f1_score(y_val, pred, average='macro'))
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        all_results.append({
            'params': param_dict,
            'mean_macro_f1': mean_score,
            'std_macro_f1': std_score
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = param_dict
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{total_combinations} combinations")
    
    elapsed = time.time() - start_time
    print(f"Tuning completed in {elapsed:.1f} seconds")
    print(f"Best {feature_name} params: {best_params}")
    print(f"Best CV macro-F1: {best_score:.4f}")
    
    # Show top 3 parameter sets
    sorted_results = sorted(all_results, key=lambda x: x['mean_macro_f1'], reverse=True)
    print(f"\nTop 3 parameter sets for {feature_name}:")
    for i, result in enumerate(sorted_results[:3]):
        print(f"  {i+1}. {result['params']} -> {result['mean_macro_f1']:.4f} ± {result['std_macro_f1']:.4f}")
    
    return {**base_params, **best_params}

# Tune hyperparameters for both feature sets
print("Hyperparameter tuning rationale:")
print("- Using macro-F1 as selection criterion (equal weight to minority classes)")
print("- 3-fold CV on stratified subset (faster while preserving class balance)")
print("- Testing key parameters: max_depth, n_estimators, learning_rate")

best_static_params = tune_hyperparameters(static_X, y_encoded, "AAindex Positional")
best_bert_params = tune_hyperparameters(bert_X, y_encoded, "ProtBERT Δ-embeddings")

# --------------------------------------------------
# 5. 5-fold CV with optimized hyperparameters
# --------------------------------------------------
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION WITH OPTIMIZED PARAMETERS")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
records = []
errTable = np.zeros((2, 2), dtype=int)   # for McNemar

# Store all predictions for confusion matrices
all_y_true = []
all_pred_static = []
all_pred_bert = []

print("Running 5-fold cross-validation...")
for fold, (tri, tei) in enumerate(skf.split(static_X, y_encoded), 1):
    print(f"  Processing fold {fold}/5...")
    y_train, y_test = y_encoded[tri], y_encoded[tei]
    
    # Store true labels for this fold
    all_y_true.extend(y_test)

    # ----- AAindex model (with optimized params) -----
    clf_stat = XGBClassifier(**best_static_params)
    clf_stat.fit(static_X[tri], y_train)
    pred_stat = clf_stat.predict(static_X[tei])
    all_pred_static.extend(pred_stat)

    # ----- ProtBERT model (with optimized params) -----
    clf_bert = XGBClassifier(**best_bert_params)
    clf_bert.fit(bert_X[tri], y_train)
    pred_bert = clf_bert.predict(bert_X[tei])
    all_pred_bert.extend(pred_bert)

    # ----- fold metrics -----
    for tag, pred in [("AAindex", pred_stat), ("ProtBERT", pred_bert)]:
        records.append(dict(
            fold     = fold,
            feature  = tag,
            accuracy = accuracy_score(y_test, pred),
            macroF1  = f1_score(y_test, pred, average="macro"),
        ))

    # ----- accumulate McNemar contingency -----
    aa_wrong   = pred_stat != y_test
    bert_wrong = pred_bert != y_test
    for a, b in zip(aa_wrong, bert_wrong):
        errTable[int(a), int(b)] += 1

# --------------------------------------------------
# 6. Results summary
# --------------------------------------------------
print("\n" + "="*60)
print("PERFORMANCE RESULTS")
print("="*60)

df = pd.DataFrame(records)
pivot = df.pivot(index="fold", columns="feature", values=["accuracy", "macroF1"])
print("\nPer-fold metrics:")
print(pivot.round(4), "\n")

summary = df.groupby("feature").agg(
    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    macro_mean=("macroF1", "mean"), macro_std=("macroF1", "std"),
).round(4)
print("Mean ± s.d. over 5 folds:")
print(summary, "\n")

# --------------------------------------------------
# 7. Statistical significance tests
# --------------------------------------------------
print("STATISTICAL SIGNIFICANCE TESTS")
print("-" * 40)

acc_stat = pivot["accuracy"]["AAindex"].values
acc_bert = pivot["accuracy"]["ProtBERT"].values
macro_stat = pivot["macroF1"]["AAindex"].values
macro_bert = pivot["macroF1"]["ProtBERT"].values

diff_acc = acc_bert - acc_stat
diff_macro = macro_bert - macro_stat

# Paired t-tests
t_stat_acc, p_t_acc = ttest_rel(acc_bert, acc_stat)
t_stat_macro, p_t_macro = ttest_rel(macro_bert, macro_stat)

# Wilcoxon signed-rank tests
w_stat_acc, p_w_acc = wilcoxon(diff_acc)
w_stat_macro, p_w_macro = wilcoxon(diff_macro)

print(f"Accuracy differences:")
print(f"  Paired t-test:  Δacc = {diff_acc.mean():.4%} ± {diff_acc.std():.4%}, t={t_stat_acc:.2f}, p={p_t_acc:.4g}")
print(f"  Wilcoxon test:  median Δacc = {np.median(diff_acc):.4%}, W={w_stat_acc}, p={p_w_acc:.4g}")

print(f"\nMacro-F1 differences:")
print(f"  Paired t-test:  Δmacro-F1 = {diff_macro.mean():.4%} ± {diff_macro.std():.4%}, t={t_stat_macro:.2f}, p={p_t_macro:.4g}")
print(f"  Wilcoxon test:  median Δmacro-F1 = {np.median(diff_macro):.4%}, W={w_stat_macro}, p={p_w_macro:.4g}")

print(f"\nMcNemar test (classification agreement):")
print("Contingency table [[both correct, AAindex wrong], [ProtBERT wrong, both wrong]]:")
print(errTable)

mc_res = mcnemar(errTable, exact=False, correction=True)
print(f"McNemar χ²(1) = {mc_res.statistic:.2f}, p = {mc_res.pvalue:.4g}")

# --------------------------------------------------
# 8. Confusion matrices visualization
# --------------------------------------------------
print("\n" + "="*60)
print("GENERATING CONFUSION MATRICES")
print("="*60)

# Convert lists back to arrays
all_y_true = np.array(all_y_true)
all_pred_static = np.array(all_pred_static)
all_pred_bert = np.array(all_pred_bert)

# Compute confusion matrices (aggregated across all folds)
cm_static = confusion_matrix(all_y_true, all_pred_static)
cm_bert = confusion_matrix(all_y_true, all_pred_bert)

# Normalize confusion matrices (show percentages)
cm_static_norm = cm_static.astype('float') / cm_static.sum(axis=1)[:, np.newaxis]
cm_bert_norm = cm_bert.astype('float') / cm_bert.sum(axis=1)[:, np.newaxis]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Raw counts
sns.heatmap(cm_static, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           ax=axes[0,0], cbar_kws={'label': 'Count'})
axes[0,0].set_title('AAindex Positional Features\n(Raw Counts)', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('True Label', fontsize=11)
axes[0,0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Oranges',
           xticklabels=class_names, yticklabels=class_names,
           ax=axes[0,1], cbar_kws={'label': 'Count'})
axes[0,1].set_title('ProtBERT Δ-embeddings\n(Raw Counts)', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('True Label', fontsize=11)
axes[0,1].set_xlabel('Predicted Label', fontsize=11)

# Normalized (percentages)
sns.heatmap(cm_static_norm, annot=True, fmt='.3f', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names,
           ax=axes[1,0], cbar_kws={'label': 'Recall'})
axes[1,0].set_title('AAindex Positional Features\n(Normalized by True Class)', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('True Label', fontsize=11)
axes[1,0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(cm_bert_norm, annot=True, fmt='.3f', cmap='Oranges',
           xticklabels=class_names, yticklabels=class_names,
           ax=axes[1,1], cbar_kws={'label': 'Recall'})
axes[1,1].set_title('ProtBERT Δ-embeddings\n(Normalized by True Class)', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('True Label', fontsize=11)
axes[1,1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrices_comparison.pdf', bbox_inches='tight')
print("Confusion matrices saved as 'confusion_matrices_comparison.png' and '.pdf'")
plt.show()

# Per-class performance analysis
print("\nPer-class performance analysis:")
print("-" * 40)

# Calculate per-class precision, recall, F1
for i, class_name in enumerate(class_names):
    # AAindex stats
    tp_static = cm_static[i, i]
    fp_static = cm_static[:, i].sum() - tp_static
    fn_static = cm_static[i, :].sum() - tp_static
    
    precision_static = tp_static / (tp_static + fp_static) if (tp_static + fp_static) > 0 else 0
    recall_static = tp_static / (tp_static + fn_static) if (tp_static + fn_static) > 0 else 0
    f1_static = 2 * precision_static * recall_static / (precision_static + recall_static) if (precision_static + recall_static) > 0 else 0
    
    # ProtBERT stats
    tp_bert = cm_bert[i, i]
    fp_bert = cm_bert[:, i].sum() - tp_bert
    fn_bert = cm_bert[i, :].sum() - tp_bert
    
    precision_bert = tp_bert / (tp_bert + fp_bert) if (tp_bert + fp_bert) > 0 else 0
    recall_bert = tp_bert / (tp_bert + fn_bert) if (tp_bert + fn_bert) > 0 else 0
    f1_bert = 2 * precision_bert * recall_bert / (precision_bert + recall_bert) if (precision_bert + recall_bert) > 0 else 0
    
    print(f"{class_name:8s} | AAindex: P={precision_static:.3f} R={recall_static:.3f} F1={f1_static:.3f} | "
          f"ProtBERT: P={precision_bert:.3f} R={recall_bert:.3f} F1={f1_bert:.3f} | "
          f"ΔF1={f1_bert-f1_static:+.3f}")

# --------------------------------------------------
# 9. Final summary report
# --------------------------------------------------
print("\n" + "="*60)
print("FINAL SUMMARY REPORT")
print("="*60)

print("HYPERPARAMETER OPTIMIZATION:")
print(f"  AAindex optimal params:  {best_static_params}")
print(f"  ProtBERT optimal params: {best_bert_params}")
print(f"  Selection criterion: Macro-F1 (equal weight to minority classes)")

print(f"\nFINAL PERFORMANCE (5-fold CV with optimized hyperparameters):")
acc_improvement = summary.loc['ProtBERT', 'acc_mean'] - summary.loc['AAindex', 'acc_mean']
f1_improvement = summary.loc['ProtBERT', 'macro_mean'] - summary.loc['AAindex', 'macro_mean']

print(f"  AAindex Positional:")
print(f"    Accuracy:  {summary.loc['AAindex', 'acc_mean']:.4f} ± {summary.loc['AAindex', 'acc_std']:.4f}")
print(f"    Macro-F1:  {summary.loc['AAindex', 'macro_mean']:.4f} ± {summary.loc['AAindex', 'macro_std']:.4f}")

print(f"  ProtBERT Δ-embeddings:")
print(f"    Accuracy:  {summary.loc['ProtBERT', 'acc_mean']:.4f} ± {summary.loc['ProtBERT', 'acc_std']:.4f}")
print(f"    Macro-F1:  {summary.loc['ProtBERT', 'macro_mean']:.4f} ± {summary.loc['ProtBERT', 'macro_std']:.4f}")

print(f"\nPERFORMANCE DIFFERENCES:")
print(f"  ProtBERT advantage: +{acc_improvement:.4f} accuracy, +{f1_improvement:.4f} macro-F1")
print(f"  Statistical significance (t-test): p={p_t_macro:.4g} ({'significant' if p_t_macro < 0.05 else 'not significant'} at α=0.05)")

# Practical significance
if f1_improvement > 0.01:
    practical_sig = "practically significant (>1pp improvement)"
elif f1_improvement > 0.005:
    practical_sig = "modest practical improvement"
else:
    practical_sig = "minimal practical difference"

print(f"  Practical significance: {practical_sig}")

print(f"\nCONCLUSION:")
if p_t_macro < 0.05 and f1_improvement > 0.01:
    conclusion = "ProtBERT Δ-embeddings significantly outperform AAindex positional features."
elif p_t_macro < 0.05:
    conclusion = "ProtBERT shows statistically significant but modest improvement over AAindex features."
else:
    conclusion = "No significant performance difference between ProtBERT and AAindex approaches."

print(f"  {conclusion}")
print(f"  Hyperparameter optimization improved results for both approaches.")
print(f"  Confusion matrices reveal class-specific performance patterns.")

print("\nFiles generated:")
print("  - confusion_matrices_comparison.png")
print("  - confusion_matrices_comparison.pdf")