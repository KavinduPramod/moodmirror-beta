# ============================================================================
# CELL 9: Test Set Evaluation
# ============================================================================
# Purpose: Comprehensive evaluation on held-out test set
# Runtime: ~5 minutes

print("="*70)
print("TEST SET EVALUATION")
print("="*70)

# ============================================================================
# STEP 1: Load Best Model
# ============================================================================

print("\n[1] Loading best model and test data...")

with open('/tmp/moodmirror_training_results.pkl', 'rb') as f:
    results_dict = pickle.load(f)

history = results_dict['history']

with open('/tmp/moodmirror_training.pkl', 'rb') as f:
    training_dict = pickle.load(f)

model = training_dict['model']
device = training_dict['device']
test_dataloader = training_dict['test_dataloader']
criterion = training_dict['criterion']

model.load_state_dict(torch.load('/tmp/best_model.pt'))
model.to(device)
model.eval()

print(f"✓ Loaded best model")
print(f"✓ Device: {device}")

# ============================================================================
# STEP 2: Generate Predictions
# ============================================================================

print("\n[2] Generating predictions on test set...")

all_predictions = []
all_labels = []
all_user_ids = []

with torch.no_grad():
    pbar = tqdm(test_dataloader, desc="Test Predictions")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']
        features = batch['features'].to(device) if batch['features'] is not None else None
        user_ids = batch['user_ids']
        
        # Model outputs LOGITS (not probabilities)
        logits = model(input_ids, attention_mask, features)

        # CRITICAL FIX: Convert logits to probabilities using sigmoid
        logits_numpy = logits.cpu().numpy().flatten()
        probabilities = 1 / (1 + np.exp(-logits_numpy))  # sigmoid: e^x / (1 + e^x)

        all_predictions.extend(probabilities)
        all_labels.extend(labels.numpy().astype(int))
        all_user_ids.extend(user_ids)

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_user_ids = np.array(all_user_ids)

print(f"✓ Generated {len(all_predictions)} predictions")
print(f"  Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
print(f"  ✓ Range should be [0, 1] (probabilities)")

# ============================================================================
# STEP 3: Compute Metrics
# ============================================================================

print("\n[3] Computing metrics at threshold = 0.5...")

threshold = 0.5
pred_labels = (all_predictions >= threshold).astype(int)

accuracy = accuracy_score(all_labels, pred_labels)
precision = precision_score(all_labels, pred_labels, zero_division=0)
recall = recall_score(all_labels, pred_labels, zero_division=0)
f1 = f1_score(all_labels, pred_labels, zero_division=0)
auc_roc = roc_auc_score(all_labels, all_predictions)

print(f"✓ Test Set Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  AUC-ROC: {auc_roc:.4f}")

# ============================================================================
# STEP 4: Confusion Matrix
# ============================================================================

print("\n[4] Computing confusion matrix...")

cm = confusion_matrix(all_labels, pred_labels)

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print(f"\n✓ Confusion Matrix:")
print(f"              Predicted")
print(f"           At-risk  Normal")
print(f"Actual")
print(f"At-risk       {TP:3d}      {FN:3d}")
print(f"Normal        {FP:3d}      {TN:3d}")

print(f"\n✓ Breakdown:")
print(f"  TP (caught at-risk): {TP}")
print(f"  TN (correctly labeled normal): {TN}")
print(f"  FP (false alarms): {FP}")
print(f"  FN (missed at-risk): {FN}")

# ============================================================================
# STEP 5: Additional Metrics
# ============================================================================

print("\n[5] Computing additional metrics...")

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"✓ Additional Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  False Positive Rate: {fpr:.4f}")
print(f"  False Negative Rate: {fnr:.4f}")

# ============================================================================
# STEP 6: Classification Report
# ============================================================================

print("\n[6] Classification report:")

report = classification_report(all_labels, pred_labels, 
                               target_names=['Normal (0)', 'At-risk (1)'],
                               digits=4)
print(f"\n{report}")

# ============================================================================
# STEP 7: Threshold Analysis
# ============================================================================

print("\n[7] Threshold analysis...")

thresholds_to_test = np.arange(0.1, 1.0, 0.1)
threshold_results = []

for thresh in thresholds_to_test:
    pred_thresh = (all_predictions >= thresh).astype(int)
    
    acc = accuracy_score(all_labels, pred_thresh)
    prec = precision_score(all_labels, pred_thresh, zero_division=0)
    rec = recall_score(all_labels, pred_thresh, zero_division=0)
    f1_t = f1_score(all_labels, pred_thresh, zero_division=0)
    
    threshold_results.append({
        'threshold': thresh,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_t
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.to_string(index=False))

# Best threshold
best_f1_idx = threshold_df['f1'].idxmax()
best_threshold = threshold_df.loc[best_f1_idx, 'threshold']
print(f"\n✓ Best threshold (max F1): {best_threshold:.1f} → F1={threshold_df.loc[best_f1_idx, 'f1']:.4f}")

# ============================================================================
# STEP 8: ROC Curve
# ============================================================================

print("\n[8] Computing ROC curve...")

fpr_curve, tpr_curve, roc_thresholds = roc_curve(all_labels, all_predictions)

print(f"✓ ROC-AUC Score: {auc_roc:.4f}")
if auc_roc >= 0.9:
    print(f"    Excellent (0.9-1.0) ✓")
elif auc_roc >= 0.8:
    print(f"    Good (0.8-0.9) ✓")
elif auc_roc >= 0.7:
    print(f"    Fair (0.7-0.8)")
else:
    print(f"    Poor (<0.7)")

# ============================================================================
# STEP 9: Save Evaluation Results
# ============================================================================

print("\n[9] Saving evaluation results...")

eval_results = {
    'predictions': all_predictions,
    'labels': all_labels,
    'user_ids': all_user_ids,
    'pred_labels': pred_labels,
    'threshold': threshold,
    'metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'fpr': fpr,
        'fnr': fnr
    },
    'confusion_matrix': cm,
    'roc_curve': {
        'fpr': fpr_curve,
        'tpr': tpr_curve,
        'thresholds': roc_thresholds
    },
    'threshold_analysis': threshold_df
}

with open('/tmp/moodmirror_eval_results.pkl', 'wb') as f:
    pickle.dump(eval_results, f)

print(f"✓ Saved evaluation results")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("TEST EVALUATION SUMMARY")
print("="*70)

print(f"\n✓ Dataset Size:")
print(f"  Total: {len(all_labels)} users")
print(f"  Normal: {(all_labels == 0).sum()} ({100*(all_labels == 0).sum()/len(all_labels):.1f}%)")
print(f"  At-risk: {(all_labels == 1).sum()} ({100*(all_labels == 1).sum()/len(all_labels):.1f}%)")

print(f"\n✓ Model Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  AUC-ROC: {auc_roc:.4f}")

print(f"\n✓ Real Impact:")
print(f"  At-risk users caught: {TP}/{(all_labels == 1).sum()} = {100*recall:.1f}%")
print(f"  Normal users correctly labeled: {TN}/{(all_labels == 0).sum()} = {100*specificity:.1f}%")

print("\n" + "="*70)
print("✓ CELL 9 COMPLETE - Test evaluation finished")
print("="*70)
