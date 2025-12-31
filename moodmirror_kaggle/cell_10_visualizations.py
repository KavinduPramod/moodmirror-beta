# ============================================================================
# CELL 10: Visualizations & Analysis
# ============================================================================
# Purpose: Create comprehensive visualizations of model performance
# Runtime: ~2-3 minutes

print("="*70)
print("VISUALIZATIONS & ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load Results
# ============================================================================

print("\n[1] Loading evaluation results...")

with open('/tmp/moodmirror_eval_results.pkl', 'rb') as f:
    eval_results = pickle.load(f)

with open('/tmp/moodmirror_history.json', 'r') as f:
    history = json.load(f)

predictions = eval_results['predictions']
labels = eval_results['labels']
pred_labels = eval_results['pred_labels']
cm = eval_results['confusion_matrix']
fpr = eval_results['roc_curve']['fpr']
tpr = eval_results['roc_curve']['tpr']
auc_roc = eval_results['metrics']['auc_roc']
threshold_df = eval_results['threshold_analysis']

print(f"✓ Loaded evaluation results")

# ============================================================================
# STEP 2: Comprehensive Evaluation Figure
# ============================================================================

print("\n[2] Creating evaluation visualizations...")

from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC={auc_roc:.3f})')
ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Random')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(cm, cmap='Blues', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['At-risk', 'Normal'])
ax2.set_yticklabels(['At-risk', 'Normal'])
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')
ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center", 
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=14, fontweight='bold')

# Threshold Impact
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(threshold_df['threshold'], threshold_df['accuracy'], marker='o', label='Accuracy')
ax3.plot(threshold_df['threshold'], threshold_df['precision'], marker='s', label='Precision')
ax3.plot(threshold_df['threshold'], threshold_df['recall'], marker='^', label='Recall')
ax3.plot(threshold_df['threshold'], threshold_df['f1'], marker='d', label='F1')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Score')
ax3.set_title('Threshold Impact', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1])

# Prediction Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(predictions[labels == 0], bins=30, alpha=0.6, label='Normal', color='blue')
ax4.hist(predictions[labels == 1], bins=30, alpha=0.6, label='At-risk', color='red')
ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Count')
ax4.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions)
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(recall_curve, precision_curve, linewidth=2.5, color='green')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 1])
ax5.set_ylim([0, 1])

# Calibration Curve with proper handling
ax6 = fig.add_subplot(gs[1, 2])
try:
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=10)
    ax6.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    ax6.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Model')
    ax6.set_xlabel('Mean Predicted Probability')
    ax6.set_ylabel('Fraction of Positives')
    ax6.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
except Exception as e:
    ax6.text(0.5, 0.5, f'Calibration plot skipped\n(Predictions valid range)', 
             ha='center', va='center', transform=ax6.transAxes, fontsize=10)
    ax6.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax6.axis('off')

# Learning Curves - Loss
ax7 = fig.add_subplot(gs[2, 0])
epochs = range(1, len(history['train_loss']) + 1)
ax7.plot(epochs, history['train_loss'], marker='o', label='Train')
ax7.plot(epochs, history['val_loss'], marker='s', label='Val')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Loss')
ax7.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# F1 Score Over Epochs
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(epochs, history['train_f1'], marker='o', label='Train')
ax8.plot(epochs, history['val_f1'], marker='s', label='Val')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('F1 Score')
ax8.set_title('F1 Score Progression', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 1])

# Recall Over Epochs
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(epochs, history['train_recall'], marker='o', label='Train')
ax9.plot(epochs, history['val_recall'], marker='s', label='Val')
ax9.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Target')
ax9.set_xlabel('Epoch')
ax9.set_ylabel('Recall')
ax9.set_title('Recall Progression', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_ylim([0, 1])

fig.suptitle('MoodMirror: Comprehensive Evaluation', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/tmp/moodmirror_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved comprehensive evaluation figure")

# ============================================================================
# STEP 3: Error Analysis
# ============================================================================

print("\n[3] Error Analysis...")

fp_mask = (labels == 0) & (pred_labels == 1)
fn_mask = (labels == 1) & (pred_labels == 0)

fp_indices = np.where(fp_mask)[0]
fn_indices = np.where(fn_mask)[0]

print(f"\n✓ Misclassification Summary:")
print(f"  Total misclassified: {len(fp_indices) + len(fn_indices)}/{len(labels)}")
print(f"  False Positives: {len(fp_indices)} (avg prob: {predictions[fp_indices].mean():.4f})")
print(f"  False Negatives: {len(fn_indices)} (avg prob: {predictions[fn_indices].mean():.4f})")

# Error Distribution Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if len(fp_indices) > 0:
    axes[0].hist(predictions[fp_indices], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0].axvline(predictions[fp_indices].mean(), color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'False Positives (N={len(fp_indices)})')
axes[0].set_xlabel('Predicted Probability')
axes[0].grid(True, alpha=0.3, axis='y')

if len(fn_indices) > 0:
    axes[1].hist(predictions[fn_indices], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(predictions[fn_indices].mean(), color='darkred', linestyle='--', linewidth=2)
axes[1].set_title(f'False Negatives (N={len(fn_indices)})')
axes[1].set_xlabel('Predicted Probability')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/moodmirror_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved error analysis figure")

# ============================================================================
# STEP 4: Metrics Summary Table
# ============================================================================

print("\n[4] Metrics Summary Table:")

metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 
               'False Positive Rate', 'False Negative Rate', 'AUC-ROC'],
    'Value': [
        f"{eval_results['metrics']['accuracy']:.4f}",
        f"{eval_results['metrics']['precision']:.4f}",
        f"{eval_results['metrics']['sensitivity']:.4f}",
        f"{eval_results['metrics']['f1']:.4f}",
        f"{eval_results['metrics']['specificity']:.4f}",
        f"{eval_results['metrics']['fpr']:.4f}",
        f"{eval_results['metrics']['fnr']:.4f}",
        f"{eval_results['metrics']['auc_roc']:.4f}"
    ],
    'Target': ['>0.80', '>0.75', '>0.85', '>0.80', '>0.70', '<0.20', '<0.15', '>0.85'],
    'Status': [
        '✓' if eval_results['metrics']['accuracy'] > 0.80 else '⚠',
        '✓' if eval_results['metrics']['precision'] > 0.75 else '⚠',
        '✓' if eval_results['metrics']['sensitivity'] > 0.85 else '⚠',
        '✓' if eval_results['metrics']['f1'] > 0.80 else '⚠',
        '✓' if eval_results['metrics']['specificity'] > 0.70 else '⚠',
        '✓' if eval_results['metrics']['fpr'] < 0.20 else '⚠',
        '✓' if eval_results['metrics']['fnr'] < 0.15 else '⚠',
        '✓' if eval_results['metrics']['auc_roc'] > 0.85 else '⚠'
    ]
})

print(metrics_table.to_string(index=False))

# ============================================================================
# STEP 5: Save Analysis
# ============================================================================

print("\n[5] Saving analysis results...")

analysis_results = {
    'false_positives': {'count': len(fp_indices), 'mean_prob': float(predictions[fp_indices].mean()) if len(fp_indices) > 0 else 0},
    'false_negatives': {'count': len(fn_indices), 'mean_prob': float(predictions[fn_indices].mean()) if len(fn_indices) > 0 else 0},
    'metrics_summary': metrics_table
}

with open('/tmp/moodmirror_analysis.pkl', 'wb') as f:
    pickle.dump(analysis_results, f)

print(f"✓ Saved analysis results")

print("\n" + "="*70)
print("✓ CELL 10 COMPLETE - Visualizations and analysis finished")
print("="*70)
