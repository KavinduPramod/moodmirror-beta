# ============================================================================
# CELL 8: Training Loop
# ============================================================================
# Purpose: Train the model with early stopping and save best weights
# Runtime: ~30-60 minutes (20 epochs max, usually stops earlier)

print("="*70)
print("TRAINING LOOP: BERT + BiLSTM Model")
print("="*70)

# ============================================================================
# STEP 1: Load Training Setup
# ============================================================================

print("\n[1] Loading training setup from Cell 7...")

with open('/tmp/moodmirror_training.pkl', 'rb') as f:
    training_dict = pickle.load(f)

model = training_dict['model']
optimizer = training_dict['optimizer']
scheduler = training_dict['scheduler']
criterion = training_dict['criterion']
device = training_dict['device']
train_dataloader = training_dict['train_dataloader']
val_dataloader = training_dict['val_dataloader']
test_dataloader = training_dict['test_dataloader']
train_epoch = training_dict['train_epoch']
evaluate = training_dict['evaluate']
EPOCHS = training_dict['EPOCHS']
EARLY_STOPPING_PATIENCE = training_dict['EARLY_STOPPING_PATIENCE']
SAVE_BEST_MODEL = training_dict['SAVE_BEST_MODEL']

print(f"✓ Loaded model, optimizer, and evaluation functions")
print(f"✓ Training on device: {device}")

# ============================================================================
# STEP 2: Initialize Tracking
# ============================================================================

print("\n[2] Initializing training history...")

history = {
    'train_loss': [], 'train_acc': [], 'train_precision': [],
    'train_recall': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_precision': [],
    'val_recall': [], 'val_f1': [],
    'learning_rates': []
}

best_val_loss = float('inf')
best_val_f1 = 0.0
patience_counter = 0

print(f"✓ Tracking: loss, accuracy, precision, recall, F1")

# ============================================================================
# STEP 3: Training Loop
# ============================================================================

print("\n[3] Starting training loop...")
print(f"✓ Max epochs: {EPOCHS}")
print(f"✓ Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print("\n" + "="*70)

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-"*70)
    
    # TRAINING
    train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, criterion, device)
    
    history['train_loss'].append(train_metrics['loss'])
    history['train_acc'].append(train_metrics['accuracy'])
    history['train_precision'].append(train_metrics['precision'])
    history['train_recall'].append(train_metrics['recall'])
    history['train_f1'].append(train_metrics['f1'])
    history['learning_rates'].append(scheduler.get_last_lr()[0])
    
    print(f"\nTrain: Loss={train_metrics['loss']:.4f} | Acc={train_metrics['accuracy']:.4f} | "
          f"P={train_metrics['precision']:.4f} | R={train_metrics['recall']:.4f} | F1={train_metrics['f1']:.4f}")
    
    # VALIDATION
    val_metrics = evaluate(model, val_dataloader, criterion, device)
    
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    
    print(f"Val:   Loss={val_metrics['loss']:.4f} | Acc={val_metrics['accuracy']:.4f} | "
          f"P={val_metrics['precision']:.4f} | R={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f}")
    
    # EARLY STOPPING
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        
        if SAVE_BEST_MODEL:
            torch.save(model.state_dict(), '/tmp/best_model.pt')
            print(f"\n✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
    else:
        patience_counter += 1
        print(f"\n⚠ No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n✓ Early stopping triggered!")
            break
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch time: {epoch_time:.1f}s")

total_time = time.time() - start_time

print("\n" + "="*70)
print(f"✓ Training complete!")
print(f"✓ Total time: {total_time/60:.1f} minutes")
print(f"✓ Best val loss: {best_val_loss:.4f}")

# ============================================================================
# STEP 4: Evaluate Best Model on Test Set
# ============================================================================

print("\n[4] Evaluating best model on test set...")

model.load_state_dict(torch.load('/tmp/best_model.pt'))
test_metrics = evaluate(model, test_dataloader, criterion, device)

print(f"\n✓ Test Set Results:")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1: {test_metrics['f1']:.4f}")

history['test_loss'] = test_metrics['loss']
history['test_acc'] = test_metrics['accuracy']
history['test_precision'] = test_metrics['precision']
history['test_recall'] = test_metrics['recall']
history['test_f1'] = test_metrics['f1']

# ============================================================================
# STEP 5: Plot Learning Curves
# ============================================================================

print("\n[5] Plotting learning curves...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('MoodMirror Training Results', fontsize=16, fontweight='bold')

epochs_range = range(1, len(history['train_loss']) + 1)

# Loss
axes[0, 0].plot(epochs_range, history['train_loss'], label='Train', marker='o')
axes[0, 0].plot(epochs_range, history['val_loss'], label='Val', marker='s')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs_range, history['train_acc'], label='Train', marker='o')
axes[0, 1].plot(epochs_range, history['val_acc'], label='Val', marker='s')
axes[0, 1].set_title('Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Precision
axes[0, 2].plot(epochs_range, history['train_precision'], label='Train', marker='o')
axes[0, 2].plot(epochs_range, history['val_precision'], label='Val', marker='s')
axes[0, 2].set_title('Precision')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_ylim([0, 1])

# Recall
axes[1, 0].plot(epochs_range, history['train_recall'], label='Train', marker='o')
axes[1, 0].plot(epochs_range, history['val_recall'], label='Val', marker='s')
axes[1, 0].set_title('Recall (catch at-risk users)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

# F1 Score
axes[1, 1].plot(epochs_range, history['train_f1'], label='Train', marker='o')
axes[1, 1].plot(epochs_range, history['val_f1'], label='Val', marker='s')
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

# Learning Rate
axes[1, 2].plot(history['learning_rates'], label='LR', marker='o', color='purple')
axes[1, 2].set_title('Learning Rate')
axes[1, 2].set_xlabel('Training Step')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.savefig('/tmp/moodmirror_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Learning curves saved")

# ============================================================================
# STEP 6: Save Results
# ============================================================================

print("\n[6] Saving training results...")

results_dict = {
    'history': history,
    'test_metrics': test_metrics,
    'model_state': torch.load('/tmp/best_model.pt'),
    'training_time': total_time
}

with open('/tmp/moodmirror_training_results.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

with open('/tmp/moodmirror_history.json', 'w') as f:
    json_history = {k: [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else float(v) if isinstance(v, (float, np.floating)) else v 
                    for k, v in history.items()}
    json.dump(json_history, f, indent=2)

print(f"✓ Saved training results")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

print(f"\n✓ Model Performance:")
print(f"  Train F1: {history['train_f1'][-1]:.4f}")
print(f"  Val F1: {history['val_f1'][-1]:.4f}")
print(f"  Test F1: {history['test_f1']:.4f}")

print(f"\n✓ Recall (Critical Metric):")
print(f"  Train: {history['train_recall'][-1]:.4f}")
print(f"  Val: {history['val_recall'][-1]:.4f}")
print(f"  Test: {history['test_recall']:.4f}")

print(f"\n✓ Training Summary:")
print(f"  Epochs: {len(history['train_loss'])}")
print(f"  Time: {total_time/60:.1f} minutes")

print("\n" + "="*70)
print("✓ CELL 8 COMPLETE - Model trained with best weights saved")
print("="*70)
