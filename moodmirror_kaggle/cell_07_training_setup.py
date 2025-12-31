# ============================================================================
# CELL 7: Training Setup
# ============================================================================
# Purpose: Configure loss function, optimizer, scheduler for BERT fine-tuning
# Runtime: ~1 minute
#
# KEY CHANGES:
# 1. Different learning rates for frozen vs fine-tuned layers
# 2. Layer-wise learning rate decay (LIRD)
# 3. Weighted loss to improve recall
# 4. Optimized warmup schedule

print("="*70)
print("TRAINING SETUP: Enhanced for Fine-Tuning")
print("="*70)

# ============================================================================
# STEP 1: Load Model
# ============================================================================

print("\n[1] Loading model from Cell 6...")

with open('/tmp/moodmirror_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
tokenizer = model_dict['tokenizer']
device = model_dict['device']
class_weights = model_dict['class_weights']
batch_size = model_dict['batch_size']

print(f"✓ Model loaded on device: {device}")
print(f"✓ Class weights: {class_weights}")

# Load DataLoaders
with open('/tmp/moodmirror_dataloaders.pkl', 'rb') as f:
    dataloader_dict = pickle.load(f)

train_dataloader = dataloader_dict['train_dataloader']
val_dataloader = dataloader_dict['val_dataloader']
test_dataloader = dataloader_dict['test_dataloader']

print(f"✓ DataLoaders loaded")

# ============================================================================
# STEP 2: Weighted Loss Function (Improved Recall)
# ============================================================================

print("\n[2] Creating weighted loss function...")

# Adjust weights for better recall
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"✓ Loss function: Weighted Binary Cross-Entropy")
print(f"  pos_weight: {pos_weight.item():.4f}")
print(f"  → At-risk class is {pos_weight.item():.2f}× heavier")

# ============================================================================
# STEP 3: Optimizer with Layer-wise Learning Rate Decay (LIRD)
# ============================================================================

print("\n[3] Creating optimizer with LIRD...")

# Define different learning rates for BERT layers and other layers
optimizer_grouped_parameters = []

# Track parameters already added to avoid duplicates
added_params = set()

# BERT layers (fine-tuned)
for layer_idx in range(12):
    layer_params = [
        p for n, p in model.bert.named_parameters()
        if f"encoder.layer.{layer_idx}" in n and p.requires_grad and id(p) not in added_params
    ]
    added_params.update(id(p) for p in layer_params)  # Mark these parameters as added
    lr = 2e-5 * (0.95 ** (11 - layer_idx))  # Layer-wise decay
    if layer_params:  # Only add if there are parameters in this group
        optimizer_grouped_parameters.append({
            'params': layer_params,
            'lr': lr
        })

# Other model parameters (BiLSTM, Dense layers)
other_params = [
    p for n, p in model.named_parameters()
    if 'bert' not in n and p.requires_grad and id(p) not in added_params
]
added_params.update(id(p) for p in other_params)  # Mark these parameters as added
if other_params:  # Only add if there are parameters in this group
    optimizer_grouped_parameters.append({
        'params': other_params,
        'lr': 2e-4  # Higher learning rate for non-BERT layers
    })

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    eps=1e-8,
    weight_decay=0.01
)

print(f"✓ Optimizer: AdamW with Layer-wise Learning Rate Decay")
print(f"  Base learning rate (BERT): 2e-5")
print(f"  Layer-wise decay: 0.95 per layer")
print(f"  Non-BERT layers learning rate: 2e-4")

# ============================================================================
# STEP 4: Optimized Learning Rate Scheduler
# ============================================================================

print("\n[4] Creating optimized learning rate scheduler...")

num_epochs = 20
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.06 * num_training_steps)  # Reduced warmup to 6%

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"✓ Learning rate scheduler: Linear Warmup + Decay")
print(f"  Total steps: {num_training_steps:,}")
print(f"  Warmup steps: {num_warmup_steps:,} ({100*num_warmup_steps/num_training_steps:.1f}%)")

# ============================================================================
# STEP 5: Training Helper Functions
# ============================================================================

def compute_metrics(predictions, labels):
    """Compute evaluation metrics from logits."""
    # Convert logits to probabilities using sigmoid
    probs = 1 / (1 + np.exp(-predictions))  # or use scipy.special.expit
    pred_labels = (probs >= 0.5).astype(int)
    
    accuracy = accuracy_score(labels, pred_labels)
    
    if len(np.unique(labels)) == 1:
        precision = recall = f1 = 0.0
    else:
        precision = precision_score(labels, pred_labels, zero_division=0)
        recall = recall_score(labels, pred_labels, zero_division=0)
        f1 = f1_score(labels, pred_labels, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).float().unsqueeze(1)
        features = batch['features'].to(device) if batch['features'] is not None else None
        
        # Forward
        outputs = model(input_ids, attention_mask, features)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_predictions.extend(outputs.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten().astype(int))
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return {'loss': avg_loss, **metrics}


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation or test set."""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)
            features = batch['features'].to(device) if batch['features'] is not None else None
            
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return {'loss': avg_loss, **metrics}


print("\n[5] Training helper functions defined")
print(f"✓ train_epoch: trains model for 1 epoch")
print(f"✓ evaluate: evaluates on validation/test set")
print(f"✓ compute_metrics: computes accuracy, precision, recall, F1")

# ============================================================================
# STEP 6: Training Hyperparameters
# ============================================================================

print("\n[6] Training hyperparameters:")

EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
SAVE_BEST_MODEL = True

print(f"✓ Max epochs: {EPOCHS}")
print(f"✓ Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"✓ Save best model: {SAVE_BEST_MODEL}")

# ============================================================================
# STEP 7: Save Updated Training Setup
# ============================================================================

print("\n[7] Saving updated training setup for Cell 8...")

training_dict = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'criterion': criterion,
    'device': device,
    'train_dataloader': train_dataloader,
    'val_dataloader': val_dataloader,
    'test_dataloader': test_dataloader,
    'tokenizer': tokenizer,
    'class_weights': class_weights,
    'train_epoch': train_epoch,
    'evaluate': evaluate,
    'EPOCHS': EPOCHS,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'SAVE_BEST_MODEL': SAVE_BEST_MODEL
}

with open('/tmp/moodmirror_training.pkl', 'wb') as f:
    pickle.dump(training_dict, f)

print(f"✓ Updated training setup saved")

print("\n" + "="*70)
print("✓ CELL 7 COMPLETE - Enhanced Training Setup Configured")
print("="*70)
