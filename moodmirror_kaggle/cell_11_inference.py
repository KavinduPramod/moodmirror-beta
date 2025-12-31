# ============================================================================
# CELL 11: Inference & Deployment
# ============================================================================
# Purpose: Create inference functions for deployment, save final model
# Runtime: ~2-3 minutes

print("="*70)
print("INFERENCE & DEPLOYMENT")
print("="*70)

# ============================================================================
# STEP 1: Load Model and Data
# ============================================================================

print("\n[1] Loading model and supporting data...")

with open('/tmp/moodmirror_training.pkl', 'rb') as f:
    training_dict = pickle.load(f)

model = training_dict['model']
device = training_dict['device']
tokenizer = training_dict['tokenizer']

with open('/tmp/moodmirror_eval_results.pkl', 'rb') as f:
    eval_results = pickle.load(f)

with open('/tmp/moodmirror_features.pkl', 'rb') as f:
    data_dict = pickle.load(f)

scaler = data_dict['scaler']
test_user_ids = data_dict['test_user_ids']

with open('/tmp/moodmirror_dataloaders.pkl', 'rb') as f:
    dataloader_dict = pickle.load(f)

test_dataset = dataloader_dict['test_dataset']

# Load best model weights
model.load_state_dict(torch.load('/tmp/best_model.pt'))
model.eval()

print(f"✓ Model loaded on device: {device}")

# ============================================================================
# STEP 2: Define Inference Function
# ============================================================================

print("\n[2] Defining inference function...")

def predict_user(user_dict, model, tokenizer, device, max_length=512):
    """
    Make a prediction for a single user.
    
    Args:
        user_dict: Dict with 'posts' and 'features' keys
        model: Trained PyTorch model
        tokenizer: BERT tokenizer
        device: torch.device
        max_length: Max sequence length (512)
        
    Returns:
        Dict with prediction, confidence, and risk assessment
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare posts
        posts = user_dict['posts']
        post_texts = [p['text'] for p in posts]
        
        max_posts = 20
        selected_posts = post_texts[:max_posts]
        combined_text = " [SEP] ".join(selected_posts)
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Prepare features
        if 'features' in user_dict and user_dict['features'] is not None:
            features = torch.tensor(user_dict['features'], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            features = None
        
        # Forward pass
        # FIXED: Model outputs LOGITS, not probabilities
        logits = model(input_ids, attention_mask, features)

        # Convert logits to probability using sigmoid
        logit_value = logits.cpu().item()
        probability = 1 / (1 + np.exp(-logit_value))  # sigmoid
    
    # Risk classification
    threshold = 0.5
    is_at_risk = probability >= threshold
    
    confidence = abs(probability - threshold)

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.5:
        risk_level = "Moderate"
    elif probability < 0.7:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    return {
        'user_id': user_dict.get('user_id', 'unknown'),
        'probability': probability,
        'is_at_risk': is_at_risk,
        'risk_level': risk_level,
        'confidence': confidence,
        'num_posts': len(posts),
        'threshold': threshold
    }


print(f"✓ Inference function defined")
print(f"  Input: User dict (posts + features)")
print(f"  Output: probability, risk classification, confidence")

# ============================================================================
# STEP 3: Batch Inference Function
# ============================================================================

print("\n[3] Defining batch inference function...")

def predict_batch(user_list, model, tokenizer, device):
    """Make predictions for multiple users."""
    predictions = []
    
    for user in tqdm(user_list, desc="Batch Predictions"):
        pred = predict_user(user, model, tokenizer, device)
        predictions.append(pred)
    
    return pd.DataFrame(predictions)


print(f"✓ Batch inference function defined")

# ============================================================================
# STEP 4: Example Predictions
# ============================================================================

print("\n[4] Running example predictions on test set...")

num_examples = 5
random_indices = np.random.choice(len(test_dataset), num_examples, replace=False)

print(f"\n✓ Running predictions on {num_examples} random test users:")
print("="*70)

example_predictions = []

for idx, test_idx in enumerate(random_indices):
    user_id = test_user_ids[test_idx]
    user_data = test_dataset.user_dict[user_id]
    features = test_dataset.X[test_idx] if test_dataset.X is not None else None
    user_dict = {**user_data, 'features': features}
    true_label = test_dataset.labels[test_idx]
    
    pred = predict_user(user_dict, model, tokenizer, device)
    example_predictions.append(pred)
    
    correct = "✓" if pred['is_at_risk'] == (true_label == 1) else "✗"
    print(f"\nExample {idx+1}: {pred['user_id']}")
    print(f"  Posts: {pred['num_posts']} | Prob: {pred['probability']:.4f} | Risk: {pred['risk_level']}")
    print(f"  Prediction: {'At-risk ⚠' if pred['is_at_risk'] else 'Normal ✓'} | True: {'At-risk' if true_label == 1 else 'Normal'} | {correct}")

# ============================================================================
# STEP 5: Risk Ranking
# ============================================================================

print("\n[5] Ranking users by risk probability...")

sample_size = min(50, len(test_dataset))
sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)

ranking_preds = []

for test_idx in tqdm(sample_indices, desc="Computing rankings"):
    user_id = test_user_ids[test_idx]
    user_data = test_dataset.user_dict[user_id]
    features = test_dataset.X[test_idx] if test_dataset.X is not None else None
    user_dict = {**user_data, 'features': features}
    
    pred = predict_user(user_dict, model, tokenizer, device)
    ranking_preds.append(pred)

ranking_df = pd.DataFrame(ranking_preds).sort_values('probability', ascending=False)

print(f"\n✓ Top 5 At-Risk Users:")
print(ranking_df[['user_id', 'probability', 'risk_level', 'num_posts']].head(5).to_string(index=False))

print(f"\n✓ Bottom 5 (Lowest Risk):")
print(ranking_df[['user_id', 'probability', 'risk_level', 'num_posts']].tail(5).to_string(index=False))

# ============================================================================
# STEP 6: Save Final Model
# ============================================================================

print("\n[6] Saving final model for production...")

torch.save(model.state_dict(), '/tmp/moodmirror_final_model.pt')

inference_config = {
    'max_length': 512,
    'max_posts_per_user': 20,
    'probability_threshold': 0.5,
    'risk_levels': {
        'Low': [0.0, 0.3],
        'Moderate': [0.3, 0.5],
        'High': [0.5, 0.7],
        'Very High': [0.7, 1.0]
    },
    'eval_metrics': {
        'accuracy': float(eval_results['metrics']['accuracy']),
        'precision': float(eval_results['metrics']['precision']),
        'recall': float(eval_results['metrics']['recall']),
        'f1': float(eval_results['metrics']['f1']),
        'auc_roc': float(eval_results['metrics']['auc_roc'])
    }
}

with open('/tmp/moodmirror_inference_config.json', 'w') as f:
    json.dump(inference_config, f, indent=2)

print(f"✓ Saved final model: /tmp/moodmirror_final_model.pt")
print(f"✓ Saved inference config: /tmp/moodmirror_inference_config.json")

# ============================================================================
# STEP 7: Deployment Report
# ============================================================================

print("\n" + "="*70)
print("DEPLOYMENT REPORT")
print("="*70)

print(f"\n✓ Model Architecture:")
print(f"  BERT (frozen) + BiLSTM (256 × 2) + Dense layers")
print(f"  Input: Reddit posts (max 512 tokens)")
print(f"  Output: Probability 0-1")

print(f"\n✓ Performance Metrics:")
print(f"  Accuracy: {eval_results['metrics']['accuracy']:.4f}")
print(f"  Precision: {eval_results['metrics']['precision']:.4f}")
print(f"  Recall: {eval_results['metrics']['recall']:.4f}")
print(f"  F1 Score: {eval_results['metrics']['f1']:.4f}")
print(f"  AUC-ROC: {eval_results['metrics']['auc_roc']:.4f}")

print(f"\n✓ Deployment Checklist:")
print(f"  ✓ Model trained and validated")
print(f"  ✓ Test performance evaluated")
print(f"  ✓ Inference functions created")
print(f"  ✓ Model weights saved")
print(f"  ✓ Configuration file created")

print(f"\n✓ Usage Instructions:")
print(f"  1. Load model: model.load_state_dict(torch.load('moodmirror_final_model.pt'))")
print(f"  2. Load config: with open('moodmirror_inference_config.json') as f: config = json.load(f)")
print(f"  3. Predict: pred = predict_user(user_dict, model, tokenizer, device)")

print(f"\n✓ Risk Level Interpretation:")
print(f"  Low (0-30%): Normal behavior")
print(f"  Moderate (30-50%): Monitor closely")
print(f"  High (50-70%): At-risk, recommend assessment")
print(f"  Very High (70-100%): Urgent intervention")

print(f"\n✓ Model Limitations:")
print(f"  - Screening tool, not diagnostic")
print(f"  - Based on behavioral proxies, not clinical labels")
print(f"  - Requires professional review")

print("\n" + "="*70)
print("✓ CELL 11 COMPLETE - Model ready for deployment")
print("="*70)

print("\n" + "="*70)
print("✓ ALL CELLS COMPLETE!")
print("✓ MoodMirror model successfully trained and evaluated")
print("✓ Key improvements applied:")
print("  ✓ BiLSTM architecture (bidirectional=True)")
print("  ✓ Behavioral proxy labels (not circular risk scores)")
print("  ✓ Clean modular code for easy debugging")
print("="*70)
