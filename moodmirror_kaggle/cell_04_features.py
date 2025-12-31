# ============================================================================
# CELL 4 COMPLETE REWRITE: Feature Engineering & Label Creation
# ============================================================================
# THIS IS THE COMPLETE CORRECT VERSION
# Replace your entire Cell 4 with this code

print("="*70)
print("FEATURE ENGINEERING & LABEL CREATION (CORRECTED)")
print("="*70)

# ============================================================================
# STEP 1: Load cleaned data
# ============================================================================

print("\n[1] Loading cleaned data...")

with open('/tmp/moodmirror_users_clean.pkl', 'rb') as f:
    users_data = pickle.load(f)

with open('/tmp/moodmirror_df_clean.pkl', 'rb') as f:
    df = pickle.load(f)

with open('/tmp/moodmirror_baseline_clean.pkl', 'rb') as f:
    population_baseline = pickle.load(f)

print(f"✓ Loaded {len(users_data)} users")

# ============================================================================
# STEP 2: Extract labels from users_data JSON (ALREADY BALANCED!)
# ============================================================================

print("\n[2] Extracting labels from users_data...")

# The users_data already has 'label' field from your balanced JSON!
labels_from_users = []
user_ids_list = []

for user in users_data:
    user_ids_list.append(user['user_id'])
    # The label was already set when you created reddit-mh-users-balanced.json
    label = user.get('label', 0)  # Default to 0 if missing
    labels_from_users.append(label)

print(f"\n✓ Extracted labels for {len(labels_from_users)} users")

# Check distribution
from collections import Counter
label_dist = Counter(labels_from_users)
print(f"  Class 0 (Normal):  {label_dist[0]} ({100*label_dist[0]/len(labels_from_users):.1f}%)")
print(f"  Class 1 (At-risk): {label_dist[1]} ({100*label_dist[1]/len(labels_from_users):.1f}%)")

# ============================================================================
# STEP 3: Calculate Z-Scores
# ============================================================================

print("\n[3] Computing z-scores for each user...")

z_scores = []

for user in users_data:
    features = user['features']
    
    # Calculate z-scores using population baseline
    z_sentiment = (features['avg_sentiment'] - population_baseline['population_mean_sentiment']) / \
                  (population_baseline['population_std_sentiment'] + 1e-6)
    
    z_frequency = (features['posting_frequency'] - population_baseline['population_mean_frequency']) / \
                  (population_baseline['population_std_frequency'] + 1e-6)
    
    z_late_night = (features['late_night_ratio'] - population_baseline['population_mean_late_night']) / \
                   (population_baseline['population_std_late_night'] + 1e-6)
    
    z_negative = (features['negative_post_ratio'] - population_baseline['population_mean_negative_ratio']) / \
                 (population_baseline['population_std_negative_ratio'] + 1e-6)
    
    z_first_person = (features['first_person_pronoun_ratio'] - population_baseline['population_mean_first_person']) / \
                     (population_baseline['population_std_first_person'] + 1e-6)
    
    z_mh_participation = (features['mental_health_participation'] - population_baseline['population_mean_mh_participation']) / \
                         (population_baseline['population_std_mh_participation'] + 1e-6)
    
    z_scores.append({
        'user_id': user['user_id'],
        'z_sentiment': z_sentiment,
        'z_frequency': z_frequency,
        'z_late_night': z_late_night,
        'z_negative': z_negative,
        'z_first_person': z_first_person,
        'z_mh_participation': z_mh_participation,
        'confidence': features['confidence_score'],
        'baseline_stability': features['baseline_stability'],
        'label': user.get('label', 0)  # ← ADD LABEL HERE!
    })

z_df = pd.DataFrame(z_scores)
print(f"✓ Computed z-scores for {len(z_df)} users")

# Verify labels in z_df
print(f"\n✓ Labels in z_df:")
z_label_dist = Counter(z_df['label'])
print(f"  Class 0: {z_label_dist[0]} ({100*z_label_dist[0]/len(z_df):.1f}%)")
print(f"  Class 1: {z_label_dist[1]} ({100*z_label_dist[1]/len(z_df):.1f}%)")

# ============================================================================
# STEP 4: Create STRATIFIED Train/Val/Test Split
# ============================================================================

print("\n[4] Creating stratified train/val/test split...")

# FIRST SPLIT: 60% train, 40% val+test (with stratification!)
train, val_test = train_test_split(
    z_df,
    test_size=0.40,
    random_state=42,
    stratify=z_df['label']  # ✓ STRATIFY!
)

# SECOND SPLIT: 50% val, 50% test from the val_test set (with stratification!)
val, test = train_test_split(
    val_test,
    test_size=0.50,
    random_state=42,
    stratify=val_test['label']  # ✓ STRATIFY!
)

# ============================================================================
# VERIFY STRATIFICATION
# ============================================================================

print(f"\n✓ Train set: {len(train)} users")
train_class_0 = (train['label'] == 0).sum()
train_class_1 = (train['label'] == 1).sum()
print(f"  - Class 0: {train_class_0} ({100 * train_class_0 / len(train):.1f}%)")
print(f"  - Class 1: {train_class_1} ({100 * train_class_1 / len(train):.1f}%)")

print(f"\n✓ Val set:   {len(val)} users")
val_class_0 = (val['label'] == 0).sum()
val_class_1 = (val['label'] == 1).sum()
print(f"  - Class 0: {val_class_0} ({100 * val_class_0 / len(val):.1f}%)")
print(f"  - Class 1: {val_class_1} ({100 * val_class_1 / len(val):.1f}%)")

print(f"\n✓ Test set:  {len(test)} users")
test_class_0 = (test['label'] == 0).sum()
test_class_1 = (test['label'] == 1).sum()
print(f"  - Class 0: {test_class_0} ({100 * test_class_0 / len(test):.1f}%)")
print(f"  - Class 1: {test_class_1} ({100 * test_class_1 / len(test):.1f}%)")

# 🔴 CHECK THIS OUTPUT!
# If all sets show ~50-50 split, the fix worked!
# If they show ~94-6, there's still a problem!

# ============================================================================
# STEP 5: Create Feature Matrices
# ============================================================================

print("\n[5] Creating feature matrices...")

# Z-score features (8 features)
feature_cols = ['z_sentiment', 'z_frequency', 'z_late_night', 'z_negative',
                'z_first_person', 'z_mh_participation', 'confidence', 'baseline_stability']

X_train = train[feature_cols].values
y_train = train['label'].values
train_user_ids = train['user_id'].values

X_val = val[feature_cols].values
y_val = val['label'].values
val_user_ids = val['user_id'].values

X_test = test[feature_cols].values
y_test = test['label'].values
test_user_ids = test['user_id'].values

print(f"✓ Feature matrices created:")
print(f"  X_train shape: {X_train.shape}, y_train distribution: {Counter(y_train)}")
print(f"  X_val shape:   {X_val.shape}, y_val distribution: {Counter(y_val)}")
print(f"  X_test shape:  {X_test.shape}, y_test distribution: {Counter(y_test)}")

# ============================================================================
# STEP 6: Calculate Class Weights
# ============================================================================

print("\n[6] Computing class weights...")

unique, counts = np.unique(y_train, return_counts=True)
class_weights = {}

total = len(y_train)
for u, c in zip(unique, counts):
    class_weights[u] = total / (2 * c)

print(f"✓ Class weights:")
print(f"  Class 0 (Normal):  {class_weights[0]:.4f}")
print(f"  Class 1 (At-risk): {class_weights[1]:.4f}")
print(f"  Ratio: {class_weights[1]/class_weights[0]:.2f}x")

# ============================================================================
# STEP 7: Normalize Features
# ============================================================================

print("\n[7] Normalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features normalized (mean≈0, std≈1)")

# ============================================================================
# STEP 8: Save for next cell
# ============================================================================

print("\n[8] Saving data for Cell 5...")

data_dict = {
    'X_train': X_train_scaled,
    'y_train': y_train,
    'X_val': X_val_scaled,
    'y_val': y_val,
    'X_test': X_test_scaled,
    'y_test': y_test,
    'train_user_ids': train_user_ids,
    'val_user_ids': val_user_ids,
    'test_user_ids': test_user_ids,
    'class_weights': class_weights,
    'scaler': scaler,
    'users_data': users_data,
    'feature_cols': feature_cols
}

with open('/tmp/moodmirror_features.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"✓ Saved feature matrices and metadata")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY")
print("="*70)

print(f"\n✓ LABELS FROM BALANCED DATASET:")
print(f"  Total at-risk users: {(y_train==1).sum() + (y_val==1).sum() + (y_test==1).sum()}")

print(f"\n✓ STRATIFIED SPLITS:")
print(f"  Train: {len(X_train)} samples ({100*(y_train==1).sum()/len(y_train):.1f}% at-risk)")
print(f"  Val:   {len(X_val)} samples ({100*(y_val==1).sum()/len(y_val):.1f}% at-risk)")
print(f"  Test:  {len(X_test)} samples ({100*(y_test==1).sum()/len(y_test):.1f}% at-risk)")

print("\n" + "="*70)
print("✓ CELL 4 COMPLETE - Features engineered, labels from balanced dataset")
print("="*70)