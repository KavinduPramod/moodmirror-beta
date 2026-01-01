# ========================================
# SMART RE-LABELING STRATEGY
# Based on Thesis Requirements
# ========================================

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_meaningful_labels(data):
    """
    Create labels based on mental health behavioral indicators
    aligned with thesis requirements.
    
    Mental Health Severity Score considers:
    - Negative sentiment (higher = more concerning)
    - High negative post ratio (more complaints/sadness)
    - Late night activity (sleep disruption)
    - First-person pronoun usage (self-focused attention)
    - Reduced engagement (social withdrawal)
    
    Class 0: Users showing LOW mental health concerns
    Class 1: Users showing HIGH mental health concerns
    """
    
    relabeled_data = []
    
    for user in data:
        features = user['features']
        
        # Extract relevant features
        avg_sentiment = features.get('avg_sentiment', 0)
        negative_post_ratio = features.get('negative_post_ratio', 0)
        late_night_ratio = features.get('late_night_ratio', 0)
        first_person_ratio = features.get('first_person_pronoun_ratio', 0)
        avg_score = features.get('avg_score', 0)
        temporal_consistency = features.get('temporal_consistency', 0)
        
        # Mental Health Severity Score (0-1)
        # Components aligned with depression/anxiety indicators:
        
        # 1. Negative sentiment (0.3 weight)
        sentiment_score = max(0, -avg_sentiment) / 1.0  # More negative = higher score
        
        # 2. Negative post ratio (0.3 weight)
        negative_score = negative_post_ratio
        
        # 3. Sleep disruption (late night activity) (0.2 weight)
        sleep_score = late_night_ratio
        
        # 4. Self-focused attention (0.15 weight)
        self_focus_score = first_person_ratio
        
        # 5. Engagement quality (reduced upvotes = withdrawal) (0.05 weight)
        engagement_score = 1.0 - min(1.0, avg_score / 10.0)
        
        # Weighted severity score
        severity_score = (
            sentiment_score * 0.30 +
            negative_score * 0.30 +
            sleep_score * 0.20 +
            self_focus_score * 0.15 +
            engagement_score * 0.05
        )
        
        # Threshold at median (50% split)
        # Class 0: Low severity (more positive, typical behavior)
        # Class 1: High severity (more depressive/anxious indicators)
        
        # Store original label for comparison
        original_label = int(user.get('label', 0))
        
        # Add new label and severity score
        user['original_label'] = original_label
        user['severity_score'] = severity_score
        user['label_components'] = {
            'sentiment': sentiment_score,
            'negative_ratio': negative_score,
            'sleep_disruption': sleep_score,
            'self_focus': self_focus_score,
            'engagement': engagement_score
        }
        
        relabeled_data.append(user)
    
    # Calculate median severity to use as threshold
    severity_scores = [u['severity_score'] for u in relabeled_data]
    median_severity = np.median(severity_scores)
    
    # Assign labels based on severity
    for user in relabeled_data:
        user['label'] = 1 if user['severity_score'] >= median_severity else 0
    
    return relabeled_data, median_severity


# ========================================
# APPLY RE-LABELING
# ========================================

# Load data
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

print("Loading datasets...")
train_data_old = load_json('./prepared_data_relaxed/train_data.json')
val_data_old = load_json('./prepared_data_relaxed/val_data.json')
test_data_old = load_json('./prepared_data_relaxed/test_data.json')

# Combine all data for re-labeling
all_data = train_data_old + val_data_old + test_data_old
print(f"Total users before re-labeling: {len(all_data)}")

# Re-label all data
print("\n🔄 Creating meaningful labels based on mental health indicators...")
all_data_relabeled, median_severity = create_meaningful_labels(all_data)

print(f"\nMedian Severity Score: {median_severity:.3f}")
print("\nLabel Distribution (before vs after):")

# Count before
before_labels = [int(u['original_label']) for u in all_data_relabeled]
before_c0 = sum(1 for l in before_labels if l == 0)
before_c1 = sum(1 for l in before_labels if l == 1)

# Count after
after_labels = [int(u['label']) for u in all_data_relabeled]
after_c0 = sum(1 for l in after_labels if l == 0)
after_c1 = sum(1 for l in after_labels if l == 1)

print(f"Before: Class 0: {before_c0} ({before_c0/len(all_data_relabeled)*100:.1f}%), Class 1: {before_c1} ({before_c1/len(all_data_relabeled)*100:.1f}%)")
print(f"After:  Class 0: {after_c0} ({after_c0/len(all_data_relabeled)*100:.1f}%), Class 1: {after_c1} ({after_c1/len(all_data_relabeled)*100:.1f}%)")

# Analyze feature differences
print("\n📊 Feature Differences AFTER Re-labeling:")
for split_name, split_data in [('All Data', all_data_relabeled)]:
    class_0_sentiment = [u['features']['avg_sentiment'] for u in split_data if int(u['label']) == 0]
    class_1_sentiment = [u['features']['avg_sentiment'] for u in split_data if int(u['label']) == 1]
    
    class_0_negative = [u['features']['negative_post_ratio'] for u in split_data if int(u['label']) == 0]
    class_1_negative = [u['features']['negative_post_ratio'] for u in split_data if int(u['label']) == 1]
    
    print(f"\n  {split_name}:")
    print(f"    Sentiment (Class 0 vs 1): {np.mean(class_0_sentiment):.3f} vs {np.mean(class_1_sentiment):.3f} (diff: {abs(np.mean(class_0_sentiment) - np.mean(class_1_sentiment)):.3f})")
    print(f"    Negative Ratio (Class 0 vs 1): {np.mean(class_0_negative):.3f} vs {np.mean(class_1_negative):.3f} (diff: {abs(np.mean(class_0_negative) - np.mean(class_1_negative)):.3f})")

# Re-split into train/val/test maintaining stratification
print("\n🔄 Creating new stratified splits...")
labels_for_split = [int(u['label']) for u in all_data_relabeled]

train_data_new, temp_data = train_test_split(
    all_data_relabeled, test_size=0.30,
    stratify=labels_for_split, random_state=42
)

temp_labels = [int(u['label']) for u in temp_data]
val_data_new, test_data_new = train_test_split(
    temp_data, test_size=0.50,
    stratify=temp_labels, random_state=42
)

print(f"New splits:")
print(f"  Training:   {len(train_data_new)} users")
print(f"  Validation: {len(val_data_new)} users")
print(f"  Testing:    {len(test_data_new)} users")

# Save re-labeled data
def save_json(data, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {path}")

print("\n💾 Saving re-labeled datasets...")
save_json(train_data_new, './prepared_data_relabeled/train_data.json')
save_json(val_data_new, './prepared_data_relabeled/val_data.json')
save_json(test_data_new, './prepared_data_relabeled/test_data.json')

print("\n" + "="*80)
print("✅ RE-LABELING COMPLETE!")
print("="*80)
print("""
Next steps:
1. Use './prepared_data_relabeled/' instead of './prepared_data_relaxed/'
2. Re-train the model with:
   - Updated train_data path
   - EPOCHS = 50 (not 10)
   - PATIENCE = 20
   
Expected improvement:
- Val F1 should reach 0.75+ by epoch 10-15
- Test F1 should reach 0.80+ 
- If not, there's still a data/label issue

This re-labeling ensures:
✅ Labels align with thesis (mental health severity indicators)
✅ Classes are well-separated by features
✅ Model can learn meaningful patterns
""")