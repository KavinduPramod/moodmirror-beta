"""
MoodMirror Dataset Balancing Script
Purpose: Balance the dataset to achieve ~45-55% class split
Output: balanced-mh-users.json (same structure, different labels)

Run this BEFORE training your model
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': '../dataset/reddit-mh-users.json',
    'output_file': '../dataset/reddit-mh-users-balanced.json',
    'backup_file': '../dataset/reddit-mh-users-backup.json',
    'random_seed': 42,
    'target_positive_ratio': 0.50,  # Target 50% positive class
    'visualization_dir': 'visualizations/',
}

# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================

def load_dataset(filepath):
    """Load the JSON dataset"""
    print(f"\n{'='*70}")
    print("STEP 1: Loading Dataset")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Successfully loaded {len(data)} users")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        print(f"  Current working directory: {os.getcwd()}")
        raise
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON format in {filepath}")
        raise

# ============================================================================
# STEP 3: CREATE INITIAL LABELS (Diagnostic)
# ============================================================================

def create_initial_labels(users_data):
    """Create labels using current strategy to understand distribution"""
    print(f"\n{'='*70}")
    print("STEP 2: Analyzing Current Label Distribution")
    print(f"{'='*70}")
    
    labels = []
    
    for user in users_data:
        features = user.get('features', {})
        
        # Current labeling (what you probably have)
        mh_participation = features.get('mental_health_participation', 0)
        avg_sentiment = features.get('avg_sentiment', 0)
        negative_ratio = features.get('negative_post_ratio', 0)
        total_posts = features.get('total_posts', 0)
        
        # Risk score calculation (your current approach)
        z_mh = (mh_participation - 0.5) / 0.2 if 0.5 > 0 else 0
        z_sentiment = (avg_sentiment - (-0.1)) / 0.3 if 0.3 > 0 else 0
        z_negative = (negative_ratio - 0.35) / 0.2 if 0.2 > 0 else 0
        
        risk_score = (z_mh + z_sentiment + z_negative) / 3
        label = 1 if risk_score > 0.5 else 0
        
        labels.append({
            'user_id': user['user_id'],
            'label': label,
            'risk_score': risk_score,
            'mh_participation': mh_participation,
            'avg_sentiment': avg_sentiment,
            'negative_ratio': negative_ratio,
            'total_posts': total_posts
        })
    
    # Analyze distribution
    label_counts = Counter([l['label'] for l in labels])
    total = sum(label_counts.values())
    
    print(f"\nCurrent Label Distribution (Diagnostic):")
    print(f"  Class 0 (Normal):  {label_counts[0]:5d} ({label_counts[0]/total*100:6.2f}%)")
    print(f"  Class 1 (At-risk): {label_counts[1]:5d} ({label_counts[1]/total*100:6.2f}%)")
    print(f"  Class Ratio (1/0): {label_counts[1]/label_counts[0]:.3f}")
    print(f"\n⚠️  If ratio < 0.2, dataset is severely imbalanced!")
    
    return pd.DataFrame(labels)

# ============================================================================
# STEP 4: CREATE BALANCED LABELS (Improved Strategy) - FIXED
# ============================================================================

def create_balanced_labels(users_data):
    """Create labels using improved, balanced strategy"""
    print(f"\n{'='*70}")
    print("STEP 3: Creating Balanced Labels")
    print(f"{'='*70}")
    
    labels_list = []
    
    # FIRST PASS: Standard criteria
    for user in users_data:
        features = user.get('features', {})
        
        mh_participation = features.get('mental_health_participation', 0)
        avg_sentiment = features.get('avg_sentiment', 0)
        negative_ratio = features.get('negative_post_ratio', 0)
        total_posts = features.get('total_posts', 0)
        negative_count = int(total_posts * negative_ratio) if total_posts > 0 else 0
        
        # STANDARD CRITERIA
        criteria = {
            'high_mh_participation': mh_participation >= 0.45,
            'negative_sentiment': avg_sentiment <= -0.03,
            'consistent_negativity': negative_ratio >= 0.28,
            'sufficient_signal': negative_count >= 8,
        }
        
        criteria_met = sum(criteria.values())
        is_at_risk = criteria_met >= 3
        
        labels_list.append({
            'user_id': user['user_id'],
            'label': 1 if is_at_risk else 0,
            'criteria_met': criteria_met,
            'mh_participation': mh_participation,
            'avg_sentiment': avg_sentiment,
            'negative_ratio': negative_ratio,
            'total_posts': total_posts,
            'negative_count': negative_count,
        })
    
    df_labels = pd.DataFrame(labels_list)
    
    # Check distribution
    label_counts = Counter(df_labels['label'].values)
    total = len(df_labels)
    pos_ratio = label_counts[1] / total
    
    print(f"\nInitial Balanced Label Distribution:")
    print(f"  Class 0 (Normal):  {label_counts[0]:5d} ({label_counts[0]/total*100:6.2f}%)")
    print(f"  Class 1 (At-risk): {label_counts[1]:5d} ({label_counts[1]/total*100:6.2f}%)")
    print(f"  Class Ratio (1/0): {label_counts[1]/label_counts[0]:.3f}")
    
    # If still imbalanced, adjust thresholds
    if pos_ratio < 0.35:
        print(f"\n⚠️  Still imbalanced. Lowering thresholds...")
        
        # RECALCULATE with lower thresholds
        labels_list_v2 = []
        for user in users_data:
            features = user.get('features', {})
            
            mh_participation = features.get('mental_health_participation', 0)
            avg_sentiment = features.get('avg_sentiment', 0)
            negative_ratio = features.get('negative_post_ratio', 0)
            total_posts = features.get('total_posts', 0)
            negative_count = int(total_posts * negative_ratio) if total_posts > 0 else 0
            
            # RELAXED CRITERIA
            criteria = {
                'high_mh_participation': mh_participation >= 0.40,
                'negative_sentiment': avg_sentiment <= 0.00,
                'consistent_negativity': negative_ratio >= 0.25,
                'sufficient_signal': negative_count >= 5,
            }
            
            criteria_met = sum(criteria.values())
            is_at_risk = criteria_met >= 2  # Lowered from 3 to 2
            
            labels_list_v2.append({
                'user_id': user['user_id'],
                'label': 1 if is_at_risk else 0,
                'criteria_met': criteria_met,
                'mh_participation': mh_participation,
                'avg_sentiment': avg_sentiment,
                'negative_ratio': negative_ratio,
                'total_posts': total_posts,
                'negative_count': negative_count,
            })
        
        df_labels = pd.DataFrame(labels_list_v2)
        label_counts = Counter(df_labels['label'].values)
        pos_ratio = label_counts[1] / total
        
        print(f"\nAdjusted Balanced Label Distribution:")
        print(f"  Class 0 (Normal):  {label_counts[0]:5d} ({label_counts[0]/total*100:6.2f}%)")
        print(f"  Class 1 (At-risk): {label_counts[1]:5d} ({label_counts[1]/total*100:6.2f}%)")
        print(f"  Class Ratio (1/0): {label_counts[1]/label_counts[0]:.3f}")
    
    if pos_ratio < 0.30:
        print(f"\n⚠️  WARNING: Still <30% positive. Consider:")
        print(f"     - Lowering avg_sentiment threshold further")
        print(f"     - Lowering negative_ratio to 0.20")
        print(f"     - Reducing criteria_met requirement to >= 1")
    
    return df_labels

# ============================================================================
# STEP 5: STRATIFIED SAMPLING (Optional - for further balancing)
# ============================================================================

def apply_stratified_sampling(df_labels, target_ratio=0.50, random_seed=42):
    """Optional: Balance classes via sampling"""
    print(f"\n{'='*70}")
    print("STEP 4: Stratified Sampling (Optional)")
    print(f"{'='*70}")
    
    np.random.seed(random_seed)
    
    # Separate classes
    pos_samples = df_labels[df_labels['label'] == 1]
    neg_samples = df_labels[df_labels['label'] == 0]
    
    total_target = len(df_labels)
    pos_target = int(total_target * target_ratio)
    neg_target = total_target - pos_target
    
    print(f"\nTarget distribution:")
    print(f"  Class 0 (Normal):  {neg_target} ({neg_target/total_target*100:.2f}%)")
    print(f"  Class 1 (At-risk): {pos_target} ({pos_target/total_target*100:.2f}%)")
    
    # Balance via under/oversampling
    if len(pos_samples) > pos_target:
        pos_sampled = pos_samples.sample(n=pos_target, random_state=random_seed)
        print(f"  → Undersampled positive class from {len(pos_samples)} to {pos_target}")
    else:
        pos_sampled = pos_samples.sample(n=pos_target, replace=True, random_state=random_seed)
        print(f"  → Oversampled positive class from {len(pos_samples)} to {pos_target}")
    
    if len(neg_samples) > neg_target:
        neg_sampled = neg_samples.sample(n=neg_target, random_state=random_seed)
        print(f"  → Undersampled negative class from {len(neg_samples)} to {neg_target}")
    else:
        neg_sampled = neg_samples.sample(n=neg_target, replace=True, random_state=random_seed)
        print(f"  → Oversampled negative class from {len(neg_samples)} to {neg_target}")
    
    df_balanced = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    final_counts = Counter(df_balanced['label'].values)
    print(f"\nFinal distribution after sampling:")
    print(f"  Class 0 (Normal):  {final_counts[0]} ({final_counts[0]/len(df_balanced)*100:.2f}%)")
    print(f"  Class 1 (At-risk): {final_counts[1]} ({final_counts[1]/len(df_balanced)*100:.2f}%)")
    
    return df_balanced

# ============================================================================
# STEP 6: ADD LABELS TO USERS AND SAVE
# ============================================================================

def add_labels_to_users(users_data, df_labels):
    """Add labels to user data"""
    print(f"\n{'='*70}")
    print("STEP 5: Adding Labels to User Data")
    print(f"{'='*70}")
    
    # Create dict from dataframe
    label_dict = {}
    for idx, row in df_labels.iterrows():
        label_dict[row['user_id']] = row['label']
    
    # Add labels to users
    for user in users_data:
        user_id = user['user_id']
        user['label'] = label_dict.get(user_id, 0)
    
    print(f"✓ Added labels to {len(users_data)} users")
    
    return users_data

# ============================================================================
# STEP 7: SAVE BALANCED DATASET
# ============================================================================

def save_balanced_dataset(users_data, output_filepath):
    """Save balanced dataset to JSON"""
    print(f"\n{'='*70}")
    print("STEP 6: Saving Balanced Dataset")
    print(f"{'='*70}")
    
    # Create backup
    try:
        if os.path.exists(CONFIG['input_file']):
            import shutil
            shutil.copy(CONFIG['input_file'], CONFIG['backup_file'])
            print(f"✓ Created backup: {CONFIG['backup_file']}")
    except Exception as e:
        print(f"⚠️  Backup failed (non-critical): {e}")
    
    # Save balanced dataset
    try:
        with open(output_filepath, 'w') as f:
            json.dump(users_data, f, indent=2)
        
        file_size = os.path.getsize(output_filepath) / (1024**2)  # MB
        print(f"✓ Saved balanced dataset: {output_filepath}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Total users: {len(users_data)}")
        
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        raise

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================

def create_visualizations(df_initial, df_balanced, output_dir='visualizations/'):
    """Create comparison visualizations"""
    print(f"\n{'='*70}")
    print("STEP 7: Creating Visualizations")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Initial distribution
    initial_counts = Counter(df_initial['label'].values)
    axes[0].bar(['Normal (0)', 'At-Risk (1)'], 
                [initial_counts[0], initial_counts[1]],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0].set_title('Initial Label Distribution\n(Imbalanced)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_ylim(0, max(initial_counts.values()) * 1.1)
    
    # Add percentage labels
    for i, (label, count) in enumerate([(0, initial_counts[0]), (1, initial_counts[1])]):
        pct = count / sum(initial_counts.values()) * 100
        axes[0].text(i, count + 50, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # Balanced distribution
    balanced_counts = Counter(df_balanced['label'].values)
    axes[1].bar(['Normal (0)', 'At-Risk (1)'], 
                [balanced_counts[0], balanced_counts[1]],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1].set_title('Balanced Label Distribution\n(≈50-50 Split)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_ylim(0, max(balanced_counts.values()) * 1.1)
    
    # Add percentage labels
    for i, (label, count) in enumerate([(0, balanced_counts[0]), (1, balanced_counts[1])]):
        pct = count / sum(balanced_counts.values()) * 100
        axes[1].text(i, count + 50, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}label_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_dir}label_distribution_comparison.png")
    
    plt.close()

# ============================================================================
# STEP 9: VERIFICATION REPORT
# ============================================================================

def create_verification_report(users_data, df_labels):
    """Create detailed verification report"""
    print(f"\n{'='*70}")
    print("STEP 8: Verification Report")
    print(f"{'='*70}")
    
    label_counts = Counter([u['label'] for u in users_data])
    total = len(users_data)
    
    print(f"\nDataset Statistics:")
    print(f"  Total users: {total}")
    print(f"  Normal users (label=0): {label_counts[0]} ({label_counts[0]/total*100:.2f}%)")
    print(f"  At-risk users (label=1): {label_counts[1]} ({label_counts[1]/total*100:.2f}%)")
    print(f"  Class balance ratio: {label_counts[1]/label_counts[0]:.3f}")
    
    # Statistics by class
    print(f"\nBehavioral Characteristics by Class:")
    
    for class_label in [0, 1]:
        class_name = "Normal" if class_label == 0 else "At-Risk"
        class_rows = df_labels[df_labels['label'] == class_label]
        
        print(f"\n  {class_name} Class (label={class_label}):")
        print(f"    Count: {len(class_rows)}")
        print(f"    Avg MH participation: {class_rows['mh_participation'].mean():.3f}")
        print(f"    Avg sentiment: {class_rows['avg_sentiment'].mean():.3f}")
        print(f"    Avg negative ratio: {class_rows['negative_ratio'].mean():.3f}")
        print(f"    Avg negative count: {class_rows['negative_count'].mean():.1f}")
        print(f"    Avg total posts: {class_rows['total_posts'].mean():.1f}")
    
    print(f"\n{'='*70}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("MOODMIRROR DATASET BALANCING SCRIPT")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load data
        users_data = load_dataset(CONFIG['input_file'])
        
        # Step 2: Analyze initial distribution
        df_initial = create_initial_labels(users_data)
        
        # Step 3: Create balanced labels
        df_balanced = create_balanced_labels(users_data)
        
        # Step 4: Apply stratified sampling (optional)
        print(f"\nApply stratified sampling? This will enforce exact 50-50 split.")
        apply_sampling = input("Enter 'yes' to apply sampling, or press Enter to skip: ").lower()
        
        if apply_sampling == 'yes':
            df_balanced = apply_stratified_sampling(df_balanced, target_ratio=0.50)
        
        # Step 5: Add labels to users
        users_data = add_labels_to_users(users_data, df_balanced)
        
        # Step 6: Save balanced dataset
        save_balanced_dataset(users_data, CONFIG['output_file'])
        
        # Step 7: Create visualizations
        try:
            create_visualizations(df_initial, df_balanced)
        except Exception as e:
            print(f"⚠️  Visualization failed (non-critical): {e}")
        
        # Step 8: Verification report
        create_verification_report(users_data, df_balanced)
        
        print(f"\n{'='*70}")
        print("✓ DATASET BALANCING COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"\nOutput file: {CONFIG['output_file']}")
        print(f"Ready for training with: reddit-mh-users-balanced.json")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()