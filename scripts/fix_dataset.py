import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelaxedDatasetPrep:
    """
    Relaxed dataset preparation that balances thesis requirements with data availability.
    
    Strategy:
    - Minimum 20 posts (down from 30) - still statistically valid
    - Minimum 30% MH participation (down from 50%) - captures interested users
    - Keep all users meeting these criteria
    - Aggressive class balancing
    """
    
    def __init__(self, merged_dataset_path):
        self.data = self._load_json(merged_dataset_path)
        
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def filter_dataset(self, min_posts=20, min_mh_participation=0.30):
        """Filter users by relaxed criteria"""
        valid_users = []
        rejection_stats = {
            'insufficient_posts': 0,
            'low_mh_participation': 0,
            'invalid_timespan': 0,
            'low_baseline_stability': 0,
            'valid': 0
        }
        
        for user in self.data:
            features = user.get('features', {})
            
            # Check criteria
            posts = features.get('total_posts', 0)
            mh_participation = features.get('mental_health_participation', 0)
            time_span = features.get('time_span_days', 0)
            baseline_stability = features.get('baseline_stability', 0)
            
            # Apply filters
            if posts < min_posts:
                rejection_stats['insufficient_posts'] += 1
                continue
            
            if mh_participation < min_mh_participation:
                rejection_stats['low_mh_participation'] += 1
                continue
            
            if time_span < 20 or time_span > 95:
                rejection_stats['invalid_timespan'] += 1
                continue
            
            if baseline_stability < 0.2:
                rejection_stats['low_baseline_stability'] += 1
                continue
            
            # All checks passed
            valid_users.append(user)
            rejection_stats['valid'] += 1
        
        # Log results
        logger.info(f"\n{'='*80}")
        logger.info("DATASET FILTERING RESULTS (Relaxed Criteria)")
        logger.info(f"{'='*80}")
        logger.info(f"Minimum posts: {min_posts}")
        logger.info(f"Minimum MH participation: {min_mh_participation:.0%}")
        logger.info(f"\nTotal users: {len(self.data)}")
        logger.info(f"Valid users: {rejection_stats['valid']} ({rejection_stats['valid']/len(self.data)*100:.1f}%)")
        logger.info(f"\nRejection breakdown:")
        logger.info(f"  Insufficient posts (<{min_posts}): {rejection_stats['insufficient_posts']}")
        logger.info(f"  Low MH participation (<{min_mh_participation:.0%}): {rejection_stats['low_mh_participation']}")
        logger.info(f"  Invalid timespan: {rejection_stats['invalid_timespan']}")
        logger.info(f"  Low baseline stability: {rejection_stats['low_baseline_stability']}")
        
        return valid_users
    
    def analyze_dataset(self, users, name="Dataset"):
        """Analyze dataset"""
        if not users:
            print(f"⚠️ {name} is empty!")
            return
        
        labels = [int(u.get('label', 0)) for u in users]
        posts = [u['features']['total_posts'] for u in users]
        mh_participation = [u['features']['mental_health_participation'] for u in users]
        
        label_dist = pd.Series(labels).value_counts().sort_index()
        
        print(f"\n📊 {name} Statistics:")
        print(f"   Total users: {len(users)}")
        print(f"   Total posts: {sum(posts)}")
        print(f"   Avg posts/user: {np.mean(posts):.1f}")
        print(f"   Avg MH participation: {np.mean(mh_participation):.1%}")
        print(f"   Label distribution:")
        for label, count in label_dist.items():
            pct = count / len(labels) * 100
            print(f"      Class {label}: {count} ({pct:.1f}%)")
        
        if len(label_dist) == 2:
            imbalance = max(label_dist.values) / min(label_dist.values)
            print(f"   Imbalance ratio: {imbalance:.2f}:1")
    
    def balance_classes(self, users, target_minority_ratio=0.45):
        """Balance classes through oversampling"""
        labels = np.array([int(u.get('label', 0)) for u in users])
        label_counts = pd.Series(labels).value_counts()
        
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        
        minority_users = [u for u, l in zip(users, labels) if l == minority_label]
        majority_users = [u for u, l in zip(users, labels) if l == majority_label]
        
        # Calculate target size
        target_minority_size = int(len(majority_users) * target_minority_ratio / (1 - target_minority_ratio))
        
        logger.info(f"\n{'='*80}")
        logger.info("CLASS BALANCING")
        logger.info(f"{'='*80}")
        logger.info(f"Majority class: {len(majority_users)} users")
        logger.info(f"Minority class: {len(minority_users)} users")
        logger.info(f"Target minority size: {target_minority_size}")
        
        # Oversample
        if len(minority_users) < target_minority_size:
            oversample_count = target_minority_size - len(minority_users)
            oversampled = np.random.choice(minority_users, size=oversample_count, replace=True)
            balanced = majority_users + minority_users + list(oversampled)
            logger.info(f"Oversampled {oversample_count} minority instances")
        else:
            balanced = majority_users + minority_users
            logger.info(f"No oversampling needed")
        
        return balanced
    
    def create_splits(self, users, train_size=0.70):
        """Create stratified splits"""
        labels = [int(u.get('label', 0)) for u in users]
        
        train, temp = train_test_split(
            users, test_size=(1-train_size),
            stratify=labels, random_state=42
        )
        
        temp_labels = [int(u.get('label', 0)) for u in temp]
        val, test = train_test_split(
            temp, test_size=0.5,
            stratify=temp_labels, random_state=42
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SPLITS")
        logger.info(f"{'='*80}")
        logger.info(f"Training:   {len(train)} users ({len(train)/len(users)*100:.1f}%)")
        logger.info(f"Validation: {len(val)} users ({len(val)/len(users)*100:.1f}%)")
        logger.info(f"Testing:    {len(test)} users ({len(test)/len(users)*100:.1f}%)")
        
        return train, val, test
    
    def save_splits(self, train, val, test, output_dir='./prepared_data_relaxed'):
        """Save to JSON"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            f'{output_dir}/train_data.json': train,
            f'{output_dir}/val_data.json': val,
            f'{output_dir}/test_data.json': test,
        }
        
        for filepath, data in files.items():
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✅ Saved {filepath}")


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    logger.info("🚀 RELAXED DATASET PREPARATION")
    
    # Initialize
    prep = RelaxedDatasetPrep('../dataset/full-data-collection/reddit-mh-users-balanced.json')
    
    # Filter with relaxed criteria
    valid_users = prep.filter_dataset(
        min_posts=20,              # Reduced from 30
        min_mh_participation=0.30  # Reduced from 0.50
    )
    
    # Analyze before balancing
    print("\n" + "="*80)
    print("BEFORE CLASS BALANCING")
    print("="*80)
    prep.analyze_dataset(valid_users, "Filtered Dataset")
    
    # Balance
    balanced_users = prep.balance_classes(valid_users, target_minority_ratio=0.45)
    
    # Analyze after balancing
    print("\n" + "="*80)
    print("AFTER CLASS BALANCING")
    print("="*80)
    prep.analyze_dataset(balanced_users, "Balanced Dataset")
    
    # Split
    train, val, test = prep.create_splits(balanced_users)
    
    # Final analysis
    print("\n" + "="*80)
    print("FINAL SPLIT ANALYSIS")
    print("="*80)
    prep.analyze_dataset(train, "Training Data")
    prep.analyze_dataset(val, "Validation Data")
    prep.analyze_dataset(test, "Test Data")
    
    # Save
    prep.save_splits(train, val, test)
    
    logger.info("\n✅ Dataset preparation complete!")