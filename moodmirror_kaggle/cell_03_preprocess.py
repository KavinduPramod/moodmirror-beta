# ============================================================================
# CELL 3: Data Preprocessing & Cleaning
# ============================================================================
# Purpose: Clean and validate data before model training
# Runtime: ~1-2 minutes

print("="*70)
print("DATA PREPROCESSING & CLEANING")
print("="*70)

# ============================================================================
# STEP 1: Load data from Cell 2
# ============================================================================

print("\n[1] Loading data from Cell 2...")

with open('/tmp/moodmirror_df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('/tmp/moodmirror_users.pkl', 'rb') as f:
    users_data = pickle.load(f)

with open('/tmp/moodmirror_baseline.pkl', 'rb') as f:
    population_baseline = pickle.load(f)

print(f"✓ Loaded {len(df)} users")
print(f"✓ Loaded {sum(len(u['posts']) for u in users_data):,} total posts")

# ============================================================================
# STEP 2: Remove users with 0 posts
# ============================================================================

print("\n[2] Check 1: Remove users with 0 posts...")

before = len(users_data)
users_data = [u for u in users_data if len(u['posts']) > 0]
df = df[df['num_posts'] > 0]
removed = before - len(users_data)

print(f"✓ Removed {removed} users with 0 posts")
print(f"  Users remaining: {len(users_data)}")

# ============================================================================
# STEP 3: Validate features (no nulls)
# ============================================================================

print("\n[3] Check 2: Validate features (no nulls)...")

invalid_users = []

for i, user in enumerate(users_data):
    features = user['features']
    required_keys = ['avg_sentiment', 'posting_frequency', 'late_night_ratio',
                     'negative_post_ratio', 'first_person_pronoun_ratio',
                     'mental_health_participation', 'confidence_score', 'baseline_stability']
    
    if any(key not in features or features[key] is None for key in required_keys):
        invalid_users.append(i)

print(f"✓ Found {len(invalid_users)} users with null features")
if invalid_users:
    users_data = [u for i, u in enumerate(users_data) if i not in invalid_users]
    print(f"  Removed {len(invalid_users)} users")
    print(f"  Users remaining: {len(users_data)}")

# Sync dataframe
user_ids_to_keep = set(u['user_id'] for u in users_data)
df = df[df['user_id'].isin(user_ids_to_keep)]

# ============================================================================
# STEP 4: Remove duplicate posts
# ============================================================================

print("\n[4] Check 3: Remove duplicate posts...")

total_duplicates = 0

for user in users_data:
    posts = user['posts']
    texts = [p['text'] for p in posts]
    
    unique_texts = set(texts)
    duplicates = len(texts) - len(unique_texts)
    total_duplicates += duplicates
    
    # Keep only unique posts
    seen = set()
    unique_posts = []
    for post in posts:
        if post['text'] not in seen:
            seen.add(post['text'])
            unique_posts.append(post)
    
    user['posts'] = unique_posts

print(f"✓ Removed {total_duplicates} duplicate posts")
print(f"  Posts after: {sum(len(u['posts']) for u in users_data):,}")

# ============================================================================
# STEP 5: Validate feature ranges
# ============================================================================

print("\n[5] Check 4: Validate feature ranges...")

invalid_features = 0

for user in users_data:
    features = user['features']
    
    # Ratios should be 0-1
    ratio_features = ['late_night_ratio', 'negative_post_ratio', 
                      'first_person_pronoun_ratio', 'mental_health_participation',
                      'confidence_score', 'baseline_stability']
    
    for feature in ratio_features:
        if feature in features:
            val = features[feature]
            if val < 0 or val > 1:
                features[feature] = max(0, min(1, val))
                invalid_features += 1
    
    # Sentiment should be -1 to 1
    if 'avg_sentiment' in features:
        val = features['avg_sentiment']
        if val < -1 or val > 1:
            features['avg_sentiment'] = max(-1, min(1, val))
            invalid_features += 1
    
    # Posting frequency should be > 0
    if features['posting_frequency'] <= 0:
        features['posting_frequency'] = 0.01
        invalid_features += 1

print(f"✓ Fixed {invalid_features} invalid feature values")

# ============================================================================
# STEP 6: Remove very short posts
# ============================================================================

print("\n[6] Check 5: Remove posts < 10 characters...")

short_posts = 0

for user in users_data:
    posts = user['posts']
    long_posts = [p for p in posts if len(p['text']) >= 10]
    short_posts += len(posts) - len(long_posts)
    user['posts'] = long_posts

print(f"✓ Removed {short_posts} posts with < 10 characters")
print(f"  Posts remaining: {sum(len(u['posts']) for u in users_data):,}")

# Remove users who now have 0 posts
before_final = len(users_data)
users_data = [u for u in users_data if len(u['posts']) > 0]
df = df[df['user_id'].isin(set(u['user_id'] for u in users_data))]
removed_final = before_final - len(users_data)

if removed_final > 0:
    print(f"⚠ Removed {removed_final} users with 0 posts after filtering")

# ============================================================================
# STEP 7: Summary
# ============================================================================

print("\n" + "="*70)
print("PREPROCESSING SUMMARY")
print("="*70)

print(f"\n✓ Final Dataset:")
print(f"  Users: {len(users_data)}")
print(f"  Posts: {sum(len(u['posts']) for u in users_data):,}")
print(f"  Avg posts/user: {sum(len(u['posts']) for u in users_data) / len(users_data):.1f}")

# ============================================================================
# STEP 8: Save cleaned data
# ============================================================================

print("\n[8] Saving cleaned data...")

with open('/tmp/moodmirror_users_clean.pkl', 'wb') as f:
    pickle.dump(users_data, f)

with open('/tmp/moodmirror_df_clean.pkl', 'wb') as f:
    pickle.dump(df, f)

with open('/tmp/moodmirror_baseline_clean.pkl', 'wb') as f:
    pickle.dump(population_baseline, f)

print("✓ Cleaned data saved for Cell 4")

print("\n" + "="*70)
print("✓ CELL 3 COMPLETE - Data preprocessed and validated")
print("="*70)
