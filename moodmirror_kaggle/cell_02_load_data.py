# ============================================================================
# CELL 2: Load & Explore Data
# ============================================================================
# Purpose: Load dataset from Kaggle, explore statistics, visualize distributions
# Runtime: ~2-3 minutes

print("="*70)
print("LOADING & EXPLORING DATA")
print("="*70)

# ============================================================================
# STEP 1: Load data from Kaggle
# ============================================================================

print("\n[1] Loading data from Kaggle...")

# Kaggle paths - adjust if your dataset name is different
DATA_PATH = '/kaggle/input/moodmirror-mh-data'

with open(f'{DATA_PATH}/reddit-mh-users-balanced.json', 'r') as f:
    users_data = json.load(f)

with open(f'{DATA_PATH}/reddit-mh-users-baseline.json', 'r') as f:
    population_baseline = json.load(f)

print(f"✓ Loaded {len(users_data)} users")
print(f"✓ Loaded population baseline")

# ============================================================================
# STEP 2: Display baseline statistics
# ============================================================================

print("\n[2] Population Baseline Statistics:")
print("-" * 70)
for key, value in population_baseline.items():
    print(f"  {key}: {value:.6f}")

# ============================================================================
# STEP 3: Create summary dataframe
# ============================================================================

print("\n[3] Creating summary dataframe...")

data_summary = []

for user in users_data:
    features = user['features']
    data_summary.append({
        'user_id': user['user_id'],
        'num_posts': len(user['posts']),
        'avg_sentiment': features['avg_sentiment'],
        'posting_frequency': features['posting_frequency'],
        'late_night_ratio': features['late_night_ratio'],
        'negative_post_ratio': features['negative_post_ratio'],
        'first_person_pronoun_ratio': features['first_person_pronoun_ratio'],
        'mental_health_participation': features['mental_health_participation'],
        'confidence_score': features['confidence_score'],
        'baseline_stability': features['baseline_stability'],
    })

df = pd.DataFrame(data_summary)
print(f"✓ Created dataframe with shape {df.shape}")

# ============================================================================
# STEP 4: Display statistics
# ============================================================================

print("\n[4] Dataset Statistics:")
print("-" * 70)
print(f"Total users: {len(df)}")
print(f"Total posts: {df['num_posts'].sum():,}")
print(f"Avg posts per user: {df['num_posts'].mean():.1f}")
print(f"Min posts per user: {df['num_posts'].min()}")
print(f"Max posts per user: {df['num_posts'].max()}")

print("\n[5] Feature Statistics:")
print("-" * 70)
print(df[['avg_sentiment', 'posting_frequency', 'late_night_ratio', 
          'negative_post_ratio', 'mental_health_participation']].describe())

# ============================================================================
# STEP 5: Visualizations
# ============================================================================

print("\n[6] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Data Distribution Exploration', fontsize=16, fontweight='bold')

# Posts per user
axes[0, 0].hist(df['num_posts'], bins=50, edgecolor='black', color='steelblue')
axes[0, 0].set_title('Posts per User')
axes[0, 0].set_xlabel('Number of Posts')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['num_posts'].mean(), color='red', linestyle='--', 
                    label=f"Mean: {df['num_posts'].mean():.1f}")
axes[0, 0].legend()

# Sentiment distribution
axes[0, 1].hist(df['avg_sentiment'], bins=50, edgecolor='black', color='green')
axes[0, 1].set_title('Average Sentiment')
axes[0, 1].set_xlabel('Sentiment Score')
axes[0, 1].axvline(df['avg_sentiment'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['avg_sentiment'].mean():.3f}")
axes[0, 1].legend()

# Posting frequency
axes[0, 2].hist(df['posting_frequency'], bins=50, edgecolor='black', color='orange')
axes[0, 2].set_title('Posting Frequency (posts/day)')
axes[0, 2].set_xlabel('Frequency')
axes[0, 2].axvline(df['posting_frequency'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['posting_frequency'].mean():.2f}")
axes[0, 2].legend()

# Late night ratio
axes[1, 0].hist(df['late_night_ratio'], bins=50, edgecolor='black', color='purple')
axes[1, 0].set_title('Late Night Posting Ratio')
axes[1, 0].set_xlabel('Ratio (0-1)')
axes[1, 0].axvline(df['late_night_ratio'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['late_night_ratio'].mean():.3f}")
axes[1, 0].legend()

# Negative post ratio
axes[1, 1].hist(df['negative_post_ratio'], bins=50, edgecolor='black', color='red')
axes[1, 1].set_title('Negative Post Ratio')
axes[1, 1].set_xlabel('Ratio (0-1)')
axes[1, 1].axvline(df['negative_post_ratio'].mean(), color='darkred', linestyle='--',
                    label=f"Mean: {df['negative_post_ratio'].mean():.3f}")
axes[1, 1].legend()

# MH participation
axes[1, 2].hist(df['mental_health_participation'], bins=50, edgecolor='black', color='brown')
axes[1, 2].set_title('Mental Health Participation')
axes[1, 2].set_xlabel('Ratio (0-1)')
axes[1, 2].axvline(df['mental_health_participation'].mean(), color='darkred', linestyle='--',
                    label=f"Mean: {df['mental_health_participation'].mean():.3f}")
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 6: Data quality check
# ============================================================================

print("\n[7] Data Quality Check:")
print("-" * 70)

# Missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"⚠ Found {missing.sum()} missing values")
else:
    print("✓ No missing values found")

# Invalid sentiment
invalid_sentiment = ((df['avg_sentiment'] < -1) | (df['avg_sentiment'] > 1)).sum()
if invalid_sentiment > 0:
    print(f"⚠ Found {invalid_sentiment} users with sentiment outside [-1, 1]")
else:
    print("✓ All sentiment values in valid range [-1, 1]")

# Users with few posts
few_posts = (df['num_posts'] < 5).sum()
print(f"ℹ Users with < 5 posts: {few_posts}")
print(f"ℹ Average baseline stability: {df['baseline_stability'].mean():.3f}")

# ============================================================================
# STEP 7: Save for next cell
# ============================================================================

print("\n[8] Saving data for Cell 3...")

with open('/tmp/moodmirror_df.pkl', 'wb') as f:
    pickle.dump(df, f)
with open('/tmp/moodmirror_users.pkl', 'wb') as f:
    pickle.dump(users_data, f)
with open('/tmp/moodmirror_baseline.pkl', 'wb') as f:
    pickle.dump(population_baseline, f)

print("\n" + "="*70)
print("✓ CELL 2 COMPLETE - Data loaded and explored")
print("="*70)