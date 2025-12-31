# ============================================================================
# CELL 5: Dataset Class & DataLoaders
# ============================================================================
# Purpose: Create PyTorch Dataset and DataLoaders for batching
# Runtime: ~1-2 minutes

print("="*70)
print("DATASET CLASS & DATALOADER CREATION")
print("="*70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n[1] Loading feature-engineered data...")

with open('/tmp/moodmirror_features.pkl', 'rb') as f:
    data_dict = pickle.load(f)

X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']
X_test = data_dict['X_test']
y_test = data_dict['y_test']

train_user_ids = data_dict['train_user_ids']
val_user_ids = data_dict['val_user_ids']
test_user_ids = data_dict['test_user_ids']
class_weights = data_dict['class_weights']
users_data = data_dict['users_data']
feature_cols = data_dict['feature_cols']

print(f"✓ Loaded feature matrices")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# ============================================================================
# STEP 2: Initialize BERT Tokenizer
# ============================================================================

print("\n[2] Initializing BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(f"✓ Tokenizer loaded")
print(f"  Vocab size: {len(tokenizer.vocab):,} tokens")
print(f"  [CLS]: {tokenizer.cls_token_id}, [SEP]: {tokenizer.sep_token_id}")

# ============================================================================
# STEP 3: Define Dataset Class
# ============================================================================

class MoodMirrorDataset(Dataset):
    """
    PyTorch Dataset for MoodMirror.
    
    For each user:
    - Retrieves their posts
    - Concatenates into a single text sequence
    - Tokenizes with BERT
    - Returns: input_ids, attention_mask, label, features
    """
    
    def __init__(self, user_ids, labels, users_data, X, tokenizer, max_length=512):
        self.user_ids = user_ids
        self.labels = labels
        self.X = X
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_dict = {u['user_id']: u for u in users_data}
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user = self.user_dict[user_id]
        label = self.labels[idx]
        features = self.X[idx] if self.X is not None else None
        
        # Get posts
        posts = user['posts']
        post_texts = [p['text'] for p in posts]
        
        # Concatenate posts (up to 20)
        max_posts = 20
        selected_posts = post_texts[:max_posts]
        combined_text = " [SEP] ".join(selected_posts)
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if features is not None:
            features_tensor = torch.tensor(features, dtype=torch.float32)
        else:
            features_tensor = torch.tensor([], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor,
            'features': features_tensor,
            'user_id': user_id,
            'num_posts': len(posts)
        }


print("\n[3] MoodMirrorDataset class defined")
print(f"✓ Concatenates up to 20 posts per user")
print(f"✓ Tokenizes with BERT (max 512 tokens)")

# ============================================================================
# STEP 4: Create Dataset Instances
# ============================================================================

print("\n[4] Creating dataset instances...")

train_dataset = MoodMirrorDataset(
    user_ids=train_user_ids,
    labels=y_train,
    users_data=users_data,
    X=X_train,
    tokenizer=tokenizer,
    max_length=512
)

val_dataset = MoodMirrorDataset(
    user_ids=val_user_ids,
    labels=y_val,
    users_data=users_data,
    X=X_val,
    tokenizer=tokenizer,
    max_length=512
)

test_dataset = MoodMirrorDataset(
    user_ids=test_user_ids,
    labels=y_test,
    users_data=users_data,
    X=X_test,
    tokenizer=tokenizer,
    max_length=512
)

print(f"✓ Train dataset: {len(train_dataset)} users")
print(f"✓ Val dataset:   {len(val_dataset)} users")
print(f"✓ Test dataset:  {len(test_dataset)} users")

# ============================================================================
# STEP 5: Collate Function
# ============================================================================

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['label'] for item in batch]
    features_list = [item['features'] for item in batch]
    user_ids_list = [item['user_id'] for item in batch]
    num_posts_list = [item['num_posts'] for item in batch]
    
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)
    
    if features_list[0].numel() > 0:
        features = torch.stack(features_list)
    else:
        features = None
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'features': features,
        'user_ids': user_ids_list,
        'num_posts': num_posts_list
    }

print("\n[5] Collate function defined")

# ============================================================================
# STEP 6: Create DataLoaders
# ============================================================================

print("\n[6] Creating DataLoaders...")

BATCH_SIZE = 32

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print(f"✓ Train DataLoader: {len(train_dataloader)} batches")
print(f"✓ Val DataLoader:   {len(val_dataloader)} batches")
print(f"✓ Test DataLoader:  {len(test_dataloader)} batches")

# ============================================================================
# STEP 7: Inspect Sample Batch
# ============================================================================

print("\n[7] Inspecting first training batch...")

sample_batch = next(iter(train_dataloader))

print(f"\n✓ Sample batch structure:")
print(f"  input_ids shape:      {sample_batch['input_ids'].shape}")
print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
print(f"  labels shape:         {sample_batch['label'].shape}")
print(f"  features shape:       {sample_batch['features'].shape if sample_batch['features'] is not None else 'None'}")

# ============================================================================
# STEP 8: Save DataLoaders
# ============================================================================

print("\n[8] Saving DataLoaders for Cell 6...")

dataloader_dict = {
    'train_dataloader': train_dataloader,
    'val_dataloader': val_dataloader,
    'test_dataloader': test_dataloader,
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset,
    'tokenizer': tokenizer,
    'class_weights': class_weights,
    'batch_size': BATCH_SIZE
}

with open('/tmp/moodmirror_dataloaders.pkl', 'wb') as f:
    pickle.dump(dataloader_dict, f)

print(f"✓ Saved DataLoaders and metadata")

print("\n" + "="*70)
print("✓ CELL 5 COMPLETE - DataLoaders ready for training")
print("="*70)
