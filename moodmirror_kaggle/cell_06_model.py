# ============================================================================
# CELL 6 UPGRADED: Model Architecture (BERT Fine-tuned + BiLSTM 512)
# ============================================================================
# Purpose: Unfreeze BERT layers + increase BiLSTM capacity for better accuracy
# Runtime: ~1-2 minutes
#
# KEY CHANGES:
# 1. Fine-tune BERT layers 10-11 (last 2 transformer layers)
# 2. Increase BiLSTM hidden size from 256 → 512
# 3. Increase overall model capacity
# 4. Better feature extraction from BERT

print("="*70)
print("MODEL ARCHITECTURE UPGRADE (BERT Fine-tuned + BiLSTM 512)")
print("="*70)

# ============================================================================
# STEP 1: Load DataLoaders
# ============================================================================

print("\n[1] Loading DataLoaders from Cell 5...")

with open('/tmp/moodmirror_dataloaders.pkl', 'rb') as f:
    dataloader_dict = pickle.load(f)

train_dataloader = dataloader_dict['train_dataloader']
val_dataloader = dataloader_dict['val_dataloader']
test_dataloader = dataloader_dict['test_dataloader']
tokenizer = dataloader_dict['tokenizer']
class_weights = dataloader_dict['class_weights']
batch_size = dataloader_dict['batch_size']

print(f"✓ Loaded DataLoaders")
print(f"  Train: {len(train_dataloader)} batches")
print(f"  Val: {len(val_dataloader)} batches")

# ============================================================================
# STEP 2: Load BERT Model (Fine-tuned)
# ============================================================================

print("\n[2] Loading pre-trained BERT model...")

bert_model = BertModel.from_pretrained('bert-base-uncased')

# Unfreeze BERT layers 10-11 (last 2 transformer layers)
for name, param in bert_model.named_parameters():
    if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print(f"✓ BERT model loaded")
print(f"  Architecture: bert-base-uncased")
print(f"  Hidden size: 768 dimensions")
print(f"  Fine-tuned layers: 10-11")
print(f"  Total BERT parameters: {sum(p.numel() for p in bert_model.parameters()):,}")
print(f"  Trainable BERT parameters: {sum(p.numel() for p in bert_model.parameters() if p.requires_grad):,}")

# ============================================================================
# STEP 3: Define MoodMirrorModel Class (with BiLSTM 512)
# ============================================================================

class MoodMirrorModel(nn.Module):
    """
    MoodMirror: BERT + BiLSTM for mental health risk detection
    
    Architecture:
    1. BERT Encoder (fine-tuned): Converts tokens → 768-dim embeddings
    2. BiLSTM Layer (trainable): 512 hidden × 2 directions = 1024-dim output
    3. Global Max Pool: Reduces sequence to single vector
    4. Dropout: Regularization
    5. Dense Layers: 1032 → 256 → 128 → 1
    6. Sigmoid: Converts to probability (0-1)
    
    KEY CHANGES:
    - Fine-tune BERT layers 10-11
    - BiLSTM hidden size increased to 512
    """
    
    def __init__(
        self,
        bert_model,
        hidden_size=512,  # Increased BiLSTM hidden size
        lstm_layers=1,
        dropout=0.3,
        num_features=8
    ):
        super(MoodMirrorModel, self).__init__()
        
        self.bert = bert_model
        self.bert_output_size = 768
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # BiLSTM layer - BIDIRECTIONAL = TRUE
        # Output size = hidden_size * 2 = 1024 (forward + backward)
        self.lstm = nn.LSTM(
            input_size=self.bert_output_size,  # 768
            hidden_size=hidden_size,            # 512
            num_layers=lstm_layers,             # 1
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True  # ✅ BIDIRECTIONAL LSTM
        )
        
        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers
        # Input: hidden_size * 2 (BiLSTM) + num_features = 1024 + 8 = 1032
        lstm_output_size = hidden_size * 2  # BiLSTM doubles the output
        
        self.dense1 = nn.Linear(lstm_output_size + num_features, 256)
        self.dense1_activation = nn.ReLU()
        self.dense1_dropout = nn.Dropout(dropout)
        
        self.dense2 = nn.Linear(256, 128)
        self.dense2_activation = nn.ReLU()
        self.dense2_dropout = nn.Dropout(dropout)
        
        self.dense_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, features=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: (batch_size, seq_length) token IDs
            attention_mask: (batch_size, seq_length) attention mask
            features: (batch_size, 8) behavioral features
            
        Returns:
            output: (batch_size, 1) probability predictions
        """
        
        # STEP 1: BERT Encoder
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Shape: (batch_size, seq_length, 768)
        sequence_output = bert_output.last_hidden_state
        
        # STEP 2: BiLSTM Layer
        # Output shape: (batch_size, seq_length, hidden_size * 2) = (B, L, 1024)
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        
        # STEP 3: Global Max Pooling
        # Shape: (batch_size, 1024)
        pooled = torch.max(lstm_output, dim=1)[0]
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # STEP 4: Concatenate with Features
        # Shape: (batch_size, 1024 + 8) = (batch_size, 1032)
        if features is not None and features.numel() > 0:
            combined = torch.cat([pooled, features], dim=1)
        else:
            combined = pooled
        
        # STEP 5: Dense Layers
        dense1 = self.dense1(combined)
        dense1 = self.dense1_activation(dense1)
        dense1 = self.dense1_dropout(dense1)
        
        dense2 = self.dense2(dense1)
        dense2 = self.dense2_activation(dense2)
        dense2 = self.dense2_dropout(dense2)
        
        logits = self.dense_out(dense2)
        
        # STEP 6: Sigmoid
        output = logits # Removed Sigmoid
        
        return output

print("\n[3] MoodMirrorModel class upgraded")
print(f"✓ Architecture components:")
print(f"  1. BERT (fine-tuned): tokens → 768-dim embeddings")
print(f"  2. BiLSTM (trainable): embeddings → 1024-dim (512 × 2 directions)")
print(f"  3. Max Pooling: sequence → single vector")
print(f"  4. Dense1: 1032 → 256 (ReLU + Dropout)")
print(f"  5. Dense2: 256 → 128 (ReLU + Dropout)")
print(f"  6. Output: 128 → 1 (Sigmoid)")

# ============================================================================
# STEP 4: Initialize Model
# ============================================================================

print("\n[4] Initializing model...")

model = MoodMirrorModel(
    bert_model=bert_model,
    hidden_size=512,
    lstm_layers=1,
    dropout=0.3,
    num_features=8
)

model = model.to(device)

print(f"✓ Model initialized on device: {device}")

# ============================================================================
# STEP 5: Count Parameters
# ============================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)

print(f"\n[5] Model parameters:")
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Trainable parameters: {trainable_params:,}")
print(f"✓ Frozen parameters (BERT): {total_params - trainable_params:,}")

print(f"\nBreakdown:")
print(f"  BERT: {sum(p.numel() for p in bert_model.parameters()):,} (frozen)")
print(f"  BiLSTM: {sum(p.numel() for p in model.lstm.parameters()):,} (trainable)")
print(f"  Dense: {sum(p.numel() for p in model.dense1.parameters()) + sum(p.numel() for p in model.dense2.parameters()) + sum(p.numel() for p in model.dense_out.parameters()):,} (trainable)")

# ============================================================================
# STEP 6: Test Forward Pass
# ============================================================================

print("\n[6] Testing forward pass...")

sample_batch = next(iter(train_dataloader))

sample_input_ids = sample_batch['input_ids'].to(device)
sample_attention_mask = sample_batch['attention_mask'].to(device)
sample_features = sample_batch['features'].to(device) if sample_batch['features'] is not None else None
sample_labels = sample_batch['label'].to(device)

with torch.no_grad():
    predictions = model(
        input_ids=sample_input_ids,
        attention_mask=sample_attention_mask,
        features=sample_features
    )

print(f"✓ Forward pass successful")
print(f"  Input: input_ids {sample_input_ids.shape}, features {sample_features.shape if sample_features is not None else 'None'}")
print(f"  Output: {predictions.shape}")
print(f"  Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")

print(f"Output min: {predictions.min():.2f}")  # Should be negative
print(f"Output max: {predictions.max():.2f}")  # Should be positive
print(f"Output range: should be unbounded (-∞ to +∞) for logits")

# ============================================================================
# STEP 7: Save Model
# ============================================================================

print("\n[7] Saving model for Cell 7...")

model_dict = {
    'model': model,
    'tokenizer': tokenizer,
    'device': device,
    'class_weights': class_weights,
    'batch_size': batch_size
}

with open('/tmp/moodmirror_model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

torch.save(model.state_dict(), '/tmp/moodmirror_model_initial.pt')

print(f"✓ Model saved")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("MODEL ARCHITECTURE SUMMARY")
print("="*70)

print(f"\n✓ MoodMirrorModel: BERT + BiLSTM (BIDIRECTIONAL!)")
print(f"  Input: Reddit posts (tokenized)")
print(f"  Output: Probability of at-risk (0-1)")

print(f"\n✓ Architecture Flow:")
print(f"  Posts → BERT (768) → BiLSTM (1024) → Pool →")
print(f"  Concat Features (+8) → Dense(256) → Dense(128) → Output(1) → Sigmoid")

print(f"\n✓ Key Fix Applied:")
print(f"  ✅ bidirectional=True (BiLSTM, not LSTM)")
print(f"  ✅ Output size: 512 × 2 = 1024 dimensions")
print(f"  ✅ Dense layer input: 1024 + 8 = 1032 dimensions")

print("\n" + "="*70)
print("✓ CELL 6 COMPLETE - BiLSTM model architecture ready")
print("="*70)
