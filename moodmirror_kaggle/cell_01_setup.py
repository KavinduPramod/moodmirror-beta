# ============================================================================
# CELL 1: Setup & Imports
# ============================================================================
# Purpose: Import all libraries, set random seed, configure device
# Runtime: ~30 seconds

import os
import warnings
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Core imports
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import Counter

# Transformers (BERT)
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# Metrics
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Progress bars
from tqdm.auto import tqdm

# ============================================================================
# Random Seed (reproducibility)
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Device Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("MOODMIRROR: Setup Complete")
print("="*70)
print(f"✓ Device: {device}")
print(f"✓ PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✓ All imports successful")
print("="*70)