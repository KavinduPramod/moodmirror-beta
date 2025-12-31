# MoodMirror - Modular Kaggle Notebook

This folder contains the MoodMirror notebook split into separate Python files for easy debugging on Kaggle.

## How to Use

1. **Upload to Kaggle**: Copy each cell file in order to your Kaggle notebook
2. **Debug**: If you get an error, you know exactly which cell to fix
3. **Run in Order**: Execute cells 1-11 sequentially

## Cell Overview

| File | Purpose | Runtime |
|------|---------|---------|
| `cell_01_setup.py` | Imports & device setup | ~30s |
| `cell_02_load_data.py` | Load & explore data | ~2min |
| `cell_03_preprocess.py` | Clean & validate data | ~1min |
| `cell_04_features.py` | Create labels & splits | ~1min |
| `cell_05_dataset.py` | PyTorch Dataset & DataLoaders | ~1min |
| `cell_06_model.py` | Model architecture (BiLSTM) | ~1min |
| `cell_07_training_setup.py` | Loss, optimizer, scheduler | ~30s |
| `cell_08_training_loop.py` | Train the model | ~30-60min |
| `cell_09_evaluation.py` | Test set metrics | ~5min |
| `cell_10_visualizations.py` | Plots & analysis | ~2min |
| `cell_11_inference.py` | Inference functions | ~2min |

## Key Fixes Applied

1. ✅ **BiLSTM** - Changed from LSTM to Bidirectional LSTM in cell_06
2. ✅ **MH-Based Labels** - Using mental health participation for labels in cell_04
3. ✅ **Clean Code** - Removed duplicate code, fixed formatting

## Expected Performance

- **F1 Score**: 0.80-0.85
- **Recall**: 0.85-0.90
- **AUC-ROC**: 0.85-0.90

## Kaggle Dataset Path

Make sure your Kaggle dataset is at:
```
/kaggle/input/moodmirror-mh-data/reddit-mh-users.json
/kaggle/input/moodmirror-mh-data/reddit-mh-users-baseline.json
```
