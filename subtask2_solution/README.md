# Subtask 2 Solution - Polarization Type Classification

Multi-label classification for detecting types of polarization in text.

## Overview

Subtask 2 is a multi-label classification task where each text can have multiple polarization types:
- **Political** - polarization related to political views
- **Racial/Ethnic** - polarization based on race or ethnicity
- **Religious** - polarization related to religion
- **Gender/Sexual** - polarization based on gender or sexual orientation
- **Other** - other types of polarization

## Files

- `merge_data_subtask2.py` - Merge all language CSV files into single train/dev files
- `train_merged_subtask2.py` - Train multi-label classification model
- `test_subtask2.py` - Evaluate model and generate predictions

## Quick Start

### 1. Merge Training Data

Combine all language files into a single merged dataset:

```bash
python merge_data_subtask2.py
```

This will create:
- `data/train_merged.csv` - All training data merged
- `data/dev_merged.csv` - All development data merged

### 2. Train Model

Train a multi-label classification model on the merged data:

```bash
python train_merged_subtask2.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --train_file ./data/train_merged.csv \
    --output_dir ./models/subtask2_model \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5
```

**Arguments:**
- `--model_name`: Pretrained model (default: Qwen/Qwen2.5-0.5B)
- `--train_file`: Path to merged training data
- `--output_dir`: Where to save the trained model
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Max sequence length (default: 128)
- `--val_split`: Validation split ratio (default: 0.1)

### 3. Test/Evaluate Model

Evaluate the trained model:

```bash
# Evaluate on merged dev data
python test_subtask2.py \
    --model_path ./models/subtask2_model/final_model \
    --merged_data ./data/dev_merged.csv \
    --output_dir ./results \
    --mode evaluate

# Evaluate by language
python test_subtask2.py \
    --model_path ./models/subtask2_model/final_model \
    --data_dir c:/hishab/semeval-2026/dev_phase/subtask2 \
    --output_dir ./results \
    --mode by_language

# Do both
python test_subtask2.py \
    --model_path ./models/subtask2_model/final_model \
    --merged_data ./data/dev_merged.csv \
    --data_dir c:/hishab/semeval-2026/dev_phase/subtask2 \
    --output_dir ./results \
    --mode all
```

**Arguments:**
- `--model_path`: Path to trained model directory
- `--data_dir`: Root directory with subtask2 data (train/dev folders)
- `--merged_data`: Path to merged CSV for evaluation (optional)
- `--output_dir`: Where to save results
- `--mode`: Testing mode (evaluate/by_language/submission/all)
- `--threshold`: Decision threshold for predictions (default: 0.5)
- `--split`: Which split to evaluate (train/dev, default: dev)

## Data Format

### Input (Training/Dev)
CSV files with columns:
- `id`: Unique identifier
- `text`: Input text
- `political`: 0 or 1
- `racial/ethnic`: 0 or 1
- `religious`: 0 or 1
- `gender/sexual`: 0 or 1
- `other`: 0 or 1
- `language`: Language code (added during merge)

### Output (Predictions)
CSV file with:
- `id`: Unique identifier
- `political`: Predicted label (0 or 1)
- `racial/ethnic`: Predicted label (0 or 1)
- `religious`: Predicted label (0 or 1)
- `gender/sexual`: Predicted label (0 or 1)
- `other`: Predicted label (0 or 1)

## Evaluation Metrics

The model is evaluated using:
- **F1 Score (Macro)** - Main metric for competition
- **F1 Score (Micro)** - Overall performance across all labels
- **Accuracy** - Exact match ratio (all labels must match)
- **Hamming Loss** - Average label-wise error
- **Per-label F1** - F1 score for each polarization type

## Model Architecture

- Uses transformer-based models (e.g., Qwen, BERT, RoBERTa)
- Multi-label classification head with sigmoid activation
- BCEWithLogitsLoss for training
- 0.5 threshold for binary predictions (adjustable)

## Tips for Better Performance

1. **Experiment with models**: Try different pretrained models
   - Multilingual: XLM-RoBERTa, mBERT
   - English-focused: RoBERTa, DeBERTa
   - Efficient: Qwen2.5, DistilBERT

2. **Tune hyperparameters**:
   - Learning rate (1e-5 to 5e-5)
   - Batch size (8, 16, 32)
   - Max sequence length (128, 256, 512)
   - Number of epochs (3-10)

3. **Adjust decision threshold**:
   - Default is 0.5
   - Try 0.3-0.7 based on validation performance
   - Can be set per-label for better balance

4. **Data augmentation**:
   - Back-translation
   - Synonym replacement
   - Handle class imbalance

5. **Ensemble methods**:
   - Train multiple models with different seeds
   - Combine predictions by voting or averaging

## Requirements

```
pandas
numpy
torch
transformers
scikit-learn
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- Subtask 2 is a multi-label task, unlike subtask 1 which is binary
- A text can have multiple polarization types simultaneously
- Some texts may have no polarization types (all zeros)
- The evaluation metric is Macro F1 across all 5 labels
