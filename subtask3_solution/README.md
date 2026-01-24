# Subtask 3 Solution - Polarization Manifestation Identification

Multi-label classification for detecting how polarization is expressed in text.

## Overview

Subtask 3 is a multi-label classification task where each text can have multiple polarization manifestations:
- **Stereotype** - Using stereotypical generalizations about groups
- **Vilification** - Attacking or defaming individuals/groups
- **Dehumanization** - Portraying people as less than human
- **Extreme Language** - Using inflammatory or extreme rhetoric
- **Lack of Empathy** - Showing no understanding or compassion
- **Invalidation** - Dismissing or negating others' experiences/views

**Note:** Some languages may exclude certain labels. Italian is not included in Subtask 3.

## Files

- `merge_data_subtask3.py` - Merge all language CSV files into single train/dev files
- `train_merged_subtask3.py` - Train multi-label classification model
- `test_subtask3.py` - Evaluate model and generate predictions
- `README.md` - This documentation file

## Quick Start

### 1. Merge Training Data

Combine all language files into a single merged dataset:

```bash
python merge_data_subtask3.py
```

This will create:
- `data/train_merged.csv` - All training data merged
- `data/dev_merged.csv` - All development data merged

### 2. Train Model

Train a multi-label classification model on the merged data:

```bash
python train_merged_subtask3.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --train_file ./data/train_merged.csv \
    --output_dir ./models/subtask3_model \
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
python test_subtask3.py \
    --model_path ./models/subtask3_model/final_model \
    --merged_data ./data/dev_merged.csv \
    --output_dir ./results \
    --mode evaluate

# Evaluate by language
python test_subtask3.py \
    --model_path ./models/subtask3_model/final_model \
    --data_dir c:/hishab/semeval-2026/dev_phase/subtask3 \
    --output_dir ./results \
    --mode by_language

# Do both
python test_subtask3.py \
    --model_path ./models/subtask3_model/final_model \
    --merged_data ./data/dev_merged.csv \
    --data_dir c:/hishab/semeval-2026/dev_phase/subtask3 \
    --output_dir ./results \
    --mode all
```

**Arguments:**
- `--model_path`: Path to trained model directory
- `--data_dir`: Root directory with subtask3 data (train/dev folders)
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
- `stereotype`: 0 or 1
- `vilification`: 0 or 1
- `dehumanization`: 0 or 1
- `extreme_language`: 0 or 1
- `lack_of_empathy`: 0 or 1
- `invalidation`: 0 or 1
- `language`: Language code (added during merge)

### Output (Predictions)
CSV file with:
- `id`: Unique identifier
- `stereotype`: Predicted label (0 or 1)
- `vilification`: Predicted label (0 or 1)
- `dehumanization`: Predicted label (0 or 1)
- `extreme_language`: Predicted label (0 or 1)
- `lack_of_empathy`: Predicted label (0 or 1)
- `invalidation`: Predicted label (0 or 1)

## Languages

18 languages are included (Italian excluded):
- Amharic (amh), Arabic (arb), Bengali (ben), Burmese (mya)
- Chinese (zho), English (eng), German (deu)
- Hausa (hau), Hindi (hin)
- Khmer (khm)
- Nepali (nep)
- Odia (ori)
- Persian (fas), Polish (pol), Punjabi (pan)
- Russian (rus)
- Spanish (spa), Swahili (swa)
- Telugu (tel), Turkish (tur)
- Urdu (urd)

## Evaluation Metrics

The model is evaluated using:
- **F1 Score (Macro)** - Main metric for competition
- **F1 Score (Micro)** - Overall performance across all labels
- **Accuracy** - Exact match ratio (all labels must match)
- **Hamming Loss** - Average label-wise error
- **Per-label F1** - F1 score for each manifestation type

## Model Architecture

- Uses transformer-based models (e.g., Qwen, BERT, RoBERTa, XLM-RoBERTa)
- Multi-label classification head with sigmoid activation
- BCEWithLogitsLoss for training
- 0.5 threshold for binary predictions (adjustable)

## Understanding Manifestations

### Stereotype
Overgeneralizations or assumptions about groups based on characteristics like race, religion, or nationality.

*Example:* "All [group] are lazy and don't want to work."

### Vilification
Direct attacks, insults, or defamation targeting individuals or groups.

*Example:* "Those [group] are scum and deserve what they get."

### Dehumanization
Language that portrays people as animals, objects, or subhuman.

*Example:* "They breed like rats and infest our neighborhoods."

### Extreme Language
Inflammatory, hyperbolic, or extreme rhetoric to provoke emotions.

*Example:* "This is a total invasion! Our country is being destroyed!"

### Lack of Empathy
Complete disregard for others' feelings, suffering, or perspectives.

*Example:* "I don't care if they're starving, not my problem."

### Invalidation
Dismissing, denying, or negating others' experiences, emotions, or identities.

*Example:* "You're just playing the victim, discrimination doesn't exist."

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

4. **Handle class imbalance**:
   - Some manifestations may be rarer than others
   - Consider weighted loss or focal loss
   - Oversample minority classes

5. **Data augmentation**:
   - Back-translation
   - Synonym replacement
   - Paraphrasing

6. **Ensemble methods**:
   - Train multiple models with different seeds
   - Combine predictions by voting or averaging
   - Use different model architectures

7. **Language-specific fine-tuning**:
   - Train separate models for language families
   - Use language-specific pretrained models
   - Consider transfer learning from high-resource languages

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

- Subtask 3 is a multi-label task with 6 manifestation types
- A text can have multiple manifestations simultaneously
- Some texts may have no manifestations (all zeros)
- The evaluation metric is Macro F1 across all 6 labels
- Italian language is NOT included in Subtask 3
- Some languages may have certain labels excluded
- Context is crucial - consider the overall meaning, not just keywords

## Differences from Subtask 2

| Aspect | Subtask 2 | Subtask 3 |
|--------|-----------|-----------|
| Task | Polarization Types | Manifestation Identification |
| Labels | 5 types | 6 manifestations |
| Languages | 22 (including Italian) | 18 (excluding Italian) |
| Focus | *What* is polarized | *How* polarization is expressed |

## Competition Information

- **Task**: SemEval 2026 Task 9 - Subtask 3
- **Evaluation**: Macro F1-score
- **Submission**: CSV with id and 6 binary labels
- **GitHub**: https://github.com/Polar-SemEval
- **Website**: https://polar-semeval.github.io/
