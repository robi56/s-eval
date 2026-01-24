"""
Testing/Evaluation Script for SemEval 2026 Task 9 - Subtask 2
Multi-label Classification for Polarization Types
Generate predictions and evaluate model performance
"""

import os
import argparse
import logging
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    classification_report,
    hamming_loss,
    multilabel_confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Label columns for subtask 2
LABEL_COLUMNS = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']


class MultiLabelDataset(torch.utils.data.Dataset):
    """Dataset for multi-label classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else [0] * 5
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        
        return item


class ModelPredictor:
    """
    Model predictor for multi-label polarization type classification
    """
    
    def __init__(self, model_path: str, device: str = None, threshold: float = 0.5):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model directory
            device: Device to use (cuda/cpu), auto-detected if None
            threshold: Decision threshold for binary classification (default: 0.5)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Decision threshold: {self.threshold}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict_batch(self, texts, batch_size=32, max_length=128):
        """
        Predict labels for a batch of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Create dataset
        dummy_labels = None  # No labels needed for inference
        dataset = MultiLabelDataset(texts, dummy_labels, self.tokenizer, max_length)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            shuffle=False
        )
        
        all_predictions = []
        all_probabilities = []
        
        # Inference
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                
                # Apply threshold to get predictions
                preds = (probs > self.threshold).int()
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_single(self, text: str):
        """
        Predict labels for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        preds, probs = self.predict_batch([text], batch_size=1)
        return preds[0], probs[0]


def evaluate_model(predictor, test_df, output_dir=None):
    """
    Evaluate model on test dataset
    
    Args:
        predictor: ModelPredictor instance
        test_df: Test DataFrame with 'text' and label columns
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating on {len(test_df)} samples...")
    
    # Get predictions
    predictions, probabilities = predictor.predict_batch(
        test_df['text'].tolist(),
        batch_size=32
    )
    
    # Get true labels
    true_labels = test_df[LABEL_COLUMNS].values
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    precision_micro = precision_score(true_labels, predictions, average='micro', zero_division=0)
    recall_micro = recall_score(true_labels, predictions, average='micro', zero_division=0)
    f1_micro = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    hamming = hamming_loss(true_labels, predictions)
    
    # Per-label metrics
    per_label_f1 = f1_score(true_labels, predictions, average=None, zero_division=0)
    per_label_precision = precision_score(true_labels, predictions, average=None, zero_division=0)
    per_label_recall = recall_score(true_labels, predictions, average=None, zero_division=0)
    
    # Classification report
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=LABEL_COLUMNS,
        digits=4,
        zero_division=0
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results - Subtask 2")
    logger.info("="*60)
    logger.info(f"Accuracy (exact match): {accuracy:.4f}")
    logger.info(f"Hamming Loss: {hamming:.4f}")
    logger.info(f"\nMacro Metrics:")
    logger.info(f"  Precision: {precision_macro:.4f}")
    logger.info(f"  Recall: {recall_macro:.4f}")
    logger.info(f"  F1 Score: {f1_macro:.4f}")
    logger.info(f"\nMicro Metrics:")
    logger.info(f"  Precision: {precision_micro:.4f}")
    logger.info(f"  Recall: {recall_micro:.4f}")
    logger.info(f"  F1 Score: {f1_micro:.4f}")
    logger.info(f"\nPer-Label F1 Scores:")
    for i, label in enumerate(LABEL_COLUMNS):
        logger.info(f"  {label}: {per_label_f1[i]:.4f} (P: {per_label_precision[i]:.4f}, R: {per_label_recall[i]:.4f})")
    logger.info("\nClassification Report:")
    logger.info("\n" + class_report)
    
    # Label distribution
    logger.info("\nLabel Distribution:")
    logger.info("True labels:")
    for i, label in enumerate(LABEL_COLUMNS):
        logger.info(f"  {label}: {true_labels[:, i].sum()}")
    logger.info("Predicted labels:")
    for i, label in enumerate(LABEL_COLUMNS):
        logger.info(f"  {label}: {predictions[:, i].sum()}")
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'hamming_loss': float(hamming),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'per_label_f1': {label: float(f1) for label, f1 in zip(LABEL_COLUMNS, per_label_f1)},
        'per_label_precision': {label: float(p) for label, p in zip(LABEL_COLUMNS, per_label_precision)},
        'per_label_recall': {label: float(r) for label, r in zip(LABEL_COLUMNS, per_label_recall)},
        'classification_report': class_report,
        'num_samples': len(test_df)
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'classification_report'}, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
        
        # Save predictions
        predictions_df = test_df.copy()
        for i, label in enumerate(LABEL_COLUMNS):
            predictions_df[f'pred_{label}'] = predictions[:, i]
            predictions_df[f'prob_{label}'] = probabilities[:, i]
        
        # Check if all labels match
        predictions_df['all_correct'] = (predictions == true_labels).all(axis=1)
        
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
    
    return results


def generate_submission(predictor, test_df, output_path):
    """
    Generate submission file for competition
    
    Args:
        predictor: ModelPredictor instance
        test_df: Test DataFrame with 'id' and 'text' columns
        output_path: Path to save submission file
    """
    logger.info(f"Generating predictions for {len(test_df)} samples...")
    
    # Get predictions
    predictions, _ = predictor.predict_batch(test_df['text'].tolist(), batch_size=32)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id']
    })
    
    # Add predictions for each label
    for i, label in enumerate(LABEL_COLUMNS):
        submission_df[label] = predictions[:, i]
    
    # Save submission
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"\nPredicted label counts:")
    for i, label in enumerate(LABEL_COLUMNS):
        logger.info(f"  {label}: {predictions[:, i].sum()}")


def test_by_language(predictor, data_dir, split='dev', output_dir=None):
    """
    Test model on each language separately
    
    Args:
        predictor: ModelPredictor instance
        data_dir: Directory containing language-specific CSV files
        split: 'train' or 'dev'
        output_dir: Directory to save results
    """
    data_path = Path(data_dir) / split
    
    languages = [
        'amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita',
        'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa',
        'tel', 'tur', 'urd', 'zho'
    ]
    
    results_by_lang = {}
    
    logger.info("\n" + "="*60)
    logger.info("Evaluating by Language")
    logger.info("="*60)
    
    for lang in languages:
        lang_file = data_path / f'{lang}.csv'
        
        if not lang_file.exists():
            logger.warning(f"File not found: {lang_file}")
            continue
        
        logger.info(f"\nEvaluating {lang}...")
        
        # Load language data
        lang_df = pd.read_csv(lang_file)
        
        if lang_df.empty or 'text' not in lang_df.columns:
            logger.warning(f"No valid data found for {lang}")
            continue
        
        # Check if labels exist
        has_labels = all(col in lang_df.columns for col in LABEL_COLUMNS)
        
        if not has_labels:
            logger.warning(f"Labels not found for {lang}, skipping evaluation")
            continue
        
        # Get predictions
        predictions, _ = predictor.predict_batch(lang_df['text'].tolist())
        true_labels = lang_df[LABEL_COLUMNS].values
        
        # Calculate metrics
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(true_labels, predictions, average='micro', zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)
        hamming = hamming_loss(true_labels, predictions)
        
        per_label_f1 = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        results_by_lang[lang] = {
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'accuracy': float(accuracy),
            'hamming_loss': float(hamming),
            'num_samples': len(lang_df),
            'per_label_f1': {label: float(f1) for label, f1 in zip(LABEL_COLUMNS, per_label_f1)}
        }
        
        logger.info(f"  F1 (macro): {f1_macro:.4f}, F1 (micro): {f1_micro:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"  Samples: {len(lang_df)}, Hamming Loss: {hamming:.4f}")
    
    # Calculate average across languages
    if results_by_lang:
        avg_f1_macro = np.mean([r['f1_macro'] for r in results_by_lang.values()])
        avg_f1_micro = np.mean([r['f1_micro'] for r in results_by_lang.values()])
        avg_accuracy = np.mean([r['accuracy'] for r in results_by_lang.values()])
        
        logger.info(f"\nAverage across languages:")
        logger.info(f"  F1 (macro): {avg_f1_macro:.4f}")
        logger.info(f"  F1 (micro): {avg_f1_micro:.4f}")
        logger.info(f"  Accuracy: {avg_accuracy:.4f}")
    
    # Save language-specific results
    if output_dir and results_by_lang:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lang_results_path = output_dir / 'results_by_language.json'
        with open(lang_results_path, 'w') as f:
            json.dump(results_by_lang, f, indent=2)
        logger.info(f"\nLanguage-specific results saved to {lang_results_path}")
    
    return results_by_lang


def main():
    parser = argparse.ArgumentParser(
        description='Test model for SemEval 2026 Task 9 - Subtask 2'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model directory'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='c:/hishab/semeval-2026/dev_phase/subtask2',
        help='Directory containing subtask2 data (train/dev folders)'
    )
    
    parser.add_argument(
        '--merged_data',
        type=str,
        default=None,
        help='Path to merged CSV file for evaluation (optional)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./subtask2_solution/results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['evaluate', 'submission', 'by_language', 'all'],
        default='all',
        help='Testing mode'
    )
    
    parser.add_argument(
        '--submission_output',
        type=str,
        default='./subtask2_solution/submission.csv',
        help='Path to save submission file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold for binary classification (default: 0.5)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev'],
        default='dev',
        help='Data split to evaluate on'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ModelPredictor(args.model_path, threshold=args.threshold)
    
    # Evaluate on merged data
    if args.mode in ['evaluate', 'all'] and args.merged_data:
        logger.info("\nEvaluating on merged dataset...")
        test_df = pd.read_csv(args.merged_data)
        evaluate_model(predictor, test_df, args.output_dir)
    
    # Evaluate by language
    if args.mode in ['by_language', 'all']:
        test_by_language(predictor, args.data_dir, split=args.split, output_dir=args.output_dir)
    
    # Generate submission (if test data is available)
    if args.mode in ['submission']:
        logger.info("\nNote: Submission generation requires test data with 'id' column")
        logger.info("Provide test data path to generate submission")
    
    logger.info("\n" + "="*60)
    logger.info("Testing Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
