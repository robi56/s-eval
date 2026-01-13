"""
Testing/Evaluation Script for SemEval 2026 Task 9 - Subtask 1
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
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from data_processing import PolarizationDataProcessor
from train_merged import PolarizationDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Model predictor for polarization detection
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model directory
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
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
        dummy_labels = [0] * len(texts)  # Dummy labels for inference
        dataset = PolarizationDataset(texts, dummy_labels, self.tokenizer, max_length)
        
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
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_single(self, text: str):
        """
        Predict label for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (prediction, probability)
        """
        preds, probs = self.predict_batch([text], batch_size=1)
        return preds[0], probs[0]


def evaluate_model(predictor, test_df, output_dir=None):
    """
    Evaluate model on test dataset
    
    Args:
        predictor: ModelPredictor instance
        test_df: Test DataFrame with 'text' and 'polarization' columns
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
    true_labels = test_df['polarization'].values
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_binary = f1_score(true_labels, predictions, average='binary', zero_division=0)
    
    # Classification report
    class_report = classification_report(
        true_labels, 
        predictions, 
        target_names=['Not Polarized', 'Polarized'],
        digits=4
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (macro): {precision:.4f}")
    logger.info(f"Recall (macro): {recall:.4f}")
    logger.info(f"F1 Score (macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (binary): {f1_binary:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + class_report)
    logger.info("\nConfusion Matrix:")
    logger.info(f"TN: {conf_matrix[0, 0]}, FP: {conf_matrix[0, 1]}")
    logger.info(f"FN: {conf_matrix[1, 0]}, TP: {conf_matrix[1, 1]}")
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_macro': float(f1_macro),
        'f1_binary': float(f1_binary),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
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
        predictions_df['predicted_polarization'] = predictions
        predictions_df['prob_not_polarized'] = probabilities[:, 0]
        predictions_df['prob_polarized'] = probabilities[:, 1]
        predictions_df['correct'] = (predictions == true_labels)
        
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
        'id': test_df['id'],
        'polarization': predictions
    })
    
    # Save submission
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Polarized: {(predictions == 1).sum()}")
    logger.info(f"Not polarized: {(predictions == 0).sum()}")


def test_by_language(predictor, data_root, output_dir):
    """
    Test model on each language separately
    
    Args:
        predictor: ModelPredictor instance
        data_root: Root directory containing subtask data
        output_dir: Directory to save results
    """
    processor = PolarizationDataProcessor(data_root=data_root)
    
    results_by_lang = {}
    
    logger.info("\n" + "="*60)
    logger.info("Evaluating by Language")
    logger.info("="*60)
    
    for lang in processor.LANGUAGES:
        logger.info(f"\nEvaluating {lang}...")
        
        # Load dev data for this language
        dev_df = processor.load_single_language(lang, 'dev')
        
        if dev_df.empty:
            logger.warning(f"No data found for {lang}")
            continue
        
        # Get predictions
        predictions, _ = predictor.predict_batch(dev_df['text'].tolist())
        true_labels = dev_df['polarization'].values
        
        # Calculate F1 scores
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        f1_binary = f1_score(true_labels, predictions, average='binary', zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)
        
        results_by_lang[lang] = {
            'f1_macro': float(f1_macro),
            'f1_binary': float(f1_binary),
            'accuracy': float(accuracy),
            'num_samples': len(dev_df)
        }
        
        logger.info(f"  F1 (macro): {f1_macro:.4f}, Accuracy: {accuracy:.4f}, Samples: {len(dev_df)}")
    
    # Save language-specific results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lang_results_path = output_dir / 'results_by_language.json'
        with open(lang_results_path, 'w') as f:
            json.dump(results_by_lang, f, indent=2)
        logger.info(f"\nLanguage-specific results saved to {lang_results_path}")
    
    return results_by_lang


def main():
    parser = argparse.ArgumentParser(
        description='Test model for SemEval 2026 Task 9 - Subtask 1'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model directory'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='./dev_phase',
        help='Root directory containing subtask data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./subtask1_solution/results',
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
        default='./subtask1_solution/submission.csv',
        help='Path to save submission file'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ModelPredictor(args.model_path)
    
    # Evaluate on merged dev set
    if args.mode in ['evaluate', 'all']:
        logger.info("\nEvaluating on merged development set...")
        processor = PolarizationDataProcessor(data_root=args.data_root)
        _, dev_df = processor.prepare_datasets(save_merged=False)
        evaluate_model(predictor, dev_df, args.output_dir)
    
    # Evaluate by language
    if args.mode in ['by_language', 'all']:
        test_by_language(predictor, args.data_root, args.output_dir)
    
    # Generate submission (if test data is available)
    if args.mode in ['submission', 'all']:
        logger.info("\nNote: Submission generation requires test data with 'id' column")
        logger.info("Use --mode submission when test data is available")
        processor = PolarizationDataProcessor(data_root=args.data_root)
        _, dev_df = processor.prepare_datasets(save_merged=False)
        generate_submission(predictor, dev_df, args.output_dir)  
    logger.info("\n" + "="*60)
    logger.info("Testing Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
