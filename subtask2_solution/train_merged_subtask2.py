"""
Training Script for Subtask 2 - Multi-label Classification
Use merged train data with train/val split for polarization type classification
Dev data has no labels, so we split train data for validation

Subtask 2: Predict the types of polarization:
- Political
- Racial/Ethnic
- Religious
- Gender/Sexual
- Other

This is a multi-label classification task (can have multiple labels per text)
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiLabelPolarizationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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


def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    """
    predictions, labels = eval_pred
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Apply threshold of 0.5
    preds = (probs.numpy() > 0.5).astype(int)
    
    # Convert to numpy arrays
    labels = labels.astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    
    hamming = hamming_loss(labels, preds)
    
    # Per-label F1 scores
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'hamming_loss': hamming,
        'f1_political': per_label_f1[0] if len(per_label_f1) > 0 else 0,
        'f1_racial_ethnic': per_label_f1[1] if len(per_label_f1) > 1 else 0,
        'f1_religious': per_label_f1[2] if len(per_label_f1) > 2 else 0,
        'f1_gender_sexual': per_label_f1[3] if len(per_label_f1) > 3 else 0,
        'f1_other': per_label_f1[4] if len(per_label_f1) > 4 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--train_file', type=str, default='./data/train_merged.csv')
    parser.add_argument('--output_dir', type=str, default='./models/subtask2_model')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SemEval 2026 Task 9 - Subtask 2 Training")
    logger.info("Multi-label Classification for Polarization Types")
    logger.info("="*60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load merged train data
    logger.info(f"Loading train data from: {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    logger.info(f"Total samples: {len(train_df)}")
    
    # Label columns
    label_columns = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    
    logger.info(f"\nLabel distribution:")
    for col in label_columns:
        logger.info(f"  {col}: {train_df[col].sum()}")
    
    # Split into train and validation
    # For multi-label, we can't use stratify directly, so we use random split
    train_data, val_data = train_test_split(
        train_df,
        test_size=args.val_split,
        random_state=args.seed
    )
    
    logger.info(f"\nAfter split:")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Prepare labels as multi-hot encoded arrays
    train_labels = train_data[label_columns].values.tolist()
    val_labels = val_data[label_columns].values.tolist()
    
    # Load tokenizer and model
    logger.info(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Multi-label classification model (num_labels = 5)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,
        problem_type="multi_label_classification",
        attn_implementation="eager"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = MultiLabelPolarizationDataset(
        train_data['text'].tolist(),
        train_labels,
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = MultiLabelPolarizationDataset(
        val_data['text'].tolist(),
        val_labels,
        tokenizer,
        max_length=args.max_length
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed
    )
    
    # Initialize Trainer
    logger.info("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("\nStarting training...")
    trainer.train()
    
    # Save model
    logger.info("\nSaving model...")
    trainer.save_model(str(output_dir / 'final_model'))
    tokenizer.save_pretrained(str(output_dir / 'final_model'))
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"F1 Score (macro): {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"F1 Score (micro): {eval_results['eval_f1_micro']:.4f}")
    logger.info(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Precision (macro): {eval_results['eval_precision_macro']:.4f}")
    logger.info(f"Recall (macro): {eval_results['eval_recall_macro']:.4f}")
    logger.info(f"Hamming Loss: {eval_results['eval_hamming_loss']:.4f}")
    logger.info(f"\nPer-label F1 scores:")
    logger.info(f"  Political: {eval_results['eval_f1_political']:.4f}")
    logger.info(f"  Racial/Ethnic: {eval_results['eval_f1_racial_ethnic']:.4f}")
    logger.info(f"  Religious: {eval_results['eval_f1_religious']:.4f}")
    logger.info(f"  Gender/Sexual: {eval_results['eval_f1_gender_sexual']:.4f}")
    logger.info(f"  Other: {eval_results['eval_f1_other']:.4f}")
    
    # Save results
    results = {
        'model': args.model_name,
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'task': 'subtask2_multilabel',
        'metrics': {
            'f1_macro': float(eval_results['eval_f1_macro']),
            'f1_micro': float(eval_results['eval_f1_micro']),
            'accuracy': float(eval_results['eval_accuracy']),
            'precision_macro': float(eval_results['eval_precision_macro']),
            'recall_macro': float(eval_results['eval_recall_macro']),
            'hamming_loss': float(eval_results['eval_hamming_loss']),
            'f1_political': float(eval_results['eval_f1_political']),
            'f1_racial_ethnic': float(eval_results['eval_f1_racial_ethnic']),
            'f1_religious': float(eval_results['eval_f1_religious']),
            'f1_gender_sexual': float(eval_results['eval_f1_gender_sexual']),
            'f1_other': float(eval_results['eval_f1_other']),
        }
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir / 'training_results.json'}")


if __name__ == '__main__':
    main()
