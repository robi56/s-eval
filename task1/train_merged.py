"""
Training Script - Use merged train data with train/val split
Dev data has no labels, so we split train data for validation
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
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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


class PolarizationDataset(Dataset):
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
        item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_binary = f1_score(labels, preds, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_binary': f1_binary
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='jhu-clsp/mmBERT-small')
    parser.add_argument('--train_file', type=str, default='./data/train_merged.csv')
    parser.add_argument('--output_dir', type=str, default='./models/gemma_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SemEval 2026 Task 9 - Subtask 1 Training")
    logger.info("="*60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load merged train data
    logger.info(f"Loading train data from: {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    logger.info(f"Total samples: {len(train_df)}")
    logger.info(f"Polarized: {(train_df['polarization'] == 1).sum()}")
    logger.info(f"Not Polarized: {(train_df['polarization'] == 0).sum()}")
    
    # Split into train and validation
    train_data, val_data = train_test_split(
        train_df,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=train_df['polarization']
    )
    
    logger.info(f"\nAfter split:")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Load tokenizer and model
    logger.info(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = PolarizationDataset(
        train_data['text'].tolist(),
        train_data['polarization'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = PolarizationDataset(
        val_data['text'].tolist(),
        val_data['polarization'].tolist(),
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
    logger.info(f"F1 Score (binary): {eval_results['eval_f1_binary']:.4f}")
    logger.info(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"Recall: {eval_results['eval_recall']:.4f}")
    
    # Save results
    results = {
        'model': args.model_name,
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'metrics': {
            'f1_macro': float(eval_results['eval_f1_macro']),
            'f1_binary': float(eval_results['eval_f1_binary']),
            'accuracy': float(eval_results['eval_accuracy']),
            'precision': float(eval_results['eval_precision']),
            'recall': float(eval_results['eval_recall'])
        }
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir / 'training_results.json'}")


if __name__ == '__main__':
    main()
