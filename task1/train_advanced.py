"""
Advanced Training Script with Enhanced Techniques
Implements class balancing, advanced schedulers, and training techniques
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
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

from data_processing import PolarizationDataProcessor
from data_augmentation import TextAugmenter
from train_merged import PolarizationDataset, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer with class-weighted loss
    """
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with class weights
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute weighted loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """
    Trainer with Focal Loss for handling class imbalance
    """
    
    def __init__(self, *args, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute focal loss
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute focal loss
        ce_loss = nn.CrossEntropyLoss(reduction='none')(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss


def get_class_weights(labels):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.FloatTensor(class_weights)


def create_weighted_sampler(labels):
    """
    Create weighted sampler for balanced training
    
    Args:
        labels: Array of labels
        
    Returns:
        WeightedRandomSampler
    """
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def train_with_advanced_techniques(args):
    """
    Train model with advanced techniques
    
    Args:
        args: Command line arguments
    """
    logger.info("="*60)
    logger.info("Advanced Training - SemEval 2026 Task 9 - Subtask 1")
    logger.info("="*60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and prepare data
    logger.info("Loading datasets...")
    processor = PolarizationDataProcessor(data_root=args.data_root)

    train_df, dev_df = processor.prepare_datasets(save_merged=True)

    # If dev_df is empty or has no labels, use 1% of train_df for validation
    if dev_df.empty or 'polarization' not in dev_df.columns or dev_df['polarization'].isnull().all():
        logger.warning("Dev set is empty or has no labels. Using 1% of training data for validation.")
        val_frac = 0.01
        val_sample = train_df.sample(frac=val_frac, random_state=args.seed)
        train_df = train_df.drop(val_sample.index).reset_index(drop=True)
        dev_df = val_sample.reset_index(drop=True)

    # Data augmentation
    if args.use_augmentation:
        logger.info("Applying data augmentation...")
        augmenter = TextAugmenter()
        train_df = augmenter.augment_dataset(
            train_df,
            augmentation_ratio=args.augmentation_ratio,
            balance_classes=args.balance_classes
        )

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Dev size: {len(dev_df)}")

    # Check class distribution
    train_class_dist = train_df['polarization'].value_counts()
    logger.info(f"Training class distribution:\n{train_class_dist}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        problem_type="single_label_classification",
        attn_implementation="eager"
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Create datasets
    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer,
        max_length=args.max_length
    )

    val_dataset = PolarizationDataset(
        dev_df['text'].tolist(),
        dev_df['polarization'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        logger.info("Computing class weights...")
        class_weights = get_class_weights(train_df['polarization'].values)
        logger.info(f"Class weights: {class_weights}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and args.use_fp16,
        bf16=torch.cuda.is_available() and args.use_bf16,
        report_to="none",
        seed=args.seed,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        disable_tqdm=False
    )
    
    # Select trainer based on loss type
    if args.loss_type == 'focal':
        logger.info("Using Focal Loss")
        trainer_class = FocalLossTrainer
        trainer_kwargs = {'alpha': args.focal_alpha, 'gamma': args.focal_gamma}
    elif args.loss_type == 'weighted':
        logger.info("Using Weighted Cross-Entropy Loss")
        trainer_class = WeightedLossTrainer
        trainer_kwargs = {'class_weights': class_weights}
    else:
        logger.info("Using standard Cross-Entropy Loss")
        trainer_class = Trainer
        trainer_kwargs = {}
    
    # Initialize trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        **trainer_kwargs
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info("Saving model...")
    final_model_dir = output_dir / 'final_model'
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Evaluate
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"F1 (macro): {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"F1 (binary): {eval_results['eval_f1_binary']:.4f}")
    logger.info(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"Recall: {eval_results['eval_recall']:.4f}")
    
    # Save results
    results = {
        'model': args.model_name,
        'timestamp': datetime.now().isoformat(),
        'advanced_techniques': {
            'augmentation': args.use_augmentation,
            'class_weights': args.use_class_weights,
            'loss_type': args.loss_type,
            'lr_scheduler': args.lr_scheduler_type,
            'fp16': args.use_fp16
        },
        'hyperparameters': vars(args),
        'eval_results': eval_results,
        'output_dir': str(final_model_dir)
    }
    
    results_path = output_dir / 'training_results_advanced.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Advanced training script')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dev_phase')
    parser.add_argument('--output_dir', type=str, default='./subtask1_solution/models_advanced')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--max_length', type=int, default=128)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    # Advanced techniques
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5)
    parser.add_argument('--balance_classes', action='store_true', help='Balance classes through augmentation')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights in loss')
    parser.add_argument('--loss_type', type=str, choices=['ce', 'weighted', 'focal'], default='ce')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial'])
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--use_bf16', action='store_true', help='Use BF16 mixed precision')
    
    args = parser.parse_args()
    
    train_with_advanced_techniques(args)


if __name__ == '__main__':
    main()
