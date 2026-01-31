"""
Data Augmentation Module for Polarization Detection
Implements various augmentation techniques to improve model robustness
"""

import random
import re
from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextAugmenter:
    """
    Text augmentation techniques for improving model generalization
    """
    
    def __init__(self, augmentation_config=None):
        """
        Initialize augmenter with configuration
        
        Args:
            augmentation_config: Dictionary with augmentation settings
        """
        self.config = augmentation_config or {}
        self.random_state = random.Random(42)
    
    def random_deletion(self, text: str, delete_prob: float = 0.1) -> str:
        """
        Randomly delete words from text
        
        Args:
            text: Input text
            delete_prob: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if self.random_state.random() > delete_prob:
                new_words.append(word)
        
        # If all words deleted, return original
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n_swaps: int = 2) -> str:
        """
        Randomly swap words in text
        
        Args:
            text: Input text
            n_swaps: Number of swaps to perform
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n_swaps):
            idx1 = self.random_state.randint(0, len(words) - 1)
            idx2 = self.random_state.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def synonym_replacement(self, text: str, n_words: int = 3) -> str:
        """
        Replace words with synonyms (simplified version - character-level)
        
        Args:
            text: Input text
            n_words: Number of words to replace
            
        Returns:
            Augmented text
        """
        # Simple implementation: randomly modify some words
        words = text.split()
        if len(words) == 0:
            return text
        
        n_replace = min(n_words, len(words))
        indices = self.random_state.sample(range(len(words)), n_replace)
        
        for idx in indices:
            word = words[idx]
            if len(word) > 3:
                # Simple character substitution
                char_idx = self.random_state.randint(1, len(word) - 2)
                word_list = list(word)
                # Swap adjacent characters
                word_list[char_idx], word_list[char_idx + 1] = word_list[char_idx + 1], word_list[char_idx]
                words[idx] = ''.join(word_list)
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, n_insertions: int = 1) -> str:
        """
        Randomly insert duplicate words
        
        Args:
            text: Input text
            n_insertions: Number of insertions
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 0:
            return text
        
        for _ in range(n_insertions):
            # Pick a random word to insert
            random_word = self.random_state.choice(words)
            random_idx = self.random_state.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def augment_sample(self, text: str, label: int, methods: List[str] = None) -> List[Tuple[str, int]]:
        """
        Apply multiple augmentation methods to a single sample
        
        Args:
            text: Input text
            label: Label for the text
            methods: List of augmentation methods to apply
            
        Returns:
            List of (augmented_text, label) tuples
        """
        if methods is None:
            methods = ['random_deletion', 'random_swap', 'synonym_replacement']
        
        augmented_samples = [(text, label)]  # Include original
        
        for method in methods:
            try:
                if method == 'random_deletion':
                    aug_text = self.random_deletion(text, delete_prob=0.1)
                elif method == 'random_swap':
                    aug_text = self.random_swap(text, n_swaps=2)
                elif method == 'synonym_replacement':
                    aug_text = self.synonym_replacement(text, n_words=3)
                elif method == 'random_insertion':
                    aug_text = self.random_insertion(text, n_insertions=1)
                else:
                    continue
                
                # Only add if different from original
                if aug_text != text:
                    augmented_samples.append((aug_text, label))
            except Exception as e:
                logger.warning(f"Error in {method}: {e}")
                continue
        
        return augmented_samples
    
    def augment_dataset(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text',
        label_column: str = 'polarization',
        augmentation_ratio: float = 0.5,
        balance_classes: bool = True
    ) -> pd.DataFrame:
        """
        Augment entire dataset
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            augmentation_ratio: Ratio of samples to augment
            balance_classes: Whether to balance classes through augmentation
            
        Returns:
            Augmented DataFrame
        """
        logger.info(f"Starting dataset augmentation...")
        logger.info(f"Original dataset size: {len(df)}")
        
        augmented_data = []
        
        # If balancing classes, focus on minority class
        if balance_classes:
            class_counts = df[label_column].value_counts()
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            logger.info(f"Minority class: {minority_class} ({class_counts[minority_class]} samples)")
            
            # Calculate how many augmented samples we need
            class_diff = class_counts[majority_class] - class_counts[minority_class]
            minority_df = df[df[label_column] == minority_class]
            
            # Augment minority class samples
            samples_to_augment = min(len(minority_df), int(class_diff / 2))
            logger.info(f"Augmenting {samples_to_augment} minority class samples")
            
            for idx, row in tqdm(minority_df.head(samples_to_augment).iterrows(), total=samples_to_augment):
                text = row[text_column]
                label = row[label_column]
                
                aug_samples = self.augment_sample(text, label)
                for aug_text, aug_label in aug_samples[1:]:  # Skip original
                    augmented_data.append({
                        **row.to_dict(),
                        text_column: aug_text,
                        label_column: aug_label
                    })
        else:
            # Random augmentation
            n_samples = int(len(df) * augmentation_ratio)
            sample_df = df.sample(n=n_samples, random_state=42)
            
            logger.info(f"Augmenting {n_samples} random samples")
            
            for idx, row in tqdm(sample_df.iterrows(), total=n_samples):
                text = row[text_column]
                label = row[label_column]
                
                aug_samples = self.augment_sample(text, label)
                for aug_text, aug_label in aug_samples[1:]:  # Skip original
                    augmented_data.append({
                        **row.to_dict(),
                        text_column: aug_text,
                        label_column: aug_label
                    })
        
        # Combine original and augmented data
        augmented_df = pd.DataFrame(augmented_data)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Augmented dataset size: {len(combined_df)}")
        logger.info(f"Added {len(augmented_df)} augmented samples")
        
        if balance_classes:
            new_class_counts = combined_df[label_column].value_counts()
            logger.info(f"New class distribution: {new_class_counts.to_dict()}")
        
        return combined_df


class BackTranslationAugmenter:
    """
    Back-translation augmentation using translation models
    Note: Requires translation models - simplified implementation
    """
    
    def __init__(self):
        self.available = False
        try:
            from transformers import pipeline
            # This would require large translation models
            logger.info("Back-translation capability available")
            self.available = True
        except Exception as e:
            logger.warning(f"Back-translation not available: {e}")
    
    def back_translate(self, text: str, intermediate_lang: str = 'de') -> str:
        """
        Translate text to intermediate language and back
        
        Args:
            text: Input text
            intermediate_lang: Intermediate language code
            
        Returns:
            Back-translated text
        """
        if not self.available:
            return text
        
        # Simplified - would need actual translation models
        return text


def create_augmented_dataset(
    input_path: str,
    output_path: str,
    augmentation_ratio: float = 0.5,
    balance_classes: bool = True
):
    """
    Create augmented dataset from input CSV
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save augmented CSV
        augmentation_ratio: Ratio of samples to augment
        balance_classes: Whether to balance classes
    """
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    
    augmenter = TextAugmenter()
    augmented_df = augmenter.augment_dataset(
        df,
        augmentation_ratio=augmentation_ratio,
        balance_classes=balance_classes
    )
    
    augmented_df.to_csv(output_path, index=False)
    logger.info(f"Saved augmented dataset to {output_path}")


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment dataset')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--ratio', type=float, default=0.5, help='Augmentation ratio')
    parser.add_argument('--balance', action='store_true', help='Balance classes')
    
    args = parser.parse_args()
    
    create_augmented_dataset(
        args.input,
        args.output,
        args.ratio,
        args.balance
    )
