"""
Data Processing Module for SemEval 2026 Task 9 - Subtask 1
Merges multilingual datasets for polarization detection (binary classification)
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolarizationDataProcessor:
    """
    Process and merge multilingual polarization detection datasets
    """
    
    # All 22 languages in the competition
    LANGUAGES = [
        'amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita',
        'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa',
        'tel', 'tur', 'urd', 'zho'
    ]
    
    def __init__(self, data_root: str = './dev_phase'):
        """
        Initialize data processor
        
        Args:
            data_root: Root directory containing subtask1, subtask2, subtask3 folders
        """
        self.data_root = Path(data_root)
        self.subtask1_path = self.data_root / 'subtask1'
        
    def load_single_language(self, language: str, split: str) -> pd.DataFrame:
        """
        Load data for a single language
        
        Args:
            language: Language code (e.g., 'eng', 'spa')
            split: 'train' or 'dev'
            
        Returns:
            DataFrame with id, text, polarization columns
        """
        file_path = self.subtask1_path / split / f'{language}.csv'
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            df['language'] = language  # Add language identifier
            logger.info(f"Loaded {len(df)} samples from {language} ({split})")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def merge_all_languages(self, split: str, languages: List[str] = None) -> pd.DataFrame:
        """
        Merge all language datasets for a given split
        
        Args:
            split: 'train' or 'dev'
            languages: List of language codes to merge (default: all languages)
            
        Returns:
            Merged DataFrame
        """
        if languages is None:
            languages = self.LANGUAGES
        
        all_data = []
        
        for lang in languages:
            df = self.load_single_language(lang, split)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No data found for split: {split}")
        
        merged_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Merged {split} set: {len(merged_df)} total samples from {len(all_data)} languages")
        
        return merged_df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get dataset statistics
        
        Args:
            df: DataFrame with polarization labels
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(df),
            'polarized': (df['polarization'] == 1).sum(),
            'not_polarized': (df['polarization'] == 0).sum(),
            'polarization_ratio': (df['polarization'] == 1).sum() / len(df),
            'languages': df['language'].nunique() if 'language' in df.columns else 1,
            'language_distribution': df['language'].value_counts().to_dict() if 'language' in df.columns else {}
        }
        return stats
    
    def prepare_datasets(self, languages: List[str] = None, save_merged: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare train and dev datasets by merging all languages
        
        Args:
            languages: List of language codes to include (default: all)
            save_merged: Whether to save merged datasets to CSV
            
        Returns:
            Tuple of (train_df, dev_df)
        """
        logger.info("Starting dataset preparation...")
        
        # Load and merge training data
        train_df = self.merge_all_languages('train', languages)
        logger.info(f"Training set statistics:")
        train_stats = self.get_statistics(train_df)
        for key, value in train_stats.items():
            if key != 'language_distribution':
                logger.info(f"  {key}: {value}")
        
        # Load and merge development data
        dev_df = self.merge_all_languages('dev', languages)
        logger.info(f"Development set statistics:")
        dev_stats = self.get_statistics(dev_df)
        for key, value in dev_stats.items():
            if key != 'language_distribution':
                logger.info(f"  {key}: {value}")
        
        # Shuffle the data
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save merged datasets
        if save_merged:
            output_dir = Path('./subtask1_solution/data')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_path = output_dir / 'train_merged.csv'
            dev_path = output_dir / 'dev_merged.csv'
            
            train_df.to_csv(train_path, index=False)
            dev_df.to_csv(dev_path, index=False)
            
            logger.info(f"Saved merged train data to {train_path}")
            logger.info(f"Saved merged dev data to {dev_path}")
        
        return train_df, dev_df


def main():
    """
    Example usage of data processor
    """
    processor = PolarizationDataProcessor(data_root='./dev_phase')
    
    # Prepare all datasets
    train_df, dev_df = processor.prepare_datasets(save_merged=True)
    
    print("\n" + "="*50)
    print("Dataset Preparation Complete!")
    print("="*50)
    print(f"Training samples: {len(train_df)}")
    print(f"Development samples: {len(dev_df)}")
    print(f"\nSample from training data:")
    print(train_df.head())


if __name__ == '__main__':
    main()
