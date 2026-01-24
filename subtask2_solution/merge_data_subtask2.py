"""
Merge train and dev datasets separately for all languages - Subtask 2
Subtask 2: Multi-label classification for polarization types
"""
import pandas as pd
from pathlib import Path

# Paths
train_dir = Path(r'c:\hishab\semeval-2026\dev_phase\subtask2\train')
dev_dir = Path(r'c:\hishab\semeval-2026\dev_phase\subtask2\dev')
output_dir = Path(r'c:\hishab\semeval-2026\subtask2_solution\data')

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Languages
languages = [
    'amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita',
    'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa',
    'tel', 'tur', 'urd', 'zho'
]

print("="*60)
print("Merging Train Data - Subtask 2")
print("="*60)

train_dfs = []
for lang in languages:
    file_path = train_dir / f'{lang}.csv'
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['language'] = lang
        train_dfs.append(df)
        print(f"Loaded {lang}: {len(df)} samples")
    else:
        print(f"Warning: {lang}.csv not found in train")

train_merged = pd.concat(train_dfs, ignore_index=True)
train_merged = train_merged.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal train samples: {len(train_merged)}")
print(f"\nLabel distribution:")
print(f"Political: {train_merged['political'].sum()}")
print(f"Racial/Ethnic: {train_merged['racial/ethnic'].sum()}")
print(f"Religious: {train_merged['religious'].sum()}")
print(f"Gender/Sexual: {train_merged['gender/sexual'].sum()}")
print(f"Other: {train_merged['other'].sum()}")

# Save train
train_output = output_dir / 'train_merged.csv'
train_merged.to_csv(train_output, index=False)
print(f"\nSaved to: {train_output}")

print("\n" + "="*60)
print("Merging Dev Data - Subtask 2")
print("="*60)

dev_dfs = []
for lang in languages:
    file_path = dev_dir / f'{lang}.csv'
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['language'] = lang
        dev_dfs.append(df)
        print(f"Loaded {lang}: {len(df)} samples")
    else:
        print(f"Warning: {lang}.csv not found in dev")

dev_merged = pd.concat(dev_dfs, ignore_index=True)

print(f"\nTotal dev samples: {len(dev_merged)}")
print(f"\nLabel distribution:")
print(f"Political: {dev_merged['political'].sum()}")
print(f"Racial/Ethnic: {dev_merged['racial/ethnic'].sum()}")
print(f"Religious: {dev_merged['religious'].sum()}")
print(f"Gender/Sexual: {dev_merged['gender/sexual'].sum()}")
print(f"Other: {dev_merged['other'].sum()}")

# Save dev
dev_output = output_dir / 'dev_merged.csv'
dev_merged.to_csv(dev_output, index=False)
print(f"\nSaved to: {dev_output}")

print("\n" + "="*60)
print("Merge Complete!")
print("="*60)
