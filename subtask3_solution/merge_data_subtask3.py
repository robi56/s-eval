"""
Merge train and dev datasets separately for all languages - Subtask 3
Subtask 3: Multi-label classification for polarization manifestations
"""
import pandas as pd
from pathlib import Path

# Paths
train_dir = Path(r'c:\hishab\semeval-2026\dev_phase\subtask3\train')
dev_dir = Path(r'c:\hishab\semeval-2026\dev_phase\subtask3\dev')
output_dir = Path(r'c:\hishab\semeval-2026\subtask3_solution\data')

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Languages (note: Italian is excluded from subtask 3)
languages = [
    'amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin',
    'khm', 'nep', 'ori', 'pan', 'spa', 'swa',
    'tel', 'tur', 'urd', 'zho'
]

print("="*60)
print("Merging Train Data - Subtask 3")
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
print(f"Stereotype: {train_merged['stereotype'].sum()}")
print(f"Vilification: {train_merged['vilification'].sum()}")
print(f"Dehumanization: {train_merged['dehumanization'].sum()}")
print(f"Extreme Language: {train_merged['extreme_language'].sum()}")
print(f"Lack of Empathy: {train_merged['lack_of_empathy'].sum()}")
print(f"Invalidation: {train_merged['invalidation'].sum()}")

# Save train
train_output = output_dir / 'train_merged.csv'
train_merged.to_csv(train_output, index=False)
print(f"\nSaved to: {train_output}")

print("\n" + "="*60)
print("Merging Dev Data - Subtask 3")
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
print(f"Stereotype: {dev_merged['stereotype'].sum()}")
print(f"Vilification: {dev_merged['vilification'].sum()}")
print(f"Dehumanization: {dev_merged['dehumanization'].sum()}")
print(f"Extreme Language: {dev_merged['extreme_language'].sum()}")
print(f"Lack of Empathy: {dev_merged['lack_of_empathy'].sum()}")
print(f"Invalidation: {dev_merged['invalidation'].sum()}")

# Save dev
dev_output = output_dir / 'dev_merged.csv'
dev_merged.to_csv(dev_output, index=False)
print(f"\nSaved to: {dev_output}")

print("\n" + "="*60)
print("Merge Complete!")
print("="*60)
