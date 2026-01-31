"""
Generate Submission Files for SemEval 2026 Task 9 - Subtask 2
Creates pred_[lang_code].csv files for each language and packages them for Codabench upload
"""

import os
import argparse
import logging
from pathlib import Path
import zipfile
import shutil

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language codes for the competition
LANGUAGES = [
    'amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin',
    'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus',
    'spa', 'swa', 'tel', 'tur', 'urd', 'zho'
]

LABEL_COLUMNS = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']

class MultiLabelPredictor:
    """Predictor for multi-label sequence classification models"""
    def __init__(self, model_path: str, device: str = None, threshold: float = 0.5):
        self.model_path = Path(model_path).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        logger.info(f"Loading model from {self.model_path}")
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    def predict(self, texts: list, batch_size: int = 32, max_length: int = 128):
        all_predictions = []
        dataloader = torch.utils.data.DataLoader(
            texts, batch_size=batch_size, shuffle=False
        )
        for batch_texts in tqdm(dataloader, desc="Predicting"):
            inputs = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).int().cpu().numpy()
                all_predictions.extend(preds)
        return all_predictions

def load_test_data(test_dir: Path, language: str):
    test_file = test_dir / f"{language}.csv"
    if not test_file.exists():
        logger.warning(f"Test file not found: {test_file}")
        return None
    df = pd.read_csv(test_file)
    if 'id' not in df.columns or 'text' not in df.columns:
        logger.error(f"Required columns 'id' and 'text' not found in {test_file}")
        return None
    return df

def generate_predictions_for_language(predictor, test_dir: Path, language: str, output_dir: Path):
    logger.info(f"\nProcessing {language}...")
    test_df = load_test_data(test_dir, language)
    if test_df is None:
        return False
    logger.info(f"  Loaded {len(test_df)} samples")
    predictions = predictor.predict(test_df['text'].tolist())
    submission_df = pd.DataFrame({'id': test_df['id']})
    for i, label in enumerate(LABEL_COLUMNS):
        submission_df[label] = [row[i] for row in predictions]
    output_file = output_dir / f"pred_{language}.csv"
    submission_df.to_csv(output_file, index=False)
    logger.info(f"  Saved predictions to {output_file}")
    for i, label in enumerate(LABEL_COLUMNS):
        logger.info(f"  {label}: {sum(submission_df[label])}")
    return True

def create_submission_package(output_dir: Path, languages: list = None):
    subtask_dir = output_dir / "subtask_2"
    subtask_dir.mkdir(exist_ok=True)
    if languages is None:
        pred_files = list(output_dir.glob("pred_*.csv"))
    else:
        pred_files = [output_dir / f"pred_{lang}.csv" for lang in languages]
        pred_files = [f for f in pred_files if f.exists()]
    if not pred_files:
        logger.error("No prediction files found!")
        return None
    logger.info(f"\nPackaging {len(pred_files)} prediction files...")
    for pred_file in pred_files:
        dest_file = subtask_dir / pred_file.name
        shutil.copy2(pred_file, dest_file)
        logger.info(f"  Added {pred_file.name}")
    zip_path = output_dir / "subtask_2_submission.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pred_file in subtask_dir.glob("pred_*.csv"):
            arcname = f"subtask_2/{pred_file.name}"
            zipf.write(pred_file, arcname)
    logger.info(f"\nSubmission package created: {zip_path}")
    logger.info(f"Contains {len(pred_files)} language predictions")
    shutil.rmtree(subtask_dir)
    return zip_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate submission files for SemEval 2026 Task 9 - Subtask 2'
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--test_dir', type=str, default='./dev_phase/subtask2/dev', help='Directory containing test CSV files (one per language)')
    parser.add_argument('--output_dir', type=str, default='./submission', help='Directory to save prediction files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for binary classification (default: 0.5)')
    parser.add_argument('--languages', type=str, nargs='+', default=None, help='Specific languages to process (default: all available)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--no_zip', action='store_true', help='Do not create zip file (only generate pred files)')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    logger.info("="*60)
    logger.info("SemEval 2026 Task 9 - Subtask 2 Submission Generator")
    logger.info("="*60)
    predictor = MultiLabelPredictor(args.model_path, threshold=args.threshold)
    if args.languages:
        languages = args.languages
        logger.info(f"Processing {len(languages)} specified languages")
    else:
        test_files = list(test_dir.glob("*.csv"))
        languages = [f.stem for f in test_files if f.stem in LANGUAGES]
        logger.info(f"Found {len(languages)} language test files")
    logger.info("\n" + "="*60)
    logger.info("Generating Predictions")
    logger.info("="*60)
    successful_languages = []
    for lang in languages:
        success = generate_predictions_for_language(
            predictor, test_dir, lang, output_dir
        )
        if success:
            successful_languages.append(lang)
    logger.info("\n" + "="*60)
    logger.info(f"Successfully generated predictions for {len(successful_languages)}/{len(languages)} languages")
    logger.info("="*60)
    if not args.no_zip and successful_languages:
        zip_path = create_submission_package(output_dir, successful_languages)
        if zip_path:
            logger.info("\n" + "="*60)
            logger.info("SUBMISSION READY!")
            logger.info("="*60)
            logger.info(f"Upload this file to Codabench: {zip_path}")
            logger.info("\nInstructions:")
            logger.info("1. Go to Codabench My Submissions page")
            logger.info("2. Upload the zip file")
            logger.info("3. Wait for evaluation results")
            logger.info("\nNote: This submission includes predictions for:")
            for lang in successful_languages:
                logger.info(f"  - {lang}")
    else:
        logger.info(f"\nPrediction files saved to: {output_dir}")
        logger.info("Use --no_zip flag was set or no successful predictions. Zip file not created.")

if __name__ == '__main__':
    main()
