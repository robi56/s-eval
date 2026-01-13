"""
Generate Submission Files for SemEval 2026 Task 9 - Subtask 1
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

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


class SequenceClassificationPredictor:
    """Predictor for sequence classification models (BERT, RoBERTa, etc.)"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading sequence classification model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def predict(self, texts: list, batch_size: int = 32, max_length: int = 128):
        """Generate predictions for a list of texts"""
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
        
        return all_predictions


class LLMPredictor:
    """Predictor for fine-tuned LLM models"""
    
    def __init__(self, model_path: str, device: str = None, use_chat_format: bool = False):
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_chat_format = use_chat_format
        
        logger.info(f"Loading LLM model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        logger.info("LLM model loaded")
    
    def format_prompt(self, text: str):
        """Format text as instruction prompt"""
        instruction = """Analyze if the following text contains polarized content. Polarized content includes:
- Division between groups (us vs them)
- Stereotyping or generalizations
- Vilification or dehumanization
- Intolerance or lack of empathy

Classify the text as either "Polarized" or "Not Polarized"."""
        
        if self.use_chat_format:
            messages = [
                {"role": "user", "content": f"{instruction}\n\nText: {text}\n\nClassification:"}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"{instruction}\n\nText: {text}\n\nClassification:"
    
    def parse_output(self, text: str) -> int:
        """Parse LLM output to binary label"""
        text_lower = text.lower().strip()
        if 'polarized' in text_lower and 'not polarized' not in text_lower:
            return 1
        return 0
    
    def predict(self, texts: list, batch_size: int = 8, max_new_tokens: int = 10):
        """Generate predictions for a list of texts"""
        all_predictions = []
        
        for text in tqdm(texts, desc="Predicting"):
            prompt = self.format_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction = self.parse_output(generated_text)
            all_predictions.append(prediction)
        
        return all_predictions


def load_test_data(test_dir: Path, language: str):
    """
    Load test data for a specific language
    
    Args:
        test_dir: Directory containing test CSV files
        language: Language code (e.g., 'eng', 'amh')
    
    Returns:
        DataFrame with 'id' and 'text' columns
    """
    test_file = test_dir / f"{language}.csv"
    
    if not test_file.exists():
        logger.warning(f"Test file not found: {test_file}")
        return None
    
    df = pd.read_csv(test_file)
    
    # Check required columns
    if 'id' not in df.columns or 'text' not in df.columns:
        logger.error(f"Required columns 'id' and 'text' not found in {test_file}")
        return None
    
    return df


def generate_predictions_for_language(predictor, test_dir: Path, language: str, output_dir: Path):
    """
    Generate predictions for a single language
    
    Args:
        predictor: Model predictor (SequenceClassificationPredictor or LLMPredictor)
        test_dir: Directory containing test data
        language: Language code
        output_dir: Directory to save prediction file
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\nProcessing {language}...")
    
    # Load test data
    test_df = load_test_data(test_dir, language)
    if test_df is None:
        return False
    
    logger.info(f"  Loaded {len(test_df)} samples")
    
    # Generate predictions
    predictions = predictor.predict(test_df['text'].tolist())
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'polarization': predictions
    })
    
    # Save prediction file
    output_file = output_dir / f"pred_{language}.csv"
    submission_df.to_csv(output_file, index=False)
    
    logger.info(f"  Saved predictions to {output_file}")
    logger.info(f"  Polarized: {sum(predictions)}, Not polarized: {len(predictions) - sum(predictions)}")
    
    return True


def create_submission_package(output_dir: Path, languages: list = None):
    """
    Package prediction files into submission structure
    
    Args:
        output_dir: Directory containing pred_*.csv files
        languages: List of languages to include (None = all found)
    
    Returns:
        Path to created zip file
    """
    # Create subtask_1 folder
    subtask_dir = output_dir / "subtask_1"
    subtask_dir.mkdir(exist_ok=True)
    
    # Copy prediction files to subtask_1 folder
    if languages is None:
        # Find all pred_*.csv files
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
    
    # Create zip file
    zip_path = output_dir / "subtask_1_submission.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pred_file in subtask_dir.glob("pred_*.csv"):
            arcname = f"subtask_1/{pred_file.name}"
            zipf.write(pred_file, arcname)
    
    logger.info(f"\nSubmission package created: {zip_path}")
    logger.info(f"Contains {len(pred_files)} language predictions")
    
    # Clean up subtask_1 folder
    shutil.rmtree(subtask_dir)
    
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate submission files for SemEval 2026 Task 9 - Subtask 1'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='./dev_phase/subtask1/dev',
        help='Directory containing test CSV files (one per language)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./submission',
        help='Directory to save prediction files'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['sequence_classification', 'llm'],
        default='sequence_classification',
        help='Type of model (sequence_classification or llm)'
    )
    
    parser.add_argument(
        '--use_chat_format',
        action='store_true',
        help='Use chat format for LLM (only for --model_type llm)'
    )
    
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=None,
        help='Specific languages to process (default: all available)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--no_zip',
        action='store_true',
        help='Do not create zip file (only generate pred files)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Initialize predictor
    logger.info("="*60)
    logger.info("SemEval 2026 Task 9 - Subtask 1 Submission Generator")
    logger.info("="*60)
    
    if args.model_type == 'sequence_classification':
        predictor = SequenceClassificationPredictor(args.model_path)
    else:  # llm
        predictor = LLMPredictor(args.model_path, use_chat_format=args.use_chat_format)
    
    # Determine languages to process
    if args.languages:
        languages = args.languages
        logger.info(f"Processing {len(languages)} specified languages")
    else:
        # Find all available test files
        test_files = list(test_dir.glob("*.csv"))
        languages = [f.stem for f in test_files if f.stem in LANGUAGES]
        logger.info(f"Found {len(languages)} language test files")
    
    # Generate predictions for each language
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
    
    # Create submission package
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
