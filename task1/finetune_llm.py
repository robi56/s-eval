"""
Fine-tuning LLM for Polarization Detection
Binary classification using instruction-tuned LLMs
"""

import os
import argparse
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import json

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune LLM for polarization detection")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="Hugging Face model ID to use for fine-tuning")
parser.add_argument("--output_dir", type=str, default="./models/finetuned_llm",
                    help="Output directory to save the model checkpoints")
parser.add_argument("--train_file", type=str, default="./data/train_merged.csv",
                    help="Path to merged training data CSV")
parser.add_argument("--num_epochs", type=int, default=3,
                    help="Number of training epochs")
parser.add_argument("--per_device_batch_size", type=int, default=4,
                    help="Batch size per device during training")
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Initial learning rate")
parser.add_argument("--max_seq_length", type=int, default=256,
                    help="Maximum sequence length for tokenization")
parser.add_argument("--val_split", type=float, default=0.1,
                    help="Validation split ratio")
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume training from")
parser.add_argument("--use_chat_format", action="store_true",
                    help="Use chat format for instruction tuning")
args = parser.parse_args()

# --- 2. Load Model and Tokenizer ---
print(f"Loading tokenizer and model: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chat template for instruction format
if args.use_chat_format:
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}"
    
    # Add special tokens if using chat format
    special_tokens_dict = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens for chat format")
    
    tokenizer.chat_template = chat_template
    print("Chat template set on tokenizer")

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Resize embeddings if tokens were added
if args.use_chat_format and num_added > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model embeddings resized to: {len(tokenizer)}")

print(f"Model loaded with {model.num_parameters():,} parameters")
print(f"Using device: {model.device}")

# --- 3. Load and Prepare Dataset ---
print(f"\nLoading training data from: {args.train_file}")
df = pd.read_csv(args.train_file)

# Filter out any missing values
df = df.dropna(subset=['text', 'polarization'])
print(f"Total samples: {len(df)}")
print(f"Polarized: {(df['polarization'] == 1).sum()}")
print(f"Not Polarized: {(df['polarization'] == 0).sum()}")

# Split train/val
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, 
    test_size=args.val_split,
    random_state=42,
    stratify=df['polarization']
)
print(f"\nTrain samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")


def format_instruction_prompt(text, label=None, use_chat=False):
    """
    Format text and label as instruction-following prompt
    
    Args:
        text: Input text
        label: Polarization label (0 or 1), None for inference
        use_chat: Whether to use chat format
    
    Returns:
        Formatted prompt string
    """
    # Define the instruction
    instruction = """Analyze if the following text contains polarized content. Polarized content includes:
- Division between groups (us vs them)
- Stereotyping or generalizations
- Vilification or dehumanization
- Intolerance or lack of empathy

Classify the text as either "Polarized" or "Not Polarized"."""
    
    if use_chat:
        # Chat format
        messages = [
            {"role": "user", "content": f"{instruction}\n\nText: {text}\n\nClassification:"}
        ]
        if label is not None:
            label_text = "Polarized" if label == 1 else "Not Polarized"
            messages.append({"role": "assistant", "content": label_text})
        
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(label is None)
        )
    else:
        # Plain instruction format
        formatted = f"{instruction}\n\nText: {text}\n\nClassification:"
        if label is not None:
            label_text = "Polarized" if label == 1 else "Not Polarized"
            formatted += f" {label_text}"
    
    return formatted


def prepare_dataset(dataframe, use_chat=False):
    """Convert dataframe to formatted prompts"""
    prompts = []
    
    print("Formatting prompts...")
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        prompt = format_instruction_prompt(
            row['text'], 
            row['polarization'],
            use_chat=use_chat
        )
        prompts.append(prompt)
    
    # Create HF dataset
    dataset = Dataset.from_dict({"text": prompts})
    return dataset


# Prepare datasets
print("\nPreparing training dataset...")
train_dataset = prepare_dataset(train_df, use_chat=args.use_chat_format)

print("Preparing validation dataset...")
val_dataset = prepare_dataset(val_df, use_chat=args.use_chat_format)


def tokenize_function(examples):
    """Tokenize the prompts"""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Tokenize datasets
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train"
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing validation"
)

print("Tokenization complete")

# --- 4. Data Collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# --- 5. Training Arguments ---
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.per_device_batch_size,
    per_device_eval_batch_size=args.per_device_batch_size * 2,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
)

# --- 6. Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. Train ---
print("\n" + "="*60)
print("Starting Training")
print("="*60)
print(f"Model: {args.model_name}")
print(f"Train samples: {len(tokenized_train)}")
print(f"Val samples: {len(tokenized_val)}")
print(f"Epochs: {args.num_epochs}")
print(f"Batch size: {args.per_device_batch_size}")
print(f"Learning rate: {args.learning_rate}")
print("="*60)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# --- 8. Save Final Model ---
print(f"\nTraining complete!")
print(f"Saving final model to {args.output_dir}/final_model")
trainer.save_model(f"{args.output_dir}/final_model")
tokenizer.save_pretrained(f"{args.output_dir}/final_model")

# Save training info
training_info = {
    "model_name": args.model_name,
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "epochs": args.num_epochs,
    "batch_size": args.per_device_batch_size,
    "learning_rate": args.learning_rate,
    "max_seq_length": args.max_seq_length,
    "use_chat_format": args.use_chat_format,
}

with open(f"{args.output_dir}/training_info.json", 'w') as f:
    json.dump(training_info, f, indent=2)

print(f"\nTraining info saved to {args.output_dir}/training_info.json")
print("\n" + "="*60)
print("Fine-tuning Complete!")
print("="*60)
print(f"\nModel saved at: {args.output_dir}/final_model")
print("\nTo use the model for inference:")
print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}/final_model')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}/final_model')")
