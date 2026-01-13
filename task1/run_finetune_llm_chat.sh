#!/bin/bash
# Run fine-tuning LLM script with chat format using nohup in background

echo "Starting LLM fine-tuning with chat format in background..."

# Run with nohup and redirect output to log file
nohup python finetune_llm.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --train_file ./data/train_merged.csv \
    --output_dir ./models/finetuned_llm_chat \
    --num_epochs 3 \
    --per_device_batch_size 4 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --val_split 0.1 \
    --use_chat_format \
    > finetune_llm_chat_output.log 2>&1 &

# Save the process ID
echo $! > finetune_llm_chat.pid

echo "Fine-tuning with chat format started!"
echo "Process ID: $(cat finetune_llm_chat.pid)"
echo ""
echo "To monitor progress:"
echo "  tail -f finetune_llm_chat_output.log"
echo ""
echo "To check status:"
echo "  ps -p $(cat finetune_llm_chat.pid)"
echo ""
echo "To stop training:"
echo "  kill $(cat finetune_llm_chat.pid)"
