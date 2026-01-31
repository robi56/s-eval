#!/bin/bash
# Run fine-tuning LLM script with nohup in background

echo "Starting LLM fine-tuning in background..."

# Run with nohup and redirect output to log file
nohup python finetune_llm_1.py \
    --model_name google/gemma-3-270m \
    --train_file ./data/train_merged.csv \
    --output_dir ./models/finetuned_llm \
    --num_epochs 3 \
    --per_device_batch_size 4 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --val_split 0.1 \
    > finetune_llm_output_gemma.log 2>&1 &

# Save the process ID
echo $! > finetune_llm.pid

echo "Fine-tuning started!"
echo "Process ID: $(cat finetune_llm.pid)"
echo ""
echo "To monitor progress:"
echo "  tail -f finetune_llm_output.log"
echo ""
echo "To check status:"
echo "  ./check_finetune_llm.sh"
echo ""
echo "To stop training:"
echo "  ./stop_finetune_llm.sh"
