#!/bin/bash

# Training script for SemEval 2026 Task 9 - Subtask 1
# Run with nohup to keep training even after logout

echo "Starting training with Qwen2.5-0.5B model..."
echo "Output will be saved to training_output_mmbert_base.log"
echo "Process ID will be saved to training.pid"

# Run training in background with nohup
nohup python train_advanced.py \
    --model_name jhu-clsp/mmBERT-base \
    --data_root "../dev_phase" \
    --output_dir "./models/mmbert_focal" \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_length 128 \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --lr_scheduler_type cosine \
    --early_stopping_patience 3 \
    --use_fp16 \
    --seed 42 > training_output_mmbert_base_focal.log 2>&1 &

# Save the process ID
echo $! > training.pid

echo "Training started with PID: $(cat training.pid)"
echo ""
echo "To monitor progress, run:"
echo "  tail -f training_output_mmbert_base.log"
echo ""
echo "To check if process is running:"
echo "  ps -p $(cat training.pid)"
echo ""
echo "To stop training:"
echo "  kill $(cat training.pid)"
