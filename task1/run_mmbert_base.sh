#!/bin/bash

# Training script for SemEval 2026 Task 9 - Subtask 1
# Run with nohup to keep training even after logout

echo "Starting training with Qwen2.5-0.5B model..."
echo "Output will be saved to training_output.log"
echo "Process ID will be saved to training.pid"

# Run training in background with nohup
nohup python train_merged.py --model_name jhu-clsp/mmBERT-base --epochs 10 --batch_size 64 > training_output_mmbert_base.log 2>&1 &

# Save the process ID
echo $! > training.pid

echo "Training started with PID: $(cat training.pid)"
echo ""
echo "To monitor progress, run:"
echo "  tail -f training_output.log"
echo ""
echo "To check if process is running:"
echo "  ps -p $(cat training.pid)"
echo ""
echo "To stop training:"
echo "  kill $(cat training.pid)"
