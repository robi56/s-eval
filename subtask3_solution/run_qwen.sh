

#!/bin/bash

# Training script for SemEval 2026 Task 9 - Subtask 3
# Run this in Ubuntu after activating your Python environment

echo "Starting training for Subtask 3..."
echo "Output will be saved to training_output_subtask3.log"
echo "Process ID will be saved to training_subtask3.pid"

nohup python train_merged_subtask3.py --model_name Qwen/Qwen3-0.6B --output_dir models_qwen --epochs 10 --batch_size 16 > training_output_subtask3.log 2>&1 &
echo $! > training_subtask3.pid

echo "Training started with PID: $(cat training_subtask3.pid)"
echo ""
echo "To monitor progress, run:"
echo "  tail -f training_output_subtask3.log"
echo ""
echo "To check if process is running:"
echo "  ps -p $(cat training_subtask3.pid)"
echo ""
echo "To stop training:"
echo "  kill $(cat training_subtask3.pid)"
