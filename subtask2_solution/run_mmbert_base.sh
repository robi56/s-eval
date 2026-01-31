#!/bin/bash

# Training script for SemEval 2026 Task 9 - Subtask 2
# Run this in Ubuntu after activating your Python environment

echo "Starting training for Subtask 2..."
echo "Output will be saved to training_output_subtask2.log"
echo "Process ID will be saved to training_subtask2.pid"

nohup python train_merged_subtask2.py > training_output_subtask2.log 2>&1 &
echo $! > training_subtask2.pid

echo "Training started with PID: $(cat training_subtask2.pid)"
echo ""
echo "To monitor progress, run:"
echo "  tail -f training_output_subtask2.log"
echo ""
echo "To check if process is running:"
echo "  ps -p $(cat training_subtask2.pid)"
echo ""
echo "To stop training:"
echo "  kill $(cat training_subtask2.pid)"
