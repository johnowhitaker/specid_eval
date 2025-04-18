#!/bin/bash

# Script to evaluate multiple models on species identification task

MODELS=(
    "gemini-2.5-flash-preview-04-17"
    "gemini-2.5-pro-preview-03-25"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-14"
    "o3-2025-04-16"
    "gpt-4.1-mini-2025-04-14"
    "gpt-4.1-nano-2025-04-14"
)

# Number of samples (remove this parameter for full dataset)
# SAMPLES="--samples 10"
SAMPLES=""

for MODEL in "${MODELS[@]}"; do
    echo "Evaluating $MODEL..."
    python run_eval.py "$MODEL" $SAMPLES
    echo "--------------------------"
done

echo "All evaluations complete!"
