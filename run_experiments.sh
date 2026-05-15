#!/bin/bash
source .venv/bin/activate

echo "Queueing Experiment 1: Fast Baseline (2 epochs)"
dvc exp run --queue -n exp1_baseline -S train.epochs=2

echo "Queueing Experiment 2: Fast Precision (3 epochs, batch 8)"
dvc exp run --queue -n exp2_precision -S train.epochs=3 -S train.batch=8

echo "Queueing Experiment 3: Fast Champion (5 epochs)"
dvc exp run --queue -n exp3_champion -S train.epochs=5

echo "Running all queued experiments in sequence..."
dvc exp run --run-all

echo "All experiments complete! Use 'dvc exp show' to see the results table."
