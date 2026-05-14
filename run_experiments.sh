#!/bin/bash
source .venv/bin/activate

echo "🚀 Starting Experiment 1: Fast Baseline (2 epochs)"
python3 src/training/train.py --epochs 2 --model yolov8n

echo "🚀 Starting Experiment 2: Fast Precision (3 epochs)"
python3 src/training/train.py --epochs 3 --batch 8 --model yolov8n

echo "🚀 Starting Experiment 3: Fast Champion (5 epochs)"
python3 src/training/train.py --epochs 5 --model yolov8n

echo "✅ All 3 experiments complete! Check them at http://localhost:5454"
