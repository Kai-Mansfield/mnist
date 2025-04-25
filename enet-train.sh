#!/bin/bash
#SBATCH --job-name=imagenet_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=compute

# Load conda from bashrc
source ~/.bashrc
conda activate mnist

# Optional info
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

# Ensure log & checkpoint dirs exist
mkdir -p logs
mkdir -p checkpoints

# Run training script
python train_imagenet.py

# Done
echo "✅ Training finished at: $(date)"
