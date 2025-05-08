#!/bin/bash
#SBATCH --job-name=effnet
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2           # 2 processes per node
#SBATCH --gpus-per-node=2             # 2 GPUs per node
#SBATCH --cpus-per-task=8             # Number of CPU threads per process (adjust as needed)
#SBATCH --mem=0                       # Use all available memory
#SBATCH --time=24:00:00
#SBATCH --partition=gpu               # Adjust to your cluster’s partition name
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load modules and activate your environment
module load cuda/12.1  # or whatever CUDA version your cluster uses
source activate mnist

# Set the number of processes (world size)
WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Launch the distributed job with torchrun
torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --node_rank=$SLURM_NODEID \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(scontrol show hostnames $SLURM_NODELIST | head -n1):29500 \
  train.py \
  --model efficientnet_b0 \
  --data-path /mnt/lustre/users/inf/kajm20/ILSVRC/Data/CLS-LOC \
  --workers 4 \
  --batch-size 64
