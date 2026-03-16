#!/bin/bash
#SBATCH --job-name=compute-dists
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/distances/slurm_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/distances/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

mkdir -p /project/jevans/maxzhuyt/narrative_project/distances

cd /project/jevans/maxzhuyt/narrative_project

echo "=== Starting at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python compute_distances.py

echo "=== Finished at $(date) ==="
