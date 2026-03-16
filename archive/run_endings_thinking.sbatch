#!/bin/bash
#SBATCH --job-name=endings-think
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=results_thinking/slurm_%j.out
#SBATCH --error=results_thinking/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Persistent cache dirs (avoid stale scratch paths)
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR

# Clean stale compile caches
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf /project/jevans/maxzhuyt/cache/torchinductor/

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

mkdir -p /project/jevans/maxzhuyt/narrative_project/results_thinking

cd /project/jevans/maxzhuyt/narrative_project

echo "=== Starting at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python run_endings_thinking.py

echo "=== Finished at $(date) ==="
