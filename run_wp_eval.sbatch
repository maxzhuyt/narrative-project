#!/bin/bash
#SBATCH --job-name=wp-eval
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results/slurm_wp_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results/slurm_wp_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf /project/jevans/maxzhuyt/cache/torchinductor/

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

mkdir -p /project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results

cd /project/jevans/maxzhuyt/narrative_project

echo "=== Starting at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python run_wp_eval.py

echo "=== Finished at $(date) ==="
