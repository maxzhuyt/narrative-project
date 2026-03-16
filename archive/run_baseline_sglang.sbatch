#!/bin/bash
#SBATCH --job-name=baseline-sglang
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

module load cuda

source /project/jevans/maxzhuyt/sglang_env_setup.sh

mkdir -p /project/jevans/maxzhuyt/narrative_project/logs
mkdir -p /project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results_sglang
cd /project/jevans/maxzhuyt/narrative_project

echo "=== Starting at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not on PATH)"
echo ""

echo "=== Running SGLang on BASELINE_CORPUS (35 stories) ==="
/project/jevans/maxzhuyt/sglang_env/bin/python run_baseline_sglang.py "$@"

echo ""
echo "=== All done at $(date) ==="
