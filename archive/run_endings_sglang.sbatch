#!/bin/bash
#SBATCH --job-name=endings-sglang
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

# ── Initialize modules so conda is available in SLURM non-interactive shell ──
module load cuda

# ── SGLang env (GCC 13.2 + CUDA 12.9 + sglang_env conda) ─────────────────
source /project/jevans/maxzhuyt/sglang_env_setup.sh

# ── Persistent cache dirs (avoid stale scratch references) ────────────────
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
rm -rf ~/.cache/vllm/torch_compile_cache/

# ── Working directories ──────────────────────────────────────────────────
mkdir -p /project/jevans/maxzhuyt/narrative_project/logs
mkdir -p /project/jevans/maxzhuyt/narrative_project/results_scaled_qwen35
cd /project/jevans/maxzhuyt/narrative_project

echo "=== Starting at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi not on PATH)"
echo ""

# ── Run offline engine (no HTTP server needed) ────────────────────────────
echo "=== Running SGLang offline engine ==="
/project/jevans/maxzhuyt/sglang_env/bin/python run_endings_sglang.py "$@"

echo ""
echo "=== All done at $(date) ==="
