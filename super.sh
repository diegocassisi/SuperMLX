#!/bin/bash
# [AI_DIRECTIVE]
# ROL: Launch script for SuperMLX server
# OBJETIVO: Activate venv with selectable MLX version overlay
# ENTRADAS: MLX_VERSION env var (default: 0.31)
# SALIDAS: Running SuperMLX server process
# REGLAS INVIOLABLES:
# - Default to stable (0.31) if no version specified
# - PYTHONPATH overlay must come before venv site-packages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_ENV="${MLX_VERSION:-0.31}"
MLX_OVERLAY="${SCRIPT_DIR}/envs/mlx-${MLX_ENV}"

if [ ! -d "$MLX_OVERLAY" ]; then
    echo "ERROR: MLX overlay not found: $MLX_OVERLAY"
    echo "Available: $(ls envs/ 2>/dev/null)"
    exit 1
fi

source "${SCRIPT_DIR}/venv/bin/activate"
export PYTHONPATH="${MLX_OVERLAY}:${PYTHONPATH}"

echo "SuperMLX launching with MLX ${MLX_ENV} overlay"
sudo sysctl iogpu.wired_limit_mb=19500 && sudo purge
nice -n 19 python SuperMLX.py
