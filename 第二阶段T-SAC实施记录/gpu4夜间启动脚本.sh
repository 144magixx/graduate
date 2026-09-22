#!/usr/bin/env bash
set -euo pipefail
test "$(hostname -s)" = "gpu4"
cd /tmp/HorizonT-SAC-20260921-tWxdou/PPO4090
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
exec /tmp/HorizonT-SAC-20260921-tWxdou/.venv-horizon/bin/python -X utf8 -m implementations.horizon_tsac_20260920.overnight \
  --plan /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/HorizonT-SAC-20260921-v1/night_plan.json \
  --durable-root /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/HorizonT-SAC-20260921-v1/PPO4090
