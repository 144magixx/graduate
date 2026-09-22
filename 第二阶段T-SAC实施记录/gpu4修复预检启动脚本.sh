#!/usr/bin/env bash
set -euo pipefail
test "$(hostname -s)" = "gpu4"
cd /dev/shm/SpectrumT-SAC-20260921-ah2j5yfv/PPO4090
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
/tmp/HorizonT-SAC-20260921-tWxdou/.venv-horizon/bin/python -X utf8 /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1/preflight.py /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1/recovery_plan.json /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1
