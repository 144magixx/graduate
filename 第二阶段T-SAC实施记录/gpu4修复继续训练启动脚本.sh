#!/usr/bin/env bash
set -euo pipefail
test "$(hostname -s)" = "gpu4"
cd /dev/shm/SpectrumT-SAC-20260921-ah2j5yfv/PPO4090
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
/tmp/HorizonT-SAC-20260921-tWxdou/.venv-horizon/bin/python -c 'import json;from pathlib import Path;d=json.loads(Path('"'"'/home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1/SpectrumGPU预检结果.json'"'"').read_text());assert d['"'"'status'"'"']=='"'"'completed'"'"' and len(d['"'"'experiments'"'"'])==2 and d['"'"'pytest_exit_code'"'"']==0'
exec /tmp/HorizonT-SAC-20260921-tWxdou/.venv-horizon/bin/python -X utf8 -m implementations.spectrum_tsac_20260921.overnight --plan /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1/recovery_plan.json --durable-root /home/u2024110224/PPO4090/implementations/graduate_horizon_20260920/SpectrumT-SAC-20260921-v1/PPO4090
