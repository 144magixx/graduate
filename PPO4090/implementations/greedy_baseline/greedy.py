#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_round_robin.py
———————————————
• 频组按波束序号轮循：group = beam_idx % 8
• 在该组内选择干扰最小的起始槽
• 共 4000 episode，输出 returns / satisfactions 到 .npz
"""

import os, math, random
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import trange
from implementations.greedy_baseline.Environment_103 import (
    SatelliteFreqState,
    BeamInfo,
    CurrentBeamState,
    Env,
)
from project_paths import get_output_dir

EPISODES = 4000
SAVE_DIR = get_output_dir("greedy_baseline", "metrics")

def build_env():
    bi = BeamInfo()
    env = Env(SatelliteFreqState(), bi, CurrentBeamState(bi.beam_info))
    return env, bi

def find_best_slot(env: Env, grp: int, slot_len: int):
    """仅在指定 grp 内寻找未占用且平均干扰最小的起始槽"""
    occ = env.satellite_freq_state.freq_pool_real       # (8,100)
    intf_vec = env._calc_group_slot_interference(grp)   # (100,)
    best_start, best_val = 0, 1e30

    for f in range(0, 100 - slot_len + 1):
        if occ[grp, f : f + slot_len].any():
            continue
        val = intf_vec[f : f + slot_len].mean()
        if val < best_val:
            best_start, best_val = f, val

    return best_start  # 若全部占用，仍返回 0

returns, sats = [], []

for ep in trange(EPISODES, desc="Greedy‑RR"):
    env, bi = build_env()
    state  = env.reset(bi)
    done   = False
    ep_ret = 0.0
    ep_sat = 0.0

    while not done:
        slot_len = random.randint(1, 10)          # 1–10 个槽
        group    = env.beam_idx % 8               # 轮循频组
        start    = find_best_slot(env, group, slot_len)
        pwr_idx  = 0                              # 固定最小功率档

        env_action = [
            group,
            start,
            slot_len,
            10 * math.log10((pwr_idx + 1) * 5),
        ]
        next_state, reward, done, info = env.step(env_action)
        state   = next_state
        ep_ret += reward
        ep_sat  = info["total_satisfaction"]

    returns.append(ep_ret)
    sats.append(ep_sat)

# ----- 保存 -----
os.makedirs(SAVE_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d%H%M")
metrics_path = SAVE_DIR / f"metrics_{ts}.npz"
np.savez(metrics_path,
         returns=np.array(returns),
         satisfactions=np.array(sats))
print(f"✔ Metrics saved to {metrics_path}")
