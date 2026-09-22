"""只读旧环境动作回放审计；不导入旧训练入口、不读取旧 checkpoint。"""
import numpy as np


def replay_legacy(csv_path, actions):
    """actions 为 (start,length,power_w)，保留旧物理、旧截短及旧奖励。"""
    import pandas as pd
    from implementations.sac_fyh_io.Environment_fyh_IO import BeamInfo, SatelliteFreqState, CurrentBeamState, Env
    saved_rng = np.random.get_state()
    try:
        beams = BeamInfo(data_df=pd.read_csv(csv_path))
        env = Env(SatelliteFreqState(), beams, CurrentBeamState(beams.beam_info))
        env.reset(beams)
        result = []
        for start, length, power_w in actions:
            _, reward, done, info = env.step([env.beam_idx % 8, start, length, 10*np.log10(power_w)])
            result.append({"step": env.beam_idx, "reward": float(reward), "done": bool(done),
                           "power_used_w": float(env.total_power), "total_satisfaction": float(env.total_satisfaction),
                           "beam_rates_bps": [float(x) for x in env.beam_bps], "success_slots": info["success_slots"]})
            if done:
                break
        return result
    finally:
        np.random.set_state(saved_rng)
