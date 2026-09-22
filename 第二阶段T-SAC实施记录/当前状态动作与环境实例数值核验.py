"""合成教学场景：只运行环境，不训练；由真实实现核验状态/动作/物理/奖励。"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PPO4090"))

import numpy as np
from project_paths import HORIZON_REPORT_DIR, PROJECT_ROOT
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.data.loader import scenario_from_rows
from implementations.horizon_tsac_20260920.env.environment import Environment
from implementations.horizon_tsac_20260920.env.action import Action


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def main():
    config = Config()
    rows = []
    for beam_id, demand_mbps, group_id, entity, diameter in (
        (10, 600, 0, True, 2.0), (20, 500, 2, True, 2.0),
        (30, 400, 1, True, 2.0), (40, 300, 0, True, 2.0),
        (50, 0, 3, True, 2.0), (60, 0, 0, False, 0.0),
    ):
        rows.append(dict(beam_id=beam_id, latitude_deg=0.0, longitude_deg=122.2,
                         demand_bps=demand_mbps*1e6, ground_diameter_deg=diameter,
                         group_id=group_id, polarization_id=group_id % 2, entity_mask=entity))
    scenario = scenario_from_rows(rows, config, source_schema="coverage.v2",
                                  scenario_id="合成教学场景-四个正需求波束",
                                  metadata={"purpose": "教学和公式核验，不是训练/验证/测试性能样本"})
    env = Environment(config)
    obs, _ = env.reset(scenario)
    actions = [Action("ALLOC", 0, 2, 9), Action("ALLOC", 0, 2, 9),
               Action("ALLOC", 0, 2, 5), Action()]
    observations = [native(obs)]
    checkpoints = []

    def snapshot_evaluation(label):
        result = env.evaluate()
        return native(dict(label=label, cursor=env.cursor, current_beam=env.current_beam_id,
                           remaining_power_w=env.remaining_power_w,
                           valid_action_count=int(env.observe()["valid_action_mask"].sum()),
                           rates_mbps=result.rate_bps/1e6, slot_power_w=result.slot_power_w[:, :4],
                           signal_w=result.signal_w[:, :4], interference_w=result.interference_w[:, :4],
                           noise_w=result.noise_w[:, :4], sinr=result.sinr_linear[:, :4],
                           metrics=result.metrics, violations=result.constraint_violations))

    checkpoints.append(snapshot_evaluation("初始"))
    transitions = []
    invalid_action_test = None
    for index, action in enumerate(actions):
        if index == 3:
            # D与A同组，重用0/1槽非法；验证失败不改变环境。
            before = native(env.state_dict())
            try:
                env.step(Action("ALLOC", 0, 2, 9))
            except ValueError as error:
                invalid_action_test = dict(rejected=True, error=str(error), state_unchanged=before == native(env.state_dict()))
            else:
                raise AssertionError("同组冲突动作未被拒绝")
            assert invalid_action_test["state_unchanged"]
        current = env.current_beam_id
        action_id = env.action_spec.encode(action)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(native(obs))
        transitions.append(native(dict(beam_id=current, action_id=action_id, action=action.to_dict(),
                                       reward=reward, terminated=terminated, truncated=truncated,
                                       reward_terms=info["reward_terms"], forced_skip=info["forced_skip"])))
        checkpoints.append(snapshot_evaluation(f"动作{index+1}之后"))

    # 独立标量公式：此特殊几何下所有真实中心同位置，夹角0且斜距=R-r。
    d = (42164.0-6371.0)*1000.0
    f = np.array([17_700_000_000.0+(s+0.5)*25_000_000.0 for s in range(2)])
    h = np.array([10**(50/10)*10**(40/10)*(299792458.0/(4*math.pi*d*x))**2 for x in f])
    noise = 1.380649e-23*290*25_000_000.0
    margin = 10**0.5
    r50 = float(sum(25_000_000.0*math.log2(1+25*x/(margin*noise)) for x in h))
    rpair = float(sum(25_000_000.0*math.log2(1+25*x/(margin*(noise+25*x))) for x in h))
    r30 = float(sum(25_000_000.0*math.log2(1+15*x/(margin*noise)) for x in h))
    expected_rates = np.array([[0, 0, 0, 0], [r50, 0, 0, 0], [rpair, rpair, 0, 0],
                               [rpair, rpair, r30, 0], [rpair, rpair, r30, 0]])
    expected_u = np.minimum(expected_rates/np.array([600e6, 500e6, 400e6, 300e6]), 1).mean(1)
    expected_reward = 100*np.diff(expected_u)
    actual_rates = np.array([x["rates_mbps"][:4] for x in checkpoints])*1e6
    actual_rewards = np.array([x["reward"] for x in transitions])
    np.testing.assert_allclose(actual_rates, expected_rates, atol=1e-5, rtol=1e-12)
    np.testing.assert_allclose(actual_rewards, expected_reward, atol=1e-10, rtol=1e-12)
    assert [x["valid_action_count"] for x in checkpoints] == [9551, 9551, 9551, 9351, 0]
    assert [x["action_id"] for x in transitions] == [20, 20, 16, 0]
    assert transitions[1]["reward"] < 0
    assert math.isclose(sum(actual_rewards), 100*checkpoints[-1]["metrics"]["mean_satisfaction"], abs_tol=1e-10)

    # 相同场景/动作再沿Spectrum包运行，避免只凭同名判断规则相同。
    from implementations.spectrum_tsac_20260921.config import Config as SpectrumConfig
    from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows as spectrum_scenario
    from implementations.spectrum_tsac_20260921.env.environment import Environment as SpectrumEnvironment
    from implementations.spectrum_tsac_20260921.env.action import Action as SpectrumAction
    spectrum_config = SpectrumConfig()
    for section in ("physics", "env", "data"):
        assert config.to_dict()[section] == spectrum_config.to_dict()[section]
    spectrum_env = SpectrumEnvironment(spectrum_config)
    spectrum_obs, _ = spectrum_env.reset(spectrum_scenario(rows, spectrum_config, source_schema="coverage.v2"))
    for key in ("legacy205", "beam_static", "beam_dynamic", "global_features", "current_slot_features", "valid_action_mask"):
        np.testing.assert_array_equal(spectrum_obs[key], np.asarray(observations[0][key]))
    for action, expected in zip(actions, transitions):
        _, reward, terminated, truncated, _ = spectrum_env.step(SpectrumAction(**action.to_dict()))
        assert reward == expected["reward"]
        assert terminated == expected["terminated"] and truncated == expected["truncated"]
    np.testing.assert_array_equal(spectrum_env.evaluate().rate_bps, env.evaluate().rate_bps)

    paths = ["env/environment.py", "env/action.py", "env/physics.py", "env/reward.py", "env/metrics.py", "env/state.py"]
    fingerprints = {}
    for path in paths:
        first = PROJECT_ROOT / "implementations/horizon_tsac_20260920" / path
        second = PROJECT_ROOT / "implementations/spectrum_tsac_20260921" / path
        fingerprints[path] = dict(horizon_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
                                  spectrum_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
                                  byte_identical=first.read_bytes() == second.read_bytes())
    result = native(dict(scope="合成教学实例；固定动作；没有模型推理或训练", config=config.to_dict(),
                         scenario=scenario.to_dict(), observations=observations, checkpoints=checkpoints,
                         transitions=transitions, illegal_action_test=invalid_action_test,
                         scalar_derivation=dict(distance_m=d, slot_frequency_hz=f, path_gain=h,
                             noise_w=noise, margin_linear=margin, isolated_50w_rate_bps=r50,
                             paired_50w_rate_bps=rpair, isolated_30w_rate_bps=r30,
                             expected_U=expected_u, expected_reward=expected_reward,
                             max_rate_abs_error_bps=float(np.max(np.abs(actual_rates-expected_rates))),
                             max_reward_abs_error=float(np.max(np.abs(actual_rewards-expected_reward)))),
                         source_fingerprints=fingerprints, spectrum_same_case_equal=True))
    destination = HORIZON_REPORT_DIR / "当前状态动作与环境实例数值核验.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(destination), checkpoints=checkpoints, transitions=transitions,
                         scalar_derivation=result["scalar_derivation"], illegal_action_test=invalid_action_test,
                         source_fingerprints=fingerprints), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
