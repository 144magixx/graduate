"""只读复核冻结PPO权重：原联合MAP、分层MAP与固定三seed采样。

从 PPO4090 根运行：.venv-horizon/Scripts/python.exe ../第二阶段T-SAC实施记录/MLP-PPO完整验证集解码复核.py
仅validation；不训练、不改变原结果、配置或部署源码。分层MAP是诊断策略，不是原主结果。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# 仅约束诊断进程的CPU线程，避免小矩阵在BLAS线程池反复调度。
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PPO4090"))
from project_paths import PROJECT_ROOT, HORIZON_REPORT_DIR, COVER_OUTPUT_DIR
from implementations.horizon_tsac_20260920.artifacts import load_checkpoint, atomic_json
from implementations.horizon_tsac_20260920.audit import sha256
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.data.loader import load_scenario
from implementations.horizon_tsac_20260920.env.environment import Environment
from implementations.horizon_tsac_20260920.paper_baselines.ppo import BaselinePPO


def main():
    started = time.perf_counter()
    print("开始只读复核：CPU计算线程=1；固定五种解码/采样设置", flush=True)
    report_dir = HORIZON_REPORT_DIR
    checkpoint = report_dir / "PPO确定性解码诊断原件/检查点.pt"
    manifest_path = report_dir / "最终结果原件20260921-0601/学习运行清单/20260920T203641_73d4515809fe/运行清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = Config.from_dict(manifest["config"])
    config.train.device = "cpu"
    torch.set_num_threads(1)
    saved = load_checkpoint(checkpoint, config, manifest["data_manifest_hash"])
    records = [x for x in manifest["data_manifest"]["records"] if x["split"] == "validation"]
    assert len(records) == 15
    paths = ("paper_baselines/ppo.py", "paper_baselines/train.py", "evaluate.py", "config.py",
             "env/environment.py", "env/action.py", "env/physics.py", "env/reward.py", "data/loader.py", "models/networks.py")
    source_checks = {}
    for suffix in paths:
        key = "implementations/horizon_tsac_20260920/" + suffix
        actual = sha256(PROJECT_ROOT / key)
        source_checks[key] = {"sha256": actual, "matches_training": actual == manifest["code_hashes"][key]}
    assert all(x["matches_training"] for x in source_checks.values())
    # 事先固定所有诊断模式与采样seed，不按结果选取。
    modes = [("joint_map", None), ("hierarchical_map_diagnostic", None),
             ("sampling", 0), ("sampling", 1), ("sampling", 2)]
    results, steps = [], []
    progress_lock = Lock()

    def run_mode(mode_seed):
        mode, seed = mode_seed
        for record in records:
            source_path = COVER_OUTPUT_DIR / record["file"]
            assert sha256(source_path) == record["source_hash"]
            order_seed = int(hashlib.sha256((str(config.train.seed) + ":" + record["scenario_id"]).encode()).hexdigest()[:16], 16)
            scenario = load_scenario(source_path, config, rng=np.random.default_rng(order_seed))
            env = Environment(config)
            obs, _ = env.reset(scenario)
            agent = BaselinePPO(config, env.action_spec, obs, hyperparameters=manifest["baseline_settings"])
            agent.load_state_dict(saved["agent"], restore_rng=False)
            agent.train(False)
            if seed is not None:
                agent.action_rng.manual_seed(seed)
            episode_return = 0.0
            skip_count = 0
            reversal_count = 0
            while not obs["terminal"]:
                p = agent.probabilities(obs)
                p_skip, p_alloc, best_alloc = float(p[0]), float(p[1:].sum(dtype=np.float64)), float(p[1:].max())
                joint_id = int(p.argmax())
                hierarchical_id = 0 if p_skip >= p_alloc else 1 + int(p[1:].argmax())
                reversal = joint_id == 0 and p_alloc > p_skip
                reversal_count += int(reversal)
                if mode == "joint_map":
                    action = agent.act(obs, deterministic=True)
                    assert env.action_spec.encode(action) == joint_id
                    steps.append(dict(scenario_id=scenario.scenario_id, step=env.cursor,
                                      beam_id=env.current_beam_id, p_skip=p_skip, p_alloc=p_alloc,
                                      max_single_alloc=best_alloc, legal_alloc_count=int(obs["valid_action_mask"][1:].sum()),
                                      joint_action_id=joint_id, hierarchical_action_id=hierarchical_id,
                                      skip_despite_alloc_majority=bool(reversal)))
                elif mode == "sampling":
                    action = agent.act(obs, deterministic=False)
                else:
                    action = env.action_spec.decode(hierarchical_id)
                skip_count += int(action.kind == "SKIP")
                obs, reward, _, _, _ = env.step(action)
                episode_return += reward
            metrics = dict(env.evaluate().metrics)
            assert np.isclose(episode_return, config.env.reward_scale * metrics["mean_satisfaction"], atol=1e-8, rtol=1e-8)
            row = dict(mode=mode, sampling_seed=seed, scenario_id=scenario.scenario_id, n_demand=scenario.n_demand,
                       U=metrics["mean_satisfaction"], skip_count=skip_count, skip_fraction=skip_count/scenario.n_demand,
                       constraint_violations=metrics["constraint_violation_count"],
                       skip_despite_alloc_majority_states=reversal_count)
            with progress_lock:
                results.append(row)
                atomic_json(report_dir / "MLP-PPO解码复核进度.json", dict(results=results, elapsed_seconds=time.perf_counter()-started))
                print(json.dumps(row, ensure_ascii=False), flush=True)

    # 每个任务自持环境、网络与动作RNG；只读共享checkpoint，各模式互不影响。
    with ThreadPoolExecutor(max_workers=len(modes)) as executor:
        list(executor.map(run_mode, modes))
    order = {record["scenario_id"]: index for index, record in enumerate(records)}
    results.sort(key=lambda row: (modes.index((row["mode"], row["sampling_seed"])), order[row["scenario_id"]]))
    summary = []
    for mode, seed in modes:
        rows = [x for x in results if x["mode"] == mode and x["sampling_seed"] == seed]
        summary.append(dict(mode=mode, sampling_seed=seed, scenes=len(rows),
                            mean_U=float(np.mean([x["U"] for x in rows])),
                            mean_skip_fraction=float(np.mean([x["skip_fraction"] for x in rows])),
                            total_skipped=sum(x["skip_count"] for x in rows),
                            all_skip_scenes=sum(x["skip_count"] == x["n_demand"] for x in rows),
                            constraint_violations=sum(x["constraint_violations"] for x in rows)))
    sampling_means = [x["mean_U"] for x in summary if x["mode"] == "sampling"]
    report = dict(scope="read_only_frozen_checkpoint_validation_only_no_training_no_test",
                  checkpoint_id=saved["checkpoint_id"], checkpoint_sha256=sha256(checkpoint),
                  counters=saved["counters"], torch_runtime=torch.__version__, original_torch_runtime=manifest["hardware"]["torch"],
                  manifest_path=str(manifest_path), data_manifest_hash=manifest["data_manifest_hash"], source_checks=source_checks,
                  modes_predeclared=modes, summary=summary, sampling_seed_mean_U=float(np.mean(sampling_means)),
                  original_joint_map_all_steps=len(steps),
                  original_joint_map_skip_despite_alloc_majority=sum(x["skip_despite_alloc_majority"] for x in steps),
                  joint_map_steps=steps, results=results, elapsed_seconds=time.perf_counter()-started,
                  limitations=["分层MAP是新的诊断决策规则，不能覆盖冻结原图或冒称原测试结果",
                               "采样seed只复用同一个训练checkpoint，不是多训练seed",
                               "本地CPU与远端CUDA/PyTorch版本不同，不声称位级一致"])
    atomic_json(report_dir / "MLP-PPO完整验证集解码复核.json", report)
    print(json.dumps({"summary": summary, "sampling_seed_mean_U": report["sampling_seed_mean_U"],
                      "joint_skip_reversals": report["original_joint_map_skip_despite_alloc_majority"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
