"""长训练前的设备容量短测：fixture仅用于测量，不进入研究训练集。"""
import argparse
import io
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from ..config import load_config


def run_probe(config, entities=220, updates=2):
    # 必须早于任何CUDA上下文初始化；与训练确定性开关保持一致。
    if config.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ..data.loader import scenario_from_rows
    from ..env.environment import Environment
    from .benchmark import process_memory
    from .replay import Transition
    from .sac import DiscreteSAC

    config.validate()
    torch.set_num_threads(config.train.torch_num_threads)
    torch.manual_seed(config.train.seed)
    torch.use_deterministic_algorithms(config.train.deterministic)
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("容量短测要求CUDA，但当前环境不可用")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = [dict(beam_id=1000+i, latitude_deg=25.+(i//20)*.05,
                 longitude_deg=108.+(i%20)*.05, demand_bps=1e8+i*1e5,
                 ground_diameter_deg=2., group_id=i % config.physics.num_groups)
            for i in range(entities)]
    scenario = scenario_from_rows(rows, config, "coverage.v2", "overnight_capacity_fixture",
        metadata={"purpose": "capacity_fixture_only", "not_training_dataset": True})
    env = Environment(config)
    observation, _ = env.reset(scenario, config.train.seed)
    agent = DiscreteSAC(config, env.action_spec, observation)
    transitions = []
    current = observation
    for step in range(min(2, entities)):
        action_id = int(np.flatnonzero(current["valid_action_mask"])[-1])
        following, reward, terminated, _, _ = env.step(action_id)
        transitions.append(Transition(current, action_id, reward, following, terminated,
                           scenario_id=scenario.scenario_id, versions=config.semantic_versions()))
        current = following
    tensor_bytes = [sum(value.nbytes for obs in (t.observation, t.next_observation)
                       for value in obs.values() if isinstance(value, np.ndarray)) for t in transitions]
    buffer = io.BytesIO()
    torch.save(transitions, buffer)
    mean_serialized_bytes = len(buffer.getbuffer()) / len(transitions)
    del buffer
    samples = [transitions[i % len(transitions)] for i in range(config.train.batch_size)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings, diagnostics = [], []
    for index in range(updates):
        synchronize()
        started = time.perf_counter()
        diagnostic = agent.update(samples)
        synchronize()
        elapsed = time.perf_counter()-started
        timings.append(elapsed)
        diagnostics.append(diagnostic)
        print(json.dumps({"update": index+1, "synchronized_seconds": elapsed,
                          "actor_loss": diagnostic["actor_loss"], "alpha": diagnostic["alpha"]}), flush=True)
    synchronize()
    started = time.perf_counter()
    action = agent.act(observation)
    synchronize()
    act_seconds = time.perf_counter()-started
    if not observation["valid_action_mask"][env.action_spec.encode(action)]:
        raise AssertionError("容量短测产生非法动作")
    replay_bytes = int(mean_serialized_bytes*config.train.replay_capacity)
    median_update = statistics.median(timings)
    return {
        "purpose": "220实体等容量fixture，仅测试设备运行性/近似容量，不计入训练与数据覆盖验收",
        "passed": True, "config": config.to_dict(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "fixture": {"entities": entities, "actions": len(env.action_spec),
                    "distinct_transitions": len(transitions), "effective_batch": len(samples),
                    "microbatch": config.train.microbatch_size,
                    "repeated_samples": True, "candidate_chunk": config.model.candidate_chunk_size},
        "timings_seconds": {"synchronized_updates": timings, "median_update": median_update,
                            "stochastic_action": act_seconds},
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "process_memory": process_memory(),
        "replay_estimates": {"observation_array_bytes_per_transition": statistics.mean(tensor_bytes),
                             "torch_serialized_bytes_per_transition": mean_serialized_bytes,
                             "at_configured_capacity_bytes": replay_bytes,
                             "minimum_arrays_during_deepcopy_bytes": int(2*statistics.mean(tensor_bytes)*config.train.replay_capacity),
                             "archive_100_full_checkpoints_bytes": 100*replay_bytes,
                             "note": "两条独立快照外推；不含Python对象、模型、优化器与日志。硬链接仅共享同一次快照，不压缩历次快照。"},
        "overnight_estimate": {"six_hour_update_only_upper_bound": int(21600/median_update),
                               "eight_hour_update_only_upper_bound": int(28800/median_update),
                               "note": "单任务短测上界，未扣环境/日志/checkpoint/评价，也未计并行竞争；不可视作实际训练承诺。"},
        "update_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--entities", type=int, default=220)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.updates < 1 or args.entities < 1:
        parser.error("updates与entities必须为正整数")
    config = load_config(args.config)
    config.train.device = args.device
    config.train.batch_size = args.batch_size
    config.train.microbatch_size = args.microbatch_size
    report = run_probe(config, args.entities, args.updates)
    if args.output:
        path = Path(args.output)
    else:
        from project_paths import get_output_dir
        path = get_output_dir("horizon_tsac_20260920", "数学验收") / "长训练设备容量短测.json"
    from ..artifacts import atomic_json
    atomic_json(path, report)
    print(json.dumps({"report": str(path), "passed": report["passed"],
                      "timings_seconds": report["timings_seconds"],
                      "cuda_peak_allocated_bytes": report["cuda_peak_allocated_bytes"],
                      "replay_estimates": report["replay_estimates"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
