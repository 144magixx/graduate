"""本地/远程共用训练入口。导入不会读数据、创建输出或启动训练。"""
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback
import uuid
import numpy as np
from project_paths import COVER_OUTPUT_DIR, HORIZON_DATA_DIR
from .config import load_config


def smoke_config(config):
    config.train.episodes = 2
    config.train.batch_size = 4
    config.train.microbatch_size = 2
    config.train.warmup_steps = 140
    config.train.update_schedule = "episode"
    config.train.updates_per_episode = 2
    config.model.d_model = 32
    config.model.attention_heads = 4
    config.model.encoder_layers = 1
    config.model.encoder = "full_pool"
    config.model.candidate_chunk_size = 256
    return config


def prepare_dataset(config, manifest_path=None):
    from .data.split import validate_split, manifest_hash
    from .data.domain import read_coverage_spec, validate_domain_annotations
    from .audit import sha256
    selected = Path(manifest_path) if manifest_path else HORIZON_DATA_DIR / "训练场景清单.json"
    if not selected.is_file():
        raise FileNotFoundError(f"冻结训练清单不存在：{selected}；先执行 data.domain 审计或用 --dataset 指定已有清单")
    manifest = json.loads(selected.read_text(encoding="utf-8-sig"))
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise ValueError("数据清单hash与完整内容不一致")
    validate_split(manifest["records"])
    if not any(record["split"] == "train" for record in manifest["records"]):
        raise ValueError("冻结清单没有训练split")
    if manifest_path is None and manifest.get("split_seed") != config.data.split_seed:
        raise ValueError("默认冻结清单split_seed与配置不同；不得隐式重新划分，请显式提供对应 --dataset")
    spec, _ = read_coverage_spec(selected.resolve().parent)
    validate_domain_annotations(manifest, spec, require_annotations=manifest_path is None)
    for record in manifest["records"]:
        path = Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR / record["file"]
        if sha256(path) != record["source_hash"]:
            raise ValueError(f"输入文件hash发生变化：{path}")
        if manifest_path is None:
            from .data.loader import load_scenario
            actual = load_scenario(path, config, rng=np.random.default_rng(0)).n_demand
            if record.get("n_demand") != actual:
                raise ValueError(f"冻结清单正需求数与源场景不符：{path}")
    return manifest


def perform_updates(config, agent, replay, counters, recorder=None, *, trigger):
    """固定调度点更新；热身未完成或没有已入池回合时不采样、不补欠账。"""
    if trigger != config.train.update_schedule or counters["env_step"] < config.train.warmup_steps or not len(replay):
        return 0
    count = config.train.updates_per_step if trigger == "env_step" else config.train.updates_per_episode
    for _ in range(count):
        batch = replay.sample(config.train.batch_size)
        diagnostics = agent.update(batch)
        diagnostics["replay"] = replay.diagnostics()
        diagnostics["sample_references"] = [{"scenario_id": t.scenario_id, "action_id": t.action_id,
            "episode_id": t.episode_id, "step_index": t.step_index, "env_step": t.env_step} for t in batch]
        counters["update_step"] += 1
        if recorder:
            recorder.record_update(diagnostics, counters["update_step"], env_step=counters["env_step"])
    return count


def run_training(config, mode="train", dataset=None, resume=None):
    if config.train.device.startswith("cuda") and config.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .artifacts import create_run, atomic_json, save_checkpoint, load_checkpoint, restore_rng
    from .data.loader import load_scenario
    from .data.augment import augment_scenario
    from .data.domain import DomainSampler
    from .env.environment import Environment
    from .rl.sac import DiscreteSAC
    from .rl.replay import ReplayBuffer, Transition
    from .telemetry import Recorder

    config.validate()
    torch.set_num_threads(config.train.torch_num_threads)
    if config.train.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前PyTorch没有可用CUDA；请明确使用CPU或远程CUDA环境")
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.use_deterministic_algorithms(config.train.deterministic)
    data_manifest = prepare_dataset(config, dataset)
    payload = load_checkpoint(resume, config, data_manifest["manifest_hash"]) if resume else None
    sampler = DomainSampler(data_manifest["records"], config.train.seed + 11)
    augmentation_rng = np.random.default_rng(config.train.seed + 23)
    exploration_rng = np.random.default_rng(config.train.seed + 37)
    order_rng = np.random.default_rng(config.train.seed + 53)
    local_rngs = {"augmentation": augmentation_rng, "exploration": exploration_rng, "order":order_rng}
    env = Environment(config)
    first_record = next(r for r in data_manifest["records"] if r["split"] == "train")
    first = load_scenario(Path(first_record["scenario_path"]) if first_record.get("scenario_path") else COVER_OUTPUT_DIR / first_record["file"], config,rng=np.random.default_rng(config.train.seed+59))
    example, _ = env.reset(first, config.train.seed)
    agent = DiscreteSAC(config, env.action_spec, example)
    replay = ReplayBuffer(config.train.replay_capacity, config.semantic_versions(), action_spec_signature=agent.action_signature(), mode=config.train.replay_mode,
                          seed=config.train.seed + 41)
    counters = {"env_step": 0, "update_step": 0, "episodes": 0}
    if payload:
        agent.load_state_dict(payload["agent"])
        replay.load_state_dict(payload["replay"])
        sampler.load_state_dict(payload["sampler"])
        counters.update(payload["counters"])
        for key, value in payload["local_rngs"].items():
            local_rngs[key].bit_generator.state = value
        restore_rng(payload["rng"])
    run_dir, manifest = create_run(config, data_manifest, payload, mode)
    from .data.domain import freeze_run_coverage
    coverage_snapshot = freeze_run_coverage(run_dir, data_manifest, config, dataset)
    manifest.update(resume_env_step=counters["env_step"], resume_update_step=counters["update_step"],
                    action_spec_signature=env.action_spec.signature(), coverage_snapshot=coverage_snapshot)
    atomic_json(run_dir / "运行清单.json", manifest)
    recorder = Recorder(run_dir, manifest, config.telemetry) if config.telemetry.enabled else None
    if recorder:
        recorder.publish_artifact(run_dir / "配置快照.json",kind="configuration")
        recorder.publish_artifact(run_dir / "数据划分.json",kind="dataset_split")
        for reference in coverage_snapshot["artifacts"].values():
            recorder.publish_artifact(run_dir / reference["path"], kind=reference["kind"], artifact_id=reference["artifact_id"])
    started = time.monotonic()
    summaries = []
    last_boundary = None
    last_heartbeat = started
    try:
        for episode_offset in range(config.train.episodes):
            if config.train.max_env_steps and counters["env_step"] >= config.train.max_env_steps:
                break
            record = sampler.sample()
            source_path = Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR / record["file"]
            scenario = load_scenario(source_path, config,rng=order_rng)
            if config.data.augment and scenario.source_schema == "legacy_lon_in_lat":
                scenario = augment_scenario(scenario, augmentation_rng)
            # New root datasets are consumed as supplied; root-level augmentation is explicit in data.domain.
            episode_id = uuid.uuid4().hex
            obs, _ = env.reset(scenario, config.train.seed + counters["episodes"])
            (run_dir / "场景快照").mkdir(exist_ok=True)
            scenario_path=run_dir / "场景快照" / f"场景-{episode_id}.json"
            atomic_json(scenario_path, scenario.to_dict())
            if recorder:
                recorder.publish_artifact(scenario_path,kind="scenario")
                snapshot=env.snapshot()
                recorder.start_episode(episode_id, scenario.scenario_id, snapshot, env_step=counters["env_step"])
            transitions, total_return = [], 0.0
            episode_enqueued = False
            episode_start = time.monotonic()
            truncated = False
            while not obs["terminal"]:
                before = snapshot if recorder else None
                if counters["env_step"] < config.train.warmup_steps:
                    action = env.action_spec.decode(int(exploration_rng.choice(np.flatnonzero(obs["valid_action_mask"]))))
                else:
                    action = agent.act(obs)
                next_obs, reward, terminated, _, info = env.step(action)
                counters["env_step"] += 1
                truncated = bool(config.train.max_env_steps and counters["env_step"] >= config.train.max_env_steps and not terminated)
                transitions.append(Transition(obs, env.action_spec.encode(action), reward, next_obs, terminated, truncated,
                                               scenario.scenario_id, config.semantic_versions(),
                                               episode_id=episode_id,step_index=env.cursor,env_step=counters["env_step"]))
                total_return += reward
                if recorder:
                    snapshot=env.snapshot()
                    recorder.record_step(episode_id, scenario.scenario_id, env.cursor, before, snapshot,
                                         {"action": action.to_dict(), "action_id": env.action_spec.encode(action), "reward": reward,
                                          "terminated": terminated, "truncated": truncated, **info}, env_step=counters["env_step"])
                obs = next_obs
                if config.train.update_schedule == "env_step":
                    # 终局动作先完成整episode入池；未完成的当前episode从不进入研究回放。
                    if terminated:
                        replay.add_episode(transitions, scenario_id=scenario.scenario_id, n_demand=scenario.n_demand,
                                           domain_cell=record.get("domain_cells"))
                        episode_enqueued = True
                    perform_updates(config, agent, replay, counters, recorder, trigger="env_step")
                current_time = time.monotonic()
                if recorder and current_time - last_heartbeat >= 5:
                    recorder.event("performance", {"steps_per_second": (counters["env_step"] - manifest["resume_env_step"]) / (current_time-started),
                                   "updates_per_second": (counters["update_step"]-manifest["resume_update_step"]) / (current_time-started),
                                   "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                                   "cpu_memory_bytes": None, "gpu_utilization": None}, env_step=counters["env_step"], phase="train")
                    last_heartbeat = current_time
                if truncated:
                    break
            if transitions and not episode_enqueued and config.train.update_schedule == "episode":
                replay.add_episode(transitions, scenario_id=scenario.scenario_id, n_demand=scenario.n_demand,
                                   domain_cell=record.get("domain_cells"), fragment=truncated)
            if transitions:
                perform_updates(config, agent, replay, counters, recorder, trigger="episode")
            counters["episodes"] += 1
            metrics = dict(env.evaluate().metrics)
            utility = metrics.get("mean_satisfaction")
            if utility is not None and not np.isclose(total_return, config.env.reward_scale*utility, rtol=1e-8, atol=1e-8):
                raise AssertionError("增量奖励望远镜恒等式失败")
            metrics.update(episode_return=total_return, episode_seconds=time.monotonic()-episode_start,
                           env_step=counters["env_step"], update_step=counters["update_step"], n_demand=scenario.n_demand,
                           episode_id=episode_id, scenario_id=scenario.scenario_id, truncated=truncated)
            summaries.append(metrics)
            if recorder:
                recorder.end_episode(episode_id, metrics, scenario_id=scenario.scenario_id,
                                     env_step=counters["env_step"], terminated=env.terminal, truncated=truncated)
            print(json.dumps({"run_id":manifest["run_id"],"episode":counters["episodes"],"env_step":counters["env_step"],
                              "updates":counters["update_step"],"U":utility,"return":total_return}, ensure_ascii=False), flush=True)
            # 精确续训首版仅在真正回合边界，截断片段可训练但不能标成完整边界checkpoint。
            wall_budget_reached=bool(config.train.max_wall_seconds and time.monotonic()-started>=config.train.max_wall_seconds)
            final_planned_episode=episode_offset+1==config.train.episodes
            env_budget_reached=bool(config.train.max_env_steps and counters["env_step"]>=config.train.max_env_steps)
            checkpoint_due=(counters["episodes"]%config.train.checkpoint_every==0 or final_planned_episode or wall_budget_reached or env_budget_reached)
            if env.terminal and checkpoint_due:
                seq = recorder.flush() if recorder else 0
                last_boundary = save_checkpoint(run_dir, agent, replay, sampler, config, manifest, counters, seq, local_rngs)
                if recorder:
                    recorder.publish_artifact(last_boundary)
                    recorder.publish_artifact(last_boundary.parent / "检查点清单.json",kind="checkpoint_manifest")
            if truncated:
                break
            if wall_budget_reached:
                break
        elapsed = time.monotonic()-started
        report = {"run_id":manifest["run_id"], "mode":mode, "counters":counters, "elapsed_seconds":elapsed,
                  "update_protocol": {"schedule": config.train.update_schedule, "warmup_steps": config.train.warmup_steps,
                      "updates_per_step": config.train.updates_per_step, "updates_per_episode": config.train.updates_per_episode,
                      "replay_eligibility": "completed_episodes_only" if config.train.update_schedule == "env_step" else "completed_episodes_or_explicit_truncated_fragments",
                      "missed_updates": "not_accumulated"},
                  "steps_per_second": (counters["env_step"]-manifest["resume_env_step"])/max(elapsed,1e-9),
                  "episodes":summaries, "checkpoint_available":last_boundary is not None,
                  "sampler":sampler.state_dict(), "replay_sampling":replay.diagnostics(),"cuda_peak_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                  "conclusion":"短运行验证工程链路，不构成性能提升或正式泛化结论" if mode=="smoke" else "训练记录；性能结论以冻结检查点的独立评估为准，不自动构成多种子泛化证据"}
        atomic_json(run_dir / ("短训练结果.json" if mode=="smoke" else "训练结果.json"), report)
        if recorder:
            recorder.close("completed")
        else:
            manifest.update(status="completed")
            atomic_json(run_dir / "运行清单.json", manifest)
        return run_dir, report
    except BaseException as error:
        atomic_json(run_dir / "中断记录.json", {"error":str(error), "traceback":traceback.format_exc(),
                     "last_boundary_checkpoint":str(last_boundary) if last_boundary else None, "counters":counters})
        if recorder:
            try:
                recorder.event("training_failed", {"error":str(error),"traceback":traceback.format_exc()}, severity="ERROR")
                recorder.close("failed")
            except Exception:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Horizon T-SAC 训练；正式长训练需显式配置预算")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("smoke","train"), default="smoke")
    parser.add_argument("--dataset", help="冻结数据划分清单")
    parser.add_argument("--resume", help="可信的新实现回合边界检查点")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-telemetry", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "smoke":
        smoke_config(config)
    if args.episodes is not None:
        config.train.episodes = args.episodes
    if args.seed is not None:
        config.train.seed = args.seed
    if args.no_telemetry:
        config.telemetry.enabled = False
    directory, report = run_training(config, args.mode, args.dataset, args.resume)
    print(json.dumps({"run_dir":str(directory), "counters":report["counters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
