"""Anchor入口：默认只预检；真实CSV训练必须显式授权。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import traceback
import uuid
import numpy as np
from project_paths import COVER_OUTPUT_DIR
from .artifacts import base_manifest
from .config import load_config


def preflight(config):
    from .env.action import ActionSpec
    config.validate()
    actions = ActionSpec(config)
    return {"status": "preflight_ok", "experiment_started": False,
            "implementation": "anchor_tsac_20260921", "algorithm_id": config.algorithm_id,
            "action_spec": actions.signature(), "manifest_preview": base_manifest(config, mode="preflight"),
            "guard": "真实CSV训练/rollout需要--allow-experiment且正预算"}


def _synthetic_observation(config, action_spec):
    occupancy = np.zeros((config.physics.num_groups, config.physics.num_slots), dtype=bool)
    observation = {"anchor205": np.zeros(205, np.float32), "terminal": False, "occupancy": occupancy,
                   "current_group": 0, "remaining_power_w": config.env.power_budget_w,
                   "observation_adapter_version": config.semantic_versions()["observation_adapter_version"],
                   "action_spec_signature": action_spec.signature()}
    observation["valid_action_mask"] = action_spec.valid_actions(observation)
    return observation


def integration_check(config):
    """只做有限合成前向，不读取CSV、checkpoint或创建run。"""
    from .env.action import ActionSpec
    from .rl import DiscreteSAC
    action_spec = ActionSpec(config)
    observation = _synthetic_observation(config, action_spec)
    agent = DiscreteSAC(config, action_spec, observation)
    probability = agent.probabilities(observation)
    manifest = base_manifest(config, mode="integration_check", is_demo=True)
    return {"status": "integration_ok", "is_demo": True, "experiment_started": False,
            "experiment_authorized": manifest["experiment_authorized"],
            "probability_sum": float(probability.sum()), "valid_action_count": int(observation["valid_action_mask"].sum()),
            "skip_id": 0, "num_actions": len(action_spec), "rl_forward_mode": agent.RL_FORWARD_MODE}


def prepare_dataset(config, manifest_path):
    """只接受显式冻结清单；不会在训练入口隐式创建或重新划分数据。"""
    from .audit import sha256
    from .data.split import manifest_hash, validate_split
    if not manifest_path:
        raise ValueError("真实训练必须显式提供--dataset冻结清单；Anchor不会改写冻结Horizon数据")
    selected = Path(manifest_path).resolve()
    manifest = json.loads(selected.read_text(encoding="utf-8-sig"))
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise ValueError("数据清单hash与完整内容不一致")
    validate_split(manifest["records"])
    if not any(record["split"] == "train" for record in manifest["records"]):
        raise ValueError("冻结清单没有训练split")
    for record in manifest["records"]:
        source = Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR / record["file"]
        if sha256(source) != record["source_hash"]:
            raise ValueError(f"输入文件hash漂移：{source}")
    return manifest


def _source_path(record):
    return Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR / record["file"]


def _budget_reached(config, counters, elapsed):
    return {"env_steps": bool(config.train.max_env_steps and counters["env_step"] >= config.train.max_env_steps),
            "wallclock": bool(config.train.max_wall_seconds and elapsed >= config.train.max_wall_seconds)}


def _warmup_uses_policy(origin_initialization):
    return origin_initialization.get("kind") == "weights_only_not_exact_resume"


def _publish_checkpoint(recorder, checkpoint):
    if recorder:
        recorder.publish_artifact(checkpoint, kind="checkpoint")
        recorder.publish_artifact(checkpoint.parent / "检查点清单.json", kind="checkpoint_manifest")


def run_training(config, dataset=None, resume=None, initialize_from=None, expected_sha256=None,
                 *, allow_experiment=False, is_demo=False):
    if not allow_experiment:
        raise RuntimeError("真实CSV rollout/训练尚未授权；必须显式传入allow_experiment=True或--allow-experiment")
    if resume and initialize_from:
        raise ValueError("--resume与--initialize-from互斥")
    if (initialize_from is None) != (expected_sha256 is None):
        raise ValueError("旧权重迁移必须同时提供--initialize-from与--expected-sha256")
    if config.train.episodes <= 0:
        raise ValueError("实验授权后仍需显式设置正episodes；默认预算0不会启动")
    config.validate()
    if config.train.device.startswith("cuda") and config.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .artifacts import create_run, atomic_json, load_checkpoint, save_checkpoint
    from .data.loader import load_scenario
    from .data.augment import augment_scenario
    from .data.domain import DomainSampler, freeze_run_coverage
    from .env.environment import Environment
    from .rl import DiscreteSAC, ReplayBuffer, Transition
    from .telemetry import Recorder
    torch.set_num_threads(config.train.torch_num_threads)
    if config.train.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA但当前不可用")
    torch.use_deterministic_algorithms(config.train.deterministic)
    torch.manual_seed(config.train.seed); random.seed(config.train.seed); np.random.seed(config.train.seed)
    data_manifest = prepare_dataset(config, dataset)
    payload = load_checkpoint(resume, config, data_manifest["manifest_hash"]) if resume else None
    sampler = DomainSampler(data_manifest["records"], config.train.seed + 11)
    augmentation_rng = np.random.default_rng(config.train.seed + 23)
    action_rng = np.random.default_rng(config.train.seed + 37)
    local_rngs = {"augmentation": augmentation_rng, "exploration": action_rng}
    env = Environment(config)
    first_record = next(record for record in data_manifest["records"] if record["split"] == "train")
    first = load_scenario(_source_path(first_record), config, rng=np.random.default_rng(config.train.seed + 59))
    example, _ = env.reset(first, config.train.seed)
    agent = DiscreteSAC(config, env.action_spec, example)
    replay = ReplayBuffer(config.train.replay_capacity, config.semantic_versions(), env.action_spec.signature(),
                          seed=config.train.seed + 41)
    counters = {"episodes": 0, "env_step": 0, "update_step": 0}
    origin_initialization = {"kind": "from_scratch", "new_trial": True}
    if initialize_from:
        from .initialization import initialize_from_legacy_full_model
        origin_initialization = initialize_from_legacy_full_model(agent, initialize_from, expected_sha256)
    if payload:
        agent.load_state_dict(payload["agent"], restore_rng=True)
        replay.load_state_dict(payload["replay"])
        sampler.load_state_dict(payload["sampler"])
        counters.update(payload["counters"])
        for key, state in payload.get("local_rngs", {}).items():
            if key in local_rngs:
                local_rngs[key].bit_generator.state = state
        origin_initialization = payload.get("initialization") or {"kind": "unknown_origin"}
        lineage = {"kind": "exact_resume", "source_run_id": payload["run_id"],
                   "source_trial_id": payload["trial_id"], "source_checkpoint_id": payload["checkpoint_id"],
                   "origin_initialization": origin_initialization}
    else:
        lineage = dict(origin_initialization)
    run_dir, manifest = create_run(config, data_manifest, parent=payload, initialization=origin_initialization,
                                   lineage=lineage, mode="train", is_demo=is_demo)
    coverage_snapshot = freeze_run_coverage(run_dir, data_manifest, config, dataset)
    manifest.update(action_spec_signature=env.action_spec.signature(), coverage_snapshot=coverage_snapshot,
                    physics_version=config.physics.physics_version, reward_version=config.env.reward_version,
                    metric_version=config.env.metric_version, model_version=config.model.model_version,
                    dataset_version=config.data.dataset_version, power_budget_w=config.env.power_budget_w,
                    resume_env_step=counters["env_step"], resume_update_step=counters["update_step"],
                    resume_episode=counters["episodes"], replay=replay.diagnostics(),
                    budget_semantics={"max_env_steps": "soft_stop_at_complete_episode_boundary",
                                      "max_wall_seconds": "soft_stop_at_complete_episode_boundary",
                                      "checkpoint_every": "cumulative_completed_episodes"},
                    warmup_collection="initialized_policy" if origin_initialization.get("kind") == "weights_only_not_exact_resume" else "uniform_legal",
                    trace_detail="basic_all动作/资源/逐束速率与满足度；逐槽详情按关键帧保存")
    atomic_json(run_dir / "运行清单.json", manifest)
    recorder = Recorder(run_dir, manifest, config.telemetry) if config.telemetry.enabled else None
    if recorder:
        recorder.publish_artifact(run_dir / "配置快照.json", kind="configuration")
        recorder.publish_artifact(run_dir / "数据划分.json", kind="dataset_split")
        for reference in coverage_snapshot["artifacts"].values():
            recorder.publish_artifact(run_dir / reference["path"], kind=reference["kind"], artifact_id=reference["artifact_id"])
    started = time.monotonic()
    last_heartbeat = started
    summaries, last_checkpoint = [], None
    new_episodes = 0
    try:
        for episode_offset in range(config.train.episodes):
            boundary = _budget_reached(config, counters, time.monotonic() - started)
            if any(boundary.values()):
                break
            record = sampler.sample()
            scenario = load_scenario(_source_path(record), config,
                                     rng=np.random.default_rng(config.train.seed + counters["episodes"] + 101))
            if config.data.augment:
                scenario = augment_scenario(scenario, augmentation_rng)
            observation, _ = env.reset(scenario, config.train.seed + counters["episodes"])
            episode_id = uuid.uuid4().hex
            scenario_dir = run_dir / "场景快照"
            scenario_path = scenario_dir / f"场景-{episode_id}.json"
            atomic_json(scenario_path, scenario.to_dict())
            if recorder:
                recorder.publish_artifact(scenario_path, kind="scenario")
                recorder.start_episode(episode_id, scenario.scenario_id, env.snapshot(), env_step=counters["env_step"])
            total_return, episode_started = 0., time.monotonic()
            while not observation["terminal"]:
                before = env.snapshot() if recorder else None
                migrated = _warmup_uses_policy(origin_initialization)
                if counters["env_step"] < config.train.warmup_steps and not migrated:
                    legal = np.flatnonzero(observation["valid_action_mask"])
                    action = env.action_spec.decode(int(action_rng.choice(legal)))
                else:
                    action = agent.act(observation)
                next_observation, reward, terminated, _, info = env.step(action)
                transition = Transition(observation, env.action_spec.encode(action), reward, next_observation,
                                        terminated, config.semantic_versions())
                replay.add(transition)
                counters["env_step"] += 1
                total_return += reward
                if len(replay) >= config.train.batch_size and counters["env_step"] >= config.train.warmup_steps:
                    for _ in range(config.train.updates_per_step):
                        diagnostics = agent.update(replay.sample(config.train.batch_size))
                        counters["update_step"] += 1
                        diagnostics.update(env_step=counters["env_step"], replay=replay.diagnostics())
                        if recorder: recorder.record_update(diagnostics, counters["update_step"], env_step=counters["env_step"])
                if recorder:
                    recorder.record_step(episode_id, scenario.scenario_id, env.cursor, before, env.snapshot(),
                                         {"action": action.to_dict(), "action_id": env.action_spec.encode(action),
                                          "reward": reward, "terminated": terminated, **info},
                                         env_step=counters["env_step"])
                    now = time.monotonic()
                    if now - last_heartbeat >= 5:
                        elapsed = max(now - started, 1e-9)
                        recorder.event("performance", {"steps_per_second": (counters["env_step"] - manifest["resume_env_step"]) / elapsed,
                            "updates_per_second": (counters["update_step"] - manifest["resume_update_step"]) / elapsed,
                            "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None},
                            env_step=counters["env_step"], phase="train")
                        last_heartbeat = now
                observation = next_observation
            counters["episodes"] += 1
            new_episodes += 1
            metrics = dict(env.evaluate().metrics)
            utility = metrics.get("mean_satisfaction")
            if utility is not None and not np.isclose(total_return, config.env.reward_scale * utility, rtol=1e-8, atol=1e-8):
                raise AssertionError("R=100U增量奖励望远镜恒等式失败")
            metrics.update(episode_return=total_return, episode_seconds=time.monotonic() - episode_started,
                           scenario_id=scenario.scenario_id, episode_id=episode_id,
                           episode=counters["episodes"], env_step=counters["env_step"], update_step=counters["update_step"])
            summaries.append(metrics)
            if recorder:
                recorder.end_episode(episode_id, metrics, scenario_id=scenario.scenario_id,
                                     env_step=counters["env_step"], terminated=True, truncated=False)
            elapsed = time.monotonic() - started
            boundary = _budget_reached(config, counters, elapsed)
            final_planned = episode_offset + 1 == config.train.episodes
            checkpoint_due = (counters["episodes"] % config.train.checkpoint_every == 0
                              or final_planned or any(boundary.values()))
            if checkpoint_due:
                durable = recorder.flush() if recorder else 0
                last_checkpoint = save_checkpoint(run_dir, agent, replay, sampler, config, manifest,
                                                  counters, durable, local_rngs)
                _publish_checkpoint(recorder, last_checkpoint)
            if any(boundary.values()):
                break
        elapsed = time.monotonic() - started
        report = {"status": "completed", "counters": counters, "new_episodes": new_episodes,
                  "elapsed_seconds": elapsed, "episodes": summaries, "initialization": origin_initialization,
                  "lineage": lineage, "replay": replay.diagnostics(), "is_demo": bool(is_demo),
                  "budget_boundary": _budget_reached(config, counters, elapsed),
                  "checkpoint_available": last_checkpoint is not None,
                  "checkpoint_path": str(last_checkpoint) if last_checkpoint else None,
                  "claim": "训练记录；不自动构成性能或泛化证据"}
        atomic_json(run_dir / "训练结果.json", report)
        if recorder:
            recorder.publish_artifact(run_dir / "训练结果.json", kind="training_summary")
            recorder.close("completed")
        else:
            manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                            counters=dict(counters), last_checkpoint_id=last_checkpoint.parent.name if last_checkpoint else None)
            atomic_json(run_dir / "运行清单.json", manifest)
        return run_dir, report
    except BaseException as error:
        failure = {"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                   "counters": counters, "last_checkpoint": str(last_checkpoint) if last_checkpoint else None}
        atomic_json(run_dir / "中断记录.json", failure)
        if recorder:
            try: recorder.close("failed")
            except Exception: pass
        else:
            manifest.update(status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat(), counters=dict(counters))
            atomic_json(run_dir / "运行清单.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description="Anchor T-SAC；默认只预检")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("preflight", "integration", "train"), default="preflight")
    parser.add_argument("--dataset")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-experiment", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None: config.train.episodes = args.episodes
    if args.seed is not None: config.train.seed = args.seed
    if args.mode == "preflight": result = preflight(config)
    elif args.mode == "integration": result = integration_check(config)
    else:
        directory, report = run_training(config, args.dataset, args.resume, args.initialize_from,
                                         args.expected_sha256, allow_experiment=args.allow_experiment, is_demo=False)
        result = {"run_dir": str(directory), "report": report}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
