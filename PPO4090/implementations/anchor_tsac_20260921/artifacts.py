"""Anchor运行身份与带hash的原子checkpoint。"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid
import numpy as np
from project_paths import PROJECT_ROOT, ANCHOR_OUTPUT_DIR
from .config import ASSUMPTIONS, Config
from .audit import sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def config_hash(config):
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in Path(__file__).parent.rglob("*.py")}


def base_manifest(config, *, initialization=None, lineage=None, mode="preflight", is_demo=False,
                  experiment_authorized=False):
    adapter_version = config.semantic_versions()["observation_adapter_version"]
    return {"schema_version": "anchor_run.v1", "implementation": "anchor_tsac_20260921",
            "algorithm_id": "anchor_tsac", "algorithm": "Anchor T-SAC",
            "algorithm_spec": {"id": "anchor_tsac", "model": "pure_transformer_14",
            "d_model": 128, "num_heads": 8, "num_layers": 2, "actor": "independent", "critic": "additive"},
            "observation": {"input": "anchor205", "adapter": config.env.observation_adapter,
                            "adapter_version": adapter_version,
                            "meaning": "legacy_scaled205使用旧尺度但始终来自新物理账本"},
            "initialization": copy.deepcopy(initialization) if initialization else {"kind": "not_started"},
            "lineage": copy.deepcopy(lineage) if lineage else {"kind": "not_started"},
            "lineage_modes": ["from_scratch", "weights_only_not_exact_resume", "exact_resume"],
            "is_demo": bool(is_demo), "mode": mode, "experiment_authorized": bool(experiment_authorized),
            "status": "created" if experiment_authorized else "not_run", "versions": config.semantic_versions(),
            "profile": {"kind": "regression_configuration", "gamma": config.train.gamma,
                        "reward": f"{config.env.reward_scale:g}_delta_U",
                        "reward_scale": config.env.reward_scale,
                        "undiscounted_reward_identity": f"episode_return={config.env.reward_scale:g}*terminal_U",
                        "discounted_objective_claim": ("gamma=1时未折扣回报保持上述望远镜恒等式"
                            if config.train.gamma == 1 else
                            f"gamma={config.train.gamma:g}的折扣优化目标不等价于terminal_U"),
                        "dropout_structure": config.model.dropout, "rl_forward_mode": "eval_dropout_disabled"},
            "legacy_architecture_evidence": {"source": "frozen_source_defaults_not_checkpoint_metadata",
                "attention_heads": 8, "dropout": 0.1,
                "limitation": "state_dict不保存head数/dropout，不能证明历史运行实际配置"},
            "policy_distribution": "flat_masked_joint_softmax_of_additive_component_logits_with_explicit_skip0",
            "policy_equivalence": "旧core/三分量头可对齐；新联合mask与SKIP策略不等价于旧独立分量采样",
            "budgets": {"episodes": config.train.episodes, "max_env_steps": config.train.max_env_steps},
            "config": config.to_dict(), "config_hash": config_hash(config), "assumptions": ASSUMPTIONS}


def create_run(config, data_manifest, parent=None, initialization=None, lineage=None, mode="train", is_demo=False):
    import torch
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
    directory = ANCHOR_OUTPUT_DIR / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(config, initialization=initialization, lineage=lineage, mode=mode, is_demo=is_demo,
                             experiment_authorized=mode == "train" and not is_demo)
    manifest.update(run_id=run_id, trial_id=parent["trial_id"] if parent else uuid.uuid4().hex,
                    attempt_id=uuid.uuid4().hex, parent_run_id=parent.get("run_id") if parent else None,
                    parent_checkpoint_id=parent.get("checkpoint_id") if parent else None,
                    created_at_utc=now.isoformat(), status="created", data_manifest=data_manifest,
                    execution_kind="engineering_fixture" if is_demo else "research_run",
                    data_manifest_hash=data_manifest["manifest_hash"], code_hashes=source_hashes(),
                    hardware={"platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
                              "numpy": np.__version__, "device": config.train.device})
    for filename, payload in (("运行清单.json", manifest), ("配置快照.json", config.to_dict()), ("数据划分.json", data_manifest)):
        atomic_json(directory / filename, payload)
    return directory, manifest


def save_checkpoint(run_dir, agent, replay, sampler, config, manifest, counters, durable_event_seq, local_rngs):
    import torch
    payload = {"format": "anchor_checkpoint.v1", "boundary": "episode", "checkpoint_id": uuid.uuid4().hex,
               "run_id": manifest["run_id"], "trial_id": manifest["trial_id"], "config": config.to_dict(),
               "versions": config.semantic_versions(), "data_manifest_hash": manifest["data_manifest_hash"],
               "agent": agent.state_dict(), "replay": replay.state_dict(), "sampler": sampler.state_dict(),
               "counters": dict(counters), "durable_event_seq": int(durable_event_seq),
               "initialization": copy.deepcopy(manifest.get("initialization")),
               "lineage": copy.deepcopy(manifest.get("lineage")),
               "local_rngs": {key: copy.deepcopy(value.bit_generator.state) for key, value in local_rngs.items()}}
    path = Path(run_dir) / "检查点.pt"
    temporary = path.with_name("检查点." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    sidecar = {"format": payload["format"], "file": path.name, "sha256": sha256(path),
               "checkpoint_id": payload["checkpoint_id"], "run_id": payload["run_id"],
               "trial_id": payload["trial_id"], "boundary": "episode", "counters": payload["counters"]}
    atomic_json(Path(run_dir) / "检查点清单.json", sidecar)
    archive = Path(run_dir) / "检查点存档" / payload["checkpoint_id"]
    archive.mkdir(parents=True, exist_ok=False)
    archived_path = archive / path.name
    try:
        os.link(path, archived_path)
    except OSError:
        shutil.copy2(path, archived_path)
    atomic_json(archive / "检查点清单.json", sidecar)
    return archived_path


def load_checkpoint(path, config=None, data_manifest_hash=None):
    import torch
    path = Path(path)
    sidecar = json.loads((path.parent / "检查点清单.json").read_text(encoding="utf-8"))
    if sidecar.get("file") != path.name or sidecar.get("sha256") != sha256(path):
        raise ValueError("checkpoint hash与清单不一致")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "anchor_checkpoint.v1" or payload.get("boundary") != "episode":
        raise ValueError("不兼容的Anchor checkpoint格式或边界")
    identity_fields = ("format", "checkpoint_id", "run_id", "trial_id", "counters", "boundary")
    mismatched = [key for key in identity_fields if sidecar.get(key) != payload.get(key)]
    if mismatched:
        raise ValueError(f"checkpoint sidecar与payload身份不一致：{mismatched}")
    counters = payload.get("counters")
    if (not isinstance(counters, dict) or set(counters) != {"episodes", "env_step", "update_step"}
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters.values())):
        raise ValueError("checkpoint计数器schema不合法")
    if payload.get("agent", {}).get("update_step") != counters["update_step"]:
        raise ValueError("checkpoint agent与运行update_step不一致")
    saved = Config.from_dict(payload["config"])
    if config:
        if payload.get("versions") != config.semantic_versions():
            raise ValueError("精确恢复语义版本不兼容")
        old, new = saved.to_dict(), config.to_dict()
        for section in ("data", "physics", "env", "model", "telemetry"):
            if old[section] != new[section]:
                raise ValueError(f"精确恢复{section}配置不兼容")
        allowed_runtime = {"episodes", "max_env_steps", "device", "torch_num_threads",
                           "checkpoint_every", "max_wall_seconds", "checkpoint_compression"}
        for key, value in old["train"].items():
            if key not in allowed_runtime and value != new["train"][key]:
                raise ValueError(f"精确恢复不允许改变train.{key}")
    if data_manifest_hash and payload.get("data_manifest_hash") != data_manifest_hash:
        raise ValueError("checkpoint数据清单hash不匹配")
    return payload
