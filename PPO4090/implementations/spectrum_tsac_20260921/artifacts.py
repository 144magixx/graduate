"""运行身份、原子工件与回合边界checkpoint。"""
import copy
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import random
import sys
import uuid
import numpy as np
from project_paths import PROJECT_ROOT, SPECTRUM_OUTPUT_DIR
from .config import ASSUMPTIONS, Config
from .audit import sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def config_hash(config):
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p)
            for p in Path(__file__).parent.rglob("*.py")}


def create_run(config, data_manifest, parent=None, mode="train"):
    import torch
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
    directory = SPECTRUM_OUTPUT_DIR / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": "run.v1", "run_id": run_id,
                "trial_id": parent["trial_id"] if parent else uuid.uuid4().hex,
                "attempt_id": uuid.uuid4().hex, "parent_run_id": parent["run_id"] if parent else None,
                "parent_checkpoint_id": parent.get("checkpoint_id") if parent else None,
                "created_at_utc": now.isoformat(), "algorithm": "CNN-SAC（同源继续训练对照）" if config.model.encoder == "cnn_local" else "Spectrum T-SAC", "mode": mode,
                "status": "created", "seed": config.train.seed, "config": config.to_dict(),
                "config_hash": config_hash(config), "versions": config.semantic_versions(),
                "data_manifest_hash": data_manifest["manifest_hash"], "data_manifest": data_manifest,
                "assumptions": ASSUMPTIONS, "code_hashes": source_hashes(),
                "resume_policy": "episode_boundary_new_run_stable_trial",
                "determinism": "CPU确定性fixture复验；跨硬件/软件版本不保证位级一致",
                "hardware": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                             "torch": torch.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
                             "device": config.train.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                "physics_version": config.physics.physics_version, "reward_version": config.env.reward_version,
                "metric_version": config.env.metric_version, "dataset_version": config.data.dataset_version,
                "power_budget_w": config.env.power_budget_w, "model": config.model.to_dict() if hasattr(config.model, "to_dict") else config.to_dict()["model"]}
    for name, payload in (("运行清单.json", manifest), ("配置快照.json", config.to_dict()), ("数据划分.json", data_manifest)):
        atomic_json(directory / name, payload)
    return directory, manifest


def capture_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value):
    import torch
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def save_checkpoint(run_dir, agent, replay, sampler, config, manifest, counters, durable_event_seq, local_rngs):
    import torch
    checkpoint_id = uuid.uuid4().hex
    payload = {"format": "horizon_checkpoint.v1", "checkpoint_id": checkpoint_id,
               "run_id": manifest["run_id"], "trial_id": manifest["trial_id"], "config": config.to_dict(),
               "versions": config.semantic_versions(), "data_manifest_hash": manifest["data_manifest_hash"],
               "agent": agent.state_dict(), "replay": replay.state_dict(), "sampler": sampler.state_dict(),
               "counters": dict(counters), "rng": capture_rng(),
               "initialization": copy.deepcopy(manifest.get("initialization")),
               "local_rngs": {key: copy.deepcopy(value.bit_generator.state) for key, value in local_rngs.items()},
               "durable_event_seq": int(durable_event_seq), "boundary": "episode",
               "normalization": {"kind": "fixed_observation_spec", "test_fitted": False}}
    path = Path(run_dir) / "检查点.pt"
    temporary = path.with_name("检查点." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        if config.train.checkpoint_compression:
            with gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1,mtime=0) as compressed:
                torch.save(payload,compressed)
        else:torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sidecar = {"format": payload["format"], "checkpoint_id": checkpoint_id,
                 "compression":"gzip" if config.train.checkpoint_compression else "none",
                 "file": path.name, "sha256": sha256(path), "durable_event_seq": int(durable_event_seq),
                 "counters": counters, "run_id": manifest["run_id"], "trial_id": manifest["trial_id"], "boundary": "episode"}
    atomic_json(Path(run_dir) / "检查点清单.json", sidecar)
    # DB引用不可变历史文件；根目录检查点只是可原子替换的最新入口。
    archive = Path(run_dir) / "检查点存档" / checkpoint_id
    archive.mkdir(parents=True,exist_ok=False)
    archived_path = archive / path.name
    try:
        os.link(path, archived_path)
    except OSError:
        import shutil
        shutil.copy2(path, archived_path)
    atomic_json(archive / "检查点清单.json", sidecar)
    return archived_path


def load_checkpoint(path, config=None, data_manifest_hash=None):
    """只加载本项目已保存且hash核验的可信本地checkpoint，不加载历史不明权重。"""
    import torch
    path = Path(path)
    sidecar = json.loads((path.parent / "检查点清单.json").read_text(encoding="utf-8"))
    if sidecar["file"] != path.name or sha256(path) != sidecar["sha256"]:
        raise ValueError("checkpoint内容hash与清单不一致")
    if sidecar.get('compression','none')=='gzip':
        with gzip.open(path,'rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=False)
    elif sidecar.get('compression','none')=='none':payload = torch.load(path, map_location="cpu", weights_only=False)
    else:raise ValueError('未知checkpoint压缩格式')
    if payload.get("format") != "horizon_checkpoint.v1" or payload.get("boundary") != "episode":
        raise ValueError("不兼容的checkpoint格式或恢复边界")
    saved = Config.from_dict(payload["config"])
    if config:
        if payload["versions"] != config.semantic_versions():
            raise ValueError("checkpoint语义版本不兼容")
        for key in ("physics", "env", "model", "data"):
            if saved.to_dict()[key] != config.to_dict()[key]:
                raise ValueError(f"checkpoint {key} 配置不兼容；改变语义需要新实验")
        allowed_runtime_changes={"episodes","max_env_steps","device","torch_num_threads","checkpoint_every","max_wall_seconds","checkpoint_compression"}
        for key,value in saved.to_dict()["train"].items():
            if key not in allowed_runtime_changes and value != config.to_dict()["train"][key]:
                raise ValueError(f"精确恢复不允许改变train.{key}；请另建具名实验")
    if data_manifest_hash and payload["data_manifest_hash"] != data_manifest_hash:
        raise ValueError("checkpoint数据划分hash不匹配")
    return payload
