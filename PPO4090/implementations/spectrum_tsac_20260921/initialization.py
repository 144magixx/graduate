"""CNN 权重初始化：保留新 trial 的优化器、回放、计数和随机源。"""
import copy
import json
from pathlib import Path

import torch

from .artifacts import load_checkpoint
from .audit import sha256
from .config import Config


NETWORKS = ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
OPTIMIZERS = ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer")


def _fresh(agent):
    if agent.update_step != 0:
        raise ValueError("weights-only 初始化仅允许全新 update_step=0 的 agent")
    if any(getattr(agent, name).state for name in OPTIMIZERS):
        raise ValueError("weights-only 初始化要求所有目标优化器状态为空")


def _tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"初始化张量缺失或类型不符：{name}")
    if not torch.isfinite(value).all():
        raise ValueError(f"初始化张量存在非有限值：{name}")


def initialize_from_checkpoint(agent, config, data_manifest, path):
    """核验可信完整 CNN checkpoint 后原子复制网络与 alpha，返回新实验谱系。

    调用者负责建立独立新 run/trial；此函数不读取源回放进行采样，不恢复任何
    optimizer/RNG/counters，不改变目标配置，也不保存或覆盖源工件。
    """
    config.validate()
    if config.to_dict() != agent.config.to_dict():
        raise ValueError("初始化配置与目标 agent 不一致")
    _fresh(agent)
    manifest_hash = data_manifest.get("manifest_hash")
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("初始化需要已冻结的非空数据清单 hash")
    path = Path(path).resolve()
    sidecar = json.loads((path.parent / "检查点清单.json").read_text(encoding="utf-8"))
    payload = load_checkpoint(path, data_manifest_hash=manifest_hash)
    source_sha = sha256(path)
    if sidecar.get("sha256") != source_sha or any(sidecar.get(key) != payload.get(key)
            for key in ("checkpoint_id", "run_id", "trial_id", "counters", "boundary")):
        raise ValueError("初始化来源清单、身份或文件在读取期间不一致")
    source = payload.get("agent", {})
    if source.get("format_version") != "discrete_sac.v1":
        raise ValueError("初始化来源必须为完整 SAC checkpoint")
    saved = Config.from_dict(payload["config"])
    saved.validate()
    if Config.from_dict(source["config"]).to_dict() != saved.to_dict():
        raise ValueError("源 checkpoint 与 agent 配置不一致")
    if source.get("versions") != payload.get("versions") or source["versions"] != saved.semantic_versions():
        raise ValueError("源 checkpoint 语义版本不一致")
    if not all(payload.get(key) for key in ("run_id", "trial_id", "checkpoint_id")):
        raise ValueError("初始化来源缺少运行身份")
    if not isinstance(payload.get("counters"), dict) or source.get("update_step") != payload["counters"].get("update_step"):
        raise ValueError("源 checkpoint 更新计数不一致")
    if saved.model.encoder != "cnn_local" or saved.model.actor != "independent" or saved.model.critic != "additive":
        raise ValueError("初始化来源仅允许 CNN 局部编码器、independent actor 与 additive critic")
    if config.model.encoder not in ("cnn_local", "cnn_attention_residual") or config.model.actor != "independent" or config.model.critic != "additive":
        raise ValueError("初始化目标仅允许 CNN 或 CNN 全局残差，保留 independent/additive")
    if saved.model.d_model != config.model.d_model:
        raise ValueError("初始化来源与目标 d_model 不一致")
    old, new = saved.to_dict(), config.to_dict()
    if old["schema_version"] != new["schema_version"]:
        raise ValueError("初始化 schema_version 不一致")
    for section in ("physics", "data", "env"):
        left, right = copy.deepcopy(old[section]), copy.deepcopy(new[section])
        if section == "env":
            left.pop("observation_version", None)
            right.pop("observation_version", None)
        if left != right:
            raise ValueError(f"初始化来源与目标 {section} 不一致")
    if source.get("action_signature") != agent.action_signature():
        raise ValueError("初始化 ActionSpec 签名不一致")

    # 全部网络预检并提前转换到目标设备；遇到晚出现的坏张量时尚未改动目标。
    staged, original, copied, added = {}, {}, {}, {}
    for name in NETWORKS:
        network = getattr(agent, name)
        current = network.state_dict()
        incoming = source.get(name)
        if not isinstance(incoming, dict):
            raise ValueError(f"初始化缺少完整网络：{name}")
        if any(key.startswith("encoder.context_") for key in incoming):
            raise ValueError(f"CNN 来源不允许含有全局残差分支：{name}")
        extra = set(incoming) - set(current)
        missing = set(current) - set(incoming)
        if extra or any(not key.startswith("encoder.context_") for key in missing):
            raise ValueError(f"初始化网络键不兼容：{name}，额外={sorted(extra)}，缺失={sorted(missing)}")
        if missing and config.model.encoder != "cnn_attention_residual":
            raise ValueError(f"非残差目标不允许新增 context 张量：{name}")
        if config.model.encoder == "cnn_attention_residual":
            outputs = [key for key in missing if key.startswith("encoder.context_out.")]
            if set(outputs) != {"encoder.context_out.weight", "encoder.context_out.bias"}:
                raise ValueError(f"残差目标必须有独立零初始化 context_out：{name}")
            if any(torch.count_nonzero(current[key]).item() for key in outputs):
                raise ValueError(f"残差输出必须为全零，保证初始 CNN 恒等：{name}")
        original[name] = {key: value.detach().clone() for key, value in current.items()}
        staged[name] = {}
        for key, destination in current.items():
            _tensor(f"{name}.{key}（目标）", destination)
            if key in incoming:
                value = incoming[key]
                _tensor(f"{name}.{key}", value)
                if value.shape != destination.shape or value.dtype != destination.dtype:
                    raise ValueError(f"初始化 shape/dtype 不匹配：{name}.{key}")
                if key == "candidates" and not torch.equal(value.cpu(), destination.cpu()):
                    raise ValueError(f"初始化候选动作字典不一致：{name}")
                staged[name][key] = value.detach().to(device=destination.device).clone()
            else:
                staged[name][key] = destination.detach().clone()
        if "candidates" not in incoming:
            raise ValueError(f"初始化网络缺少候选动作字典：{name}")
        copied[name], added[name] = sorted(incoming), sorted(missing)
    alpha = source.get("log_alpha")
    _tensor("log_alpha", alpha)
    if alpha.shape != agent.log_alpha.shape or alpha.dtype != agent.log_alpha.dtype:
        raise ValueError("初始化 log_alpha shape/dtype 不匹配")
    if not torch.isfinite(alpha.exp()).all() or not (alpha.exp() > 0).all():
        raise ValueError("初始化 alpha 必须为有限正数")
    alpha = alpha.detach().to(device=agent.log_alpha.device).clone()
    original_alpha = agent.log_alpha.detach().clone()
    # 严格 load 在常规模块预检后不会失败；仍保留回滚防线。
    try:
        for name in NETWORKS:
            getattr(agent, name).load_state_dict(staged[name], strict=True)
        with torch.no_grad():
            agent.log_alpha.copy_(alpha)
    except Exception:
        for name in NETWORKS:
            getattr(agent, name).load_state_dict(original[name], strict=True)
        with torch.no_grad():
            agent.log_alpha.copy_(original_alpha)
        raise
    _fresh(agent)
    return {
        "initialization_version": "cnn_weights_only.v1",
        "kind": "weights_only_not_exact_resume",
        "source_run_id": payload["run_id"], "source_trial_id": payload["trial_id"],
        "source_checkpoint_id": payload["checkpoint_id"], "source_sha256": source_sha,
        "source_path": str(path), "source_counters": copy.deepcopy(payload["counters"]),
        "data_manifest_hash": manifest_hash, "source_model": copy.deepcopy(old["model"]),
        "target_model": copy.deepcopy(new["model"]), "copied_tensor_keys": copied,
        "new_tensor_keys": added, "copied_log_alpha": float(alpha.cpu()),
        "preserved_new_state": ["optimizer", "replay", "sampler", "RNG", "counters", "config", "new_trial_identity"],
        "new_update_step": 0,
    }
