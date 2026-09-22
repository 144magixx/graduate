"""可信旧full_model的五网络weights-only事务迁移。"""
import copy
from pathlib import Path
import torch
from .audit import sha256


NETWORK_SOURCE_KEYS = {
    "actor": "actor_state_dict", "critic_1": "critic_1", "critic_2": "critic_2",
    "target_critic_1": "target_critic_1", "target_critic_2": "target_critic_2",
}
OPTIMIZERS = ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer")


def _fresh(agent):
    if agent.update_step != 0 or any(getattr(agent, name).state for name in OPTIMIZERS):
        raise ValueError("weights-only迁移仅允许优化器为空且update_step=0的新agent")


def _legacy_name(destination_key):
    return destination_key[5:] if destination_key.startswith("core.") else destination_key


def _expected_legacy_keys(network):
    return {_legacy_name(key) for key in network.state_dict()
            if key != "candidates" and not key.startswith("skip_head.") and not key.startswith("q_skip.")}


def initialize_from_legacy_full_model(agent, path, expected_sha256):
    """全预检后一次提交；不迁移alpha、优化器、Replay、RNG或计数。"""
    _fresh(agent)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("必须显式提供64位期望SHA256")
    path = Path(path).resolve()
    actual_sha = sha256(path)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError("旧权重SHA256不匹配")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("当前PyTorch不支持安全weights_only加载，拒绝降级反序列化") from exc
    if not isinstance(payload, dict) or set(payload) != set(NETWORK_SOURCE_KEYS.values()):
        raise ValueError("旧full_model必须且只能包含五网络")
    staged, originals, copied, added = {}, {}, {}, {}
    for target_name, source_name in NETWORK_SOURCE_KEYS.items():
        network = getattr(agent, target_name)
        destination = network.state_dict()
        incoming = payload[source_name]
        if not isinstance(incoming, dict) or set(incoming) != _expected_legacy_keys(network):
            raise ValueError(f"{source_name}键集合不匹配，拒绝未知/错模型")
        originals[target_name] = {key: value.detach().clone() for key, value in destination.items()}
        staged[target_name], copied[target_name], added[target_name] = {}, [], []
        for key, destination_tensor in destination.items():
            if not isinstance(destination_tensor, torch.Tensor) or not torch.isfinite(destination_tensor).all():
                raise ValueError(f"目标张量非法：{target_name}.{key}")
            legacy_key = _legacy_name(key)
            if legacy_key in incoming:
                source_tensor = incoming[legacy_key]
                if not isinstance(source_tensor, torch.Tensor) or not torch.isfinite(source_tensor).all():
                    raise ValueError(f"来源张量非法：{source_name}.{legacy_key}")
                if source_tensor.shape != destination_tensor.shape or source_tensor.dtype != destination_tensor.dtype:
                    raise ValueError(f"来源shape/dtype不匹配：{source_name}.{legacy_key}")
                staged[target_name][key] = source_tensor.detach().to(destination_tensor.device).clone()
                copied[target_name].append(key)
            else:
                if key != "candidates" and not key.startswith(("skip_head.", "q_skip.")):
                    raise ValueError(f"非声明新增张量：{target_name}.{key}")
                staged[target_name][key] = destination_tensor.detach().clone()
                added[target_name].append(key)
        if "candidates" not in added[target_name]:
            raise ValueError("新动作字典必须由目标实现声明初始化")
    original_alpha = agent.log_alpha.detach().clone()
    try:
        for name in NETWORK_SOURCE_KEYS:
            getattr(agent, name).load_state_dict(staged[name], strict=True)
    except Exception:
        for name in NETWORK_SOURCE_KEYS:
            getattr(agent, name).load_state_dict(originals[name], strict=True)
        with torch.no_grad(): agent.log_alpha.copy_(original_alpha)
        raise
    _fresh(agent)
    return {"initialization_version": "legacy_transformer_weights_only.v1",
            "kind": "weights_only_not_exact_resume", "new_trial": True,
            "source_path": str(path), "source_sha256": actual_sha,
            "source_data_seen_risk": "旧权重可能已见现有全部CSV；仅作回归诊断",
            "model_equivalence": "仅匹配相同205输入下的Transformer core与三分量头；新9551联合mask/SKIP策略不等价",
            "copied_tensor_keys": copied, "new_tensor_keys": added,
            "preserved_new_state": ["alpha", "optimizers", "replay", "RNG", "counters", "new_trial_identity"],
            "rl_forward_mode": agent.RL_FORWARD_MODE, "new_update_step": 0}
