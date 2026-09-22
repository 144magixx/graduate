import copy
import torch
import pytest

from implementations.anchor_tsac_20260921.audit import sha256
from implementations.anchor_tsac_20260921.config import Config
from implementations.anchor_tsac_20260921.env.action import ActionSpec
from implementations.anchor_tsac_20260921.initialization import initialize_from_legacy_full_model, NETWORK_SOURCE_KEYS
from implementations.anchor_tsac_20260921.rl import DiscreteSAC
from implementations.sac_fyh_io.sac_fyh_IO import PolicyNet, ValueNet


def observation(config, spec):
    value = {"anchor205": torch.zeros(205).numpy(), "terminal": False,
             "occupancy": torch.zeros(8, 100, dtype=torch.bool).numpy(), "current_group": 0,
             "remaining_power_w": 6000., "observation_adapter_version": config.semantic_versions()["observation_adapter_version"],
             "action_spec_signature": spec.signature()}
    value["valid_action_mask"] = spec.valid_actions(value)
    return value


def payload():
    torch.manual_seed(17)
    actor = PolicyNet(205, 128, 100, 10, 10)
    return {"actor_state_dict": actor.state_dict(), "critic_1": ValueNet(205, 128, 100, 10, 10).state_dict(),
            "critic_2": ValueNet(205, 128, 100, 10, 10).state_dict(),
            "target_critic_1": ValueNet(205, 128, 100, 10, 10).state_dict(),
            "target_critic_2": ValueNet(205, 128, 100, 10, 10).state_dict()}


def agent():
    config = Config(); spec = ActionSpec(config)
    return DiscreteSAC(config, spec, observation(config, spec))


def snapshot(value):
    return {name: {key: tensor.clone() for key, tensor in getattr(value, name).state_dict().items()}
            for name in NETWORK_SOURCE_KEYS}


def assert_same(value, before):
    for name, state in before.items():
        for key, tensor in state.items():
            assert torch.equal(tensor, getattr(value, name).state_dict()[key]), (name, key)


def test_five_network_migration_copies_core_heads_only_and_preserves_new_state(tmp_path):
    path = tmp_path / "trusted-old.pth"; torch.save(payload(), path)
    target = agent(); before_alpha = target.log_alpha.detach().clone()
    lineage = initialize_from_legacy_full_model(target, path, sha256(path))
    assert lineage["kind"] == "weights_only_not_exact_resume"
    assert lineage["new_trial"] is True and "旧权重可能已见" in lineage["source_data_seen_risk"]
    assert torch.equal(before_alpha, target.log_alpha)
    assert target.update_step == 0 and all(not getattr(target, name).state for name in (
        "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"))
    assert all(any(key.startswith(("skip_head.", "q_skip.")) for key in keys)
               for keys in lineage["new_tensor_keys"].values())


@pytest.mark.parametrize("damage", ["nan", "shape", "dtype", "missing", "extra", "unknown_top"])
def test_bad_late_tensor_or_unknown_model_is_rejected_without_partial_commit(tmp_path, damage):
    source = payload()
    if damage == "unknown_top": source["mystery"] = {}
    else:
        state = source["target_critic_2"]
        key = next(key for key, value in state.items() if value.is_floating_point() and value.ndim and value.shape[0] > 1)
        if damage == "nan": state[key].flatten()[0] = float("nan")
        elif damage == "shape": state[key] = state[key][:1]
        elif damage == "dtype": state[key] = state[key].double()
        elif damage == "missing": del state[key]
        elif damage == "extra": state["unknown"] = torch.zeros(1)
    path = tmp_path / "bad-old.pth"; torch.save(source, path)
    target = agent(); before = snapshot(target)
    with pytest.raises(ValueError): initialize_from_legacy_full_model(target, path, sha256(path))
    assert_same(target, before)


def test_wrong_expected_hash_and_nonfresh_target_are_rejected(tmp_path):
    path = tmp_path / "old.pth"; torch.save(payload(), path)
    target = agent(); before = snapshot(target)
    with pytest.raises(ValueError, match="SHA256"): initialize_from_legacy_full_model(target, path, "0" * 64)
    assert_same(target, before)
    target.update_step = 1
    with pytest.raises(ValueError, match="update_step"): initialize_from_legacy_full_model(target, path, sha256(path))
