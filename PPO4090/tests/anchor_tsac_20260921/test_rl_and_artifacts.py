import json
import numpy as np
import pytest
import torch

from implementations.anchor_tsac_20260921.artifacts import base_manifest, load_checkpoint, save_checkpoint
from implementations.anchor_tsac_20260921.audit import sha256
from implementations.anchor_tsac_20260921.config import Config
from implementations.anchor_tsac_20260921.env.action import ActionSpec
from implementations.anchor_tsac_20260921.rl import DiscreteSAC, ReplayBuffer, Transition
from implementations.anchor_tsac_20260921.train import preflight, integration_check, run_training


class State:
    def state_dict(self): return {"fixture": True}


def obs(config, spec, terminal=False):
    value = {"anchor205": np.zeros(205, np.float32), "terminal": terminal,
             "occupancy": np.zeros((8, 100), bool), "current_group": 0,
             "remaining_power_w": 6000., "observation_adapter_version": config.semantic_versions()["observation_adapter_version"],
             "action_spec_signature": spec.signature()}
    value["valid_action_mask"] = spec.valid_actions(value)
    return value


def test_rl_paths_keep_eval_mode_and_synthetic_single_update_calls_optimizers():
    torch.set_num_threads(1)
    config, spec = Config(), ActionSpec(Config())
    before, after = obs(config, spec), obs(config, spec, terminal=True)
    agent = DiscreteSAC(config, spec, before)
    transition = Transition(before, 0, .25, after, True, config.semantic_versions())
    metrics = agent.update([transition])
    assert metrics["update_step"] == 1 and metrics["rl_forward_mode"] == "eval_dropout_disabled"
    assert metrics["actor_gradient_norm"] > 0
    assert all(not network.training for network in (agent.actor, agent.critic_1, agent.critic_2,
                                                      agent.target_critic_1, agent.target_critic_2))


def test_adapter_mismatch_and_checkpoint_version_are_rejected(tmp_path):
    config, spec = Config(), ActionSpec(Config())
    example = obs(config, spec); agent = DiscreteSAC(config, spec, example)
    wrong = dict(example, observation_adapter_version="modern205.v1")
    with pytest.raises(ValueError, match="适配器"): agent.probabilities(wrong)
    replay = ReplayBuffer(4, config.semantic_versions(), spec.signature())
    manifest = {"run_id": "synthetic", "trial_id": "synthetic-trial", "data_manifest_hash": "fixture",
                "initialization": {"kind": "from_scratch"}}
    path = save_checkpoint(tmp_path, agent, replay, State(), config, manifest,
                           {"episodes": 0, "env_step": 0, "update_step": 0}, 0,
                           {"fixture": np.random.default_rng(1)})
    assert load_checkpoint(path, config, "fixture")["format"] == "anchor_checkpoint.v1"
    sidecar = path.parent / "检查点清单.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8")); metadata["trial_id"] = "wrong-trial"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="身份不一致"): load_checkpoint(path)
    metadata["trial_id"] = "synthetic-trial"; sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    payload = torch.load(path, weights_only=False); payload["format"] = "unknown.v9"; torch.save(payload, path)
    metadata = json.loads(sidecar.read_text(encoding="utf-8")); metadata["sha256"] = sha256(path)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="不兼容"): load_checkpoint(path)


def test_default_entrypoints_are_nonexperimental_and_training_guard_fires_before_data():
    config = Config()
    preview = preflight(config)
    assert preview["experiment_started"] is False
    assert preview["manifest_preview"]["initialization"]["kind"] == "not_started"
    assert preview["manifest_preview"]["algorithm"] == "Anchor T-SAC"
    integrated = integration_check(config)
    assert integrated["is_demo"] is True and integrated["experiment_authorized"] is False
    with pytest.raises(RuntimeError, match="尚未授权"):
        run_training(config, dataset="does-not-exist.json")


def test_manifest_profile_is_derived_from_reward_scale_and_gamma():
    config = Config(); config.env.reward_scale = 77.; config.train.gamma = .5
    profile = base_manifest(config)["profile"]
    assert profile["reward"] == "77_delta_U" and profile["reward_scale"] == 77.
    assert profile["undiscounted_reward_identity"] == "episode_return=77*terminal_U"
    assert profile["discounted_objective_claim"] == "gamma=0.5的折扣优化目标不等价于terminal_U"
    config.train.gamma = 1.
    assert "gamma=1" in base_manifest(config)["profile"]["discounted_objective_claim"]


def test_checkpoint_compression_is_explicitly_rejected_in_v1():
    config = Config(); config.train.checkpoint_compression = True
    with pytest.raises(ValueError, match="未压缩原子格式"):
        config.validate()
