"""可信 CNN checkpoint 的新实验初始化；所有合成数据仅为接口测试 fixture。"""
import copy
import json
import math
import pickle
import random

import numpy as np
import pytest
import torch

from implementations.horizon_tsac_20260920.config import Config as OldConfig
from implementations.horizon_tsac_20260920.rl.sac import DiscreteSAC as OldSAC
from implementations.horizon_tsac_20260920.rl.replay import Transition as OldTransition
from implementations.horizon_tsac_20260920.artifacts import save_checkpoint
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.initialization import initialize_from_checkpoint, NETWORKS, OPTIMIZERS
from implementations.spectrum_tsac_20260921.rl.sac import DiscreteSAC
from implementations.spectrum_tsac_20260921.rl.replay import Transition


class FixtureState:
    def state_dict(self):
        return {"fixture_only": True}


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(219)
    old = OldConfig()
    old.model.encoder, old.model.actor, old.model.critic = "cnn_local", "independent", "additive"
    old.model.model_version = "paper_adapted_cnn_sac.v1"
    old.env.observation_version = "obs_legacy205"
    old.physics.num_slots, old.physics.num_groups = 8, 2
    old.env.max_block_length, old.env.power_levels_w = 2, (5., 10.)
    old.model.d_model, old.model.attention_heads, old.model.encoder_layers = 16, 4, 1
    new = Config.from_dict(old.to_dict())
    new.env.observation_version = "obs_full_v2"
    new.model.encoder, new.model.model_version = "cnn_attention_residual", "spectrum_tsac.v1"
    rows = [dict(beam_id=50+i, latitude_deg=29.+i*.1, longitude_deg=111.+i*.1,
                 demand_bps=(3-i)*1e8, ground_diameter_deg=2., group_id=i % 2) for i in range(3)]
    scenario = scenario_from_rows(rows, new, "coverage.v2", "synthetic-initialization-fixture")
    env = Environment(new)
    obs, _ = env.reset(scenario)
    agent = OldSAC(old, env.action_spec, obs)
    # 确保源优化器确有状态，检测误继承，而非空状态的平凡用例。
    for name in ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer"):
        optimizer = getattr(agent, name)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter) * .01
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    agent.log_alpha.grad = torch.ones_like(agent.log_alpha)
    agent.alpha_optimizer.step()
    agent.alpha_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        agent.log_alpha.fill_(math.log(.037))
    agent.update_step = 7
    manifest = {"run_id": "synthetic-source-run", "trial_id": "synthetic-source-trial", "data_manifest_hash": "fixture-data-hash"}
    counters = {"env_step": 21, "update_step": 7, "episodes": 7}
    path = save_checkpoint(tmp_path, agent, FixtureState(), FixtureState(), old, manifest, counters, 0,
                           {"fixture": np.random.default_rng(17)})
    return dict(source=agent, config=new, env=env, obs=obs, path=path, manifest={"manifest_hash": "fixture-data-hash"})


def target(fixture, mode="cnn_attention_residual"):
    config = copy.deepcopy(fixture["config"])
    config.model.encoder = mode
    agent = DiscreteSAC(config, fixture["env"].action_spec, fixture["obs"])
    return agent, config


def snapshot(agent):
    return {name: copy.deepcopy(getattr(agent, name).state_dict()) for name in NETWORKS}, agent.log_alpha.detach().clone()


def unchanged(agent, before):
    states, alpha = before
    for name in NETWORKS:
        for key, value in states[name].items():
            assert torch.equal(value, getattr(agent, name).state_dict()[key]), (name, key)
    assert torch.equal(alpha, agent.log_alpha.detach())


def rewrite(path, mutate):
    value = torch.load(path, map_location="cpu", weights_only=False)
    mutate(value)
    torch.save(value, path)
    sidepath = path.parent / "检查点清单.json"
    side = json.loads(sidepath.read_text(encoding="utf-8"))
    side["sha256"] = sha256(path)
    sidepath.write_text(json.dumps(side), encoding="utf-8")


@pytest.mark.parametrize("mode", ["cnn_local", "cnn_attention_residual"])
def test_t0_actor_all_critics_td_target_and_lineage_match(source, mode):
    agent, config = target(source, mode)
    fresh_context = {n: {k: t.clone() for k, t in getattr(agent, n).state_dict().items()
                         if k.startswith("encoder.context_")} for n in NETWORKS}
    lineage = initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    observations = [source["obs"]]
    obs, _, _, _, _ = source["env"].step(1)
    observations.append(obs)
    old = source["source"]
    batch = agent._batch(observations)
    with torch.no_grad():
        for left, right in zip(old.actor(batch), agent.actor(batch)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name in NETWORKS[1:]:
            torch.testing.assert_close(old._candidate_q(getattr(old, name), batch),
                                       agent._candidate_q(getattr(agent, name), batch), rtol=0, atol=0)
        for n in NETWORKS:
            for k, t in fresh_context[n].items():
                assert torch.equal(t, getattr(agent, n).state_dict()[k])
    old_t = OldTransition(observations[0], 1, .25, observations[1], False,
                       scenario_id="fixture", versions=old.config.semantic_versions())
    new_t = Transition(observations[0], 1, .25, observations[1], False,
                       scenario_id="fixture", versions=config.semantic_versions())
    torch.testing.assert_close(old.targets([old_t]), agent.targets([new_t]), rtol=0, atol=0)
    assert lineage["kind"] == "weights_only_not_exact_resume"
    assert lineage["source_counters"] == {"env_step": 21, "update_step": 7, "episodes": 7}
    assert lineage["source_run_id"] == "synthetic-source-run"
    assert lineage["source_sha256"] == sha256(source["path"])
    assert all(k.startswith("encoder.context_") for keys in lineage["new_tensor_keys"].values() for k in keys)


def test_keeps_new_rng_counters_and_optimizers_without_importing_source_state(source):
    agent, config = target(source)
    random.seed(293)
    np.random.seed(293)
    torch.manual_seed(293)
    agent.action_rng.manual_seed(923)
    before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
              torch.get_rng_state().clone(), agent.action_rng.get_state().clone())
    parameters = [copy.deepcopy(getattr(agent, name).param_groups[0]["lr"]) for name in OPTIMIZERS]
    initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    assert pickle.dumps(random.getstate()) == before[0]
    assert pickle.dumps(np.random.get_state()) == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert torch.equal(agent.action_rng.get_state(), before[3])
    assert agent.update_step == 0
    assert all(not getattr(agent, name).state for name in OPTIMIZERS)
    assert [getattr(agent, name).param_groups[0]["lr"] for name in OPTIMIZERS] == parameters
    assert abs(float(agent.alpha.detach()) - .037) < 1e-7
    assert source["source"].update_step == 7


@pytest.mark.parametrize("change", [
    lambda c: setattr(c.physics, "sinr_margin_db", 6.),
    lambda c: setattr(c.data, "traffic_scale", .5),
    lambda c: setattr(c.env, "power_budget_w", 5000.),
    lambda c: setattr(c.env, "reward_scale", 10.),
    lambda c: setattr(c.env, "power_levels_w", (5., 15.)),
])
def test_semantic_change_rejected_without_partial_mutation(source, change):
    agent, config = target(source)
    change(config)
    agent.config = copy.deepcopy(config)
    before = snapshot(agent)
    with pytest.raises(ValueError, match="初始化来源与目标"):
        initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    unchanged(agent, before)


@pytest.mark.parametrize("damage", ["shape", "nan", "dtype", "candidates", "missing", "extra", "alpha", "signature", "source_encoder"])
def test_late_bad_source_tensor_or_schema_does_not_partly_load(source, damage):
    def mutate(payload):
        last = payload["agent"]["target_critic_2"]
        key = next(k for k, v in last.items() if v.is_floating_point())
        if damage == "shape": last[key] = last[key][:1]
        elif damage == "nan": last[key].flatten()[0] = float("nan")
        elif damage == "dtype": last[key] = last[key].double()
        elif damage == "candidates": last["candidates"][1, 0] += 1
        elif damage == "missing": del last[key]
        elif damage == "extra": last["unknown"] = torch.zeros(1)
        elif damage == "alpha": payload["agent"]["log_alpha"] = torch.tensor(1000.)
        elif damage == "signature": payload["agent"]["action_signature"]["num_slots"] += 1
        elif damage == "source_encoder":
            for c in (payload["config"], payload["agent"]["config"]): c["model"]["encoder"] = "mlp_local"
    rewrite(source["path"], mutate)
    agent, config = target(source)
    before = snapshot(agent)
    with pytest.raises(ValueError):
        initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    unchanged(agent, before)


def test_nonzero_residual_refused_before_any_copy(source):
    agent, config = target(source)
    with torch.no_grad(): agent.target_critic_2.encoder.context_out.bias[0] = .1
    before = snapshot(agent)
    with pytest.raises(ValueError, match="残差输出必须为全零"):
        initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    unchanged(agent, before)


@pytest.mark.parametrize("reason", ["trained", "optimizer", "data_hash", "file_hash", "identity"])
def test_nonfresh_target_and_untrusted_identity_are_refused(source, reason):
    agent, config = target(source)
    manifest = source["manifest"].copy()
    if reason == "trained": agent.update_step = 1
    elif reason == "optimizer": agent.alpha_optimizer.state[agent.log_alpha] = {"step": torch.tensor(1.)}
    elif reason == "data_hash": manifest["manifest_hash"] = "different"
    elif reason == "file_hash":
        with source["path"].open("ab") as f: f.write(b"bad-hash")
    elif reason == "identity":
        path = source["path"].parent / "检查点清单.json"
        side = json.loads(path.read_text(encoding="utf-8")); side["checkpoint_id"] = "wrong"
        path.write_text(json.dumps(side), encoding="utf-8")
    before = snapshot(agent)
    with pytest.raises(ValueError):
        initialize_from_checkpoint(agent, config, manifest, source["path"])
    unchanged(agent, before)


def test_commit_exception_rolls_back_all_networks(source, monkeypatch):
    agent, config = target(source)
    before = snapshot(agent)
    original = agent.target_critic_2.load_state_dict
    calls = []
    def one_failure(state, strict=True):
        if not calls:
            calls.append(True)
            raise RuntimeError("synthetic commit failure")
        return original(state, strict=strict)
    monkeypatch.setattr(agent.target_critic_2, "load_state_dict", one_failure)
    with pytest.raises(RuntimeError, match="synthetic commit failure"):
        initialize_from_checkpoint(agent, config, source["manifest"], source["path"])
    unchanged(agent, before)
