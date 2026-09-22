"""论文MLP-PPO合法联合策略、GAE、裁剪及真实on-policy更新验收。"""
import copy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.data.loader import scenario_from_rows
from implementations.horizon_tsac_20260920.env.environment import Environment
from implementations.horizon_tsac_20260920.paper_baselines.ppo import (
    BaselinePPO, PPOHyperparameters, clipped_policy_loss, generalized_advantages)
from implementations.horizon_tsac_20260920.rl.replay import Transition


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def setup(beams=3, hyperparameters=None):
    config = Config()
    config.physics.num_slots = 4
    config.physics.num_groups = 2
    config.env.max_block_length = 2
    config.env.power_levels_w = (5., 10.)
    config.env.power_budget_w = 40.
    config.model.d_model = 8
    config.model.attention_heads = 2
    config.train.actor_lr = .003
    config.train.critic_lr = .003
    config.train.gamma = .9
    rows = [dict(beam_id=i+10, latitude_deg=28.+i*.1, longitude_deg=110.+i*.1,
                 demand_bps=1e8+i*1e6, ground_diameter_deg=2., group_id=i % 2) for i in range(beams)]
    scenario = scenario_from_rows(rows, config, "coverage.v2", "ppo-test-fixture")
    env = Environment(config)
    observation, _ = env.reset(scenario)
    agent = BaselinePPO(config, env.action_spec, observation, hyperparameters=hyperparameters)
    return config, env, observation, agent


def rollout(agent, env):
    observation, _ = env.reset(env.scenario)
    samples, logps, values = [], [], []
    while not observation["terminal"]:
        action, logp, value = agent.collect(observation)
        following, reward, terminated, _, _ = env.step(action)
        samples.append(Transition(observation, env.action_spec.encode(action), reward, following, terminated,
                       scenario_id=env.scenario.scenario_id, versions=env.config.semantic_versions(),
                       episode_id="fixture-episode", step_index=env.cursor, env_step=env.cursor))
        logps.append(logp)
        values.append(value)
        observation = following
    return samples, logps, values


def test_gae_matches_hand_calculation_and_terminal_zero_bootstrap():
    advantage, returns = generalized_advantages([1., 2.], [.5, 1.], [1., 99.], [False, True], [False, False], .9, .8)
    np.testing.assert_allclose(advantage, [2.12, 1.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(returns, [2.62, 2.], rtol=0, atol=1e-12)


def test_truncation_bootstraps_real_next_state_but_stops_next_episode_gae():
    advantage, returns = generalized_advantages([1., 100.], [.5, 7.], [2., 999.],
                                                [False, True], [True, False], .9, .95)
    np.testing.assert_allclose(advantage, [2.3, 93.], atol=1e-12)
    np.testing.assert_allclose(returns, [2.8, 100.], atol=1e-12)
    independent, _ = generalized_advantages([1., 100.], [.5, 7.], [2., 999.],
                                             [False, True], [False, False], .9, .95, [True, False])
    np.testing.assert_allclose(independent, advantage, atol=1e-12)


def test_ppo_initial_ratio_is_one_and_clip_has_correct_advantage_sign():
    old = torch.log(torch.tensor([.2, .4], dtype=torch.float64))
    loss, ratios = clipped_policy_loss(old, old, torch.tensor([1., -2.], dtype=torch.float64))
    torch.testing.assert_close(ratios, torch.ones_like(ratios), rtol=0, atol=0)
    assert loss.item() == pytest.approx(.5)
    for ratio, advantage, expected, zero_gradient in ((1.5, 1., -1.2, True), (.5, -1., .8, True),
                                                     (.5, 1., -.5, False), (1.5, -1., 1.5, False)):
        logp = torch.tensor([math.log(ratio)], dtype=torch.float64, requires_grad=True)
        value, _ = clipped_policy_loss(logp, torch.zeros_like(logp), torch.tensor([advantage], dtype=torch.float64))
        value.backward()
        assert value.item() == pytest.approx(expected)
        assert (logp.grad.item() == 0) is zero_gradient


def test_input_dimension_comes_from_real_small_environment():
    config, env, obs, agent = setup()
    assert len(obs["legacy205"]) == 5+2*config.physics.num_slots == agent.input_dim
    malformed = copy.deepcopy(obs)
    malformed["legacy205"] = np.zeros(205, dtype=np.float32)
    with pytest.raises(ValueError, match="维度"):
        agent.act(malformed)


def test_masked_distribution_zero_illegal_and_forced_skip_finite():
    _, env, obs, agent = setup()
    partial = copy.deepcopy(obs)
    partial["occupancy"][partial["current_group"], 1] = True
    partial["valid_action_mask"] = env.action_spec.valid_actions(partial)
    probability = agent.probabilities(partial)
    assert probability.sum() == pytest.approx(1., abs=1e-6)
    assert np.all(probability[~partial["valid_action_mask"]] == 0)
    forced = copy.deepcopy(partial)
    forced["occupancy"][:] = True
    forced["valid_action_mask"] = env.action_spec.valid_actions(forced)
    probability = agent.probabilities(forced)
    np.testing.assert_array_equal(probability, np.r_[1., np.zeros(len(env.action_spec)-1)])
    action, logp, value = agent.collect(forced)
    assert action.kind == "SKIP" and logp == 0. and np.isfinite(value)


def test_deterministic_action_is_global_joint_argmax_not_gate_argmax():
    _, env, obs, agent = setup()
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.gate.bias.copy_(torch.log(torch.tensor([.2, .8])))
    probability = agent.probabilities(obs)
    assert probability[1:].sum() == pytest.approx(.8, abs=1e-6)
    assert probability[0] > probability[1:].max()
    assert agent.act(obs, deterministic=True).kind == "SKIP"
    action, logp, value = agent.collect(obs)
    assert logp == pytest.approx(float(np.log(probability[env.action_spec.encode(action)])), abs=1e-6)
    assert value == pytest.approx(agent.value(obs))


def test_positive_advantage_increases_selected_joint_action_probability():
    config, env, obs, agent = setup(beams=1, hyperparameters=dict(epochs=1, minibatch_size=1,
                    entropy_coefficient=0., value_coefficient=0., normalize_advantages=False))
    with torch.no_grad():
        for module in (agent.actor, agent.critic):
            for parameter in module.parameters():
                parameter.zero_()
    action_id = 1
    initial = agent.probabilities(obs)
    following, _, done, _, _ = env.step(action_id)
    sample = Transition(obs, action_id, 10., following, done, scenario_id=env.scenario.scenario_id,
                        versions=config.semantic_versions())
    result = agent.update_rollout([sample], [float(np.log(initial[action_id]))], [0.])
    assert result["initial_ratio_max_error"] < 1e-6
    assert agent.probabilities(obs)[action_id] > initial[action_id]
    assert result["optimizer_steps"] == 1
    with pytest.raises(ValueError, match="陈旧"):
        agent.update_rollout([sample], [float(np.log(initial[action_id]))], [0.])


def test_collected_rollout_has_ratio_one_and_updates_without_replay():
    _, env, _, agent = setup(hyperparameters=dict(epochs=2, minibatch_size=2))
    samples, logps, values = rollout(agent, env)
    diagnostics = agent.update_rollout(samples, logps, values)
    assert diagnostics["initial_ratio_max_error"] < 1e-5
    assert diagnostics["optimizer_steps"] == 4
    assert [row["update_step"] for row in diagnostics["mini_updates"]] == [1, 2, 3, 4]
    assert [row["batch_size"] for row in diagnostics["mini_updates"]] == [2, 1, 2, 1]
    assert diagnostics["aggregation"]["first_update_step"] == 1
    assert diagnostics["aggregation"]["last_update_step"] == 4
    assert all(np.isfinite([row["actor_loss"], row["value_loss"], row["entropy"], row["actor_gradient_norm"]]).all()
               for row in diagnostics["mini_updates"])
    expected = sum(row["actor_loss"]*row["batch_size"] for row in diagnostics["mini_updates"])/6
    assert diagnostics["actor_loss"] == pytest.approx(expected)
    assert diagnostics["rollout_policy"] == "on_policy_no_replay"
    assert np.isfinite([diagnostics[key] for key in ("actor_loss", "value_loss", "entropy", "approx_kl")]).all()
    with pytest.raises(ValueError, match="同时提供"):
        agent.update_rollout(samples, old_log_probs=logps)


def test_noncontiguous_fragments_from_same_episode_do_not_share_gae(monkeypatch):
    from implementations.horizon_tsac_20260920.paper_baselines import ppo
    _, env, _, agent = setup(hyperparameters=dict(epochs=1))
    samples, logps, values = rollout(agent, env)
    captured = {}
    original = ppo.generalized_advantages

    def capture(*args, **kwargs):
        captured["boundaries"] = np.array(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(ppo, "generalized_advantages", capture)
    agent.update_rollout([samples[0], samples[2]], [logps[0], logps[2]], [values[0], values[2]])
    assert captured["boundaries"].tolist() == [True, False]


def test_old_samples_cannot_be_reused_by_omitting_behavior_records():
    _, env, _, agent = setup(hyperparameters=dict(epochs=1))
    samples, logps, values = rollout(agent, env)
    agent.update_rollout(samples, logps, values)
    before = copy.deepcopy(agent.actor.state_dict())
    previous_updates = agent.update_step
    with pytest.raises(ValueError, match="必须同时提供collect采集"):
        agent.update_rollout(samples)
    with pytest.raises(ValueError, match="陈旧"):
        agent.update_rollout(samples, logps, values)
    assert agent.update_step == previous_updates
    for key, tensor in agent.actor.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("metadata", ["all_missing", "episode_missing", "step_missing", "env_step_missing", "complete"])
def test_gae_only_links_adjacent_samples_with_complete_continuity_metadata(monkeypatch, metadata):
    from implementations.horizon_tsac_20260920.paper_baselines import ppo
    _, env, _, agent = setup(hyperparameters=dict(epochs=1))
    samples, logps, values = rollout(agent, env)
    # 复现审查：清空元数据后同一场景的第0和第2步不能默认接续GAE。
    indices = [0, 2] if metadata == "all_missing" else [0, 1]
    updates = {"all_missing": dict(episode_id="", step_index=None, env_step=None),
               "episode_missing": dict(episode_id=""), "step_missing": dict(step_index=None),
               "env_step_missing": dict(env_step=None), "complete": {}}[metadata]
    selected = [replace(samples[index], **updates) for index in indices]
    captured = {}
    original = ppo.generalized_advantages

    def capture(*args, **kwargs):
        captured["boundaries"] = np.array(args[-1])
        result = original(*args, **kwargs)
        captured["advantages"] = result[0]
        captured["first_one_step"] = args[0][0]+args[5]*args[2][0]-args[1][0]
        return result

    monkeypatch.setattr(ppo, "generalized_advantages", capture)
    agent.update_rollout(selected, [logps[index] for index in indices], [values[index] for index in indices])
    assert captured["boundaries"].tolist() == [metadata != "complete", False]
    if metadata != "complete":
        assert captured["advantages"][0] == pytest.approx(captured["first_one_step"])


def test_checkpoint_restores_sampling_and_minibatch_updates_exactly():
    config, env, obs, left = setup(hyperparameters=dict(epochs=2, minibatch_size=2))
    left.update_rollout(*rollout(left, env))
    state = left.state_dict()
    assert state["hyperparameters"]["gae_lambda"] == .95
    assert "未刊" in state["publication_status"]
    right = BaselinePPO(config, env.action_spec, obs, hyperparameters=state["hyperparameters"])
    global_rng = torch.get_rng_state().clone()
    right.load_state_dict(state)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    assert [left.act(obs) for _ in range(12)] == [right.act(obs) for _ in range(12)]
    samples, logps, values = rollout(left, env)
    a = left.update_rollout(samples, logps, values)
    b = right.update_rollout(samples, logps, values)
    for key in ("actor_loss", "value_loss", "entropy"):
        assert a[key] == b[key]
    for model_name in ("actor", "critic"):
        for first, second in zip(getattr(left, model_name).parameters(), getattr(right, model_name).parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cross_device_inference_skips_both_local_generators():
    config, env, obs, source = setup()
    state = source.state_dict()
    state["config"]["train"]["device"] = "cuda:0"
    state["action_rng"] = torch.tensor([1], dtype=torch.uint8)
    state["minibatch_rng"] = torch.tensor([1], dtype=torch.uint8)
    target = BaselinePPO(config, env.action_spec, obs)
    action_rng, minibatch_rng = target.action_rng.get_state().clone(), target.minibatch_rng.get_state().clone()
    target.load_state_dict(state, restore_rng=False)
    torch.testing.assert_close(target.action_rng.get_state(), action_rng, rtol=0, atol=0)
    torch.testing.assert_close(target.minibatch_rng.get_state(), minibatch_rng, rtol=0, atol=0)
    np.testing.assert_array_equal(target.probabilities(obs), source.probabilities(obs))
    with pytest.raises(ValueError, match="跨设备类型"):
        target.load_state_dict(state, restore_rng=True)


@pytest.mark.parametrize("changed", ["budget", "gradient_clip"])
def test_exact_resume_rejects_changed_learning_or_environment_before_mutation(changed):
    config, env, obs, source = setup()
    state = source.state_dict()
    changed_config = copy.deepcopy(config)
    if changed == "budget":
        changed_config.env.power_budget_w *= 10
    else:
        changed_config.train.gradient_clip_norm = 1e-5
    target = BaselinePPO(changed_config, env.action_spec, obs)
    before = copy.deepcopy(target.actor.state_dict())
    with pytest.raises(ValueError, match="精确续训.*配置不兼容"):
        target.load_state_dict(state)
    for key, tensor in target.actor.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)


def test_exact_resume_allows_explicit_runtime_budget_changes():
    config, env, obs, source = setup()
    state = source.state_dict()
    config.train.episodes += 1
    config.train.max_wall_seconds = 123.
    config.train.checkpoint_every = 5
    config.train.checkpoint_compression = True
    target = BaselinePPO(config, env.action_spec, obs)
    target.load_state_dict(state)
    np.testing.assert_array_equal(target.probabilities(obs), source.probabilities(obs))


def test_terminal_policy_and_altered_mask_are_rejected():
    _, env, obs, agent = setup(beams=1)
    altered = copy.deepcopy(obs)
    altered["valid_action_mask"][1] = False
    with pytest.raises(ValueError, match="mask"):
        agent.act(altered)
    final, _, _, _, _ = env.step(0)
    with pytest.raises(ValueError, match="终止"):
        agent.act(final)
    assert agent.value(final) == 0.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA真实跨设备PPO验收")
def test_real_cuda_checkpoint_can_be_evaluated_on_cpu():
    config, env, obs, _ = setup()
    config.train.device = "cuda:0"
    source = BaselinePPO(config, env.action_spec, obs)
    state = source.state_dict()
    config.train.device = "cpu"
    target = BaselinePPO(config, env.action_spec, obs)
    target.load_state_dict(state, restore_rng=False)
    assert target.probabilities(obs).sum() == pytest.approx(1., abs=1e-6)
