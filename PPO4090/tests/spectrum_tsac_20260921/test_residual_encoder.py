"""合成场景上的CNN等价起点与上下文残差验收；不代表学习性能提升。"""
import copy
import math

import numpy as np
import pytest
import torch

from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.models import Actor, Critic, ObservationEncoder, collate_observations
from implementations.spectrum_tsac_20260921.rl import DiscreteSAC, Transition


ENTITY_FIELDS = ("beam_static", "beam_dynamic", "group_id", "order_rank", "entity_mask",
                 "demand_mask", "status", "allocation_start", "allocation_length", "allocation_power_w")


@pytest.fixture(autouse=True)
def deterministic_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(3127)


def setup(slots=8):
    config = Config()
    config.physics.num_slots = slots
    config.physics.num_groups = 4
    config.env.max_block_length = 2
    config.env.power_levels_w = (5., 10.)
    config.model.d_model = 16
    config.model.attention_heads = 4
    config.model.encoder_layers = 1
    rows = [dict(beam_id=50+i, latitude_deg=28+i*.1, longitude_deg=110+i*.1,
                 demand_bps=(5-i)*1e8, ground_diameter_deg=2.+i*.2, group_id=i)
            for i in range(4)]
    scenario = scenario_from_rows(rows, config, "coverage.v2", "synthetic-residual-fixture")
    env = Environment(config)
    observation, _ = env.reset(scenario)
    return config, env, observation


def local_output(encoder, batch):
    x = batch["legacy205"]
    spectrum = torch.stack((x[:, 5:5+encoder.num_slots], x[:, 5+encoder.num_slots:]), dim=1)
    return encoder.local_network(torch.cat((x[:, :5], encoder.local_convolutions(spectrum).flatten(1)), dim=1))


def nonzero_out(encoder):
    with torch.no_grad():
        encoder.context_out.weight.normal_(0., .07)
        encoder.context_out.bias.normal_(0., .03)


def test_defaults_and_context_configuration_are_explicit():
    config = Config()
    assert config.model.model_version == "spectrum_tsac.v1"
    assert (config.model.encoder, config.model.actor, config.model.critic) == (
        "cnn_attention_residual", "independent", "additive")
    assert (config.model.context_layers, config.model.context_residual_scale) == (1, .25)
    assert Config.from_dict(config.to_dict()) == config
    for scale in (0., 1.):
        config.model.context_residual_scale = scale
        config.validate()
    for scale in (-.01, 1.01, float("nan"), float("inf"), True, "0.25"):
        config.model.context_residual_scale = scale
        with pytest.raises(ValueError, match="context_residual_scale"):
            config.validate()
    config.model.context_residual_scale = .25
    for layers in (0, -1, 1.5, True):
        config.model.context_layers = layers
        with pytest.raises(ValueError, match="context_layers"):
            config.validate()


@pytest.mark.parametrize("slots", [8, 100])
def test_cnn_weights_give_exact_actor_critics_and_td_targets_at_initialization(slots):
    config, env, obs = setup(slots)
    cnn_config = copy.deepcopy(config)
    cnn_config.model.encoder = "cnn_local"
    source = DiscreteSAC(cnn_config, env.action_spec, obs)
    target = DiscreteSAC(config, env.action_spec, obs)
    network_names = ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
    for name in network_names:
        original, residual = getattr(source, name), getattr(target, name)
        result = residual.load_state_dict(original.state_dict(), strict=False)
        assert not result.unexpected_keys
        assert result.missing_keys and all(key.startswith("encoder.context_") for key in result.missing_keys)
        assert all(name.startswith("encoder.context_") for name in set(residual.state_dict())-set(original.state_dict()))
        assert torch.count_nonzero(residual.encoder.context_out.weight) == 0
        assert torch.count_nonzero(residual.encoder.context_out.bias) == 0

    transitions, observations = [], []
    while not obs["terminal"]:
        observations.append(obs)
        action_id = int(np.flatnonzero(obs["valid_action_mask"])[1]) if len(observations) % 2 else 0
        following, reward, terminated, truncated, _ = env.step(action_id)
        transitions.append(Transition(obs, action_id, reward, following, terminated, truncated,
            scenario_id=env.scenario.scenario_id, versions=config.semantic_versions()))
        obs = following
    batch = collate_observations(observations)
    with torch.no_grad():
        for left, right in zip(source.actor(batch), target.actor(batch)):
            torch.testing.assert_close(left, right, rtol=0., atol=0.)
        for name in network_names[1:]:
            left, right = getattr(source, name), getattr(target, name)
            z1, z2 = left.encoder(batch), right.encoder(batch)
            torch.testing.assert_close(z1, z2, rtol=0., atol=0.)
            ids = torch.arange(len(env.action_spec))
            torch.testing.assert_close(left.values_from_encoded(z1, ids), right.values_from_encoded(z2, ids), rtol=0., atol=0.)
        torch.testing.assert_close(source.targets(transitions), target.targets(transitions), rtol=0., atol=0.)


def test_context_parameters_receive_gradients_after_zero_exit_opens():
    config, _, obs = setup()
    encoder = ObservationEncoder(config, obs)
    batch = collate_observations([obs])
    optimizer = torch.optim.SGD(encoder.parameters(), lr=.2)
    with torch.no_grad():
        desired = encoder(batch)+torch.linspace(.3, 1., encoder.width)
    first = {}
    second = {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (encoder(batch)-desired).square().mean()
        loss.backward()
        gradients = first if step == 0 else second
        for name, parameter in encoder.named_parameters():
            if name.startswith("context_"):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                gradients[name] = float(parameter.grad.abs().sum())
        optimizer.step()
    assert first["context_out.weight"] > 0 and first["context_out.bias"] > 0
    assert all(value == 0 for name, value in first.items() if not name.startswith("context_out."))
    for prefix in ("context_beam_mlp.", "context_group_embedding.", "context_status_embedding.",
                   "context_attention.", "context_global_mlp.", "context_query.", "context_read_beams."):
        assert sum(value for name, value in second.items() if name.startswith(prefix)) > 0, prefix


def test_nonzero_context_preserves_entity_permutation_padding_and_batching():
    config, _, obs = setup()
    encoder = ObservationEncoder(config, obs).eval()
    nonzero_out(encoder)
    permutation = np.array([2, 0, 3, 1])
    permuted = copy.deepcopy(obs)
    for key in ENTITY_FIELDS:
        permuted[key] = obs[key][permutation]
    permuted["current_beam_index"] = int(np.flatnonzero(permutation == obs["current_beam_index"])[0])
    padded = copy.deepcopy(obs)
    for key in ENTITY_FIELDS:
        value = padded[key]
        padded[key] = np.concatenate((value, np.zeros((3,)+value.shape[1:], dtype=value.dtype)))
    with torch.no_grad():
        expected = encoder(collate_observations([obs]))
        for other in (permuted, padded):
            torch.testing.assert_close(expected, encoder(collate_observations([other])), rtol=1e-5, atol=1e-6)
        batch = encoder(collate_observations([obs, padded]))
        torch.testing.assert_close(batch, expected.expand(2, -1), rtol=1e-5, atol=1e-6)


def test_context_intervention_changes_features_with_local_input_frozen():
    config, _, obs = setup()
    encoder = ObservationEncoder(config, obs).eval()
    nonzero_out(encoder)
    changed_globals, changed_beams = copy.deepcopy(obs), copy.deepcopy(obs)
    changed_globals["global_features"] = changed_globals["global_features"]+.8
    changed_beams["beam_static"][1, 2] += 2.
    with torch.no_grad():
        original = encoder(collate_observations([obs]))
        for changed in (changed_globals, changed_beams):
            np.testing.assert_array_equal(obs["legacy205"], changed["legacy205"])
            assert not torch.allclose(original, encoder(collate_observations([changed])), rtol=1e-6, atol=1e-7)


def test_residual_is_bounded_and_scale_factor_does_not_backpropagate():
    config, _, obs = setup()
    encoder = ObservationEncoder(config, obs)
    batch = collate_observations([obs])
    nonzero_out(encoder)
    with torch.no_grad():
        encoder.local_network[-2].bias.fill_(8.)
        local = local_output(encoder, batch)
        result = encoder(batch)
        bound = config.model.context_residual_scale*local.square().mean(-1, keepdim=True).sqrt().clamp_min(1.)
        assert torch.all((result-local).abs() <= bound+1e-6)
        encoder.context_out.weight.zero_()
        encoder.context_out.bias.fill_(math.atanh(.3))
    encoder.zero_grad(set_to_none=True)
    local_output(encoder, batch).sum().backward()
    expected = {name: parameter.grad.clone() for name, parameter in encoder.named_parameters() if name.startswith("local_")}
    encoder.zero_grad(set_to_none=True)
    encoder(batch).sum().backward()
    for name, parameter in encoder.named_parameters():
        if name.startswith("local_"):
            torch.testing.assert_close(parameter.grad, expected[name], rtol=0., atol=0.)
    encoder.context_residual_scale = 0.
    torch.testing.assert_close(encoder(batch), local_output(encoder, batch), rtol=0., atol=0.)


def test_terminal_and_empty_entity_batches_are_finite_after_context_training():
    config, env, obs = setup()
    encoder = ObservationEncoder(config, obs).eval()
    nonzero_out(encoder)
    while not obs["terminal"]:
        obs, *_ = env.step(0)
    empty = copy.deepcopy(obs)
    for key in ENTITY_FIELDS:
        empty[key] = empty[key][:0]
    with torch.no_grad():
        assert torch.isfinite(encoder(collate_observations([obs]))).all()
        assert torch.isfinite(encoder(collate_observations([empty]))).all()
        assert torch.isfinite(encoder(collate_observations([empty, obs]))).all()


@pytest.mark.parametrize("mode", ["legacy14", "summary", "full_pool", "full_attention", "mlp_local", "cnn_local", "cnn_attention_residual"])
def test_existing_encoders_and_action_heads_remain_available(mode):
    config, env, obs = setup(100)
    config.model.encoder = mode
    batch = collate_observations([obs])
    for actor_mode, critic_mode in (("independent", "additive"), ("conditional", "joint")):
        config.model.actor, config.model.critic = actor_mode, critic_mode
        config.validate()
        actor = Actor(config, env.action_spec, obs)
        critic = Critic(config, env.action_spec, obs)
        probability, log_probability = actor(batch)
        assert torch.isfinite(probability).all() and torch.isfinite(log_probability).all()
        torch.testing.assert_close(probability.sum(-1), torch.ones(1), rtol=1e-5, atol=1e-6)
        assert torch.isfinite(critic(batch, torch.tensor([1]))).all()
