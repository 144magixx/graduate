import torch

from implementations.anchor_tsac_20260921.config import Config
from implementations.anchor_tsac_20260921.env.action import ActionSpec
from implementations.anchor_tsac_20260921.models.networks import Actor, Critic
from implementations.sac_fyh_io.sac_fyh_IO import PolicyNet, ValueNet


def observation(config, spec, batch=3):
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(batch, 205, generator=generator)
    mask = torch.ones(batch, len(spec), dtype=torch.bool)
    return {"anchor205": values, "valid_action_mask": mask}


def copy_actor(old, new):
    state = new.state_dict()
    for key, value in old.state_dict().items():
        target = "core." + key if key not in ("freq_head.weight", "freq_head.bias", "slots_head.weight",
                                                "slots_head.bias", "power_head.weight", "power_head.bias") else key
        state[target] = value.clone()
    new.load_state_dict(state)


def copy_critic(old, new):
    state = new.state_dict()
    for key, value in old.state_dict().items():
        target = "core." + key if not key.startswith(("q_freq.", "q_slots.", "q_power.")) else key
        state[target] = value.clone()
    new.load_state_dict(state)


def test_old_actor_core_and_component_heads_match_eval_and_seeded_train_dropout():
    torch.manual_seed(7)
    config, spec = Config(), ActionSpec(Config())
    old = PolicyNet(205, 128, 100, 10, 10)
    new = Actor(config, spec)
    copy_actor(old, new)
    obs = observation(config, spec)
    old.eval(); new.eval()
    legacy = old(obs["anchor205"])
    _, freq, slots, power = new.component_logits(obs)
    for expected, actual in zip(legacy, map(lambda x: torch.softmax(x, -1), (freq, slots, power))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old.train(); new.train()
    torch.manual_seed(991); legacy = old(obs["anchor205"])
    torch.manual_seed(991); _, freq, slots, power = new.component_logits(obs)
    for expected, actual in zip(legacy, map(lambda x: torch.softmax(x, -1), (freq, slots, power))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_old_critic_core_and_component_heads_match():
    torch.manual_seed(11)
    config, spec = Config(), ActionSpec(Config())
    old, new = ValueNet(205, 128, 100, 10, 10), Critic(config, spec)
    copy_critic(old, new)
    obs = observation(config, spec)
    old.eval(); new.eval()
    expected = old(obs["anchor205"])
    _, *actual = new.component_values(obs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_position_is_zero_and_transformer_protocol_is_frozen():
    config, spec = Config(), ActionSpec(Config())
    actor = Actor(config, spec)
    assert torch.count_nonzero(actor.core.pos_embed) == 0
    layer = actor.core.encoder.layers[0]
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.out_features == 512
    assert layer.norm_first is False
    assert layer.dropout.p == .1
