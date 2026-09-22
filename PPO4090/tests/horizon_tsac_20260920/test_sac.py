"""W3/W4数学验收；合成观察仅作模型接口fixture，不作为训练数据。"""
import copy
import math
import pickle
import unittest
import numpy as np
import torch
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.env.action import ActionSpec
from implementations.horizon_tsac_20260920.models import Actor, Critic, ObservationEncoder, collate_observations, masked_distribution
from implementations.horizon_tsac_20260920.rl import DiscreteSAC, ReplayBuffer, Transition
from implementations.horizon_tsac_20260920.rl.sac import exact_expectations, temperature_loss


def small_config(n=4, length=2, powers=(5., 10.)):
    c = Config()
    c.physics.num_slots, c.physics.num_groups = n, 2
    c.env.max_block_length, c.env.power_levels_w = length, powers
    c.model.d_model, c.model.attention_heads, c.model.encoder_layers = 8, 2, 1
    c.model.candidate_chunk_size = 3
    c.train.microbatch_size = 2
    c.train.actor_lr = .001
    c.train.critic_lr = .001
    return c


def observation(config, beams=3, terminal=False, remaining=20.):
    n, g = config.physics.num_slots, config.physics.num_groups
    rng = np.random.default_rng(7)
    obs = dict(beam_static=rng.normal(size=(beams, 8)).astype(np.float32),
        beam_dynamic=rng.normal(size=(beams, 5)).astype(np.float32),
        group_id=np.arange(beams) % g, order_rank=np.arange(beams),
        entity_mask=np.ones(beams, bool), demand_mask=np.ones(beams, bool),
        status=np.ones(beams, np.int64), allocation_start=np.full(beams, -1, np.int64),
        allocation_length=np.zeros(beams, np.int64), allocation_power_w=np.zeros(beams, np.float32),
        occupancy=np.zeros((g, n), bool), current_slot_features=rng.normal(size=(n, 4)).astype(np.float32),
        global_features=rng.normal(size=16+g).astype(np.float32),
        current_beam_index=-1 if terminal else 0, terminal=terminal,
        current_group=0, remaining_power_w=remaining, legacy205=rng.normal(size=205).astype(np.float32))
    obs["action_spec_signature"] = ActionSpec(config).signature()
    obs["valid_action_mask"] = ActionSpec(config).valid_actions(obs)
    return obs


def episode(config, length=3, scenario="fixture", remaining=20., beams=3):
    obs = observation(config, beams=beams, remaining=remaining)
    return [Transition(obs, 0, float(i), observation(config, beams=beams, terminal=i == length-1, remaining=remaining),
                       i == length-1, scenario_id=scenario, versions=config.semantic_versions()) for i in range(length)]


class SACMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_action_oracle_and_gradient(self):
        logits = torch.tensor([[math.log(.2), math.log(.3), math.log(.5)]], dtype=torch.float64, requires_grad=True)
        p, lp = masked_distribution(logits, torch.ones_like(logits, dtype=torch.bool))
        q1, q2 = torch.tensor([[1., 3., 4.]]), torch.tensor([[2., 2., 5.]])
        v, loss, h = exact_expectations(p, lp, q1, q2, .1)
        expected_h = -sum(x * math.log(x) for x in (.2, .3, .5))
        self.assertAlmostEqual(h.item(), expected_h, places=12)
        self.assertAlmostEqual(v.item(), 2.8 + .1 * expected_h, places=12)
        self.assertAlmostEqual(loss.item(), -v.item(), places=12)
        loss.backward()
        analytical = logits.grad.detach().clone()
        for i in range(3):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[0, i] += 1e-6
            minus[0, i] -= 1e-6
            values = []
            for x in (plus, minus):
                pp, ll = masked_distribution(x, torch.ones_like(x, dtype=torch.bool))
                values.append(exact_expectations(pp, ll, q1, q2, .1)[1].item())
            self.assertAlmostEqual(analytical[0, i].item(), (values[0] - values[1]) / 2e-6, places=8)

    def test_high_q_action_probability_increases_without_critic_gradient(self):
        logits = torch.zeros(2, requires_grad=True)
        q1 = torch.tensor([0., 10.], requires_grad=True)
        q2 = torch.tensor([0., 12.], requires_grad=True)
        optimizer = torch.optim.SGD([logits], lr=.1)
        p, lp = masked_distribution(logits, torch.ones(2, dtype=torch.bool))
        loss = exact_expectations(p, lp, q1.detach(), q2.detach(), .01)[1]
        loss.backward()
        optimizer.step()
        self.assertGreater(torch.softmax(logits, 0)[1].item(), .5)
        self.assertIsNone(q1.grad)
        self.assertIsNone(q2.grad)

    def test_mask_zero_and_forced_skip(self):
        for actor_kind in ("conditional", "independent"):
            c = small_config()
            c.model.actor = actor_kind
            spec = ActionSpec(c)
            normal, forced = observation(c), observation(c, remaining=4.)
            normal["occupancy"][0, 1] = True
            normal["valid_action_mask"] = spec.valid_actions(normal)
            actor = Actor(c, spec, normal)
            batch = collate_observations([normal, forced])
            p, lp = actor(batch)
            torch.testing.assert_close(p.sum(-1), torch.ones(2))
            self.assertEqual(float(p[~batch["valid_action_mask"]].abs().sum()), 0.)
            self.assertEqual(float(p[1, 0]), 1.)
            self.assertTrue(torch.isfinite(lp).all())
            (p * lp).sum().backward()
            self.assertTrue(all(x.grad is None or torch.isfinite(x.grad).all() for x in actor.parameters()))

    def test_conditional_global_argmax_counterexample(self):
        c = small_config(2, 2, (5.,))
        obs, spec = observation(c), ActionSpec(c)
        agent = DiscreteSAC(c, spec, obs)
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.zero_()
            agent.actor.gate.bias.copy_(torch.tensor([math.log(.1), math.log(.9)]))
            agent.actor.start.bias.copy_(torch.tensor([math.log(.51), math.log(.49)]))
        probabilities = agent.probabilities(obs)
        self.assertEqual(agent.act(obs, deterministic=True).start, 1)
        self.assertAlmostEqual(float(probabilities[spec.encode(spec.actions[-1])]), .441, places=6)
        self.assertEqual(int(agent.actor.start.bias.argmax()), 0)

    def test_additive_critic_sums_before_twin_min(self):
        c = small_config()
        c.model.critic = "additive"
        spec, obs = ActionSpec(c), observation(c)
        first, second = Critic(c, spec, obs), Critic(c, spec, obs)
        with torch.no_grad():
            for net in (first, second):
                for parameter in net.parameters():
                    parameter.zero_()
            first.start.bias.fill_(10.)
            second.length.bias.fill_(10.)
        batch = collate_observations([obs])
        q1, q2 = first(batch, [1]), second(batch, [1])
        self.assertEqual(torch.minimum(q1, q2).item(), 10.)
        self.assertEqual(first(batch, [0]).item(), 0.)

    def test_temperature_direction_and_forced_exclusion(self):
        for entropy_value, increase in ((0., True), (math.log(3), False)):
            log_alpha = torch.tensor(0., requires_grad=True)
            loss, _ = temperature_loss(log_alpha, torch.tensor([entropy_value, 0.]), torch.tensor([3, 1]), .5)
            loss.backward()
            next_alpha = torch.exp(log_alpha.detach() - .1 * log_alpha.grad)
            self.assertEqual(next_alpha.item() > 1., increase)
            self.assertAlmostEqual(log_alpha.grad.item(), entropy_value - .5 * math.log(3), places=6)
        self.assertIsNone(temperature_loss(torch.tensor(0., requires_grad=True), torch.zeros(2), torch.ones(2), .5)[0])

    def test_forced_only_batch_does_not_step_alpha(self):
        c = small_config()
        agent = DiscreteSAC(c, ActionSpec(c), observation(c))
        before = agent.log_alpha.detach().clone()
        metrics = agent.update(episode(c, 2, remaining=4.))
        self.assertEqual(metrics["nonforced_count"], 0)
        self.assertIsNone(metrics["alpha_loss"])
        self.assertFalse(agent.alpha_optimizer.state)
        torch.testing.assert_close(before, agent.log_alpha, atol=0, rtol=0)

    def test_actual_actor_gradient_finite_difference(self):
        c = small_config()
        obs, spec = observation(c), ActionSpec(c)
        actor = Actor(c, spec, obs).double()
        batch = collate_observations([obs])
        batch = {k: v.double() if v.is_floating_point() else v for k, v in batch.items()}
        q = torch.arange(len(spec), dtype=torch.float64).unsqueeze(0)
        p, lp = actor(batch)
        exact_expectations(p, lp, q, q + 1., .01)[1].sum().backward()
        gradient = actor.gate.bias.grad[1].item()
        base = actor.gate.bias[1].item()
        losses = []
        for delta in (1e-6, -1e-6):
            with torch.no_grad():
                actor.gate.bias[1] = base + delta
                p, lp = actor(batch)
                losses.append(exact_expectations(p, lp, q, q + 1., .01)[1].item())
        self.assertAlmostEqual(gradient, (losses[0] - losses[1]) / 2e-6, places=7)

    def test_terminal_truncated_targets(self):
        c = small_config()
        obs, spec = observation(c), ActionSpec(c)
        agent = DiscreteSAC(c, spec, obs)
        with torch.no_grad():
            for net in (agent.target_critic_1, agent.target_critic_2):
                for parameter in net.parameters():
                    parameter.zero_()
                net.q[-1].bias.fill_(2.)
        terminal = Transition(obs, 0, 3., observation(c, terminal=True), True)
        truncated = Transition(obs, 0, 3., obs, False, True)
        ordinary = Transition(obs, 0, 3., obs, False)
        y = agent.targets([terminal, truncated, ordinary])
        self.assertEqual(y[0].item(), 3.)
        self.assertGreater(y[1].item(), 5.)
        self.assertEqual(y[1].item(), y[2].item())

    def test_soft_update_all_parameters_and_buffers(self):
        c = small_config()
        c.train.tau = .2
        agent = DiscreteSAC(c, ActionSpec(c), observation(c))
        before = copy.deepcopy(agent.target_critic_1.state_dict())
        with torch.no_grad():
            for p in agent.critic_1.parameters():
                p.add_(1.)
        agent.soft_update()
        for name, parameter in agent.target_critic_1.named_parameters():
            torch.testing.assert_close(parameter, before[name] + .2)
        self.assertTrue(all(x.grad is None for x in agent.target_critic_1.parameters()))

    def test_chunk_microbatch_gradient_equivalence(self):
        c = small_config()
        c.train.gradient_clip_norm = 10000.
        c.train.microbatch_size = 1
        left = DiscreteSAC(c, ActionSpec(c), observation(c))
        other = copy.deepcopy(c)
        other.train.microbatch_size = 8
        other.model.candidate_chunk_size = 10000
        right = DiscreteSAC(other, ActionSpec(other), observation(other))
        right.load_state_dict(left.state_dict())
        values = episode(c, 3) + episode(c, 1, remaining=4.)
        captured = []
        for agent in (left, right):
            gradients = {}
            for name in ("actor", "critic_1", "critic_2"):
                opt = getattr(agent, name + "_optimizer")
                step = opt.step
                def wrapped(*args, _name=name, _model=getattr(agent, name), _step=step, _grad=gradients, **kwargs):
                    _grad[_name] = [p.grad.detach().clone() if p.grad is not None else None for p in _model.parameters()]
                    return _step(*args, **kwargs)
                opt.step = wrapped
            captured.append(gradients)
        a, b = left.update(values), right.update(values)
        for metric in ("actor_loss", "critic_1_loss", "critic_2_loss", "alpha_loss", "entropy", "target_mean"):
            self.assertAlmostEqual(a[metric], b[metric], places=5)
        for name in captured[0]:
            for grad_a, grad_b in zip(captured[0][name], captured[1][name]):
                if grad_a is not None:
                    torch.testing.assert_close(grad_a, grad_b, atol=2e-6, rtol=2e-4)
        self.assertEqual(a["nonforced_count"], 3)
        for agent in (left, right):
            for name in ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
                self.assertTrue(all(float(s["step"]) == 1 for s in getattr(agent, name).state.values()))

    def test_checkpoint_next_sample_update_and_incompatibility(self):
        c = small_config()
        obs, spec = observation(c), ActionSpec(c)
        left = DiscreteSAC(c, spec, obs)
        left.update(episode(c))
        state = left.state_dict()
        right = DiscreteSAC(c, spec, obs)
        right.load_state_dict(state)
        self.assertEqual([left.act(obs) for _ in range(10)], [right.act(obs) for _ in range(10)])
        a, b = left.update(episode(c)), right.update(episode(c))
        self.assertEqual(a["actor_loss"], b["actor_loss"])
        for pa, pb in zip(left.actor.parameters(), right.actor.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        bad = copy.deepcopy(state)
        bad["versions"]["physics_version"] = "wrong"
        with self.assertRaises(ValueError):
            right.load_state_dict(bad)

    def test_changed_power_grid_rejected_despite_identical_mask(self):
        c = small_config()
        changed = copy.deepcopy(c)
        changed.env.power_levels_w = (6., 12.)
        a, b = observation(c), observation(changed)
        np.testing.assert_array_equal(a["valid_action_mask"], b["valid_action_mask"])
        agent = DiscreteSAC(c, ActionSpec(c), a)
        with self.assertRaisesRegex(ValueError, "ActionSpec"):
            agent.act(b)
        with self.assertRaisesRegex(ValueError, "ActionSpec"):
            agent.update(episode(changed))
        replay = ReplayBuffer(100, c.semantic_versions())
        replay.add_episode(episode(c))
        with self.assertRaisesRegex(ValueError, "ActionSpec"):
            replay.add_episode(episode(changed))

    def test_full_encoder_entity_permutation_and_padding(self):
        c = small_config()
        for mode in ("full_pool", "full_attention"):
            for slots in ("slots", "blocks10"):
                c.model.encoder, c.model.spectrum_tokens = mode, slots
                obs = observation(c)
                encoder = ObservationEncoder(c, obs)
                permutation = np.array([2, 0, 1])
                other = copy.deepcopy(obs)
                entity_keys = ("beam_static", "beam_dynamic", "group_id", "order_rank", "entity_mask", "demand_mask", "status", "allocation_start", "allocation_length", "allocation_power_w")
                for key in entity_keys:
                    other[key] = other[key][permutation]
                other["current_beam_index"] = 1
                padded = copy.deepcopy(obs)
                for key in entity_keys:
                    arr = padded[key]
                    padded[key] = np.concatenate((arr, np.zeros((2,) + arr.shape[1:], dtype=arr.dtype)))
                original = encoder(collate_observations([obs]))
                torch.testing.assert_close(original, encoder(collate_observations([other])), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(original, encoder(collate_observations([padded])), atol=1e-6, rtol=1e-5)

    def test_legacy_token_and_summary_information(self):
        c = small_config(100)
        obs = observation(c)
        for slots, count in (("blocks10", 14), ("slots", 104)):
            c.model.encoder, c.model.spectrum_tokens = "legacy14", slots
            encoder = ObservationEncoder(c, obs)
            self.assertEqual(encoder.legacy_positions.shape[1], count)
            self.assertTrue(torch.isfinite(encoder(collate_observations([obs]))).all())
        c.model.encoder = "summary"
        encoder = ObservationEncoder(c, obs)
        other = copy.deepcopy(obs)
        other["global_features"] += 10.
        self.assertFalse(torch.allclose(encoder(collate_observations([obs])), encoder(collate_observations([other]))))


class ReplayTests(unittest.TestCase):
    def test_transition_event_references_survive_replay_restore(self):
        c = small_config()
        obs = observation(c)
        transition = Transition(obs, 0, 0., observation(c, terminal=True), True, False, "scenario", c.semantic_versions(),
                                episode_id="episode-unique", step_index=1, env_step=121)
        replay = ReplayBuffer(10, c.semantic_versions())
        replay.add_episode([transition])
        restored = ReplayBuffer(10, c.semantic_versions())
        restored.load_state_dict(pickle.loads(pickle.dumps(replay.state_dict())))
        result = restored.sample(1)[0]
        self.assertEqual((result.episode_id, result.step_index, result.env_step), ("episode-unique", 1, 121))

    def test_immutable_whole_episode_eviction_and_versions(self):
        c = small_config()
        original = observation(c)
        t = Transition(original, 0, 1., observation(c, terminal=True), True, versions=c.semantic_versions())
        original["beam_static"][:] = 999.
        self.assertFalse(np.all(t.observation["beam_static"] == 999.))
        with self.assertRaises(ValueError):
            t.observation["beam_static"][0, 0] = 1.
        with self.assertRaises(TypeError):
            t.observation["terminal"] = True
        replay = ReplayBuffer(4, c.semantic_versions())
        replay.add_episode(episode(c, 2, "first"))
        replay.add_episode(episode(c, 3, "second"))
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay.evicted_episodes, 1)
        self.assertTrue(all(t.scenario_id == "second" for t in replay.sample(20)))
        with self.assertRaises(ValueError):
            replay.add_episode([Transition(original, 0, 0., original, True, versions={})])
        with self.assertRaises(ValueError):
            replay.add_episode(episode(c, 5))

    def test_bucket_episode_sampling_and_rng_restore(self):
        c = small_config()
        replay = ReplayBuffer(100, c.semantic_versions(), bucket_boundaries=(2,))
        replay.add_episode(episode(c, 1, "short", beams=1), n_demand=1)
        replay.add_episode(episode(c, 9, "long", beams=9), n_demand=9)
        samples = replay.sample(5000)
        fraction = sum(t.scenario_id == "short" for t in samples) / len(samples)
        self.assertTrue(.47 < fraction < .53)
        restored = ReplayBuffer(100, c.semantic_versions(), bucket_boundaries=(2,))
        # 标准pickle往返验证数组仍不可写，RNG及语义可恢复。
        restored.load_state_dict(pickle.loads(pickle.dumps(replay.state_dict())))
        left, right = replay.sample(100), restored.sample(100)
        self.assertEqual([(t.scenario_id, t.reward) for t in left], [(t.scenario_id, t.reward) for t in right])
        self.assertFalse(right[0].observation["beam_static"].flags.writeable)

    def test_incomplete_episode_and_explicit_truncated_fragment(self):
        c = small_config()
        obs = observation(c)
        values = [Transition(obs, 0, 0., obs, False, True, versions=c.semantic_versions())]
        replay = ReplayBuffer(10, c.semantic_versions())
        with self.assertRaises(ValueError):
            replay.add_episode(values)
        replay.add_episode(values, fragment=True)
        self.assertEqual(len(replay), 1)


if __name__ == "__main__":
    unittest.main()
