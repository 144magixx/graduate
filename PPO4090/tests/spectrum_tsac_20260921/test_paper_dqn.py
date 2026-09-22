"""论文三head求和 DQN 的数学、合法性与恢复回归。"""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch

from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.env.action import Action, ActionSpec
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.paper_baselines.dqn import BaselineDQN
from implementations.spectrum_tsac_20260921.rl.replay import Transition
from tests.spectrum_tsac_20260921.test_data import fixture


def small_config(slots=2, powers=(5., 10.), budget=20.):
    config = Config()
    config.physics.num_slots = slots
    config.env.max_block_length = 2
    config.env.power_levels_w = powers
    config.env.power_budget_w = budget
    config.train.batch_size = 4
    config.train.microbatch_size = 2
    config.train.critic_lr = .01
    config.train.gamma = .5
    config.train.tau = .1
    return config


def environment_example(config, beams=2):
    env = Environment(config)
    observation, _ = env.reset(fixture(beams, config))
    return env, observation


def set_heads(network, start, length, power, skip):
    with torch.no_grad():
        for value in network.parameters():
            value.zero_()
        network.start_head.bias.copy_(torch.tensor(start))
        network.length_head.bias.copy_(torch.tensor(length))
        network.power_head.bias.copy_(torch.tensor(power))
        network.skip_head.bias.fill_(skip)


def completed_actions(config):
    result = []
    for action, diagnostic_reward in ((Action(), 0.), (Action("ALLOC", 0, 1, 0), 10.)):
        env, obs = environment_example(config, 1)
        nxt, _, terminated, _, _ = env.step(action)
        # 此确定性reward仅为DQN学习验收的两动作oracle，不是生产物理数据。
        result.append(Transition(obs, env.action_spec.encode(action), diagnostic_reward, nxt, terminated,
                                 scenario_id="dqn_math_fixture", versions=config.semantic_versions()))
    return result


class PaperDQNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_action_sum_argmax_respects_boundary_not_headwise_greedy(self):
        config = small_config()
        env, obs = environment_example(config)
        agent = BaselineDQN(config, env.action_spec, obs)
        set_heads(agent.online, [10., 11.], [0., 100.], [2., 3.], -5.)
        # 三head逐个argmax会请求start=1,length=2并越界；合法全集最优在start=0。
        self.assertEqual(agent.act(obs, deterministic=True), Action("ALLOC", 0, 2, 1))
        q = agent.q_values(obs)
        self.assertEqual(q[env.action_spec.encode(Action("ALLOC", 0, 2, 1))], 113.)
        self.assertEqual(q[0], -5.)
        self.assertEqual(agent.interaction_step, 0)

    def test_masked_non_double_target_terminal_and_truncated_hand_oracle(self):
        config = small_config(budget=5.)
        env, obs = environment_example(config)
        nxt, _, _, _, _ = env.step(Action())
        terminal_obs, _, _, _, _ = env.step(Action())
        agent = BaselineDQN(config, env.action_spec, obs)
        set_heads(agent.target, [1., 2.], [3., 4.], [5., 100.], 7.)
        set_heads(agent.online, [0., 0.], [0., 0.], [0., 0.], 100.)
        # 10W档虽有100的head值但不可行；target合法max=10，online偏好SKIP不参与选择。
        normal = Transition(obs, 0, 2., nxt, False, versions=config.semantic_versions())
        truncated = Transition(obs, 0, 2., nxt, False, True, versions=config.semantic_versions())
        terminal = Transition(nxt, 0, 3., terminal_obs, True, versions=config.semantic_versions())
        torch.testing.assert_close(agent.targets([normal, truncated, terminal]), torch.tensor([7., 7., 3.]), rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))

    def test_epsilon_decay_legal_sampling_forced_skip_and_evaluation_is_pure(self):
        config = small_config()
        env, obs = environment_example(config)
        agent = BaselineDQN(config, env.action_spec, obs, epsilon_start=1., epsilon_end=.05, epsilon_decay_steps=10)
        self.assertEqual(agent.epsilon, 1.)
        for _ in range(5):
            action = agent.act(obs)
            self.assertTrue(obs["valid_action_mask"][env.action_spec.encode(action)])
        self.assertAlmostEqual(agent.epsilon, .525)
        before_rng, before_step = copy.deepcopy(agent.rng.bit_generator.state), agent.interaction_step
        for _ in range(3):
            agent.act(obs, deterministic=True)
        self.assertEqual(before_rng, agent.rng.bit_generator.state)
        self.assertEqual(before_step, agent.interaction_step)
        agent.set_interaction_step(100)
        self.assertAlmostEqual(agent.epsilon, .05)
        exhausted = copy.deepcopy(obs)
        exhausted["remaining_power_w"] = 4.
        exhausted["valid_action_mask"] = env.action_spec.valid_actions(exhausted)
        for _ in range(20):
            self.assertEqual(agent.act(exhausted), Action())
        self.assertEqual(agent.interaction_step, 120)

    def test_full_space_exploration_and_no_policy_call_on_terminal(self):
        config = small_config()
        env, obs = environment_example(config)
        agent = BaselineDQN(config, env.action_spec, obs, epsilon_start=1., epsilon_end=1.)
        seen = {env.action_spec.encode(agent.act(obs)) for _ in range(300)}
        self.assertEqual(seen, set(np.flatnonzero(obs["valid_action_mask"])))
        env.step(Action())
        terminal, *_ = env.step(Action())
        count, rng = agent.interaction_step, copy.deepcopy(agent.rng.bit_generator.state)
        with self.assertRaises(ValueError):
            agent.act(terminal)
        self.assertEqual(agent.interaction_step, count)
        self.assertEqual(agent.rng.bit_generator.state, rng)

    def test_update_learns_high_value_action_and_soft_updates_target(self):
        config = small_config()
        config.train.critic_lr = .05
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        set_heads(agent.online, [0., 0.], [0., 0.], [0., 0.], 0.)
        agent.target.load_state_dict(agent.online.state_dict())
        self.assertEqual(agent.act(obs, deterministic=True), Action())
        samples = completed_actions(config)
        previous_target = {name: value.detach().clone() for name, value in agent.target.named_parameters()}
        first_loss = agent.update(samples)["dqn_loss"]
        online_parameters = dict(agent.online.named_parameters())
        for name, value in agent.target.named_parameters():
            expected = previous_target[name]*(1-config.train.tau)+online_parameters[name].detach()*config.train.tau
            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-8)
        for _ in range(59):
            diagnostics = agent.update(samples)
        self.assertLess(diagnostics["dqn_loss"], first_loss*.2)
        self.assertEqual(agent.act(obs, deterministic=True), Action("ALLOC", 0, 1, 0))
        self.assertEqual(agent.update_step, 60)
        self.assertEqual(agent.interaction_step, 0)  # 优化器更新不推进epsilon环境步。
        self.assertFalse(agent.target.training)
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
        self.assertGreater(float(agent.target.power_head.bias[0]), 0.)

    def test_microbatch_uses_one_effective_batch_update(self):
        torch.manual_seed(123)
        config = small_config()
        env, obs = environment_example(config, 1)
        left = BaselineDQN(config, env.action_spec, obs)
        other = copy.deepcopy(config)
        other.train.microbatch_size = 20
        right = BaselineDQN(other, env.action_spec, obs)
        right.load_state_dict(left.state_dict(), restore_rng=False)
        batch = completed_actions(config)*3
        a, b = left.update(batch), right.update(batch)
        np.testing.assert_allclose(a["dqn_loss"], b["dqn_loss"], rtol=2e-7, atol=2e-6)
        self.assertEqual((left.update_step, right.update_step), (1, 1))
        for name, value in left.online.state_dict().items():
            torch.testing.assert_close(value, right.online.state_dict()[name], rtol=1e-5, atol=2e-7)

    def test_checkpoint_restores_epsilon_rng_optimizer_and_next_update_exactly(self):
        config = small_config()
        env, obs = environment_example(config, 1)
        left = BaselineDQN(config, env.action_spec, obs, epsilon_decay_steps=50)
        samples = completed_actions(config)
        left.update(samples)
        for _ in range(7):
            left.act(obs)
        left.train(False)
        state = left.state_dict()
        right = BaselineDQN(config, env.action_spec, obs, epsilon_decay_steps=50)
        right.load_state_dict(state)
        self.assertFalse(right.online.training)
        self.assertFalse(right.target.training)
        self.assertEqual(left.epsilon, right.epsilon)
        self.assertEqual([left.act(obs) for _ in range(30)], [right.act(obs) for _ in range(30)])
        a, b = left.update(samples), right.update(samples)
        self.assertEqual(a["dqn_loss"], b["dqn_loss"])
        for name, value in left.online.state_dict().items():
            self.assertTrue(torch.equal(value, right.online.state_dict()[name]), name)
        for name, value in left.target.state_dict().items():
            self.assertTrue(torch.equal(value, right.target.state_dict()[name]), name)

    def test_malformed_mask_signature_and_terminal_signature_rejected_before_update(self):
        config = small_config()
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        bad = copy.deepcopy(obs)
        bad["valid_action_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "mask"):
            agent.act(bad)
        bad = copy.deepcopy(obs)
        bad["action_spec_signature"]["power_levels_w"] = [6., 12.]
        with self.assertRaisesRegex(ValueError, "签名"):
            agent.act(bad)
        sample = completed_actions(config)[0]
        bad_next = {**sample.next_observation, "action_spec_signature": bad["action_spec_signature"]}
        transition = Transition(sample.observation, sample.action_id, sample.reward, bad_next, True, versions=config.semantic_versions())
        saved = agent.state_dict()
        with self.assertRaisesRegex(ValueError, "签名"):
            agent.update([transition])
        self.assertEqual(agent.update_step, 0)
        for name, value in agent.online.state_dict().items():
            self.assertTrue(torch.equal(value, saved["online"][name]))

    def test_checkpoint_rejects_changed_epsilon_and_power_semantics(self):
        config = small_config()
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        state = agent.state_dict()
        other = BaselineDQN(config, env.action_spec, obs, epsilon_decay_steps=200)
        with self.assertRaisesRegex(ValueError, "配置"):
            other.load_state_dict(state)
        state["action_signature"]["power_levels_w"] = [6., 12.]
        with self.assertRaisesRegex(ValueError, "配置"):
            agent.load_state_dict(state)

    def test_cross_device_exact_resume_rejected_but_evaluation_skips_cuda_rng(self):
        config = small_config()
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        state = agent.state_dict()
        state["training_device_type"] = "cuda"
        state["cuda_rng"] = [torch.ones(8, dtype=torch.uint8)]
        with self.assertRaisesRegex(ValueError, "设备类型"):
            agent.load_state_dict(state)
        with patch("torch.cuda.set_rng_state_all") as cuda_restore:
            agent.load_state_dict(state, restore_rng=False)
            cuda_restore.assert_not_called()

    def test_direct_exact_resume_checks_physics_budget_data_model_and_train(self):
        config = small_config()
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        saved = agent.state_dict()
        changes = (("physics", "noise_temperature_k", 300.), ("env", "power_budget_w", 15.),
                   ("data", "traffic_scale", .5), ("model", "d_model", 64),
                   ("train", "batch_size", 8), ("train", "microbatch_size", 1),
                   ("train", "update_schedule", "env_step"), ("train", "seed", 100))
        for section, key, value in changes:
            altered = copy.deepcopy(config)
            setattr(getattr(altered, section), key, value)
            other_env, other_obs = environment_example(altered, 1)
            other = BaselineDQN(altered, other_env.action_spec, other_obs)
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "精确恢复"):
                other.load_state_dict(saved)
        altered = copy.deepcopy(config)
        altered.train.episodes += 2
        altered.train.max_env_steps = 100
        altered.train.max_wall_seconds = 20.
        altered.train.checkpoint_every = 2
        altered.train.torch_num_threads = 2
        other = BaselineDQN(altered, env.action_spec, obs)
        other.load_state_dict(saved)  # 明确允许的运行限制变化不伪装为数学参数变化。

    def test_default_100_slot_input_and_no_state_alias(self):
        config = Config()
        env, obs = environment_example(config, 1)
        agent = BaselineDQN(config, env.action_spec, obs)
        self.assertEqual(agent.input_dim, 205)
        self.assertEqual(agent.online.encoder[0].out_features, 128)
        self.assertEqual(agent.online.encoder[2].out_features, 128)
        self.assertEqual(agent.q_values(obs).shape, (9551,))
        state = agent.state_dict()
        with torch.no_grad():
            agent.online.skip_head.bias.add_(10.)
        self.assertFalse(torch.equal(state["online"]["skip_head.bias"], agent.online.skip_head.bias))


if __name__ == "__main__":
    unittest.main()
