"""论文启发式：规则正确性、完整合法候选和同环境硬约束回归。"""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.env.action import Action, ActionSpec
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.paper_baselines.heuristics import HeuristicPolicy


def scenario(config, demands, groups=None):
    groups = list(range(len(demands))) if groups is None else groups
    rows = [dict(beam_id=10 + i * 3, latitude_deg=30 + i * .001, longitude_deg=110 + i * .001,
                 demand_bps=d, ground_diameter_deg=2., group_id=groups[i]) for i, d in enumerate(demands)]
    return scenario_from_rows(rows, config, "coverage.v2", "paper_heuristic_fixture",
                              metadata={"purpose": "interface_test_not_training_data"})


def small_config(slots=6, length=3, budget=6000., powers=(20., 25., 30.)):
    config = Config()
    config.data.service_order = "raw"
    config.physics.num_slots = slots
    config.env.max_block_length = length
    config.env.power_budget_w = budget
    config.env.power_levels_w = powers
    return config


class PaperHeuristicTests(unittest.TestCase):
    def test_fixed_keeps_group_service_order_and_sequential_starts(self):
        config = small_config()
        source = scenario(config, [1e9] * 10, [i % 8 for i in range(10)])
        before = source.to_dict()
        env = Environment(config)
        obs, _ = env.reset(source)
        policy = HeuristicPolicy("Fixed", env.action_spec, source, config, fixed_length=2)
        starts = []
        while not obs["terminal"]:
            action = policy.act(obs)
            starts.append(action.start)
            self.assertEqual(action.length, 2)
            self.assertEqual(env.action_spec.power_levels_w[action.power_index], 30.)
            obs, _, _, _, info = env.step(action)
            self.assertEqual(info["constraint_violations"], [])
        self.assertEqual(starts, [0] * 8 + [2, 2])
        self.assertEqual(source.to_dict(), before)
        self.assertEqual(env.scenario.to_dict(), before)

    def test_fixed_scans_holes_and_never_crosses_band_boundary(self):
        config = small_config(slots=6, length=3)
        source = scenario(config, [1e9] * 2, [0, 0])
        env = Environment(config)
        env.reset(source)
        obs, *_ = env.step(Action("ALLOC", 1, 2, 0))
        policy = HeuristicPolicy("fixed", env.action_spec, source, config, fixed_length=2)
        action = policy.act(obs)
        self.assertEqual(action, Action("ALLOC", 3, 2, 2))
        self.assertLessEqual(action.start + action.length, config.physics.num_slots)
        self.assertTrue(obs["valid_action_mask"][env.action_spec.encode(action)])

    def test_fixed_does_not_shorten_to_fit_fragmented_holes(self):
        config = small_config(slots=3, length=2)
        source = scenario(config, [1e9] * 2, [0, 0])
        env = Environment(config)
        env.reset(source)
        obs, *_ = env.step(Action("ALLOC", 1, 1, 0))
        self.assertGreater(obs["valid_action_mask"].sum(), 1)
        policy = HeuristicPolicy("fixed", env.action_spec, source, config, fixed_length=2)
        self.assertEqual(policy.act(obs), Action())

    def test_fixed_does_not_reduce_power_but_honors_budget_tolerance(self):
        for budget, expected in ((29.999, "SKIP"), (30. - 5e-9, "ALLOC")):
            config = small_config(budget=budget)
            source = scenario(config, [1e9])
            env = Environment(config)
            obs, _ = env.reset(source)
            policy = HeuristicPolicy("fixed", env.action_spec, source, config, fixed_length=2)
            action = policy.act(obs)
            self.assertEqual(action.kind, expected)
            env.step(action)
            self.assertGreaterEqual(env.remaining_power_w, -config.env.budget_tolerance_w)

    def test_greedy_counts_all_groups_and_the_whole_window(self):
        config = small_config(slots=5, length=2)
        source = scenario(config, [1e9] * 3, [1, 2, 0])
        env = Environment(config)
        env.reset(source)
        env.step(Action("ALLOC", 1, 1, 0))
        obs, *_ = env.step(Action("ALLOC", 1, 1, 0))
        policy = HeuristicPolicy("greedy", env.action_spec, source, config, greedy_length=2)
        # slot 0 is empty, but its length-2 window includes two existing beams.
        # The first zero-count complete window is [2,4); ties prefer start 2.
        self.assertEqual(policy.act(obs), Action("ALLOC", 2, 2, 0))
        self.assertEqual(int(obs["occupancy"][:, 1].sum()), 2)

    def test_greedy_uses_stable_candidate_id_and_respects_current_group_holes(self):
        config = small_config(slots=6, length=2)
        source = scenario(config, [1e9] * 2, [0, 0])
        env = Environment(config)
        env.reset(source)
        obs, *_ = env.step(Action("ALLOC", 1, 1, 0))
        policy = HeuristicPolicy("greedy", env.action_spec, source, config, greedy_length=2)
        self.assertEqual(policy.act(obs), Action("ALLOC", 2, 2, 0))
        self.assertEqual(policy.act(obs, deterministic=False), policy.act(obs))

    def test_greedy_positive_demand_tertiles_and_equal_boundary_rule(self):
        config = small_config()
        source = scenario(config, [0., 3., 6., 9., 12.])
        env = Environment(config)
        obs, _ = env.reset(source)
        policy = HeuristicPolicy("greedy", env.action_spec, source, config, greedy_length=2)
        self.assertEqual(policy.greedy_demand_thresholds_bps, (6., 9.))
        selected = []
        while not obs["terminal"]:
            action = policy.act(obs)
            selected.append(env.action_spec.power_levels_w[action.power_index])
            obs, *_ = env.step(action)
        self.assertEqual(selected, [20., 20., 25., 30.])

    def test_greedy_explicit_thresholds_and_no_undeclared_power_fallback(self):
        config = small_config(budget=24.)
        source = scenario(config, [150.])
        env = Environment(config)
        obs, _ = env.reset(source)
        policy = HeuristicPolicy("greedy", env.action_spec, source, config, greedy_length=2,
                                 greedy_demand_thresholds_bps=(100., 200.))
        self.assertGreater(obs["valid_action_mask"].sum(), 1)
        self.assertEqual(policy.act(obs), Action())
        self.assertEqual(policy.metadata()["parameters"]["threshold_source"], "explicit_frozen_thresholds_bps")

    def test_all_policies_skip_only_action_and_reject_terminal(self):
        config = small_config(budget=0.)
        source = scenario(config, [1e9])
        for name in ("fixed", "greedy", "random"):
            env = Environment(config)
            obs, _ = env.reset(source)
            policy = HeuristicPolicy(name, env.action_spec, source, config, fixed_length=2, greedy_length=2)
            self.assertEqual(policy.act(obs), Action())
            obs, *_ = env.step(policy.act(obs))
            with self.assertRaisesRegex(ValueError, "terminal"):
                policy.act(obs)

    def test_random_samples_every_joint_legal_combination_uniformly(self):
        config = small_config(slots=2, length=2, powers=(20., 30.))
        source = scenario(config, [1e9])
        env = Environment(config)
        obs, _ = env.reset(source)
        policy = HeuristicPolicy("random", env.action_spec, source, config, seed=81)
        counts = np.zeros(len(env.action_spec), dtype=int)
        for _ in range(6000):
            counts[env.action_spec.encode(policy.act(obs))] += 1
        legal = np.flatnonzero(obs["valid_action_mask"])[1:]
        self.assertEqual(len(legal), 6)
        self.assertEqual(counts[0], 0)
        self.assertTrue(np.all(np.abs(counts[legal] - 1000) < 120), counts)
        self.assertEqual(int(counts.sum()), 6000)

    def test_random_reproducible_independent_rng_and_masked_budget(self):
        config = small_config(slots=4, length=3, budget=40.)
        source = scenario(config, [1e9] * 2, [0, 0])
        env = Environment(config)
        env.reset(source)
        obs, *_ = env.step(Action("ALLOC", 1, 2, 0))
        a = HeuristicPolicy("random", env.action_spec, source, config, seed=42)
        b = HeuristicPolicy("random", env.action_spec, source, config, seed=42)
        before = copy.deepcopy(np.random.get_state())
        first = [a.act(obs) for _ in range(100)]
        second = [b.act(obs, deterministic=False) for _ in range(100)]
        after = np.random.get_state()
        self.assertEqual(first, second)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(set(first), {Action("ALLOC", 0, 1, 0), Action("ALLOC", 3, 1, 0)})

    def test_random_with_skip_is_explicitly_named_diagnostic(self):
        config = small_config(slots=1, length=1, powers=(20.,))
        source = scenario(config, [1e9])
        env = Environment(config)
        obs, _ = env.reset(source)
        policy = HeuristicPolicy("random", env.action_spec, source, config, random_include_skip=True)
        self.assertEqual(policy.name, "uniform_with_skip")
        self.assertEqual(policy.metadata()["algorithm"], "uniform_with_skip")
        sampled = [policy.act(obs).kind for _ in range(200)]
        self.assertTrue(70 < sampled.count("SKIP") < 130)

    def test_no_physics_rollout_or_future_reward_is_used_by_act(self):
        config = small_config()
        source = scenario(config, [1e9])
        env = Environment(config)
        obs, _ = env.reset(source)
        for name in ("fixed", "greedy", "random"):
            policy = HeuristicPolicy(name, env.action_spec, source, config, fixed_length=2, greedy_length=2)
            with patch.object(Environment, "step", side_effect=AssertionError("future transition")), \
                 patch("implementations.spectrum_tsac_20260921.env.physics.evaluate_allocation", side_effect=AssertionError("future reward")):
                action = policy.act(obs)
            self.assertTrue(obs["valid_action_mask"][env.action_spec.encode(action)])

    def test_invalid_parameters_and_mismatched_source_group_rejected(self):
        config = small_config()
        source = scenario(config, [1e9])
        spec = ActionSpec(config)
        for values in ({"fixed_length": 4}, {"fixed_length": True}, {"fixed_power_w": 29.}, {"fixed_length": 1.5}):
            with self.assertRaises(ValueError):
                HeuristicPolicy("fixed", spec, source, config, **values)
        with self.assertRaises(ValueError):
            HeuristicPolicy("greedy", spec, source, config, greedy_length=2, greedy_demand_thresholds_bps=(20., 10.))
        env = Environment(config)
        obs, _ = env.reset(source)
        obs["current_group"] = 1
        policy = HeuristicPolicy("fixed", spec, source, config, fixed_length=2)
        with self.assertRaisesRegex(ValueError, "group"):
            policy.act(obs)


if __name__ == "__main__":
    unittest.main()
