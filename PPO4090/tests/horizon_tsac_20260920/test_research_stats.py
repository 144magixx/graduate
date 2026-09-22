"""统计接口fixture；以下数值均为合成数学测试，不能作为研究实验结果。"""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.research_stats import summarize_runs, compare_runs


def run_fixture(seed=1, values=(.2, .8), algorithm="left", trial=None, run_id=None, created="2026-09-20T10:00:00Z"):
    config = Config()
    config.train.seed = seed
    manifest = dict(run_id=run_id or f"{algorithm}-{seed}", trial_id=trial or f"trial-{algorithm}-{seed}",
        parent_run_id=None, seed=seed, status="completed", algorithm=algorithm, phase="test", created_at_utc=created,
        config=config.to_dict(), versions=config.semantic_versions(), data_manifest_hash="frozen-fixture-manifest",
        dataset_version=config.data.dataset_version,
        data_manifest=dict(strict_root_independence=True, records=[dict(scenario_id=f"scene-{i}", split="test",
            root_scene_id=f"root-{i}", split_group_id=f"lineage-{i}") for i in range(len(values))]))
    rows = [dict(scenario_id=f"scene-{i}", mean_satisfaction=value, episode_return=None if value is None else value*100,
                 phase="test", end_to_end_metric_available=False) for i, value in enumerate(values)]
    return dict(manifest=manifest, episodes=rows)


class ResearchStatsTests(unittest.TestCase):
    def test_single_seed_reports_scene_diagnostic_only(self):
        result = summarize_runs([run_fixture()], resamples=500)
        self.assertAlmostEqual(result["mean"], .5)
        self.assertEqual(result["n_seeds"], 1)
        self.assertEqual(result["n_scenarios"], 2)
        self.assertEqual(result["scenario_ci95"], [.2, .8])
        self.assertIsNone(result["seed_ci95"])
        self.assertFalse(result["ranking_available"])

    def test_paired_scene_delta_direction_and_values(self):
        result = compare_runs([run_fixture(values=(.2, .4))], [run_fixture(values=(.4, .8), algorithm="right")], resamples=500)
        self.assertTrue(result["comparison_available"])
        self.assertEqual(result["delta_direction"], "right_minus_left")
        self.assertAlmostEqual(result["mean_difference"], .3)
        np.testing.assert_allclose([r["difference"] for r in result["paired_differences"]], [.2, .4])
        self.assertIsNone(result["seed_ci95"])

    def test_multiseed_hierarchical_ci_retains_shared_scene_variation(self):
        first = [run_fixture(seed, (.1, .1)) for seed in range(5)]
        second = [run_fixture(seed, (.2, .4), "right") for seed in range(5)]
        result = compare_runs(first, second, expected_seeds=list(range(5)), resamples=1000)
        self.assertEqual(result["n_seed_pairs"], 5)
        self.assertAlmostEqual(result["mean_difference"], .2)
        # 五个seed有完全相同的场景效应，不能把共享场景误当10个独立样本而缩窄区间。
        np.testing.assert_allclose(result["seed_ci95"], [.1, .3])
        self.assertTrue(result["seed_interval_available"])
        self.assertTrue(result["ranking_available"])
        repeat = compare_runs(first, second, expected_seeds=list(range(5)), resamples=1000)
        self.assertEqual(result["seed_ci95"], repeat["seed_ci95"])

    def test_constant_paired_effect_has_exact_degenerate_interval(self):
        first = [run_fixture(seed, (.1*seed, .1*seed+.1)) for seed in range(3)]
        second = [run_fixture(seed, (.1*seed+.25, .1*seed+.35), "right") for seed in range(3)]
        result = compare_runs(first, second, resamples=500)
        np.testing.assert_allclose(result["seed_ci95"], [.25, .25], atol=1e-15)

    def test_parent_restore_selects_latest_completed_once(self):
        parent = run_fixture(values=(.1, .2), trial="shared", run_id="parent")
        child = run_fixture(values=(.8, .9), trial="shared", run_id="child", created="2026-09-20T11:00:00Z")
        child["manifest"]["parent_run_id"] = "parent"
        result = summarize_runs([parent, child], resamples=500)
        self.assertEqual(result["n_seeds"], 1)
        self.assertEqual(result["selected_run_ids"], ["child"])
        self.assertAlmostEqual(result["mean"], .85)
        self.assertIsNone(result["seed_ci95"])

    def test_failed_child_is_reported_and_does_not_add_seed(self):
        parent = run_fixture(run_id="parent", trial="same")
        failed = run_fixture(run_id="failed", trial="same", created="2026-09-21")
        failed["manifest"].update(parent_run_id="parent", status="failed")
        result = summarize_runs([parent, failed], expected_seeds=[1, 2], resamples=500)
        self.assertEqual(result["selected_run_ids"], ["parent"])
        self.assertEqual(result["failed_runs"], ["failed"])
        self.assertEqual(result["missing_seeds"], [2])
        self.assertFalse(result["ranking_available"])

    def test_same_seed_new_trial_is_not_independent_seed(self):
        first = run_fixture(run_id="one", trial="first")
        repeat = run_fixture(values=(.3, .7), run_id="two", trial="second", created="2026-09-21")
        result = summarize_runs([first, repeat], resamples=500)
        self.assertEqual(result["n_seeds"], 1)
        self.assertEqual(result["selected_run_ids"], ["two"])
        self.assertEqual(len(result["duplicate_seed_trials"]), 1)

    def test_ambiguous_same_lineage_evaluation_order_is_not_score_selected(self):
        first = run_fixture(run_id="one", trial="same", values=(.1, .2))
        second = run_fixture(run_id="two", trial="same", values=(.8, .9))
        result = summarize_runs([first, second], resamples=100)
        self.assertEqual(result["n_seeds"], 0)
        self.assertIsNone(result["mean"])
        self.assertTrue(all(row["reason"] == "ambiguous_evaluation_order_same_lineage" for row in result["excluded_runs"]))

    def test_two_by_two_configs_only_change_actor_and_critic(self):
        folder = Path(__file__).resolve().parents[2]/"implementations"/"horizon_tsac_20260920"/"configs"
        files = sorted(folder.glob("完整观察-*配置.json"))
        self.assertEqual(len(files), 4)
        normalized, factors = [], set()
        for path in files:
            config = json.loads(path.read_text(encoding="utf-8"))
            Config.from_dict(config)
            self.assertEqual(config["model"]["encoder"], "full_attention")
            self.assertEqual(config["model"]["spectrum_tokens"], "slots")
            factors.add((config["model"].pop("actor"), config["model"].pop("critic")))
            normalized.append(config)
        self.assertEqual(factors, {("independent", "additive"), ("independent", "joint"), ("conditional", "additive"), ("conditional", "joint")})
        self.assertTrue(all(config == normalized[0] for config in normalized))

    def test_incompatible_physics_budget_and_versions_block_comparison(self):
        first = run_fixture()
        mutations = [lambda m: m["config"]["physics"].update(sinr_margin_db=6.),
                     lambda m: m["config"]["env"].update(power_budget_w=3000.),
                     lambda m: m["versions"].update(metric_version="other.v1"),
                     lambda m: m["config"]["env"].update(power_levels_w=[6., 12.]),
                     lambda m: m["config"]["data"].update(traffic_scale=.5)]
        for mutation in mutations:
            second = run_fixture(algorithm="right")
            mutation(second["manifest"])
            result = compare_runs([first], [second], resamples=100)
            self.assertFalse(result["compatible"])
            self.assertFalse(result["ranking_available"])
            self.assertIsNone(result["mean_difference"])

    def test_reward_difference_blocks_return_but_allows_business_metric(self):
        left, right = run_fixture(), run_fixture(algorithm="right")
        right["manifest"]["versions"]["reward_version"] = "different_reward.v1"
        right["manifest"]["config"]["env"]["reward_version"] = "different_reward.v1"
        utility = compare_runs([left], [right], resamples=100)
        self.assertTrue(utility["compatible"])
        self.assertTrue(utility["comparison_available"])
        self.assertFalse(utility["return_comparable"])
        returns = compare_runs([left], [right], metric="episode_return", resamples=100)
        self.assertFalse(returns["compatible"])
        self.assertIsNone(returns["mean_difference"])

    def test_unknown_root_and_assignment_are_not_fabricated(self):
        runs = [run_fixture(seed) for seed in range(5)]
        for run in runs:
            run["manifest"]["data_manifest"]["strict_root_independence"] = False
            for row in run["episodes"]:
                row["J_root"] = .9
        result = summarize_runs(runs, resamples=100)
        self.assertTrue(result["seed_interval_available"])
        self.assertFalse(result["ranking_available"])
        self.assertTrue(any("root" in text for text in result["support_limitations"]))
        root = summarize_runs(runs, metric="J_root", resamples=100)
        self.assertFalse(root["compatible"])
        self.assertIsNone(root["mean"])

    def test_declared_independence_cannot_hide_missing_or_shared_root_ids(self):
        runs = [run_fixture(seed) for seed in range(5)]
        for run in runs:
            for record in run["manifest"]["data_manifest"]["records"]:
                record["root_scene_id"] = "same-root"
        result = summarize_runs(runs, expected_seeds=list(range(5)), resamples=100)
        self.assertFalse(result["ranking_available"])
        self.assertTrue(any("共享root" in text for text in result["support_limitations"]))

    def test_nonfinite_missing_values_not_replaced_with_zero(self):
        run = run_fixture(values=(.2, None))
        run["episodes"].append(dict(scenario_id="nan", mean_satisfaction=float("nan"), phase="test"))
        result = summarize_runs([run], resamples=100)
        self.assertEqual(result["mean"], .2)
        self.assertEqual(result["n_scenarios"], 1)
        self.assertIn("scene-1", result["missing_scenarios"])
        self.assertIsNone(result["scenario_ci95"])

    def test_mixed_policy_cohort_is_rejected_without_cross_strategy_dedup(self):
        left, right = run_fixture(trial="same"), run_fixture(algorithm="right", trial="same")
        mixed = summarize_runs([left, right], resamples=100)
        self.assertFalse(mixed["compatible"])
        paired = compare_runs([left], [right], resamples=100)
        self.assertEqual(paired["n_seed_pairs"], 1)
        self.assertTrue(paired["compatible"])

    def test_missing_seeds_and_incomplete_common_scenes_are_explicit(self):
        left = [run_fixture(1), run_fixture(2)]
        right = [run_fixture(2, algorithm="right"), run_fixture(3, algorithm="right")]
        right[0]["episodes"] = right[0]["episodes"][:1]
        result = compare_runs(left, right, expected_seeds=[1, 2, 3], resamples=100)
        self.assertEqual(result["unmatched_seeds"], {"left": [1], "right": [3]})
        self.assertEqual(result["n_scenario_pairs"], 1)
        self.assertEqual(result["missing_seeds"], {"left": [3], "right": [1]})
        self.assertFalse(result["ranking_available"])

    def test_repeated_scene_rows_are_not_independent_scenes(self):
        run = run_fixture(values=(.2, .8))
        run["episodes"].append(dict(scenario_id="scene-0", mean_satisfaction=.6, phase="test"))
        result = summarize_runs([run], resamples=100)
        self.assertEqual(result["n_scenarios"], 2)
        self.assertAlmostEqual(result["mean"], .6)

    def test_train_phase_rejected_and_rng_is_local(self):
        with self.assertRaises(ValueError):
            summarize_runs([run_fixture()], phase="train")
        unlabelled = run_fixture()
        unlabelled["manifest"].pop("phase")
        for row in unlabelled["episodes"]:
            row.pop("phase")
        self.assertIsNone(summarize_runs([unlabelled], resamples=100)["mean"])
        np.random.seed(991)
        expected = np.random.random(5)
        np.random.seed(991)
        summarize_runs([run_fixture()], resamples=100)
        np.testing.assert_array_equal(np.random.random(5), expected)


if __name__ == "__main__":
    unittest.main()
