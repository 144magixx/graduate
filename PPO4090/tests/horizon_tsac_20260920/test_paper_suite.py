"""Suite依赖门与归档预算模拟；所有数据仅临时诊断，不启动训练子进程。"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.paper_baselines import suite


def make_plan(campaign="suite-fixture", final_test=True):
    config = Config()
    config.train.max_wall_seconds = 60
    config.train.checkpoint_every = 25
    config.telemetry.enabled = True
    return {"campaign_id": campaign, "final_test": final_test, "wait_seconds": 1, "poll_seconds": .01,
            "experiments": [{"name": algorithm, "algorithm": algorithm, "config": config.to_dict(), "settings": {}}
                            for algorithm in suite.LEARNED_IDS]}


def make_run(path, checkpoints=((25, 75, 20), (100, 300, 100)), config=None):
    path.mkdir(parents=True)
    config = config or Config()
    manifest = {"run_id": path.name, "trial_id": path.name, "seed": 42, "status": "completed", "config": config.to_dict(),
                "versions": config.semantic_versions(), "data_manifest_hash": "diagnostic_split_hash",
                "dataset_version": config.data.dataset_version, "algorithm": "diagnostic_fixture"}
    suite._write(path/"运行清单.json", manifest)
    conn = sqlite3.connect(path/suite.DB_NAME)
    with conn:
        conn.execute("CREATE TABLE runs(status TEXT)")
        conn.execute("INSERT INTO runs VALUES('completed')")
    conn.close()
    for episodes, steps, updates in checkpoints:
        identity = f"diagnostic-{episodes}-{steps}-{updates}"
        folder = path/"检查点存档"/identity
        folder.mkdir(parents=True)
        content = ("NOT_A_MODEL_UNIT_TEST_ONLY:"+identity).encode()
        checkpoint = folder/"检查点.pt"
        checkpoint.write_bytes(content)
        suite._write(folder/"检查点清单.json", {"checkpoint_id": identity, "file": checkpoint.name,
                     "sha256": hashlib.sha256(content).hexdigest(),
                     "counters": {"episodes": episodes, "env_step": steps, "update_step": updates}})
    return path


def parent_row(path, eta, value):
    counters = {"episodes": 100, "env_step": 300, "update_step": 100}
    report = {"split": "validation", "results": {"policy": [
        {"scenario_id": f"parent-fixture-{i}", "mean_satisfaction": value} for i in range(15)]}}
    return {"name": "eta"+str(eta), "target_entropy_ratio": eta, "run_dir": str(path), "durable_copy": str(path),
            "exit_code": 0, "status": "trained", "final_evaluation": {"status": "completed", "checkpoint_counters": counters, "report": report}}


class PaperSuiteTests(unittest.TestCase):
    def test_import_before_gate_does_not_load_torch_or_numpy(self):
        command = "import sys; import implementations.horizon_tsac_20260920.paper_baselines.suite; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parent_gate_waits_for_all_flags_and_has_finite_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"父进度.json"
            suite._write(path, {"finished": False})
            clock = [0.]
            def sleep(value):
                clock[0] += value
                if clock[0] >= .5:
                    suite._write(path, {"finished": True, "all_training_successful": True,
                                      "all_final_evaluations_available": True, "all_durable_copies_available": True})
            value = suite.wait_for_parent(path, 1., .5, sleeper=sleep, clock=lambda: clock[0])
            self.assertTrue(value["finished"])
            self.assertEqual(clock[0], .5)
            suite._write(path, {"finished": True, "all_training_successful": False})
            with self.assertRaises(suite.SuiteBlocked):
                suite.wait_for_parent(path, 1., .5, sleeper=sleep, clock=lambda: clock[0])
            suite._write(path, {"finished": False})
            clock[0] = 0.
            with self.assertRaisesRegex(suite.SuiteBlocked, "超时"):
                suite.wait_for_parent(path, 1., .5, sleeper=lambda value: clock.__setitem__(0, clock[0]+value), clock=lambda: clock[0])

    def test_failed_parent_never_calls_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"父进度.json"
            suite._write(path, {"finished": True, "all_training_successful": False,
                              "all_final_evaluations_available": True, "all_durable_copies_available": True})
            with patch.object(suite, "_supervise") as train, patch.object(suite, "_evaluate") as evaluate:
                folder, result = suite.run_suite(make_plan(), root/"durable", path)
                self.assertEqual(result["status"], "blocked")
                train.assert_not_called()
                evaluate.assert_not_called()
                self.assertEqual(suite._read(folder/"队列状态.json")["status"], "blocked")

    def test_failed_parent_matched_evaluation_blocks_even_when_basic_flags_are_true(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"父进度.json"
            for status in ("failed", "unavailable", "scheduled"):
                suite._write(path, {"finished": True, "all_training_successful": True,
                                   "all_final_evaluations_available": True, "all_durable_copies_available": True,
                                   "matched_comparison": {"status": status}})
                with self.subTest(status=status), self.assertRaisesRegex(suite.SuiteBlocked, "补评"):
                    suite.wait_for_parent(path, 1., .1)

    def test_plan_freezes_five_algorithms_100_episodes_and_distinct_adapted_hashes(self):
        plan = suite.normalize_suite_plan(make_plan(), Path.cwd())
        self.assertFalse(plan["quick_evaluation"])
        self.assertFalse(plan["matched_budget_fallback"])
        self.assertTrue(plan["gpu_preflight"])
        self.assertEqual(len({row["config_hash"] for row in plan["experiments"]}), 5)
        for row in plan["experiments"]:
            self.assertEqual(row["config"]["train"]["episodes"], 100)
            self.assertEqual(row["final_policies"], ["policy"])
            self.assertIn(row["algorithm"], row["config"]["model"]["model_version"])
        bad = make_plan()
        bad["experiments"].pop()
        with self.assertRaises(ValueError):
            suite.normalize_suite_plan(bad, Path.cwd())

    def test_budget_matches_environment_not_ppo_optimizer_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {"a": make_run(root/"a", ((25, 75, 20), (100, 300, 100))),
                     "ppo": make_run(root/"ppo", ((25, 75, 300), (100, 300, 2000))),
                     "reference": make_run(root/"ref", ((25, 75, 20), (100, 300, 100)))}
            result = suite.select_common_environment_budget(paths)
            self.assertEqual((result["episodes"], result["env_step"]), (100, 300))
            self.assertFalse(result["optimizer_counts_required_equal"])
            self.assertEqual(result["checkpoints"]["ppo"]["counters"]["update_step"], 2000)
            wrong = make_run(root/"wrong", ((100, 301, 100),))
            self.assertEqual(suite.select_common_environment_budget({**paths, "wrong": wrong})["status"], "missing")

    def test_parent_reference_selected_only_from_full_validation(self):
        rows = [parent_row(Path("a"), .2, .4), parent_row(Path("b"), .5, .5), parent_row(Path("c"), .8, .7)]
        rows[0]["test_evaluation"] = {"mean_U": 1.}
        choice = suite.select_parent_references({"experiments": rows})
        self.assertEqual(choice["primary"]["target_entropy_ratio"], .5)
        self.assertEqual(choice["supplemental"]["target_entropy_ratio"], .8)
        rows[2]["final_evaluation"]["report"]["results"]["policy"].pop()
        with self.assertRaises(suite.SuiteBlocked):
            suite.select_parent_references({"experiments": rows})

    def test_parent_unequal_final_budgets_use_matched_validation_to_select_eta(self):
        rows = [parent_row(Path("a"), .2, .4), parent_row(Path("b"), .5, .5), parent_row(Path("c"), .8, .9)]
        rows[-1]["final_evaluation"]["checkpoint_counters"] = {"episodes": 90, "env_step": 270, "update_step": 90}
        for row, score in zip(rows, (.8, .7, .6)):
            row["matched_evaluation"] = copy.deepcopy(row["final_evaluation"])
            row["matched_evaluation"]["checkpoint_counters"] = {"episodes": 25, "env_step": 75, "update_step": 20}
            for value in row["matched_evaluation"]["report"]["results"]["policy"]:
                value["mean_satisfaction"] = score
        parent = {"experiments": rows, "matched_comparison": {"status": "completed"}}
        choice = suite.select_parent_references(parent)
        self.assertEqual(choice["comparison_basis"], "matched_evaluation")
        self.assertEqual(choice["supplemental"]["target_entropy_ratio"], .2)
        parent["matched_comparison"]["status"] = "failed"
        with self.assertRaises(suite.SuiteBlocked):
            suite.select_parent_references(parent)

    def test_missing_parent_local_directory_is_copied_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            durable = root/"durable"
            source = make_run(durable/"source")
            row = {"run_dir": str(root/"expired"), "durable_copy": str(source)}
            with patch.object(suite, "_node_output_root", return_value=root/"node"):
                target = suite.ensure_node_local(row, durable, "fixture", "parent")
            self.assertTrue(target.is_dir())
            self.assertNotEqual(target, source)
            self.assertNotIn(durable, target.parents)
            self.assertTrue((source/"检查点存档").is_dir())

    def test_copy_rejects_active_nested_evaluation_database(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_run(root/"run")
            nested = make_run(source/"独立评估"/"still-active")
            conn = sqlite3.connect(nested/suite.DB_NAME)
            with conn:
                conn.execute("UPDATE runs SET status='active'")
            conn.close()
            with self.assertRaisesRegex(suite.SuiteBlocked, "active"):
                suite._copy_closed(source, root/"durable")
            self.assertFalse((root/"durable").exists())

    def _full_case(self, root, final_test=True, failed=False, supplemental_matched=True, gpu_preflight=False, preflight_fail=False):
        durable, node = root/"durable", root/"node"
        parents = [parent_row(make_run(node/("eta"+str(eta)), ((100, 300, 100),) if eta != .8 or supplemental_matched else ((90, 270, 90),)), eta, value)
                   for eta, value in ((.2, .4), (.5, .5), (.8, .7))]
        parent = {"finished": True, "all_training_successful": True, "all_final_evaluations_available": True,
                  "all_durable_copies_available": True, "experiments": parents}
        parent_path = root/"父进度.json"
        suite._write(parent_path, parent)
        children = []
        for index, algorithm in enumerate(suite.LEARNED_IDS):
            path = make_run(node/algorithm, ((100, 300, 100+index*30),))
            children.append({"name": algorithm, "run_dir": str(path), "durable_copy": str(path),
                             "status": "failed" if failed and index == 0 else "trained",
                             "exit_code": 1 if failed and index == 0 else 0,
                             "final_evaluation": {"status": "completed"}})
        calls = []
        plan = make_plan(final_test=final_test)
        plan["gpu_preflight"] = gpu_preflight
        order = []
        original_wait = suite.wait_for_parent
        def tracked_wait(*args, **kwargs):
            value = original_wait(*args, **kwargs)
            order.append("parent_gate")
            return value
        def fake_preflight(normalized, folder):
            self.assertEqual(order, ["parent_gate"])
            order.append("preflight")
            return {"status": "failed" if preflight_fail else "completed", "experiments": [
                {"algorithm": key, "status": "failed" if preflight_fail and i == 0 else "completed"}
                for i, key in enumerate(suite.LEARNED_IDS)]}
        def fake_supervise(normalized, destination):
            self.assertTrue(suite._read(parent_path)["finished"])
            order.append("supervise")
            return durable/"训练模拟", {"finished": True, "experiments": children}
        def fake_evaluate(request, folder, timeout):
            frozen = suite._read(Path(folder)/"测试前选择清单.json")
            self.assertFalse(frozen["test_set_consulted"])
            self.assertEqual(frozen["supplemental_eta"], .8)
            self.assertNotIn(durable, Path(request["run_dir"]).parents)
            calls.append((request["split"], request["method_id"], request["policies"]))
            output = Path(request["run_dir"])/"独立评估"/("unit-"+request["split"]+"-"+request["method_id"])
            output.mkdir(parents=True, exist_ok=True)
            suite._write(output/"诊断说明.json", {"purpose": "mock_evaluation_no_training"})
            for policy in request["policies"]:
                make_run(output/policy, checkpoints=())
            rows = {policy: [{"scenario_id": request["split"]+f"-fixture-{i:02d}", "phase": request["split"], "n_demand": 3,
                              "total_demand_bps": 3e9, "power_budget_w": 6000., "mean_satisfaction": .5,
                              "constraint_violation_count": 0, "constraint_violations": []} for i in range(15)]
                    for policy in request["policies"]}
            return {"status": "completed", "report": {"split": request["split"], "results": rows, "output_dir": str(output),
                    "checkpoint_id": request["checkpoint"]["checkpoint_id"], "evaluated_training_counters": request["checkpoint"]["counters"]}}
        with patch.object(suite, "wait_for_parent", side_effect=tracked_wait), patch.object(suite, "_preflight", side_effect=fake_preflight), \
             patch.object(suite, "_supervise", side_effect=fake_supervise), patch.object(suite, "_evaluate", side_effect=fake_evaluate), \
             patch.object(suite, "_node_output_root", return_value=node), patch("subprocess.Popen", side_effect=AssertionError("不能启动真实长训")):
            directory, result = suite.run_suite(plan, durable, parent_path)
        self.last_execution_order = order
        return directory, result, calls

    def test_gate_then_preflight_then_training_order(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result, _ = self._full_case(Path(temp), gpu_preflight=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(self.last_execution_order, ["parent_gate", "preflight", "supervise"])

    def test_preflight_failure_blocks_formal_training(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result, calls = self._full_case(Path(temp), gpu_preflight=True, preflight_fail=True)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(self.last_execution_order, ["parent_gate", "preflight"])
            self.assertEqual(calls, [])

    def test_preflight_launches_five_diagnostic_one_episode_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            node = root/"node"
            result_folder = root/"results"
            result_folder.mkdir()
            plan = suite.normalize_suite_plan(make_plan(), root)
            seen = []
            class Child:
                returncode = 0
                def wait(self, timeout):
                    return 0
            def launch(command, **kwargs):
                self.assertIn("--preflight", command)
                algorithm = command[command.index("--algorithm")+1]
                config = suite._read(command[command.index("--config")+1])
                settings = suite._read(command[command.index("--settings")+1])
                self.assertEqual(config["train"]["episodes"], 1)
                self.assertEqual(config["train"]["warmup_steps"], 0)
                self.assertEqual(config["train"]["update_schedule"], "episode")
                self.assertEqual(config["train"]["updates_per_episode"], 1)
                self.assertEqual(config["train"]["batch_size"], 2)
                self.assertEqual(config["train"]["microbatch_size"], 2)
                self.assertEqual(config["train"]["max_wall_seconds"], 300.)
                self.assertTrue(config["train"]["device"].startswith("cuda"))
                if algorithm == "mlp_ppo":
                    self.assertEqual((settings["epochs"], settings["minibatch_size"]), (1, 64))
                path = make_run(node/algorithm, checkpoints=(), config=Config.from_dict(config))
                manifest = suite._read(path/"运行清单.json")
                manifest.update(mode="paper_baseline_preflight", paper_baseline={"algorithm_id": algorithm})
                suite._write(path/"运行清单.json", manifest)
                suite._write(path/"基线训练结果.json", {"counters": {"episodes": 1, "env_step": 3, "update_step": 1},
                            "episodes": [{"constraint_violation_count": 0}]})
                kwargs["stdout"].write(json.dumps({"run_dir": str(path)})+"\n")
                kwargs["stdout"].flush()
                seen.append(algorithm)
                return Child()
            with patch.object(suite, "_node_output_root", return_value=node), patch("subprocess.Popen", side_effect=launch):
                result = suite._preflight(plan, result_folder)
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(seen, list(suite.LEARNED_IDS))
            self.assertTrue(all(row["research_data"] is False for row in result["experiments"]))

    def test_summary_aggregates_real_metrics_and_keeps_formal_ranking_disabled(self):
        rows = [{"scenario_id": str(i), "n_demand": 3, "total_demand_bps": 3e9, "power_budget_w": 6000.,
                 "mean_satisfaction": .5, "constraint_violation_count": 0, "sgm_mean_positive_demand": value,
                 "mean_slot_sinr_db": 2.+4*i, "p05_slot_sinr_db": -1.+2*i, "sinr_sample_count": 3+4*i,
                 "power_used_w": 5.+10*i, "skip_fraction": .5*i, "delivered_bps": 100. if i == 0 else None}
                for i, value in enumerate((.5, 1.))]
        record = {"rows": rows, "training_required": True, "checkpoint": {"counters": {"episodes": 100, "env_step": 300, "update_step": 100}}}
        result = suite.summarize_split({"records": {"horizon_primary": record}, "missing": []})
        summary = result["methods"]["horizon_primary"]
        self.assertEqual(summary["scene_equal_weight_means"]["sgm_mean_positive_demand"], .75)
        self.assertEqual(summary["scene_equal_weight_means"]["mean_slot_sinr_db"], 4.)
        self.assertEqual(summary["scene_equal_weight_means"]["p05_slot_sinr_db"], 0.)
        self.assertEqual(summary["sinr_sample_count_total"], 10)
        self.assertIsNone(summary["scene_equal_weight_means"]["delivered_bps"])
        self.assertFalse(result["ranking_available"])

    def test_mock_complete_suite_has_seven_baselines_control_and_preselected_references(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, result, calls = self._full_case(Path(temp))
            self.assertEqual(result["status"], "completed", result)
            self.assertTrue(result["test_set_used"])
            self.assertEqual(set(result["validation"]["methods"]), set(suite.PAPER_IDS)|{"tsac_205", "horizon_primary", "horizon_supplemental"})
            self.assertEqual(sum(split == "validation" and method == "heuristics" for split, method, _ in calls), 1)
            self.assertEqual(sum(split == "test" and method == "heuristics" for split, method, _ in calls), 1)
            self.assertFalse(result["formal_multiseed_complete"])
            self.assertFalse(result["validation"]["ranking_available"])
            self.assertTrue(result["validation"]["descriptive_order_available"])
            self.assertTrue((directory/"验证集逐场景结果.json").is_file())
            self.assertTrue((directory/"测试集逐场景结果.json").is_file())

    def test_failed_method_reports_missing_and_never_opens_test(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, result, calls = self._full_case(Path(temp), failed=True)
            self.assertEqual(result["status"], "missing")
            self.assertEqual(result["missing_methods"], ["mlp_sac"])
            self.assertFalse(result["test_set_used"])
            self.assertEqual(calls, [])
            self.assertEqual(suite._read(directory/"队列状态.json")["status"], "missing")

    def test_unmatched_supplemental_remains_validation_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result, calls = self._full_case(Path(temp), supplemental_matched=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["supplemental"]["status"], "validation_diagnostic_only")
            self.assertNotIn("horizon_supplemental", result["test"]["methods"])
            self.assertFalse(any(method == "horizon_supplemental" for _, method, _ in calls))


if __name__ == "__main__":
    unittest.main()
