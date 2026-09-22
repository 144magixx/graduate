"""独立更新调度审查：真实SAC恢复和截断边界，不修改控制器。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from implementations.horizon_tsac_20260920 import artifacts, train
from implementations.horizon_tsac_20260920.audit import sha256
from implementations.horizon_tsac_20260920.data.split import manifest_hash
from implementations.horizon_tsac_20260920.rl.replay import ReplayBuffer
from tests.horizon_tsac_20260920.test_data import fixture
from tests.horizon_tsac_20260920.test_schedule import tiny_config


class ScheduleIndependentReview(unittest.TestCase):
    def prepare(self, root, config):
        source = root / "调度审查场景.json"
        source.write_text(json.dumps(fixture(3, config).to_dict()), encoding="utf-8")
        manifest = {"dataset_version": config.data.dataset_version, "source": "diagnostic_fixture_only",
            "records": [{"scenario_id": "diagnostic_fixture", "scenario_path": str(source),
                         "source_hash": sha256(source), "split": "train", "domain_cells": ["diagnostic"]}]}
        manifest["manifest_hash"] = manifest_hash(manifest)
        path = root / "调度审查清单.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def run_captured(self, root, config, manifest_path, resume=None):
        ticks, episodes = [], []
        original_update, original_add = train.perform_updates, ReplayBuffer.add_episode
        def track_update(cfg, agent, replay, counters, recorder=None, *, trigger):
            changed = original_update(cfg, agent, replay, counters, recorder, trigger=trigger)
            if changed:
                ticks.append((counters["env_step"], changed))
                self.assertEqual(agent.update_step, counters["update_step"])
            return changed
        def track_add(replay, transitions, **kwargs):
            values = tuple(transitions)
            episodes.append((values[-1].env_step, values[-1].terminated, values[-1].truncated))
            return original_add(replay, values, **kwargs)
        with patch.object(artifacts, "HORIZON_OUTPUT_DIR", root/"运行"), \
             patch.object(train, "perform_updates", side_effect=track_update), \
             patch.object(ReplayBuffer, "add_episode", new=track_add), contextlib.redirect_stdout(io.StringIO()):
            directory, report = train.run_training(config, "independent_schedule_review", manifest_path, resume)
        return directory, report, ticks, episodes

    def assert_checkpoint_numerics_equal(self, first, second):
        a, b = artifacts.load_checkpoint(first), artifacts.load_checkpoint(second)
        self.assertEqual(a["counters"], b["counters"])
        for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            for key, value in a["agent"][name].items():
                torch.testing.assert_close(value, b["agent"][name][key], rtol=0, atol=0)
        torch.testing.assert_close(a["agent"]["log_alpha"], b["agent"]["log_alpha"], rtol=0, atol=0)
        self.assertEqual(a["sampler"], b["sampler"])

    def test_env_step_resume_reproduces_next_three_updates_exactly(self):
        config = tiny_config(warmup=0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.prepare(root, config)
            whole, report, ticks, _ = self.run_captured(root, config, manifest)
            checkpoints = sorted((whole/"检查点存档").glob("*/检查点.pt"), key=lambda p: artifacts.load_checkpoint(p)["counters"]["episodes"])
            resumed = copy.deepcopy(config)
            resumed.train.episodes = 1
            child, child_report, child_ticks, added = self.run_captured(root, resumed, manifest, checkpoints[0])
            self.assertEqual(ticks, [(3, 1), (4, 1), (5, 1), (6, 1)])
            self.assertEqual(child_ticks, [(4, 1), (5, 1), (6, 1)])
            self.assertEqual(added, [(6, True, False)])
            self.assertEqual(report["counters"], child_report["counters"])
            self.assert_checkpoint_numerics_equal(whole/"检查点.pt", child/"检查点.pt")

    def test_resume_before_warmup_does_not_reset_gate_or_accumulate_debt(self):
        config = tiny_config(warmup=5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.prepare(root, config)
            whole, _, ticks, _ = self.run_captured(root, config, manifest)
            checkpoints = sorted((whole/"检查点存档").glob("*/检查点.pt"), key=lambda p: artifacts.load_checkpoint(p)["counters"]["episodes"])
            self.assertEqual(artifacts.load_checkpoint(checkpoints[0])["counters"]["update_step"], 0)
            resumed = copy.deepcopy(config)
            resumed.train.episodes = 1
            child, _, child_ticks, _ = self.run_captured(root, resumed, manifest, checkpoints[0])
            self.assertEqual(ticks, [(5, 1), (6, 1)])
            self.assertEqual(child_ticks, ticks)
            self.assert_checkpoint_numerics_equal(whole/"检查点.pt", child/"检查点.pt")

    def test_truncated_second_episode_uses_only_old_completed_replay(self):
        config = tiny_config()
        config.train.max_env_steps = 4
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory, report, ticks, added = self.run_captured(root, config, self.prepare(root, config))
            self.assertEqual(ticks, [(3, 1), (4, 1)])
            self.assertEqual(added, [(3, True, False)])
            self.assertEqual(report["counters"], {"env_step": 4, "update_step": 2, "episodes": 2})
            # 第4步已真实更新但不构成完整回合边界；最新可恢复checkpoint仍是第3步。
            saved = artifacts.load_checkpoint(directory/"检查点.pt")
            self.assertEqual(saved["counters"], {"env_step": 3, "update_step": 1, "episodes": 1})
            self.assertEqual(saved["replay"]["transition_count"], 3)
            self.assertTrue(report["episodes"][-1]["truncated"])

    def test_resume_rejects_schedule_or_update_ratio_changes(self):
        config = tiny_config()
        config.train.episodes = 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory, _, _, _ = self.run_captured(root, config, self.prepare(root, config))
            for field, value in (("update_schedule", "episode"), ("updates_per_step", 2), ("warmup_steps", 5)):
                changed = copy.deepcopy(config)
                setattr(changed.train, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    artifacts.load_checkpoint(directory/"检查点.pt", changed)


if __name__ == "__main__":
    unittest.main()
