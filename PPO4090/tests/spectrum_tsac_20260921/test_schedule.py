"""真实三束环境/网络的有界更新调度回归；关闭遥测并使用临时产物。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from implementations.spectrum_tsac_20260921.config import Config, load_config
from implementations.spectrum_tsac_20260921 import artifacts, train
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.data.split import manifest_hash
from implementations.spectrum_tsac_20260921.rl.replay import ReplayBuffer
from tests.spectrum_tsac_20260921.test_data import fixture


def tiny_config(schedule="env_step", warmup=0):
    config = Config()
    config.physics.num_slots = 4
    config.env.max_block_length = 2
    config.env.power_levels_w = (5., 10.)
    config.env.power_budget_w = 20.
    config.model.d_model = 8
    config.model.attention_heads = 2
    config.model.encoder_layers = 1
    config.model.encoder = "full_pool"
    config.model.candidate_chunk_size = 8
    config.train.batch_size = 2
    config.train.microbatch_size = 1
    config.train.replay_capacity = 12
    config.train.episodes = 2
    config.train.warmup_steps = warmup
    config.train.update_schedule = schedule
    config.train.updates_per_episode = 2
    config.train.torch_num_threads = 1
    config.data.augment = False
    config.data.dataset_version = "schedule_diagnostic.v1"
    config.telemetry.enabled = False
    return config


class ScheduleTests(unittest.TestCase):
    def run_case(self, config):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"三束诊断场景.json"
            source.write_text(json.dumps(fixture(3, config).to_dict()), encoding="utf-8")
            manifest = {"dataset_version": config.data.dataset_version, "source": "diagnostic_fixture",
                        "records": [{"scenario_id": "diagnostic_fixture", "scenario_path": str(source), "source_hash": sha256(source),
                                     "split": "train", "domain_cells": ["diagnostic"]}]}
            manifest["manifest_hash"] = manifest_hash(manifest)
            manifest_path = root/"诊断数据清单.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            update_ticks, episodes_added = [], []
            original_updates = train.perform_updates
            original_add = ReplayBuffer.add_episode
            def track_updates(cfg, agent, replay, counters, recorder=None, *, trigger):
                changed = original_updates(cfg, agent, replay, counters, recorder, trigger=trigger)
                if changed:
                    update_ticks.append((counters["env_step"], changed))
                return changed
            def track_episode(replay, transitions, **kwargs):
                transitions = tuple(transitions)
                episodes_added.append((len(transitions), transitions[-1].env_step, transitions[-1].terminated, kwargs.get("fragment", False)))
                return original_add(replay, transitions, **kwargs)
            with patch.object(artifacts, "SPECTRUM_OUTPUT_DIR", root/"运行"), patch.object(train, "perform_updates", side_effect=track_updates), \
                 patch.object(ReplayBuffer, "add_episode", new=track_episode), contextlib.redirect_stdout(io.StringIO()):
                run_dir, report = train.run_training(config, "schedule_diagnostic", manifest_path)
            if report["checkpoint_available"]:
                saved = artifacts.load_checkpoint(run_dir/"检查点.pt", config, manifest["manifest_hash"])
                self.assertEqual(saved["counters"], report["counters"])
                self.assertEqual(saved["agent"]["update_step"], report["counters"]["update_step"])
            return report, update_ticks, episodes_added

    def test_env_step_waits_for_complete_episode_and_counts_every_eligible_step(self):
        report, ticks, added = self.run_case(tiny_config())
        self.assertEqual(report["counters"], {"env_step": 6, "update_step": 4, "episodes": 2})
        self.assertEqual(ticks, [(3, 1), (4, 1), (5, 1), (6, 1)])
        self.assertEqual(added, [(3, 3, True, False), (3, 6, True, False)])
        self.assertEqual(report["update_protocol"]["replay_eligibility"], "completed_episodes_only")

    def test_warmup_threshold_and_updates_per_step_multiplier(self):
        config = tiny_config(warmup=5)
        config.train.updates_per_step = 2
        report, ticks, _ = self.run_case(config)
        self.assertEqual(ticks, [(5, 2), (6, 2)])
        self.assertEqual(report["counters"]["update_step"], 4)

    def test_unfinished_first_episode_has_no_replay_or_updates(self):
        config = tiny_config()
        config.train.max_env_steps = 2
        report, ticks, added = self.run_case(config)
        self.assertEqual(ticks, [])
        self.assertEqual(added, [])
        self.assertEqual(report["counters"]["update_step"], 0)
        self.assertFalse(report["checkpoint_available"])
        self.assertTrue(report["episodes"][0]["truncated"])

    def test_episode_protocol_retains_two_updates_per_eligible_episode(self):
        report, ticks, added = self.run_case(tiny_config(schedule="episode", warmup=5))
        self.assertEqual(ticks, [(6, 2)])
        self.assertEqual(report["counters"]["update_step"], 2)
        self.assertEqual(len(added), 2)

    def test_old_config_defaults_and_smoke_override(self):
        old = Config().to_dict()
        old["train"].pop("update_schedule")
        old["train"].pop("updates_per_step")
        restored = Config.from_dict(old)
        self.assertEqual(restored.train.update_schedule, "episode")
        self.assertEqual(restored.train.updates_per_step, 1)
        restored.train.update_schedule = "env_step"
        train.smoke_config(restored)
        self.assertEqual(restored.train.update_schedule, "episode")
        self.assertEqual(restored.train.updates_per_episode, 2)

    def test_old_checkpoint_metadata_defaults_and_schedule_change_rejection(self):
        import torch
        config = Config()
        old = config.to_dict()
        old["train"].pop("update_schedule")
        old["train"].pop("updates_per_step")
        # 只测试外层兼容门，网络/优化器真实保存重载已由 run_case 覆盖。
        payload = {"format": "horizon_checkpoint.v1", "boundary": "episode", "config": old,
                   "versions": config.semantic_versions(), "data_manifest_hash": "diagnostic"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root/"检查点.pt"
            torch.save(payload, checkpoint)
            (root/"检查点清单.json").write_text(json.dumps({"file": checkpoint.name, "sha256": sha256(checkpoint)}), encoding="utf-8")
            artifacts.load_checkpoint(checkpoint, config, "diagnostic")
            config.train.update_schedule = "env_step"
            with self.assertRaisesRegex(ValueError, "update_schedule"):
                artifacts.load_checkpoint(checkpoint, config, "diagnostic")

    def test_research_presets_share_explicit_schedule(self):
        directory = Path(train.__file__).parent/"configs"
        paths = list(directory.glob("*.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            config = load_config(path)
            self.assertEqual(config.train.update_schedule, "env_step", path.name)
            self.assertEqual(config.train.updates_per_step, 1, path.name)

    def test_invalid_protocol_and_noninteger_multiplier_rejected(self):
        for schedule, multiplier in (("unknown", 1), ("env_step", -1), ("env_step", 1.5), ("env_step", True)):
            config = tiny_config(schedule)
            config.train.updates_per_step = multiplier
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
