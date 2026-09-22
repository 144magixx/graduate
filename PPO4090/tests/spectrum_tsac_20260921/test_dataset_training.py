"""冻结域标签接入真实训练与run级离线证据回归。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from implementations.spectrum_tsac_20260921 import artifacts, train
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.domain import (DomainSampler, coverage_spec, freeze_run_coverage, validate_domain_annotations)
from implementations.spectrum_tsac_20260921.data.split import legacy_manifest, manifest_hash
from implementations.spectrum_tsac_20260921.telemetry.recorder import export_run
from implementations.spectrum_tsac_20260921.telemetry.storage import DB_NAME, readonly
from project_paths import COVER_OUTPUT_DIR, SPECTRUM_DATA_DIR
from tests.spectrum_tsac_20260921.test_data import fixture
from tests.spectrum_tsac_20260921.test_schedule import tiny_config


def annotated_fixture(root, config):
    root.mkdir(parents=True, exist_ok=True)
    source = root/"诊断输入.json"
    source.write_text(json.dumps(fixture(3, config).to_dict()), encoding="utf-8")
    spec = coverage_spec()
    spec["mandatory_cells"] = [{"id": "diagnostic_small", "conditions": {"n_demand": [1, 4]}}]
    spec["fixture_only"] = True
    (root/"训练场景覆盖规格.yaml").write_text(json.dumps(spec), encoding="utf-8")
    manifest = {"dataset_version": config.data.dataset_version, "split_seed": config.data.split_seed,
                "coverage_spec_hash": manifest_hash(spec), "records": [{"scenario_id": "diagnostic_fixture", "scenario_path": str(source),
                "source_hash": sha256(source), "split": "train", "root_scene_id": None, "n_demand": 3,
                "domain_cells": ["diagnostic_small"], "provenance_status": "diagnostic_fixture"}]}
    manifest["manifest_hash"] = manifest_hash(manifest)
    path = root/"训练场景清单.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, spec


class DatasetTrainingTests(unittest.TestCase):
    def test_default_uses_frozen_real_domain_labels_without_inventing_roots(self):
        manifest = train.prepare_dataset(Config())
        stored = json.loads((SPECTRUM_DATA_DIR/"训练场景清单.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, stored)
        self.assertEqual(len(manifest["records"]), 100)
        self.assertTrue(all(record["domain_cells"] for record in manifest["records"]))
        self.assertTrue(all(record["root_scene_id"] is None for record in manifest["records"]))
        sampler = DomainSampler(manifest["records"], seed=17, warmup_draws=30)
        for _ in range(30):
            self.assertEqual(sampler.sample()["split"], "train")
        counts = sampler.state_dict()["collected_by_domain_cell"]
        self.assertNotIn("unknown", counts)
        self.assertEqual(sum(counts.values()), 30)
        self.assertTrue(all(key.startswith("scale_") for key in counts))
        checkpoint = sampler.state_dict()
        incompatible = DomainSampler(manifest["records"], seed=17, warmup_draws=30, mixture=(1., 0., 0.))
        with self.assertRaisesRegex(ValueError, "采样协议"):
            incompatible.load_state_dict(checkpoint)
        legacy_state = {key: checkpoint[key] for key in ("rng", "draws", "counts", "branch_counts")}
        sampler.load_state_dict(legacy_state)
        self.assertEqual(sampler.state_dict()["collected_by_domain_cell"], counts)

    def test_explicit_old_manifest_remains_byte_semantics_compatible(self):
        old = legacy_manifest(COVER_OUTPUT_DIR.glob("cover_output_*.csv"), Config().data.split_seed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"原数据划分.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            received = train.prepare_dataset(Config(), path)
            self.assertEqual(received, old)
            self.assertFalse(any("domain_cells" in record for record in received["records"]))
            self.assertNotEqual(received["manifest_hash"], train.prepare_dataset(Config())["manifest_hash"])

    def test_default_rejects_domain_or_spec_tampering_even_after_manifest_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            config = tiny_config()
            directory = Path(temp)
            path, manifest, spec = annotated_fixture(directory, config)
            with patch.object(train, "SPECTRUM_DATA_DIR", directory):
                self.assertEqual(train.prepare_dataset(config)["manifest_hash"], manifest["manifest_hash"])
                bad = copy.deepcopy(manifest)
                bad["records"][0]["domain_cells"] = ["not_the_declared_cell"]
                bad["manifest_hash"] = manifest_hash(bad)
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "域标签"):
                    train.prepare_dataset(config)
                bad = copy.deepcopy(manifest)
                bad["records"][0]["n_demand"] = 4  # 同一规模桶，仍必须与实际场景核验。
                bad["manifest_hash"] = manifest_hash(bad)
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "源场景"):
                    train.prepare_dataset(config)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                spec["independent_roots_min"]["train"] = 1
                (directory/"训练场景覆盖规格.yaml").write_text(json.dumps(spec), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "规格hash"):
                    train.prepare_dataset(config)

    def test_source_hash_and_frozen_split_seed_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            config = tiny_config()
            directory = Path(temp)
            path, manifest, _ = annotated_fixture(directory, config)
            with patch.object(train, "SPECTRUM_DATA_DIR", directory):
                config.data.split_seed += 1
                with self.assertRaisesRegex(ValueError, "split_seed"):
                    train.prepare_dataset(config)
                config.data.split_seed -= 1
                source = Path(manifest["records"][0]["scenario_path"])
                source.write_text(source.read_text(encoding="utf-8")+" ", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "输入文件hash"):
                    train.prepare_dataset(config)

    def test_unknown_legacy_cannot_be_promoted_to_verified_root_by_annotation(self):
        manifest = train.prepare_dataset(Config())
        manifest["records"][0]["root_scene_id"] = "invented-root"
        with self.assertRaisesRegex(ValueError, "未知legacy"):
            validate_domain_annotations(manifest, coverage_spec(), require_annotations=True)

    def test_run_snapshot_is_immutable_when_source_spec_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = tiny_config()
            path, manifest, spec = annotated_fixture(root/"data", config)
            run_dir = root/"run"
            run_dir.mkdir()
            snapshot = freeze_run_coverage(run_dir, manifest, config, path)
            spec["support_domain"]["positive_beam_count"] = [999, 1000]
            (path.parent/"训练场景覆盖规格.yaml").write_text(json.dumps(spec), encoding="utf-8")
            for reference in snapshot["artifacts"].values():
                self.assertEqual(sha256(run_dir/reference["path"]), reference["sha256"])
            audit = json.loads((run_dir/"训练场景覆盖审计.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["source_manifest_hash"], manifest["manifest_hash"])
            self.assertEqual(audit["unknown_provenance_candidates"], 1)
            self.assertFalse(audit["strict_acceptance_passed"])
            self.assertIsNone(audit["collected_samples"])

    def test_actual_training_publishes_frozen_audit_and_preserves_offline_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = tiny_config(schedule="episode")
            config.train.episodes = 1
            config.telemetry.enabled = True
            path, manifest, spec = annotated_fixture(root/"data", config)
            with patch.object(artifacts, "SPECTRUM_OUTPUT_DIR", root/"runs"), contextlib.redirect_stdout(io.StringIO()):
                run_dir, report = train.run_training(config, "dataset_diagnostic", path)
            counts = report["sampler"]["collected_by_domain_cell"]
            self.assertEqual(counts, {"diagnostic_small": 1})
            saved = artifacts.load_checkpoint(run_dir/"检查点.pt", config, manifest["manifest_hash"])
            update_counts = saved["replay"]["sampled_by_domain_cell"]
            self.assertTrue(update_counts)
            self.assertTrue(all("diagnostic_small" in key for key in update_counts))
            conn = readonly(run_dir/DB_NAME)
            try:
                rows = conn.execute("SELECT kind,path,sha256 FROM artifacts WHERE kind IN ('coverage_spec','coverage_audit')").fetchall()
                self.assertEqual({row["kind"] for row in rows}, {"coverage_spec", "coverage_audit"})
                for row in rows:
                    self.assertEqual(sha256(run_dir/row["path"]), row["sha256"])
            finally:
                conn.close()
            package = export_run(run_dir, root/"离线包")
            # 上游全局证据修改后，离线文件保持当时的内容与hash。
            (path.parent/"训练场景覆盖规格.yaml").write_text("{}", encoding="utf-8")
            self.assertEqual((run_dir/"训练场景覆盖审计.json").read_bytes(), (package/"训练场景覆盖审计.json").read_bytes())
            self.assertEqual((run_dir/"训练场景覆盖规格.yaml").read_bytes(), (package/"训练场景覆盖规格.yaml").read_bytes())
            from fastapi.testclient import TestClient
            from implementations.spectrum_tsac_20260921.dashboard.api import create_app
            client = TestClient(create_app(package, datasets_root=package))
            response = client.get(f"/api/v1/datasets/{config.data.dataset_version}/coverage").json()
            self.assertEqual(response["status"], "recorded")
            self.assertEqual(response["data"]["source_manifest_hash"], manifest["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
