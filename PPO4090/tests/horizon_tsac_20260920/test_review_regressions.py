"""独立代码审查发现的环境/数据负向回归。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from project_paths import COVER_OUTPUT_DIR
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.data.loader import scenario_from_rows
from implementations.horizon_tsac_20260920.data.domain import RootScene, CoverageCandidate, DomainSampler
from implementations.horizon_tsac_20260920.data.split import legacy_manifest, manifest_hash
from implementations.horizon_tsac_20260920.env.physics import evaluate_allocation
from implementations.horizon_tsac_20260920.env.state import Allocation
from implementations.horizon_tsac_20260920.train import prepare_dataset


class IndependentReviewRegressions(unittest.TestCase):
    def test_full_manifest_hash_and_training_rejects_changed_split(self):
        manifest = legacy_manifest(sorted(COVER_OUTPUT_DIR.glob("cover_output_*.csv"))[:3])
        self.assertEqual(manifest["manifest_hash"], manifest_hash(manifest))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            prepare_dataset(Config(), path)
            manifest["records"][0]["split"] = "train" if manifest["records"][0]["split"] != "train" else "test"
            self.assertNotEqual(manifest["manifest_hash"], manifest_hash(manifest))
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_dataset(Config(), path)

    def test_empty_domain_cells_support_sampling_and_restore(self):
        records = [dict(scenario_id="fixture", root_scene_id="fixture", source_hash="fixture",
                        split="train", domain_cells=[])]
        sampler = DomainSampler(records, seed=7, warmup_draws=5)
        self.assertTrue(all(sampler.sample()["scenario_id"] == "fixture" for _ in range(100)))
        self.assertEqual(sampler.counts["cell:unknown"], 100)

    def test_assignment_conservation_is_insufficient_without_geometry(self):
        root = RootScene("root", "family", "fixture", np.array([1]), np.array([[30., 110.]]),
                         np.array([100.]), {"name": "test"}, {"longitude": 122.2}, "fixture.v1", 0)
        rows = [dict(beam_id=1, latitude_deg=35., longitude_deg=110., demand_bps=100., ground_diameter_deg=1.)]
        scenario = scenario_from_rows(rows, source_schema="coverage.v2")
        candidate = CoverageCandidate("root", "candidate", "fixture.v1", {}, scenario, np.array([[1.]]), 0., {}, "fixture")
        with self.assertRaisesRegex(ValueError, "覆盖范围"):
            candidate.validate(root)
        rows[0]["latitude_deg"] = 30.1
        candidate.beam_table = scenario_from_rows(rows, source_schema="coverage.v2")
        self.assertTrue(candidate.validate(root)["conservation_passed"])

    def test_evaluator_rejects_non_grid_power_and_idle_skip(self):
        rows = [dict(beam_id=i, latitude_deg=30., longitude_deg=110., demand_bps=100. if i == 0 else 0.,
                     ground_diameter_deg=1.) for i in range(2)]
        scenario = scenario_from_rows(rows, source_schema="coverage.v2")
        result = evaluate_allocation(scenario, {0: Allocation(0, "allocated", 0, 1, 7.)})
        self.assertIn("invalid_power_level", [v["kind"] for v in result.constraint_violations])
        continuous = evaluate_allocation(scenario, {0: Allocation(0, "allocated", 0, 1, 7.)}, power_mode="continuous")
        self.assertFalse(continuous.constraint_violations)
        skipped = evaluate_allocation(scenario, {1: Allocation(1, "skipped")})
        self.assertIn("non_demand_beam_in_ledger", [v["kind"] for v in skipped.constraint_violations])

    def test_nonfinite_physics_cannot_be_reported_as_valid(self):
        rows = [dict(beam_id=0, latitude_deg=30., longitude_deg=110., demand_bps=100.,
                     ground_diameter_deg=1., tx_gain_peak_dbi=20000.)]
        scenario = scenario_from_rows(rows, source_schema="coverage.v2")
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(FloatingPointError):
            evaluate_allocation(scenario, {0: Allocation(0, "allocated", 0, 1, 5.)})


if __name__ == "__main__":
    unittest.main()
