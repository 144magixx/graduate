import unittest
import numpy as np
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.data.domain import (RootScene, CoverageCandidate, root_metric,
    augment_root_candidates, DomainSampler, coverage_spec, coverage_audit, generate_candidates)
from implementations.spectrum_tsac_20260921.data.split import grouped_split, validate_split


def root_fixture():
    return RootScene("r1", "lineage1", "hotspot", np.array([10,20,30]), np.array([[30.,100.],[30.2,100.2],[31.,101.]]),
                     np.array([100.,200.,300.]), {"name":"fixture"}, {"satellite_longitude_deg":122.2}, "fixture.v1", 1)


def candidate_fixture(root, matrix):
    matrix = np.asarray(matrix, dtype=float)
    demands = matrix.T @ root.offered_demand_bps
    rows = [dict(beam_id=10+i*3, latitude_deg=30+i*.2, longitude_deg=100+i*.2, demand_bps=float(d), ground_diameter_deg=3.) for i,d in enumerate(demands)]
    scenario = scenario_from_rows(rows, Config(), "standard", "candidate", metadata={"source":"unit_test_only"})
    return CoverageCandidate(root.root_scene_id, "candidate", "fixture.v1", {"seed":1}, scenario, matrix,
                             float((1-matrix.sum(1)) @ root.offered_demand_bps), {}, "fixture")


class DomainTests(unittest.TestCase):
    def test_two_candidate_common_denominator_and_uncovered(self):
        root = root_fixture()
        a = candidate_fixture(root, [[1,0],[0,1],[0,0]])
        b = candidate_fixture(root, [[1,0,0],[0,1,0],[0,0,1]])
        self.assertTrue(a.validate(root)["conservation_passed"])
        self.assertEqual(root_metric(root,a,[100.,200.])["J_root"], .5)
        self.assertEqual(root_metric(root,b,[100.,200.,300.])["J_root"], 1.)
        self.assertFalse(root_metric(None,None,None)["end_to_end_metric_available"])

    def test_duplicate_assignment_rejected(self):
        root = root_fixture()
        a = candidate_fixture(root, [[1,1],[0,1],[0,0]])
        with self.assertRaisesRegex(ValueError,"重复计数"):
            a.validate(root)

    def test_root_augmentation_shared_and_conserved(self):
        root = root_fixture()
        a = candidate_fixture(root, [[1,0],[0,1],[0,0]])
        b = candidate_fixture(root, [[1,0,0],[0,1,0],[0,0,1]])
        aug, candidates = augment_root_candidates(root,[a,b],np.random.default_rng(5))
        self.assertEqual(candidates[0].beam_table.metadata["augmented_root_id"], candidates[1].beam_table.metadata["augmented_root_id"])
        for candidate in candidates:
            self.assertTrue(candidate.validate(aug)["conservation_passed"])
        np.testing.assert_array_equal(a.beam_table.group_id,candidates[0].beam_table.group_id)

    def test_transitive_lineage_and_hash_never_cross_split(self):
        records = [dict(scenario_id=str(i), root_scene_id=f"r{i}",split_group_id=f"g{i}",source_hash=f"hash{i}") for i in range(20)]
        records[1]["root_scene_id"] = "r0"
        records[2]["split_group_id"] = "g1"
        records[3]["source_hash"] = "hash2"
        split = grouped_split(records)
        self.assertTrue(validate_split(split))
        self.assertEqual(len({x["split"] for x in split[:4]}),1)
        split[3]["split"] = "test" if split[2]["split"] != "test" else "train"
        with self.assertRaises(ValueError): validate_split(split)

    def test_sampler_resume_root_balancing_and_test_exclusion(self):
        records = [dict(scenario_id=str(i),root_scene_id="many" if i<9 else "one",source_hash=str(i),split="train") for i in range(10)]
        records.append(dict(scenario_id="test",root_scene_id="test",source_hash="test",split="test"))
        sampler = DomainSampler(records,5,mixture=(1.,0.,0.))
        counts = {"many":0,"one":0}
        for _ in range(1000): counts[sampler.sample()["root_scene_id"]] += 1
        self.assertTrue(400 < counts["one"] < 600)
        state = sampler.state_dict()
        expected = [sampler.sample()["scenario_id"] for _ in range(20)]
        sampler.load_state_dict(state)
        self.assertEqual(expected,[sampler.sample()["scenario_id"] for _ in range(20)])

    def test_unknown_roots_cannot_meet_coverage(self):
        records = [dict(scenario_id=str(i),split="train",root_scene_id=None,n_demand=60) for i in range(100)]
        report = coverage_audit(records,coverage_spec())
        self.assertEqual(report["cell_coverage_fraction"],0.)
        self.assertFalse(report["strict_acceptance_passed"])
        with self.assertRaisesRegex(RuntimeError,"生成器"):
            generate_candidates(root_fixture())


if __name__ == "__main__": unittest.main()
