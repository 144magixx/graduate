import copy
import unittest
import numpy as np
from project_paths import COVER_OUTPUT_DIR
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.loader import load_scenario, scenario_from_rows
from implementations.spectrum_tsac_20260921.data.schema import CoverageScenario
from implementations.spectrum_tsac_20260921.data.augment import augment_scenario


def fixture(n=3, config=None, demand=1e9):
    rows = [dict(beam_id=10+i*3, latitude_deg=30+i*.001, longitude_deg=110+i*.001,
                 demand_bps=demand, ground_diameter_deg=2., group_id=i%8) for i in range(n)]
    return scenario_from_rows(rows, config, "coverage.v2", "diagnostic_fixture", metadata={"purpose": "interface_test_not_training_data"})


class DataTests(unittest.TestCase):
    def test_all_source_counts_and_mapping(self):
        scenarios = [load_scenario(p) for p in COVER_OUTPUT_DIR.glob("cover_output_*.csv")]
        self.assertEqual(len(scenarios), 100)
        self.assertEqual(sum(s.n_demand for s in scenarios), 14130)
        self.assertEqual(sum((~s.entity_mask).sum() for s in scenarios), 5870)
        self.assertEqual((min(s.n_demand for s in scenarios), max(s.n_demand for s in scenarios)), (52, 168))
        first = load_scenario(COVER_OUTPUT_DIR/"cover_output_0.csv")
        self.assertAlmostEqual(first.latitude_deg[0], 29.17056274847715)
        self.assertAlmostEqual(first.longitude_deg[0], 111.8837661840735)
        self.assertEqual(first.demand_bps[0], 600e6)
        self.assertGreater(max(s.ground_diameter_deg.max() for s in scenarios), 2.)

    def test_idle_distinct_from_padding_and_roundtrip(self):
        rows = [dict(latitude_deg=30, longitude_deg=110, demand_bps=0., ground_diameter_deg=1.),
                dict(latitude_deg=0, longitude_deg=0, demand_bps=0., ground_diameter_deg=0., entity_mask=False)]
        s = scenario_from_rows(rows, source_schema="coverage.v2")
        self.assertEqual(s.n_entities, 1)
        self.assertEqual(s.n_demand, 0)
        self.assertEqual(CoverageScenario.from_dict(s.to_dict()).to_dict(), s.to_dict())

    def test_invalid_sources_rejected(self):
        source = fixture(2).to_dict()
        for field, value in (("latitude_deg", [91, 0]), ("demand_bps", [-1, 1]), ("ground_diameter_deg", [0, 1]),
                             ("beam_id", [10, 10]), ("beam_id", [1.2, 3]), ("noise_temperature_k", [0, 290]),
                             ("entity_mask", [2, 1]),
                             ("longitude_deg", [float("nan"), 100]), ("service_order", [10, 10])):
            bad = copy.deepcopy(source)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CoverageScenario.from_dict(bad)

    def test_augments_only_demand_and_preserves_group(self):
        s = load_scenario(COVER_OUTPUT_DIR/"cover_output_0.csv")
        a = augment_scenario(s, np.random.default_rng(7))
        b = augment_scenario(s, np.random.default_rng(7))
        self.assertEqual(a.to_dict(), b.to_dict())
        np.testing.assert_array_equal(s.group_id, a.group_id)
        np.testing.assert_array_equal(s.ground_diameter_deg, a.ground_diameter_deg)
        self.assertTrue(np.all(a.demand_bps[~s.demand_mask] == 0))
        with self.assertRaises(ValueError):
            augment_scenario(s, np.random.default_rng(0), split="test")
        with self.assertRaises(ValueError):
            augment_scenario(s, np.random.default_rng(0), split="validation")

    def test_stable_ties_and_220_no_truncation(self):
        s = fixture(220)
        self.assertEqual(s.n_demand, 220)
        self.assertEqual(len(s.service_order), 220)
        np.testing.assert_array_equal(s.service_order, s.beam_id)
        self.assertEqual(s.service_order[-1], 667)

    def test_explicit_random_order_rng_preserves_canonical_groups(self):
        path = COVER_OUTPUT_DIR/"cover_output_0.csv"
        reference = load_scenario(path)
        config = Config()
        config.data.service_order = "random"
        with self.assertRaises(ValueError):
            load_scenario(path, config)
        a = load_scenario(path, config, rng=np.random.default_rng(71))
        b = load_scenario(path, config, rng=np.random.default_rng(71))
        np.testing.assert_array_equal(a.service_order, b.service_order)
        np.testing.assert_array_equal(a.group_id, reference.group_id)
        self.assertFalse(np.array_equal(a.service_order, reference.service_order))
        self.assertEqual(sorted(a.service_order), sorted(reference.service_order))

    def test_source_schema_never_guessed(self):
        row = [{"lat": 110., "lon": 30., "rate": 100., "beamwidth": 1.}]
        with self.assertRaises((ValueError, KeyError)):
            scenario_from_rows(row, source_schema="standard")

    def test_legacy_reference_frozen_sequence_and_rng(self):
        from implementations.spectrum_tsac_20260921.legacy_reference import replay_legacy
        before = np.random.get_state()
        trace = replay_legacy(COVER_OUTPUT_DIR/"cover_output_0.csv", [(0, 1, 5), (10, 2, 10), (20, 3, 15)])
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_allclose([t["reward"] for t in trace], [-4.525778073594062, -3.894080845910622, -3.558148814471596], rtol=1e-12)
        np.testing.assert_allclose(trace[-1]["beam_rates_bps"], [111721926.4059258, 268412718.9851345, 441307991.41201794], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
