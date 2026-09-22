import copy
import json
import math
import unittest
import numpy as np
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.schema import CoverageScenario
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.env.action import Action, ActionSpec
from implementations.spectrum_tsac_20260921.env.state import Allocation
from implementations.spectrum_tsac_20260921.env.physics import evaluate_allocation, channel_geometry, BOLTZMANN, LIGHT_SPEED, coverage_polygon
from tests.spectrum_tsac_20260921.test_data import fixture


class EnvironmentTests(unittest.TestCase):
    def test_evaluator_rejects_skip_on_idle_and_nonfinite_physics(self):
        s = scenario_from_rows([dict(beam_id=10, latitude_deg=30, longitude_deg=110, demand_bps=0., ground_diameter_deg=1)], source_schema="coverage.v2")
        result = evaluate_allocation(s, {10: Allocation(10, "skipped")})
        self.assertEqual(result.constraint_violations[0]["kind"], "non_demand_beam_in_ledger")
        value = fixture(1).to_dict()
        value["tx_gain_peak_dbi"] = [1e9]
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(FloatingPointError):
            evaluate_allocation(CoverageScenario.from_dict(value), {})

    def test_non_grid_power_requires_explicit_continuous_mode(self):
        s = fixture(1)
        ledger = {10: Allocation(10, "allocated", 0, 1, 7.)}
        discrete = evaluate_allocation(s, ledger)
        self.assertEqual(discrete.constraint_violations[0]["kind"], "invalid_power_level")
        continuous = evaluate_allocation(s, ledger, power_mode="continuous")
        self.assertEqual(continuous.constraint_violations, [])
        self.assertEqual(continuous.metrics["power_used_w"], 7.)

    def test_required_sizes_empty_and_last_allocate(self):
        for size in (0, 1, 4, 52, 168, 170, 200, 220):
            env = Environment()
            obs, info = env.reset(fixture(size))
            self.assertEqual(int(obs["demand_mask"].sum()), size)
            self.assertEqual(obs["terminal"], size == 0)
        env.reset(fixture(1))
        obs, _, done, _, info = env.step(Action("ALLOC", 99, 1, 0))
        self.assertTrue(done)
        self.assertTrue(obs["occupancy"][0, 99])
        self.assertEqual(info["requested_action"], info["executed_action"])
        self.assertEqual(env.snapshot()["resources"]["beam_ids"][0][99], 10)

    def test_dictionary_counts_and_mask_prefix(self):
        spec = ActionSpec()
        self.assertEqual(len(spec), 9551)
        self.assertEqual(spec.encode(spec.decode(9550)), 9550)
        cfg = Config()
        cfg.physics.num_slots = 4
        cfg.env.max_block_length = 3
        cfg.env.power_budget_w = 5
        env = Environment(cfg)
        obs, _ = env.reset(fixture(2, cfg))
        masks = env.action_spec.conditional_masks(obs)
        self.assertEqual(int(obs["valid_action_mask"].sum()), 10)  # 4+3+2 allocations, SKIP
        self.assertFalse(masks["power"][:, :, 1:].any())
        self.assertFalse(masks["length"][3, 1:].any())

    def test_hand_calculated_single_beam_and_power_conservation(self):
        cfg = Config()
        s = fixture(1)
        ledger = {10: Allocation(10, "allocated", 0, 2, 10.)}
        result = evaluate_allocation(s, ledger, cfg)
        sat = np.array([cfg.physics.satellite_radius_km*math.cos(math.radians(122.2)), cfg.physics.satellite_radius_km*math.sin(math.radians(122.2)), 0])
        lat, lon = math.radians(30), math.radians(110)
        ground = 6371*np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])
        distance = np.linalg.norm(ground-sat)*1000
        expected = []
        for slot in range(2):
            frequency = 17.7e9+(slot+.5)*25e6
            signal = 5.*10**9*(LIGHT_SPEED/(4*math.pi*distance*frequency))**2
            sinr = signal/(10**.5*BOLTZMANN*290*25e6)
            expected.append(25e6*math.log2(1+sinr))
        self.assertAlmostEqual(result.rate_bps[0]/sum(expected), 1., places=12)
        np.testing.assert_allclose(result.slot_power_w.sum(axis=1), [10.])
        np.testing.assert_array_equal(result.interference_w, 0.)

    def test_victim_rx_gain_and_polarization(self):
        s = fixture(3).to_dict()
        s["group_id"] = [0, 2, 1]
        s["polarization_id"] = [0, 0, 1]
        s["latitude_deg"] = [30, 30, 30]
        s["longitude_deg"] = [110, 110, 110]
        s["rx_gain_peak_dbi"] = [30, 40, 50]
        s = CoverageScenario.from_dict(s)
        c = channel_geometry(s, Config().physics)["coupling"]
        self.assertAlmostEqual(c[0, 1]/c[1, 0], 10.)
        allocations = {int(b): Allocation(int(b), "allocated", 2, 2, 10.) for b in s.beam_id}
        result = evaluate_allocation(s, allocations)
        self.assertGreater(result.interference_w[0, 2], 0.)
        self.assertEqual(result.interference_w[2, 2], 0.)
        self.assertAlmostEqual(result.interference_w[1, 2]/result.interference_w[0, 2], 10.)

    def test_more_interference_cannot_improve_victim_or_touch_other_slots(self):
        value = fixture(2).to_dict()
        value["group_id"], value["polarization_id"] = [0, 2], [0, 0]
        s = CoverageScenario.from_dict(value)
        low = {10: Allocation(10, "allocated", 0, 2, 10.), 13: Allocation(13, "allocated", 0, 1, 5.)}
        high = dict(low)
        high[13] = Allocation(13, "allocated", 0, 1, 50.)
        a, b = evaluate_allocation(s, low), evaluate_allocation(s, high)
        self.assertGreater(a.sinr_linear[0, 0], b.sinr_linear[0, 0])
        self.assertEqual(a.sinr_linear[0, 1], b.sinr_linear[0, 1])

    def test_negative_delta_and_padding_invariance(self):
        value = fixture(2).to_dict()
        value["demand_bps"] = [1e8, 1e12]
        value["group_id"], value["polarization_id"] = [0, 2], [0, 0]
        value["latitude_deg"], value["longitude_deg"] = [30, 30], [110, 110]
        a, b = Environment(), Environment()
        a.reset(CoverageScenario.from_dict(value))
        padded = copy.deepcopy(value)
        for field in ("beam_id", "latitude_deg", "longitude_deg", "demand_bps", "ground_diameter_deg", "tx_gain_peak_dbi", "rx_gain_peak_dbi",
                      "noise_temperature_k", "group_id", "polarization_id", "entity_mask", "demand_mask"):
            padded[field].append(999 if field == "beam_id" else False if field.endswith("mask") else 0.)
        b.reset(CoverageScenario.from_dict(padded))
        first = a.step(Action("ALLOC", 0, 1, 9))
        first_b = b.step(Action("ALLOC", 0, 1, 9))
        last = a.step(Action("ALLOC", 0, 1, 9))
        last_b = b.step(Action("ALLOC", 0, 1, 9))
        self.assertLess(last[1], 0.)
        self.assertEqual(first[1:], first_b[1:])
        self.assertEqual(last[1:], last_b[1:])

    def test_legacy_observation_alias_is_resolved_by_full_state(self):
        value = fixture(2).to_dict()
        other = copy.deepcopy(value)
        value["demand_bps"][0], other["demand_bps"][0] = 1e7, 1e10
        a, b = Environment(), Environment()
        a.reset(CoverageScenario.from_dict(value))
        b.reset(CoverageScenario.from_dict(other))
        oa, *_ = a.step(Action("ALLOC", 0, 1, 0))
        ob, *_ = b.step(Action("ALLOC", 0, 1, 0))
        np.testing.assert_array_equal(oa["legacy205"], ob["legacy205"])
        self.assertFalse(np.array_equal(oa["beam_static"], ob["beam_static"]))

    def test_illegal_action_is_atomic_and_rng_unchanged(self):
        cfg = Config()
        cfg.env.power_budget_w = 5
        env = Environment(cfg)
        env.reset(fixture(2), seed=17)
        before = json.dumps(env.state_dict(), sort_keys=True)
        for bad in (Action("ALLOC", 99, 2, 0), Action("ALLOC", 0, 1, 1), Action("ALLOC", .5, 1, 0), -1, True, Action("SKIP", 0)):
            with self.assertRaises(ValueError):
                env.step(bad)
            self.assertEqual(before, json.dumps(env.state_dict(), sort_keys=True))

    def test_budget_exhaustion_and_final_ledger(self):
        cfg = Config()
        cfg.env.power_budget_w = 5
        env = Environment(cfg)
        obs, _ = env.reset(fixture(3))
        obs, reward, done, _, _ = env.step(Action("ALLOC", 99, 1, 0))
        self.assertFalse(done)
        self.assertEqual(obs["valid_action_mask"].sum(), 1)
        env.step(Action())
        obs, _, done, _, info = env.step(Action())
        self.assertTrue(done)
        self.assertEqual(obs["current_beam_index"], -1)
        self.assertEqual(obs["pending_mask"].sum(), 0)
        self.assertTrue(env.snapshot()["resources"]["occupancy"][0][99])
        self.assertEqual(info["metrics"]["skip_fraction"], 2/3)
        self.assertAlmostEqual(reward, 100*info["metrics"]["mean_satisfaction"])
        with self.assertRaises(RuntimeError):
            env.step(Action())

    def test_empty_demand_and_real_idle(self):
        s = scenario_from_rows([dict(latitude_deg=30, longitude_deg=110, demand_bps=0., ground_diameter_deg=1)], source_schema="coverage.v2")
        env = Environment()
        obs, info = env.reset(s)
        self.assertTrue(obs["terminal"])
        self.assertIsNone(info["metrics"]["mean_satisfaction"])
        self.assertIsNone(info["metrics"]["mean_slot_sinr_db"])
        self.assertEqual(len(env.snapshot()["beams"]), 1)

    def test_random_legal_trajectory_telescopes_and_independent_evaluation(self):
        cfg = Config()
        cfg.env.power_budget_w = 63.
        cfg.physics.num_slots = 8
        cfg.env.max_block_length = 4
        env = Environment(cfg)
        obs, _ = env.reset(fixture(25))
        rng, total = np.random.default_rng(17), 0.
        while not obs["terminal"]:
            action_id = int(rng.choice(np.flatnonzero(obs["valid_action_mask"])))
            obs, reward, _, _, info = env.step(action_id)
            total += reward
            evaluation = env.evaluate()
            self.assertEqual(evaluation.constraint_violations, [])
            np.testing.assert_allclose(evaluation.rate_bps, env._evaluation.rate_bps, rtol=1e-13)
            self.assertGreaterEqual(env.remaining_power_w, -1e-8)
        self.assertAlmostEqual(total, 100*info["metrics"]["mean_satisfaction"], places=12)

    def test_final_order_invariance_and_historical_loss(self):
        value = fixture(2).to_dict()
        value["group_id"], value["polarization_id"] = [0, 2], [0, 0]
        value["latitude_deg"], value["longitude_deg"] = [30, 30], [110, 110]
        s = CoverageScenario.from_dict(value)
        a, b = Environment(), Environment()
        a.reset(s)
        a.step(Action("ALLOC", 0, 2, 1))
        _, _, _, _, info = a.step(Action("ALLOC", 0, 2, 1))
        self.assertGreater(info["reward_terms"]["historical_utility_loss"], 0.)
        value["service_order"] = list(reversed(value["service_order"]))
        b.reset(CoverageScenario.from_dict(value))
        b.step(Action("ALLOC", 0, 2, 1))
        b.step(Action("ALLOC", 0, 2, 1))
        np.testing.assert_allclose(a.evaluate().rate_bps, b.evaluate().rate_bps, rtol=0, atol=0)

    def test_contiguous_occupied_and_budget_boundaries(self):
        cfg = Config()
        cfg.physics.num_slots = 4
        cfg.env.max_block_length = 3
        value = fixture(2).to_dict()
        value["group_id"] = [0, 0]
        value["polarization_id"] = [0, 0]
        env = Environment(cfg)
        env.reset(CoverageScenario.from_dict(value))
        obs, *_ = env.step(Action("ALLOC", 1, 2, 0))
        mask = obs["valid_action_mask"]
        self.assertFalse(mask[env.action_spec.encode(Action("ALLOC", 0, 2, 0))])
        self.assertTrue(mask[env.action_spec.encode(Action("ALLOC", 3, 1, 0))])
        before = env.state_dict()
        with self.assertRaises(ValueError):
            env.step(Action("ALLOC", 0, 2, 0))
        self.assertEqual(before, env.state_dict())
        obs["remaining_power_w"] = 4.999
        self.assertEqual(env.action_spec.valid_actions(obs).sum(), 1)
        obs["remaining_power_w"] = 5.
        self.assertEqual(env.action_spec.valid_actions(obs).sum(), 3)

    def test_checkpoint_continuation_and_stable_ids(self):
        a, b = Environment(), Environment()
        a.reset(fixture(3), seed=8)
        a.step(Action("ALLOC", 0, 1, 0))
        b.load_state_dict(a.state_dict())
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertEqual(a.current_beam_id, 13)
        self.assertEqual(a.step(Action())[1:], b.step(Action())[1:])

    def test_geodesic_contour_coordinates(self):
        poly = coverage_polygon(30, 110, 2)
        coords = np.asarray(poly["coordinates"][0])
        np.testing.assert_allclose(coords[0], [110, 31], atol=1e-10)
        np.testing.assert_array_equal(coords[0], coords[-1])
        dlat, dlon = np.deg2rad(coords[:, 1]-30), np.deg2rad(coords[:, 0]-110)
        angle = 2*np.arcsin(np.sqrt(np.sin(dlat/2)**2+np.cos(np.deg2rad(30))*np.cos(np.deg2rad(coords[:, 1]))*np.sin(dlon/2)**2))
        np.testing.assert_allclose(np.rad2deg(angle), 1., atol=1e-10)

    def test_220_positive_terminal_and_pending_sinr_null(self):
        cfg = Config()
        cfg.env.power_budget_w = 0
        env = Environment(cfg)
        obs, _ = env.reset(fixture(220))
        self.assertIsNone(env.snapshot()["beams"][0]["sinr_db"])
        for _ in range(220):
            obs, *_ = env.step(0)
        self.assertTrue(obs["terminal"])
        self.assertEqual(env.evaluate().metrics["skip_fraction"], 1.)
        self.assertEqual(env.evaluate().metrics["mean_satisfaction"], 0.)


if __name__ == "__main__":
    unittest.main()
