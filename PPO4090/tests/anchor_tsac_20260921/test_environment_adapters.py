import copy
import numpy as np
import pytest

from implementations.anchor_tsac_20260921.config import Config
from implementations.anchor_tsac_20260921.data.loader import scenario_from_rows
from implementations.anchor_tsac_20260921.env.action import Action
from implementations.anchor_tsac_20260921.env.environment import Environment
from implementations.anchor_tsac_20260921.env.observation import adapt_observation


def fixture(config=None):
    rows = [
        {"beam_id": 10, "latitude_deg": 30., "longitude_deg": 110., "demand_bps": 5e8,
         "ground_diameter_deg": 1.5, "group_id": 0},
        {"beam_id": 11, "latitude_deg": 30.2, "longitude_deg": 110.2, "demand_bps": 4e8,
         "ground_diameter_deg": 1., "group_id": 0},
    ]
    return scenario_from_rows(rows, config or Config(), "coverage.v2", "synthetic-anchor")


def test_two_adapters_share_new_physics_and_same_action_outcome():
    legacy = Config()
    modern = copy.deepcopy(legacy); modern.env.observation_adapter = "modern205"
    left, right = Environment(legacy), Environment(modern)
    old_obs, _ = left.reset(fixture(legacy)); new_obs, _ = right.reset(fixture(modern))
    assert old_obs["observation_adapter_version"] == "legacy_scaled205_new_physics.v1"
    assert new_obs["observation_adapter_version"] == "modern205.v1"
    assert not np.array_equal(old_obs["anchor205"], new_obs["anchor205"])
    action = Action("ALLOC", 3, 2, 1)
    left.step(action); right.step(action)
    np.testing.assert_allclose(left.evaluate().rate_bps, right.evaluate().rate_bps, rtol=0, atol=0)
    np.testing.assert_allclose(left.evaluate().slot_power_w, right.evaluate().slot_power_w, rtol=0, atol=0)


def test_legacy_scale_formula_and_scalar_validation():
    view = {"latitude_deg": 29., "longitude_deg": 104., "demand_bps": 5e8,
            "ground_diameter_deg": 3., "remaining_power_w": 3000.,
            "interference_w": np.zeros(100), "noise_w": np.ones(100) * 1e-13,
            "occupancy": np.zeros(100, bool)}
    values, version = adapt_observation("legacy_scaled205", view)
    np.testing.assert_allclose(values[:5], [0., 0., 1., 1., .5])
    assert version == "legacy_scaled205_new_physics.v1"
    with pytest.raises(ValueError, match="标量"):
        adapt_observation("modern205", dict(view, demand_bps=float("nan")))


def test_mask_keeps_skip_zero_and_forces_it_when_power_unavailable():
    config = Config(); config.env.power_budget_w = 5.
    env = Environment(config)
    obs, _ = env.reset(fixture(config))
    assert len(env.action_spec) == 9551 and env.action_spec.decode(0).kind == "SKIP"
    obs, *_ = env.step(Action("ALLOC", 0, 1, 0))
    assert np.flatnonzero(obs["valid_action_mask"]).tolist() == [0]


def test_invalid_config_physics_and_semantics_rejected():
    for mutate in (
        lambda c: setattr(c.physics, "noise_temperature_k", float("nan")),
        lambda c: setattr(c.physics, "satellite_radius_km", 1000.),
        lambda c: setattr(c.data, "rate_unit", "bps"),
        lambda c: setattr(c.model, "encoder", "cnn_local"),
        lambda c: setattr(c.env, "power_levels_w", tuple(range(1, 11))),
        lambda c: setattr(c.env, "action_version", "unknown"),
    ):
        config = Config(); mutate(config)
        with pytest.raises(ValueError): config.validate()
