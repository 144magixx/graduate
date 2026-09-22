"""A tiny, explicitly non-scientific dashboard fixture.

It never imports a model, checkpoint, optimizer, or CSV loader.  The three fixed
manual actions exercise the environment and telemetry contracts only.
"""
from pathlib import Path

import numpy as np

from ..config import Config
from ..data.schema import CoverageScenario
from ..env.action import Action
from ..env.environment import Environment
from ..telemetry.recorder import Recorder


def _scenario():
    """Three artificial demand beams, deliberately unlike a research dataset."""
    ids = np.array([101, 102, 103], dtype=np.int64)
    return CoverageScenario(
        scenario_id="anchor-engineering-demo-3beams",
        source_hash="synthetic-engineering-fixture-not-a-dataset",
        source_schema="synthetic_engineering_fixture.v1",
        beam_id=ids,
        latitude_deg=np.array([20.0, 22.0, 24.0]),
        longitude_deg=np.array([110.0, 114.0, 118.0]),
        demand_bps=np.array([80e6, 100e6, 120e6]),
        ground_diameter_deg=np.array([1.0, 1.1, 1.2]),
        tx_gain_peak_dbi=np.array([50.0, 50.0, 50.0]),
        rx_gain_peak_dbi=np.array([40.0, 40.0, 40.0]),
        noise_temperature_k=np.array([290.0, 290.0, 290.0]),
        group_id=np.array([0, 1, 2], dtype=np.int64),
        polarization_id=np.array([0, 1, 0], dtype=np.int64),
        entity_mask=np.array([True, True, True]),
        demand_mask=np.array([True, True, True]),
        service_order=ids,
        metadata={"is_demo": True, "fixture": "three_manual_actions"},
    )


def create_demo_run(root):
    """Write one immutable engineering-demo run beneath *root* and return its path."""
    destination = Path(root).resolve() / "工程示例-三个人工波束"
    config = Config()
    environment = Environment(config)
    environment.reset(_scenario(), seed=0)
    manifest = {
        "run_id": "anchor-engineering-demo",
        "trial_id": "anchor-engineering-demo",
        "attempt_id": "anchor-engineering-demo",
        "implementation": "anchor_tsac_20260921",
        "algorithm_id": "anchor_tsac",
        "algorithm": {"id": "anchor_tsac", "model": "pure_transformer_14", "token_count": 14,
                      "d_model": 128, "num_heads": 8, "num_layers": 2, "actor": "independent", "critic": "additive"},
        "observation_adapter": config.env.observation_adapter,
        "observation_adapter_version": config.semantic_versions()["observation_adapter_version"],
        "initialization": "engineering_fixture_not_applicable",
        "lineage": {"kind": "engineering_fixture_not_applicable"},
        "lineage_modes": ["from_scratch", "weights_only_not_exact_resume", "exact_resume"],
        "parent_checkpoint_id": None,
        "source_hash": "synthetic-engineering-fixture-not-a-dataset",
        "source": "diagnostic_fixture",
        "mode": "demo",
        "is_demo": True,
        "experiment_authorized": False,
        "config": config.to_dict(),
        "versions": config.semantic_versions(),
        "action_spec_signature": environment.action_spec.signature(),
        "note": "工程示例：3个人工波束和3个固定手工合法动作；无模型初始化或训练谱系，非训练结果，不能用于科学比较。",
    }
    # Three explicit manual allocations.  They use separate groups and a 5 W
    # total each, so their validity comes from the public ActionSpec/Environment.
    actions = (Action("ALLOC", 0, 1, 0), Action("ALLOC", 1, 1, 0), Action("ALLOC", 2, 1, 0))
    with Recorder(destination, manifest) as recorder:
        episode_id = "engineering-demo-episode"
        recorder.start_episode(episode_id, environment.scenario.scenario_id, environment.snapshot(), phase="demo", env_step=0)
        for step, action in enumerate(actions, start=1):
            before = environment.snapshot()
            _, reward, terminated, truncated, info = environment.step(action)
            after = environment.snapshot()
            recorder.record_step(episode_id, environment.scenario.scenario_id, step, before, after,
                                 {**info, "reward": reward, "manual_action": True,
                                  "demo_disclaimer": "engineering fixture; not an optimization update"},
                                 env_step=step, phase="demo", event_id=f"demo-manual-step-{step}")
        recorder.end_episode(episode_id, environment.evaluate().metrics, scenario_id=environment.scenario.scenario_id,
                             env_step=len(actions), terminated=terminated, truncated=truncated, phase="demo")
    return destination
