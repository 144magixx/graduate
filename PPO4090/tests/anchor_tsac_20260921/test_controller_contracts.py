import copy
import json
from pathlib import Path
import pickle
import numpy as np
import pytest

from implementations.anchor_tsac_20260921 import artifacts
from implementations.anchor_tsac_20260921.config import Config
from implementations.anchor_tsac_20260921.data.loader import scenario_from_rows
from implementations.anchor_tsac_20260921.env.action import ActionSpec
from implementations.anchor_tsac_20260921.rl import DiscreteSAC, ReplayBuffer, Transition
from implementations.anchor_tsac_20260921 import train


def scenario(config):
    return scenario_from_rows([{"beam_id": 7, "latitude_deg": 30., "longitude_deg": 110.,
                                "demand_bps": 2e8, "ground_diameter_deg": 1., "group_id": 0}],
                              config, "coverage.v2", "synthetic-controller", "synthetic-hash")


def manifest():
    return {"manifest_hash": "synthetic-manifest-hash", "dataset_version": "synthetic-fixture.v1",
            "records": [{"scenario_id": "synthetic-controller", "scenario_path": "synthetic-fixture.json",
                         "source_hash": "synthetic-hash", "split": "train", "root_scene_id": None,
                         "split_group_id": None, "semantic_hash": "synthetic-semantic",
                         "provenance_status": "synthetic_fixture", "n_demand": 1}]}


def install_fixture(monkeypatch, tmp_path, config):
    from implementations.anchor_tsac_20260921.data import loader
    monkeypatch.setattr(artifacts, "ANCHOR_OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(train, "prepare_dataset", lambda _config, _path: copy.deepcopy(manifest()))
    monkeypatch.setattr(loader, "load_scenario", lambda *_args, **_kwargs: scenario(config))


def run_fixture(monkeypatch, tmp_path, config, **kwargs):
    install_fixture(monkeypatch, tmp_path, config)
    return train.run_training(config, dataset=tmp_path / "冻结合成清单.json",
                              allow_experiment=True, is_demo=True, **kwargs)


def test_controller_checkpoint_frequency_artifacts_and_telemetry_manifest(monkeypatch, tmp_path):
    config = Config(); config.train.episodes = 3; config.train.warmup_steps = 999
    config.train.checkpoint_every = 2; config.telemetry.enabled = True
    run_dir, report = run_fixture(monkeypatch, tmp_path, config)
    archives = sorted((run_dir / "检查点存档").iterdir())
    assert len(archives) == 2
    assert (run_dir / "检查点.pt").is_file() and (run_dir / "检查点清单.json").is_file()
    assert all((directory / "检查点.pt").is_file() and (directory / "检查点清单.json").is_file()
               for directory in archives)
    assert len(list((run_dir / "场景快照").glob("场景-*.json"))) == 3
    for name in ("配置快照.json", "数据划分.json", "训练场景覆盖规格.yaml", "训练场景覆盖审计.json", "训练结果.json"):
        assert (run_dir / name).is_file(), name
    saved = json.loads((run_dir / "运行清单.json").read_text(encoding="utf-8"))
    assert saved["algorithm"] == "Anchor T-SAC" and saved["algorithm_spec"]["num_heads"] == 8
    assert saved["status"] == "completed" and saved["experiment_authorized"] is False
    assert saved["execution_kind"] == "engineering_fixture"
    assert report["counters"] == {"episodes": 3, "env_step": 3, "update_step": 0}
    assert all(np.isclose(row["episode_return"], 100 * row["mean_satisfaction"]) for row in report["episodes"])


def test_wallclock_stops_at_complete_boundary_and_telemetry_off_updates_manifest(monkeypatch, tmp_path):
    config = Config(); config.train.episodes = 5; config.train.warmup_steps = 999
    config.train.max_wall_seconds = 1e-12; config.telemetry.enabled = False
    run_dir, report = run_fixture(monkeypatch, tmp_path, config)
    assert report["new_episodes"] == 1 and report["budget_boundary"]["wallclock"] is True
    assert report["checkpoint_available"] is True
    saved = json.loads((run_dir / "运行清单.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completed" and saved["counters"]["episodes"] == 1


def test_env_step_soft_limit_finishes_current_episode_then_checkpoints(monkeypatch, tmp_path):
    config = Config(); config.train.episodes = 4; config.train.max_env_steps = 1
    config.train.warmup_steps = 999; config.telemetry.enabled = False
    install_fixture(monkeypatch, tmp_path, config)
    from implementations.anchor_tsac_20260921.data import loader
    two = scenario_from_rows([
        {"beam_id": 7, "latitude_deg": 30., "longitude_deg": 110., "demand_bps": 2e8,
         "ground_diameter_deg": 1., "group_id": 0},
        {"beam_id": 8, "latitude_deg": 31., "longitude_deg": 111., "demand_bps": 1e8,
         "ground_diameter_deg": 1., "group_id": 1}], config, "coverage.v2", "synthetic-two", "synthetic-hash")
    monkeypatch.setattr(loader, "load_scenario", lambda *_args, **_kwargs: two)
    run_dir, report = train.run_training(config, dataset=tmp_path / "冻结合成清单.json",
                                         allow_experiment=True, is_demo=True)
    assert report["new_episodes"] == 1 and report["counters"]["env_step"] == 2
    assert report["budget_boundary"]["env_steps"] is True and report["checkpoint_available"] is True
    assert json.loads((run_dir / "运行清单.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_telemetry_off_failure_marks_manifest_failed(monkeypatch, tmp_path):
    config = Config(); config.train.episodes = 1; config.train.warmup_steps = 999; config.telemetry.enabled = False
    install_fixture(monkeypatch, tmp_path, config)
    from implementations.anchor_tsac_20260921.env.environment import Environment
    monkeypatch.setattr(Environment, "step", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        train.run_training(config, dataset=tmp_path / "冻结合成清单.json",
                           allow_experiment=True, is_demo=True)
    run_dir = next((tmp_path / "outputs").iterdir())
    saved = json.loads((run_dir / "运行清单.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and (run_dir / "中断记录.json").is_file()


def test_exact_resume_keeps_trial_origin_and_uniform_warmup(monkeypatch, tmp_path):
    config = Config(); config.train.episodes = 1; config.train.warmup_steps = 999; config.telemetry.enabled = False
    first_dir, first_report = run_fixture(monkeypatch, tmp_path, config)
    first_manifest = json.loads((first_dir / "运行清单.json").read_text(encoding="utf-8"))
    checkpoint = Path(first_report["checkpoint_path"])
    original_act = DiscreteSAC.act
    def forbidden(*_args, **_kwargs):
        raise AssertionError("from_scratch精确恢复在warmup内必须继续uniform采集")
    monkeypatch.setattr(DiscreteSAC, "act", forbidden)
    second_dir, second_report = train.run_training(config, dataset=tmp_path / "冻结合成清单.json",
                                                    resume=checkpoint, allow_experiment=True, is_demo=True)
    monkeypatch.setattr(DiscreteSAC, "act", original_act)
    second_manifest = json.loads((second_dir / "运行清单.json").read_text(encoding="utf-8"))
    assert second_manifest["trial_id"] == first_manifest["trial_id"]
    assert second_manifest["parent_checkpoint_id"] == checkpoint.parent.name
    assert second_manifest["lineage"]["kind"] == "exact_resume"
    assert second_manifest["initialization"]["kind"] == "from_scratch"
    assert second_manifest["warmup_collection"] == "uniform_legal"
    assert second_report["counters"]["episodes"] == 2
    assert train._warmup_uses_policy({"kind": "weights_only_not_exact_resume"}) is True
    assert train._warmup_uses_policy({"kind": "from_scratch"}) is False


def _observation(config, spec, terminal=False):
    value = {"anchor205": np.zeros(205, np.float32), "terminal": terminal,
             "occupancy": np.zeros((8, 100), bool), "current_group": 0, "remaining_power_w": 6000.,
             "observation_adapter_version": config.semantic_versions()["observation_adapter_version"],
             "action_spec_signature": spec.signature()}
    value["valid_action_mask"] = spec.valid_actions(value)
    return value


def test_replay_restore_is_atomic_validated_and_rng_reproducible():
    config, spec = Config(), ActionSpec(Config())
    before, terminal = _observation(config, spec), _observation(config, spec, True)
    transition = Transition(before, 0, .25, terminal, True, config.semantic_versions())
    source = ReplayBuffer(4, config.semantic_versions(), spec.signature(), seed=17)
    source.add(transition); source.add(transition)
    state = source.state_dict()
    left = ReplayBuffer(4, config.semantic_versions(), spec.signature(), seed=99)
    right = ReplayBuffer(4, config.semantic_versions(), spec.signature(), seed=100)
    left.load_state_dict(state); right.load_state_dict(state)
    assert left.sample(1)[0].reward == right.sample(1)[0].reward
    before_state = pickle.dumps(left.state_dict())
    broken = copy.deepcopy(state); broken["values"][-1].reward = float("nan")
    with pytest.raises(ValueError, match="reward"): left.load_state_dict(broken)
    assert pickle.dumps(left.state_dict()) == before_state
    with pytest.raises(ValueError, match="action_id"): source.add(Transition(before, True, 0., terminal, True, config.semantic_versions()))
    with pytest.raises(ValueError, match="不合法"): source.add(Transition(before, len(spec), 0., terminal, True, config.semantic_versions()))


def test_agent_exact_rng_rejects_cross_device_but_inference_load_allows_it():
    config, spec = Config(), ActionSpec(Config())
    observation = _observation(config, spec)
    agent = DiscreteSAC(config, spec, observation)
    state = agent.state_dict(); state["device_type"] = "cuda"
    with pytest.raises(ValueError, match="跨设备"): agent.load_state_dict(state, restore_rng=True)
    agent.load_state_dict(state, restore_rng=False)
