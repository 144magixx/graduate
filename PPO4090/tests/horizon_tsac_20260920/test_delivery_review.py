"""Independent delivery review: adversarial stage-three artifacts and path identities."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from implementations.horizon_tsac_20260920.artifacts import atomic_json
from implementations.horizon_tsac_20260920.audit import sha256
from implementations.horizon_tsac_20260920.data.loader import scenario_from_rows
from implementations.horizon_tsac_20260920.data.split import manifest_hash
from implementations.horizon_tsac_20260920.env.environment import Environment
from implementations.horizon_tsac_20260920.export import export_allocation, reload_and_evaluate
from implementations.horizon_tsac_20260920.train import run_training
from implementations.horizon_tsac_20260920.evaluate import evaluate_run
from tests.horizon_tsac_20260920.test_integration import tiny_config


@pytest.fixture
def exported(tmp_path):
    config = tiny_config()
    scenario = scenario_from_rows([dict(beam_id=7, latitude_deg=30., longitude_deg=110.,
        demand_bps=1e8, ground_diameter_deg=1.)], config, "coverage.v2", "review_fixture")
    env = Environment(config)
    env.reset(scenario)
    env.step(0)
    directory = tmp_path / "交接验收"
    export_allocation(env, directory)
    return directory


def mutate_archive(directory, mutation):
    metadata_path = directory / "第三阶段交接清单.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    array_path = directory / metadata["file"]
    with np.load(array_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    mutation(arrays)
    np.savez_compressed(array_path, **arrays)
    # Re-sign the container hash deliberately: the verifier must also check
    # semantic consistency, not merely transport integrity.
    metadata["sha256"] = sha256(array_path)
    atomic_json(metadata_path, metadata)


def test_resigned_wrong_frequency_grid_is_rejected(exported):
    mutate_archive(exported, lambda arrays: arrays["frequency_hz"].__iadd__(1e9))
    with pytest.raises(AssertionError):
        reload_and_evaluate(exported)


def test_resigned_unknown_status_is_rejected(exported):
    mutate_archive(exported, lambda arrays: arrays["status"].__setitem__(0, 99))
    with pytest.raises(ValueError, match="未知枚举"):
        reload_and_evaluate(exported)


@pytest.mark.parametrize("name,value", [("X", 1), ("rate_bps", float("nan"))])
def test_nonfinite_or_wrong_mask_dtype_is_rejected(exported, name, value):
    def change(arrays):
        if name == "X":
            arrays[name] = arrays[name].astype(np.int64)
        else:
            arrays[name][0] = value
    mutate_archive(exported, change)
    with pytest.raises(ValueError):
        reload_and_evaluate(exported)


def test_evaluation_scenario_identity_cannot_escape_output(tmp_path):
    config = tiny_config()
    config.train.episodes = 1
    records = []
    malicious_id = "../../../escaped-output"
    for split, identity in (("train", "train-fixture"), ("validation", malicious_id)):
        scenario = scenario_from_rows([dict(beam_id=5, latitude_deg=30., longitude_deg=110.,
            demand_bps=1e8, ground_diameter_deg=1.)], config, "coverage.v2", identity)
        source = tmp_path / f"{split}.json"
        atomic_json(source, scenario.to_dict())
        records.append({"scenario_id": identity, "scenario_path": str(source), "source_hash": sha256(source), "split": split})
    manifest = {"dataset_version": "review_fixture", "records": records}
    manifest["manifest_hash"] = manifest_hash(manifest)
    dataset = tmp_path / "验收数据划分.json"
    atomic_json(dataset, manifest)
    with patch("implementations.horizon_tsac_20260920.artifacts.HORIZON_OUTPUT_DIR", tmp_path / "runs"):
        run, _ = run_training(config, "review_fixture", dataset)
        result = evaluate_run(run / "运行清单.json", "validation", 1, ("random",))
    output = Path(result["output_dir"])
    exported = list(output.glob("random/场景-*/第三阶段交接清单.json"))
    assert len(exported) == 1
    assert json.loads(exported[0].read_text(encoding="utf-8"))["scenario_id"] == malicious_id
    assert not list(tmp_path.rglob("escaped-output"))
