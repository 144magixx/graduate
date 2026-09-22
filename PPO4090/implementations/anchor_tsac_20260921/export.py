"""第三阶段固定频谱/连续功率交接，NPZ无pickle并独立重载复算。"""
import argparse
import json
import os
from pathlib import Path
import uuid
import numpy as np
from .artifacts import atomic_json
from .audit import sha256


def export_allocation(env, directory, run_id="standalone"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scenario, config, result = env.scenario, env.config, env.evaluate()
    if result.constraint_violations:
        raise ValueError("违反硬约束的方案不能作为第三阶段交接")
    obs = env.observe()
    arrays = {"beam_id": scenario.beam_id, "entity_mask":scenario.entity_mask, "demand_mask":scenario.demand_mask,
              "status":obs["status"], "group_id":scenario.group_id, "polarization_id":scenario.polarization_id,
              "X":result.slot_power_w > 0, "p_total_w":result.slot_power_w.sum(1), "p_slot_w":result.slot_power_w,
              "demand_bps":scenario.demand_bps, "ground_diameter_deg":scenario.ground_diameter_deg,
              "noise_temperature_k":scenario.noise_temperature_k,
              "frequency_hz":config.physics.frequency_start_hz + (np.arange(config.physics.num_slots)+.5)*config.physics.slot_bandwidth_hz,
              "rate_bps":result.rate_bps, "sinr_linear":result.sinr_linear}
    path = directory / "第二阶段分配结果.npz"
    temporary = path.with_name("分配." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    manifest = {"schema_version":"allocation.v1", "run_id":run_id, "scenario_id":scenario.scenario_id,
                "source_hash":scenario.source_hash, "scenario":scenario.to_dict(), "config":config.to_dict(),
                "versions":config.semantic_versions(), "file":path.name, "sha256":sha256(path),
                "metrics":result.metrics, "terminal":env.terminal, "U_definition":"positive_demand_mean_clipped_rate_ratio",
                "SGM_beta":result.metrics["sgm_beta_bps"], "SGM_beta_rule":"current_positive_demand_mean_bps",
                "stage3_objective":"显式选择SGM或U；精修后同时复算两者，U不下降需另设接受规则",
                "power_semantics":"beam_total_uniform_slots", "gain_source":"legacy_fixed_peak_approximation_unconfirmed"}
    atomic_json(directory / "第三阶段交接清单.json", manifest)
    verification = reload_and_evaluate(directory)
    atomic_json(directory / "导出复算验收.json", verification)
    return path, verification


def reload_and_evaluate(directory):
    from .config import Config
    from .data.schema import CoverageScenario
    from .env.state import Allocation, IDLE, PENDING, ALLOCATED, SKIPPED
    from .env.physics import evaluate_allocation
    directory = Path(directory)
    metadata = json.loads((directory / "第三阶段交接清单.json").read_text(encoding="utf-8"))
    path = directory / metadata["file"]
    if path.parent.resolve() != directory.resolve() or sha256(path) != metadata["sha256"]:
        raise ValueError("交接文件路径/hash不匹配")
    scenario = CoverageScenario.from_dict(metadata["scenario"])
    config = Config.from_dict(metadata["config"])
    ledger = {}
    with np.load(path, allow_pickle=False) as arrays:
        n, slots = len(scenario.beam_id),config.physics.num_slots
        matrix_keys=("X","p_slot_w","sinr_linear")
        vector_keys=("beam_id","entity_mask","demand_mask","status","group_id","polarization_id","p_total_w",
                     "demand_bps","ground_diameter_deg","noise_temperature_k","rate_bps")
        expected=set(matrix_keys+vector_keys+("frequency_hz",))
        if set(arrays.files)!=expected:
            raise ValueError("交接数组字段与schema不一致")
        for key in expected:
            value=arrays[key]
            shape=(n,slots) if key in matrix_keys else (slots,) if key=="frequency_hz" else (n,)
            if value.shape!=shape or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"交接数组{key}形状/类型/有限性非法")
        for key in ("beam_id","status","group_id","polarization_id"):
            if arrays[key].dtype.kind not in "iu":raise ValueError(f"{key}必须为整数数组")
        for key in ("entity_mask","demand_mask","X"):
            if arrays[key].dtype.kind!="b":raise ValueError(f"{key}必须为布尔数组")
        if not np.isin(arrays["status"],[IDLE,PENDING,ALLOCATED,SKIPPED]).all():
            raise ValueError("status包含未知枚举")
        if np.any(arrays["status"][~scenario.demand_mask]!=IDLE) or np.any(arrays["status"][scenario.demand_mask]==IDLE):
            raise ValueError("status与业务mask不一致")
        if metadata["terminal"] and np.any(arrays["status"]==PENDING):raise ValueError("终局含pending业务")
        expected_frequency=config.physics.frequency_start_hz+(np.arange(slots)+.5)*config.physics.slot_bandwidth_hz
        np.testing.assert_allclose(arrays["frequency_hz"],expected_frequency,rtol=0,atol=1e-5)
        for key in ("p_total_w","p_slot_w","rate_bps","sinr_linear"):
            if (arrays[key]<0).any():raise ValueError(f"{key}不得为负")
        for key in ("beam_id","entity_mask","demand_mask","group_id","polarization_id","demand_bps","ground_diameter_deg","noise_temperature_k"):
            np.testing.assert_array_equal(arrays[key],getattr(scenario,key))
        np.testing.assert_allclose(arrays["p_slot_w"].sum(1),arrays["p_total_w"],rtol=1e-12,atol=1e-12)
        for i, beam in enumerate(scenario.beam_id):
            selected = np.flatnonzero(arrays["X"][i])
            if arrays["status"][i] == ALLOCATED:
                if not len(selected) or not np.array_equal(selected,np.arange(selected[0],selected[-1]+1)):
                    raise ValueError("分配不是非空完整连续块")
                ledger[int(beam)] = Allocation(int(beam),"allocated",int(selected[0]),len(selected),float(arrays["p_total_w"][i]))
            elif arrays["status"][i] == SKIPPED:
                ledger[int(beam)] = Allocation(int(beam),"skipped",-1,0,0.)
            if arrays["status"][i] != ALLOCATED and (len(selected) or arrays["p_total_w"][i] != 0):
                raise ValueError("未分配束含资源")
        evaluation = evaluate_allocation(scenario,ledger,config)
        if evaluation.constraint_violations:
            raise ValueError("重载方案违反约束")
        np.testing.assert_allclose(evaluation.slot_power_w,arrays["p_slot_w"],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(evaluation.rate_bps,arrays["rate_bps"],rtol=1e-8,atol=1e-5)
        np.testing.assert_allclose(evaluation.sinr_linear,arrays["sinr_linear"],rtol=1e-8,atol=1e-12)
    return {"passed":True,"schema_version":"allocation.v1","sha256":metadata["sha256"],"allow_pickle":False,
            "power_conservation":True,"rate_recomputation":True,"constraint_violation_count":0,"metrics":evaluation.metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-experiment", action="store_true", help="授权从模型执行真实场景评价")
    args = parser.parse_args()
    directory = Path(args.manifest).parent
    if args.verify_only:
        result = reload_and_evaluate(directory)
    else:
        if not args.allow_experiment:
            raise RuntimeError("导出前的真实CSV rollout尚未授权；请显式使用 --allow-experiment")
        from .evaluate import evaluate_run
        result = evaluate_run(args.manifest, split="validation", limit=1, export_results=True, allow_experiment=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__": main()
