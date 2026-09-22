"""第一阶段root/candidate接口、需求守恒、域审计与可恢复采样。

不包含臆造的覆盖生成器；缺少真实root/assignment时不生成J_root。
"""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
from pathlib import Path
import uuid
from typing import Protocol
import numpy as np
from project_paths import SPECTRUM_DATA_DIR, COVER_OUTPUT_DIR, SPECTRUM_REPORT_DIR
from .split import legacy_manifest, validate_split, manifest_hash


@dataclass
class RootScene:
    root_scene_id: str
    split_group_id: str
    generator_family_id: str
    demand_id: np.ndarray
    demand_location: np.ndarray  # latitude, longitude degrees
    offered_demand_bps: np.ndarray
    service_area: dict
    satellite_geometry: dict
    demand_generator_version: str
    seed: int
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if not all((self.root_scene_id, self.split_group_id, self.generator_family_id, self.demand_generator_version)):
            raise ValueError("root身份、谱系与生成版本不能为空")
        ids = np.asarray(self.demand_id)
        loc = np.asarray(self.demand_location, dtype=np.float64)
        demand = np.asarray(self.offered_demand_bps, dtype=np.float64)
        if ids.ndim != 1 or demand.shape != ids.shape or loc.shape != (len(ids), 2):
            raise ValueError("root业务ID/坐标/需求维度不一致")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("重复业务ID")
        if not np.isfinite(loc).all() or not np.isfinite(demand).all() or (demand < 0).any():
            raise ValueError("root存在非法坐标/需求")
        if (np.abs(loc[:, 0]) > 90).any() or (np.abs(loc[:, 1]) > 180).any():
            raise ValueError("root坐标超物理范围")
        if not self.service_area or not self.satellite_geometry:
            raise ValueError("缺少服务区域或卫星几何来源")
        return self


@dataclass
class CoverageCandidate:
    root_scene_id: str
    coverage_candidate_id: str
    coverage_generator_version: str
    parameters_and_seed: dict
    beam_table: object
    assignment: np.ndarray
    uncovered_demand_bps: float
    source_quality_metrics: dict
    physics_profile_id: str
    provenance_status: str = "provided_upstream"

    def validate(self, root, config=None):
        root.validate()
        if self.root_scene_id != root.root_scene_id or not self.coverage_candidate_id or not self.coverage_generator_version:
            raise ValueError("候选身份与root不一致或生成版本缺失")
        self.beam_table.validate(config.physics.num_groups if config else 8)
        assignment = np.asarray(self.assignment, dtype=np.float64)
        demand = np.asarray(root.offered_demand_bps, dtype=np.float64)
        if assignment.shape != (len(demand), len(self.beam_table.beam_id)):
            raise ValueError("assignment轴必须对应原始业务ID与候选beam_id")
        if not np.isfinite(assignment).all() or (assignment < 0).any() or (assignment.sum(axis=1) > 1 + 1e-12).any():
            raise ValueError("业务归属负值/非有限/重复计数")
        if (assignment[:, ~self.beam_table.entity_mask] != 0).any():
            raise ValueError("业务不得归属padding")
        # 正归属必须落在声明的球面覆盖轮廓中，不能只让数字守恒而几何不合法。
        locations = np.deg2rad(np.asarray(root.demand_location))
        lat = np.deg2rad(self.beam_table.latitude_deg)
        lon = np.deg2rad(self.beam_table.longitude_deg)
        delta_lat = locations[:,0,None] - lat[None,:]
        delta_lon = locations[:,1,None] - lon[None,:]
        hav = np.sin(delta_lat/2)**2 + np.cos(locations[:,0,None])*np.cos(lat[None,:])*np.sin(delta_lon/2)**2
        separation = np.rad2deg(2*np.arcsin(np.sqrt(np.clip(hav,0,1))))
        if np.any((assignment>0) & (separation > self.beam_table.ground_diameter_deg[None,:]/2 + 1e-8)):
            raise ValueError("业务assignment超出候选声明的模型覆盖范围")
        expected = assignment.T @ demand
        uncovered = float((1 - assignment.sum(axis=1)) @ demand)
        if not np.allclose(expected, self.beam_table.demand_bps, rtol=1e-8, atol=1e-5):
            raise ValueError("聚合波束需求与原始业务归属不守恒")
        if not math.isfinite(self.uncovered_demand_bps) or not math.isclose(uncovered, self.uncovered_demand_bps, rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError("未覆盖业务量不守恒")
        if not math.isclose(float(expected.sum() + uncovered), float(demand.sum()), rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError("业务总量不守恒")
        return {"root_total_demand_bps": float(demand.sum()), "covered_demand_bps": float(expected.sum()),
                "uncovered_demand_bps": uncovered, "conservation_passed": True}


class CoverageGenerator(Protocol):
    """上游必须实现并验证几何/业务归属，第二阶段只消费合法候选。"""
    version: str
    def generate(self, root: RootScene, parameters: dict, seed: int) -> list[CoverageCandidate]: ...


def generate_candidates(root, generator=None, parameters=None, seed=0, config=None):
    if generator is None:
        raise RuntimeError("缺少已验证的第一阶段覆盖生成器；不能随机坐标凑规模")
    candidates = generator.generate(root.validate(), parameters or {}, seed)
    seen = set()
    for candidate in candidates:
        candidate.validate(root, config)
        if candidate.coverage_candidate_id in seen:
            raise ValueError("同root候选ID重复")
        seen.add(candidate.coverage_candidate_id)
    return candidates


def root_metric(root, candidate, rates_bps):
    if root is None or candidate is None:
        return {"end_to_end_metric_available": False, "J_root": None,
                "reason": "缺少原始业务和assignment"}
    totals = candidate.validate(root)
    rates = np.asarray(rates_bps, dtype=np.float64)
    if rates.shape != candidate.beam_table.demand_bps.shape or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("速率数组不合法")
    denominator = totals["root_total_demand_bps"]
    return {"end_to_end_metric_available": denominator > 0,
            "J_root": float(np.minimum(rates, candidate.beam_table.demand_bps).sum() / denominator) if denominator > 0 else None,
            **totals, "interpretation": "聚合可交付业务代理，非逐用户调度结果"}


def augment_root_candidates(root, candidates, rng, scale_range=(.8, 1.2), noise_std=.02):
    """所有同root候选共享相同业务增强；几何和assignment保持已验证值。"""
    root.validate()
    factor = float(rng.uniform(*scale_range))
    noise = np.maximum(1 + rng.normal(0, noise_std, len(root.demand_id)), 1e-8)
    new_root = copy.deepcopy(root)
    new_root.offered_demand_bps = np.asarray(root.offered_demand_bps) * factor * noise
    augmented_id = hashlib.sha256(new_root.offered_demand_bps.tobytes()).hexdigest()
    new_root.metadata.update(augmented_root_id=augmented_id, demand_scale=factor, noise_std=noise_std)
    results = []
    for candidate in candidates:
        candidate.validate(root)
        new = copy.deepcopy(candidate)
        data = new.beam_table.to_dict()
        data["demand_bps"] = (new.assignment.T @ new_root.offered_demand_bps).tolist()
        data["demand_mask"] = (np.asarray(data["demand_bps"]) > 0).tolist()
        # Stable mapping remains; service order is explicitly recomputed by demand + beam ID.
        order = sorted(np.flatnonzero(data["demand_mask"]), key=lambda i: (-data["demand_bps"][i], data["beam_id"][i]))
        data["service_order"] = [data["beam_id"][i] for i in order]
        data["metadata"]["augmented_root_id"] = augmented_id
        new.beam_table = type(new.beam_table).from_dict(data)
        new.uncovered_demand_bps = float((1 - new.assignment.sum(axis=1)) @ new_root.offered_demand_bps)
        new.validate(new_root)
        results.append(new)
    return new_root, results


def coverage_spec():
    bins = [[48, 80], [81, 120], [121, 168], [169, 220], [221, 256]]
    cells = [{"id": f"scale_{a}_{b}", "conditions": {"n_demand": [a, b]}} for a, b in bins]
    cells += [{"id": f"spatial_{shape}", "conditions": {"spatial_shape": shape}}
              for shape in ("uniform", "single_hotspot", "multiple_hotspots", "long_tail")]
    cells += [{"id": f"scale_{a}_{b}_load_{load}", "conditions": {"n_demand": [a, b], "load_band": load}}
              for a, b in bins for load in ("low", "medium", "high")]
    cells += [{"id": "large_high_tight", "conditions": {"n_demand": [169, 256], "load_band": "high", "pressure_tag": "tight"}},
              {"id": "dense_reuse_weak", "conditions": {"overlap_band": "dense", "reuse_band": "same_polarization", "link_band": "weak"}},
              {"id": "longtail_wide_spectrum", "conditions": {"spatial_shape": "long_tail", "width_band": "heterogeneous", "spectrum_pressure": "tight"}}]
    return {"dataset_version": "coverage_domain.v1", "spec_status": "provisional_pending_upstream_feasibility",
            "root_definition": "upstream_business_instance", "support_domain": {"positive_beam_count": [48, 256],
            "geometry_profile_ids": [], "main_power_budget_w": 6000, "deployment_confirmed": False},
            "axes_and_bins": {"n_demand": bins, "spatial_shape": ["uniform", "single_hotspot", "multiple_hotspots", "long_tail"],
                              "load_band": {"values": ["low", "medium", "high"], "threshold_status": "pending_business_spec_or_train_only_fit"}},
            "feasible_combination_rules": [], "mandatory_cells": cells,
            "independent_roots_min": {"train": 20, "validation": 10, "test": 10},
            "required_cell_coverage": 1.0, "split_unit": "split_group_id",
            "sampling_protocol_version": "root_mixture.v1", "mixture": [.7, .2, .1],
            "holdout_sizes_provisional": {"validation_unseen": [104, 184, 240], "test_unseen": [120, 170, 220]},
            "rule": "不能使用验证/测试拟合负荷阈值；额外未见尺寸与范围内共同锚点评估分别统计"}


def in_cell(record, cell):
    for key, expected in cell["conditions"].items():
        actual = record.get(key)
        if actual is None:
            return False
        if isinstance(expected, list):
            if not expected[0] <= actual <= expected[1]:
                return False
        elif actual != expected:
            return False
    return True


def coverage_audit(records, spec):
    validate_split(records)
    result = []
    for cell in spec["mandatory_cells"]:
        matching = [r for r in records if in_cell(r, cell)]
        roots = {s: len({r["root_scene_id"] for r in matching if r.get("root_scene_id") and r["split"] == s})
                 for s in ("train", "validation", "test")}
        candidates_by_split = {s: sum(r["split"] == s for r in matching) for s in ("train", "validation", "test")}
        sizes_by_split = {s: sorted({int(r["n_demand"]) for r in matching if r["split"] == s and r.get("n_demand") is not None})
                          for s in ("train", "validation", "test")}
        passed = all(roots[s] >= spec["independent_roots_min"][s] for s in roots)
        result.append({**cell, "independent_roots": roots, "candidate_count": len(matching), "candidates_by_split": candidates_by_split,
                       "observed_n_demand_by_split": sizes_by_split, "passed": passed,
                       "missing_reasons": [] if passed else ["独立root配额不足或来源未知"]})
    return {"dataset_version": spec["dataset_version"], "spec_status": spec["spec_status"],
            "cells": result, "cell_coverage_fraction": sum(r["passed"] for r in result) / len(result) if result else None,
            "strict_acceptance_passed": bool(result) and all(r["passed"] for r in result) and spec["support_domain"]["deployment_confirmed"],
            "unknown_provenance_candidates": sum(not r.get("root_scene_id") for r in records),
            "support_domain": spec["support_domain"], "collected_samples": 0, "updated_samples": 0}


def read_coverage_spec(directory=None):
    """只读取所选清单旁的规格；不把其他run的全局审计当本run证据。"""
    path = Path(directory or SPECTRUM_DATA_DIR) / "训练场景覆盖规格.yaml"
    if not path.is_file():
        return coverage_spec(), "implementation_default_missing_spec_sidecar"
    text = path.read_text(encoding="utf-8-sig")
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        spec = yaml.safe_load(text)
    if not isinstance(spec, dict) or not isinstance(spec.get("mandatory_cells"), list):
        raise ValueError("训练场景覆盖规格缺少mandatory_cells")
    ids = [cell.get("id") for cell in spec["mandatory_cells"]]
    if not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("覆盖单元ID必须非空且唯一")
    return spec, str(path.resolve())


def validate_domain_annotations(manifest, spec, *, require_annotations=False):
    """域标签只来自冻结字段；没有root身份的数据绝不提升为独立root。"""
    expected_hash = manifest.get("coverage_spec_hash")
    if require_annotations and not expected_hash:
        raise ValueError("默认冻结清单缺少coverage_spec_hash；先执行 data.domain 更新域审计，旧恢复显式沿用 --dataset")
    if expected_hash and expected_hash != manifest_hash(spec):
        raise ValueError("覆盖规格hash与冻结数据清单不一致")
    for record in manifest["records"]:
        if str(record.get("provenance_status", "")).startswith("unknown_legacy") and (record.get("root_scene_id") or record.get("split_group_id")):
            raise ValueError("未知legacy来源不得声明未经确认的root_scene_id或split_group_id")
        labels = record.get("domain_cells")
        if labels is None:
            if require_annotations:
                raise ValueError("默认冻结清单缺少domain_cells；请重新执行数据审计")
            continue  # 显式旧清单不被自动补标签；维持旧采样和checkpoint语义。
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("domain_cells必须为不重复的字符串列表")
        expected = {cell["id"] for cell in spec["mandatory_cells"] if in_cell(record, cell)}
        if require_annotations and set(labels) != expected:
            raise ValueError(f"域标签与冻结场景字段不一致：{record.get('scenario_id')}")
    return True


def freeze_run_coverage(run_dir, manifest, config, manifest_path=None):
    """按本次清单重算并冻结审计；副本不随工作区后续D0更新而改变。"""
    from ..artifacts import atomic_json
    from ..audit import sha256
    directory = Path(manifest_path).resolve().parent if manifest_path else SPECTRUM_DATA_DIR
    spec, origin = read_coverage_spec(directory)
    validate_domain_annotations(manifest, spec, require_annotations=manifest_path is None)
    audit = coverage_audit(manifest["records"], spec)
    audit.update(dataset_version=manifest["dataset_version"], coverage_spec_version=spec["dataset_version"],
                 source_manifest_hash=manifest["manifest_hash"], coverage_spec_hash=manifest_hash(spec),
                 coverage_spec_origin=origin, audit_origin="recomputed_from_exact_frozen_run_manifest",
                 input_domain_labels={"labeled_candidates": sum(bool(r.get("domain_cells")) for r in manifest["records"]),
                                      "unlabeled_candidates": sum(not r.get("domain_cells") for r in manifest["records"])},
                 collected_samples=None, updated_samples=None,
                 sampling_counts={"status": "static_inventory_see_run_counters", "collected_source": "checkpoint.sampler.collected_by_domain_cell / 短训练结果.json sampler",
                                  "updated_source": "optimizer diagnostics replay.sampled_by_domain_cell / checkpoint.replay",
                                  "counting_rule": "多标签单元分别计数；同一采集或回放样本可贡献多个单元，不相加冒充总样本数"},
                 runtime_resource_context={"power_budget_w": config.env.power_budget_w,
                                           "max_power_w": max(config.env.power_levels_w), "num_slots": config.physics.num_slots},
                 runtime_profile_matches_spec=config.env.power_budget_w == spec["support_domain"].get("main_power_budget_w"))
    if not audit["runtime_profile_matches_spec"]:
        audit["strict_acceptance_passed"] = False
    run_dir = Path(run_dir)
    result = {"source_manifest_hash": manifest["manifest_hash"], "dataset_version": manifest["dataset_version"],
              "scope": "immutable_run_local", "artifacts": {}}
    for key, filename, payload in (("spec", "训练场景覆盖规格.yaml", spec), ("audit", "训练场景覆盖审计.json", audit)):
        path = run_dir / filename
        atomic_json(path, payload)
        result["artifacts"][key] = {"path": filename, "sha256": sha256(path), "artifact_id": uuid.uuid4().hex,
                                     "kind": "coverage_spec" if key == "spec" else "coverage_audit"}
    return result


class DomainSampler:
    """先选采样分支/单元，再root均匀、candidate均匀；不按候选数量加权root。"""
    def __init__(self, records, seed=42, warmup_draws=0, mixture=(.7, .2, .1)):
        self.records = [dict(r) for r in records if r["split"] == "train"]
        if not self.records:
            raise ValueError("训练split为空")
        if len(mixture) != 3 or any(x < 0 for x in mixture) or not math.isclose(sum(mixture), 1):
            raise ValueError("采样混合比例需非负且和为1")
        self.rng = np.random.default_rng(seed)
        self.warmup_draws, self.mixture = warmup_draws, mixture
        self.draws = 0
        self.counts = {}
        self.branch_counts = {}

    def sample(self):
        branch = "balanced" if self.draws < self.warmup_draws else self.rng.choice(["deployment", "rare", "boundary"], p=self.mixture)
        pool = self.records
        if branch == "boundary":
            pool = [r for r in pool if r.get("boundary", False)] or pool
        elif branch in ("rare", "balanced"):
            cells = sorted({cell for r in pool for cell in (r.get("domain_cells") or ["unknown"])})
            weights = np.array([1 / (1 + self.counts.get("cell:" + cell, 0)) for cell in cells]) if branch == "rare" else np.ones(len(cells))
            cell = self.rng.choice(cells, p=weights / weights.sum())
            pool = [r for r in pool if cell in (r.get("domain_cells") or ["unknown"])]
        groups = {}
        for record in pool:
            identity = record.get("root_scene_id") or record["source_hash"]
            groups.setdefault(identity, []).append(record)
        # Supplied deployment weights are root-level; absent weights mean explicitly uniform roots.
        keys = sorted(groups)
        weights = np.array([float(groups[k][0].get("deployment_root_weight", 1.0)) for k in keys]) if branch == "deployment" else np.ones(len(keys))
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("非法部署root权重")
        key = self.rng.choice(keys, p=weights / weights.sum())
        selected = groups[key][int(self.rng.integers(len(groups[key])))]
        self.counts[key] = self.counts.get(key, 0) + 1
        for cell in (selected.get("domain_cells") or ["unknown"]):
            self.counts["cell:" + cell] = self.counts.get("cell:" + cell, 0) + 1
        self.branch_counts[branch] = self.branch_counts.get(branch, 0) + 1
        self.draws += 1
        return selected

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "draws": self.draws,
                "counts": copy.deepcopy(self.counts), "branch_counts": copy.deepcopy(self.branch_counts),
                "collected_by_domain_cell": {key[5:]: value for key, value in self.counts.items() if key.startswith("cell:")},
                "source_identity_rule": "declared_root_scene_id_else_source_hash_not_independent_root",
                "domain_counting_rule": "multi_label_episode_memberships",
                "sampling_protocol": {"version": "root_mixture.v1", "warmup_draws": self.warmup_draws, "mixture": list(self.mixture)}}

    def load_state_dict(self, state):
        if state.get("sampling_protocol") is not None and state["sampling_protocol"] != self.state_dict()["sampling_protocol"]:
            raise ValueError("恢复采样协议与当前warmup/mixture不同")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.draws, self.counts, self.branch_counts = state["draws"], copy.deepcopy(state["counts"]), copy.deepcopy(state["branch_counts"])


def write_legacy_coverage():
    import csv
    manifest = legacy_manifest(COVER_OUTPUT_DIR.glob("cover_output_*.csv"))
    spec = coverage_spec()
    manifest.update(coverage_spec_hash=manifest_hash(spec), coverage_annotation_version="legacy_count_cells.v1")
    # Semantic duplicate keys are exact numeric equality; near-duplicate audit uses six-decimal canonical rows.
    near_groups = {}
    for record in manifest["records"]:
        with (COVER_OUTPUT_DIR / record["file"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        n = sum(float(r["rate"]) > 0 and float(r["beamwidth"]) > 0 for r in rows)
        record.update(n_demand=n, pressure_ratio=6000 / (n * 50) if n else None)
        record["domain_cells"] = [cell["id"] for cell in spec["mandatory_cells"] if in_cell(record, cell)]
        rounded = sorted(tuple(round(float(r[k]), 6) for k in ("lat", "lon", "rate", "beamwidth")) for r in rows)
        near_hash = hashlib.sha256(json.dumps(rounded).encode()).hexdigest()
        near_groups.setdefault(near_hash, []).append(record["scenario_id"])
    # This limited numerical audit cannot establish geographic independence.
    manifest["manifest_hash"] = manifest_hash(manifest)
    audit = coverage_audit(manifest["records"], spec)
    audit.update(dataset_version=manifest["dataset_version"], coverage_spec_version=spec["dataset_version"],
                 source_manifest_hash=manifest["manifest_hash"], coverage_spec_hash=manifest_hash(spec))
    audit["near_duplicate_audit"] = {"method": "row_order_invariant_round6_all_four_fields", "precision": 6,
                                    "groups": [v for v in near_groups.values() if len(v) > 1],
                                    "limitation": "不替代空间相似性/已知生成谱系审计，无法确认真实root独立"}
    audit["available_n_demand"] = sorted({r["n_demand"] for r in manifest["records"]})
    audit["source_generator_available"] = False
    SPECTRUM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("训练场景覆盖规格.yaml", spec), ("训练场景清单.json", manifest),
                              ("训练场景覆盖审计.json", audit),
                              ("场景缺口与补样清单.json", {"missing_cells": [x for x in audit["cells"] if not x["passed"]],
                              "required_inputs": ["第一阶段覆盖生成器及可行域规格", "原始业务RootScene", "业务到波束assignment及漏覆盖量", "真实root/split_group生成谱系"]})):
        (SPECTRUM_DATA_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SPECTRUM_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (SPECTRUM_REPORT_DIR / "训练场景覆盖审计.md").write_text(
        "# 训练场景覆盖审计\n\nD0–D3接口已建设，正式广覆盖数据验收尚未通过。\n\n"
        f"现有100个聚合CSV，正需求范围52–168，hash划分70/15/15；未知root来源100个。严格独立root必覆盖单元达标率为{audit['cell_coverage_fraction']:.0%}。"
        "48–256为候选建设域，未获第一阶段可行性确认。数据/规格/缺口位于PPO4090/data/spectrum_tsac_20260921。\n\n"
        "接口验证包括同root多候选守恒、重复业务拒绝、跨split传递闭包、root层增强共享、端到端共同分母、采样恢复。合成fixture只用于测试，不计入生产训练覆盖。\n\n"
        "近重复审计仅对四字段六位小数、忽略行顺序做一致性检查；不能证明地理或母场景独立。后续需上游谱系、部署参数及验证过的生成器，再冻结完整轴/二阶/三阶可行单元。不得删除困难单元来提高覆盖率。\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    result = write_legacy_coverage()
    print(json.dumps({"strict_acceptance_passed": result["strict_acceptance_passed"],
                      "cell_coverage_fraction": result["cell_coverage_fraction"],
                      "unknown_provenance_candidates": result["unknown_provenance_candidates"]}, ensure_ascii=False))
