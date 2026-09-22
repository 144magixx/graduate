"""仅训练聚合 CSV 的需求增强。新 root 数据须在业务层增强。"""
import numpy as np
from .schema import CoverageScenario


def augment_scenario(scenario, rng, level="normal", split="train"):
    if split != "train":
        raise ValueError("验证和测试场景禁止增强")
    if scenario.metadata.get("root_scene_id") or scenario.metadata.get("assignment") is not None:
        raise ValueError("具有 root/assignment 的场景必须在 root 原始业务层增强")
    if level not in ("normal", "hard"):
        raise ValueError("增强 level 只能为 normal/hard")
    scale_range, noise_std = ((.8, 1.2), .02) if level == "normal" else ((.5, 1.6), .06)
    scale = float(rng.uniform(*scale_range))
    factors = 1.0 + rng.normal(0, noise_std, scenario.n_demand)
    while np.any(factors <= 0):
        bad = factors <= 0
        factors[bad] = 1.0 + rng.normal(0, noise_std, int(bad.sum()))
    value = scenario.to_dict()
    demand = scenario.demand_bps.copy()
    demand[scenario.demand_mask] *= scale * factors
    value["demand_bps"] = demand
    if scenario.metadata.get("service_order") == "demand_desc":
        order = np.lexsort((scenario.beam_id, -demand))
        value["service_order"] = scenario.beam_id[order[scenario.demand_mask[order]]]
    value["metadata"]["augmentation"] = {"version": "legacy_demand_only.v1", "level": level, "rate_scale": scale,
                                            "noise_std": noise_std, "factors": factors.tolist(), "width_augmented": False}
    return CoverageScenario.from_dict(value)
