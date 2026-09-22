"""覆盖实体契约：稳定 ID、真实实体与正需求分别保存。"""
from dataclasses import dataclass, fields
from copy import deepcopy
import numpy as np


ARRAY_DTYPES = {
    "beam_id": np.int64, "latitude_deg": np.float64, "longitude_deg": np.float64,
    "demand_bps": np.float64, "ground_diameter_deg": np.float64,
    "tx_gain_peak_dbi": np.float64, "rx_gain_peak_dbi": np.float64,
    "noise_temperature_k": np.float64, "group_id": np.int64,
    "polarization_id": np.int64, "entity_mask": bool, "demand_mask": bool,
    "service_order": np.int64,
}


@dataclass
class CoverageScenario:
    scenario_id: str
    source_hash: str
    source_schema: str
    beam_id: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    demand_bps: np.ndarray
    ground_diameter_deg: np.ndarray
    tx_gain_peak_dbi: np.ndarray
    rx_gain_peak_dbi: np.ndarray
    noise_temperature_k: np.ndarray
    group_id: np.ndarray
    polarization_id: np.ndarray
    entity_mask: np.ndarray
    demand_mask: np.ndarray
    service_order: np.ndarray
    metadata: dict

    def __post_init__(self):
        for name, dtype in ARRAY_DTYPES.items():
            raw = np.asarray(getattr(self, name))
            if dtype == bool and raw.size and (raw.dtype.kind != "b" and not np.isin(raw, [0, 1]).all()):
                raise ValueError(f"{name} 必须为布尔值")
            if dtype == np.int64 and raw.dtype.kind == "b":
                raise ValueError(f"{name} 必须为整数而非布尔值")
            if dtype == np.int64 and raw.size and (not np.issubdtype(raw.dtype, np.integer)):
                if not np.isfinite(raw.astype(float)).all() or not np.equal(raw.astype(float), np.floor(raw.astype(float))).all():
                    raise ValueError(f"{name} 必须为整数")
            value = np.array(raw, dtype=dtype, copy=True)
            value.setflags(write=False)
            setattr(self, name, value)
        self.metadata = deepcopy(self.metadata)
        self.validate()

    @property
    def n_demand(self):
        return int(self.demand_mask.sum())

    @property
    def n_entities(self):
        return int(self.entity_mask.sum())

    def validate(self, num_groups=None):
        size = len(self.beam_id)
        for name in ARRAY_DTYPES:
            value = getattr(self, name)
            if value.ndim != 1 or (name != "service_order" and len(value) != size):
                raise ValueError(f"{name} 的实体轴不一致")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} 含非有限值")
        if len(np.unique(self.beam_id)) != size:
            raise ValueError("beam_id 必须唯一")
        if np.any(np.abs(self.latitude_deg) > 90) or np.any(np.abs(self.longitude_deg) > 180):
            raise ValueError("坐标超出地理范围；必须显式选择 source_schema")
        if np.any(self.demand_bps < 0) or np.any(self.ground_diameter_deg < 0) or np.any(self.ground_diameter_deg > 180):
            raise ValueError("需求非负，地面角直径应在 0–180 度内")
        if np.any(self.demand_bps[~self.entity_mask] != 0):
            raise ValueError("padding 不能含真实需求")
        expected = self.entity_mask & (self.demand_bps > 0)
        if not np.array_equal(expected, self.demand_mask):
            raise ValueError("demand_mask 必须等于实体且正需求")
        if np.any(self.ground_diameter_deg[self.entity_mask] <= 0):
            raise ValueError("真实实体必须具有正宽度")
        if np.any(self.noise_temperature_k[self.entity_mask] <= 0):
            raise ValueError("噪声温度必须为正")
        if np.any(self.group_id < 0) or (num_groups is not None and np.any(self.group_id >= num_groups)):
            raise ValueError("group_id 越界")
        if np.any(self.polarization_id < 0):
            raise ValueError("polarization_id 必须非负")
        if sorted(self.service_order.tolist()) != sorted(self.beam_id[self.demand_mask].tolist()):
            raise ValueError("service_order 必须恰好遍历每个正需求 beam_id 一次")
        return self

    def to_dict(self):
        return {f.name: getattr(self, f.name).tolist() if f.name in ARRAY_DTYPES else deepcopy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
