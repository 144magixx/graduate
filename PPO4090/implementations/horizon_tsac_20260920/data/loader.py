"""显式旧 CSV 适配与标准 coverage.v2 加载；绝不截断或重复实体。"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .schema import CoverageScenario
from ..config import Config


def _bool(value):
    if isinstance(value, str):
        if value.lower() not in ("true", "false", "1", "0"):
            raise ValueError("entity_mask 必须为布尔值")
        return value.lower() in ("true", "1")
    if value not in (True, False, 1, 0):
        raise ValueError("entity_mask 必须为布尔值")
    return bool(value)


def scenario_from_rows(rows, config=None, source_schema=None, scenario_id="scenario", source_hash="", metadata=None, rng=None):
    config = config or Config()
    config.validate()
    source_schema = source_schema or config.data.source_schema
    if source_schema not in ("legacy_lon_in_lat", "coverage.v2", "standard"):
        raise ValueError(f"未知 source_schema: {source_schema}")
    legacy = source_schema == "legacy_lon_in_lat"
    rows = list(rows)
    n = len(rows)
    col = lambda name, default=None: np.asarray([r[name] if name in r else default for r in rows], dtype=np.float64)
    beam_id = col("beam_id") if rows and "beam_id" in rows[0] else np.arange(n, dtype=np.int64)
    lat, lon = (col("lon"), col("lat")) if legacy else (col("latitude_deg"), col("longitude_deg"))
    if legacy:
        if config.data.rate_unit != "Mbps":
            raise ValueError("旧 rate 数据必须显式使用 Mbps 单位")
        raw_rate, width = col("rate"), col("beamwidth")
        entity = ~((raw_rate == 0) & (width == 0))
        demand = raw_rate * 1e6 * config.data.traffic_scale
    else:
        demand, width = col("demand_bps"), col("ground_diameter_deg")
        entity = np.asarray([_bool(r.get("entity_mask", True)) for r in rows], dtype=bool)
    positive = entity & (demand > 0)
    canonical = np.lexsort((beam_id, -demand))
    canonical = canonical[positive[canonical]]
    groups = np.zeros(n, dtype=np.int64)
    groups[canonical] = np.arange(len(canonical)) % config.physics.num_groups
    supplied = bool(rows) and all("group_id" in r for r in rows)
    if supplied:
        groups = col("group_id")
    elif any("group_id" in r for r in rows):
        raise ValueError("group_id 不允许部分缺失")
    pol = col("polarization_id") if rows and all("polarization_id" in r for r in rows) else groups % 2
    if config.data.service_order == "demand_desc":
        order = canonical
    elif config.data.service_order == "raw":
        order = np.flatnonzero(positive)
    elif config.data.service_order == "random":
        if rng is None:
            raise ValueError("随机服务顺序需要显式局部 rng")
        order = rng.permutation(np.flatnonzero(positive))
    else:
        raise ValueError("未知 service_order")
    meta = dict(metadata or {})
    meta.update(rate_unit="bps", raw_rate_unit="Mbps" if legacy else "bps", traffic_scale=config.data.traffic_scale if legacy else 1.0,
                beamwidth_kind=config.data.beamwidth_kind, group_assignment="source" if supplied else config.data.group_assignment,
                service_order=config.data.service_order, provenance_status=meta.get("provenance_status", "unknown_legacy" if legacy else "declared"),
                template_rows=n)
    scenario = CoverageScenario(scenario_id, source_hash, source_schema, beam_id, lat, lon, demand, width,
                              col("tx_gain_peak_dbi", config.physics.tx_gain_peak_dbi), col("rx_gain_peak_dbi", config.physics.rx_gain_peak_dbi),
                              col("noise_temperature_k", config.physics.noise_temperature_k), groups, pol, entity, positive,
                              beam_id[order], meta)
    return scenario.validate(config.physics.num_groups)


def load_scenario(path, config=None, source_schema=None, rng=None):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix.lower() == ".json":
        return CoverageScenario.from_dict(json.loads(content.decode("utf-8-sig")))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return scenario_from_rows(rows, config, source_schema, path.stem, hashlib.sha256(content).hexdigest(), {"source_path": str(path.resolve())}, rng=rng)


load_legacy_csv = load_scenario
