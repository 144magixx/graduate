"""float64 物理纯函数；所有速率从线性槽功率和完整账本重算。"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from ..config import Config, PhysicsConfig
from .state import Allocation

BOLTZMANN = 1.380649e-23
LIGHT_SPEED = 299792458.0


def earth_points(latitude_deg, longitude_deg, radius_km):
    lat, lon = np.deg2rad(latitude_deg), np.deg2rad(longitude_deg)
    return radius_km * np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def ground_diameter_to_satellite_width(ground_diameter_deg, physics):
    """继承旧星下点圆锥换算近似；地面宽度不是星载天线角宽。"""
    theta = np.deg2rad(np.asarray(ground_diameter_deg, np.float64)) / 2
    r, orbital = physics.earth_radius_km, physics.satellite_radius_km
    slant = np.sqrt(orbital**2 + r**2 - 2*r*orbital*np.cos(theta))
    return 2*np.rad2deg(np.arcsin(np.clip(r*np.sin(theta)/slant, -1, 1)))


def antenna_gain_linear(peak_dbi, angle_deg, width_deg):
    """具名旧方向图近似（包括 log10(2) 与 1e-10 旁瓣底）。"""
    width = np.maximum(np.asarray(width_deg, np.float64), np.finfo(float).tiny)
    angle = np.abs(np.asarray(angle_deg, np.float64))
    peak = np.asarray(peak_dbi, np.float64)
    main = 10**((peak - 3*(angle/(width/2))**2)/10)
    sigma = width/(2*np.sqrt(2*np.log10(2)))
    side = np.exp(-angle**2/(2*sigma**2))*10**(peak/10) + 1e-10
    return np.where(angle <= width/2, main, side)


def channel_geometry(scenario, physics):
    """返回 C[j,i]/f_s² 的可分离耦合；Rx 增益必须取受害端 i。"""
    points = earth_points(scenario.latitude_deg, scenario.longitude_deg, physics.earth_radius_km)
    satellite = earth_points(np.array([physics.satellite_latitude_deg]), np.array([physics.satellite_longitude_deg]), physics.satellite_radius_km)[0]
    vectors = points-satellite
    slant_km = np.linalg.norm(vectors, axis=1)
    unit = vectors/slant_km[:, None]
    cos_angles = np.clip(unit @ unit.T, -1, 1)
    np.fill_diagonal(cos_angles, 1.0)
    angles = np.rad2deg(np.arccos(cos_angles))
    widths = ground_diameter_to_satellite_width(np.where(scenario.entity_mask, scenario.ground_diameter_deg, 1.0), physics)
    tx = antenna_gain_linear(scenario.tx_gain_peak_dbi[:, None], angles, widths[:, None])
    rx = 10**(scenario.rx_gain_peak_dbi/10)
    coupling = tx*rx[None, :]*(LIGHT_SPEED/(4*np.pi*slant_km[None, :]*1000))**2
    coupling *= scenario.entity_mask[:, None] & scenario.entity_mask[None, :]
    freq = physics.frequency_start_hz + (np.arange(physics.num_slots)+.5)*physics.slot_bandwidth_hz
    if not np.isfinite(coupling).all() or not np.isfinite(slant_km).all():
        raise FloatingPointError("信道几何/增益产生非有限值；拒绝该场景")
    return {"coupling": coupling, "frequency_factor": 1/freq**2, "slant_km": slant_km}


def coverage_polygon(latitude_deg, longitude_deg, ground_diameter_deg, vertices=36):
    """球面大圆轮廓，返回 GeoJSON 经度/纬度，不假定平面公里。"""
    lat, lon, radius = np.deg2rad([latitude_deg, longitude_deg, ground_diameter_deg/2])
    bearing = np.linspace(0, 2*np.pi, vertices+1)
    latitude = np.arcsin(np.sin(lat)*np.cos(radius)+np.cos(lat)*np.sin(radius)*np.cos(bearing))
    longitude = lon+np.arctan2(np.sin(bearing)*np.sin(radius)*np.cos(lat), np.cos(radius)-np.sin(lat)*np.sin(latitude))
    coords = np.column_stack(((np.rad2deg(longitude)+180)%360-180, np.rad2deg(latitude))).tolist()
    coords[-1] = coords[0][:]
    return {"type": "Polygon", "coordinates": [coords]}


@dataclass
class Evaluation:
    rate_bps: np.ndarray
    slot_power_w: np.ndarray
    signal_w: np.ndarray
    interference_w: np.ndarray
    noise_w: np.ndarray
    sinr_linear: np.ndarray
    occupancy: np.ndarray
    beam_ids: np.ndarray
    metrics: dict
    constraint_violations: list


def evaluate_allocation(scenario, ledger, config=None, *, channel_cache=None, power_mode="discrete"):
    """无 history_cinr /累计满意度；独立调用时总是重建几何与账本。"""
    config = config or Config()
    if power_mode not in ("discrete", "continuous"):
        raise ValueError("power_mode 必须显式为 discrete 或 continuous")
    if isinstance(config, PhysicsConfig):
        config = Config(physics=config)
    physics, env = config.physics, config.env
    scenario.validate(physics.num_groups)
    n, slots = len(scenario.beam_id), physics.num_slots
    power = np.zeros((n, slots), np.float64)
    occupancy = np.zeros((physics.num_groups, slots), bool)
    owners = np.full((physics.num_groups, slots), -1, np.int64)
    violations = []
    index = {int(b): i for i, b in enumerate(scenario.beam_id)}
    entries = ledger.values() if isinstance(ledger, dict) else ledger
    seen, used = set(), 0.0
    for entry in entries:
        a = Allocation(**entry) if isinstance(entry, dict) else entry
        if a.beam_id not in index or a.beam_id in seen:
            violations.append({"kind": "unknown_or_duplicate_beam", "beam_id": a.beam_id})
            continue
        seen.add(a.beam_id)
        i = index[a.beam_id]
        if isinstance(ledger, dict) and (a.beam_id not in ledger or ledger[a.beam_id] is not entry):
            violations.append({"kind": "ledger_key_mismatch", "beam_id": a.beam_id})
        if not scenario.demand_mask[i]:
            violations.append({"kind": "non_demand_beam_in_ledger", "beam_id": a.beam_id})
            continue
        if a.status == "skipped":
            if (a.start, a.length, a.power_total_w) != (-1, 0, 0):
                violations.append({"kind": "skip_has_resources", "beam_id": a.beam_id})
            continue
        if a.status != "allocated":
            violations.append({"kind": "invalid_status_or_demand", "beam_id": a.beam_id})
            continue
        if not isinstance(a.start, (int, np.integer)) or not isinstance(a.length, (int, np.integer)) or a.start < 0 or a.length < 1 or a.length > env.max_block_length or a.start+a.length > slots:
            violations.append({"kind": "invalid_contiguous_block", "beam_id": a.beam_id})
            continue
        if not np.isfinite(a.power_total_w) or a.power_total_w <= 0:
            violations.append({"kind": "invalid_power", "beam_id": a.beam_id})
            continue
        used += a.power_total_w
        if power_mode == "discrete" and not np.any(np.abs(np.asarray(env.power_levels_w)-a.power_total_w) <= env.budget_tolerance_w):
            violations.append({"kind": "invalid_power_level", "beam_id": a.beam_id, "power_total_w": float(a.power_total_w)})
        if a.power_total_w > max(env.power_levels_w) + env.budget_tolerance_w:
            violations.append({"kind": "per_beam_power", "beam_id": a.beam_id, "excess_w": float(a.power_total_w-max(env.power_levels_w))})
        group, interval = int(scenario.group_id[i]), slice(a.start, a.start+a.length)
        if occupancy[group, interval].any():
            violations.append({"kind": "same_pool_overlap", "beam_id": a.beam_id})
        occupancy[group, interval] = True
        owners[group, interval] = a.beam_id
        power[i, interval] = a.power_total_w/a.length
    if used > env.power_budget_w + env.budget_tolerance_w:
        violations.append({"kind": "total_power", "excess_w": float(used-env.power_budget_w)})
    channel = channel_cache if channel_cache is not None else channel_geometry(scenario, physics)
    coupling, factor = channel["coupling"], channel["frequency_factor"]
    diagonal = np.diag(coupling)
    signal = power * diagonal[:, None] * factor[None, :]
    interference_coupling = coupling * (scenario.polarization_id[:, None] == scenario.polarization_id[None, :])
    np.fill_diagonal(interference_coupling, 0.)
    interference = (interference_coupling.T @ power) * factor[None, :]
    noise = BOLTZMANN*scenario.noise_temperature_k[:, None]*physics.slot_bandwidth_hz
    noise = np.broadcast_to(noise, power.shape).copy()
    # padding 温度不参与计算，避免 schema 允许的 padding 0 K 导致 0/0。
    denominator = 10**(physics.sinr_margin_db/10)*(noise+interference)
    sinr = np.divide(signal, denominator, out=np.zeros_like(signal), where=denominator>0)
    rate = (physics.slot_bandwidth_hz*np.log1p(sinr)/np.log(2)).sum(axis=1)
    for name, value in (("slot_power_w", power), ("signal_w", signal), ("interference_w", interference),
                        ("noise_w", noise), ("sinr_linear", sinr), ("rate_bps", rate)):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"物理计算 {name} 出现非有限值；拒绝提交")
    result = Evaluation(rate, power, signal, interference, noise, sinr, occupancy, owners, {}, violations)
    from .metrics import compute_metrics
    result.metrics = compute_metrics(scenario, ledger, result, config)
    return result

