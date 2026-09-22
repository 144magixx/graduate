"""从同一新物理账本构造两种具名205维观察。"""
import numpy as np


MODERN_VERSION = "modern205.v1"
LEGACY_SCALED_VERSION = "legacy_scaled205_new_physics.v1"


def _validate_ledger_view(view):
    required = ("latitude_deg", "longitude_deg", "demand_bps", "ground_diameter_deg",
                "remaining_power_w", "interference_w", "noise_w", "occupancy")
    missing = [name for name in required if name not in view]
    if missing:
        raise ValueError(f"观察账本缺少字段：{missing}")
    scalars = np.asarray([view["latitude_deg"], view["longitude_deg"], view["demand_bps"],
                          view["ground_diameter_deg"], view["remaining_power_w"]], dtype=np.float64)
    if not np.isfinite(scalars).all() or scalars[2] < 0 or scalars[3] < 0 or scalars[4] < 0:
        raise ValueError("观察账本标量必须有限且需求、宽度、剩余功率非负")
    if abs(scalars[0]) > 90 or abs(scalars[1]) > 180:
        raise ValueError("观察账本经纬度超物理范围")
    interference = np.asarray(view["interference_w"], dtype=np.float64)
    noise = np.asarray(view["noise_w"], dtype=np.float64)
    occupancy = np.asarray(view["occupancy"], dtype=bool)
    if interference.shape != (100,) or noise.shape != (100,) or occupancy.shape != (100,):
        raise ValueError("Anchor观察固定需要100槽")
    if (interference < 0).any() or (noise <= 0).any() or not np.isfinite(interference).all() or not np.isfinite(noise).all():
        raise ValueError("新物理账本功率必须有限且噪声为正")
    return interference, noise, occupancy


def modern205(view):
    interference, noise, occupancy = _validate_ledger_view(view)
    result = np.empty(205, dtype=np.float32)
    result[:5] = (float(view["latitude_deg"]) / 90., float(view["longitude_deg"]) / 180.,
                  float(view["demand_bps"]) / 1e9, float(view["ground_diameter_deg"]),
                  float(view["remaining_power_w"]) / 6000.)
    result[5:105] = np.log1p(interference / noise)
    result[105:] = occupancy
    return result


def legacy_scaled205(view):
    """恢复旧数值尺度；interference仍来自统一重算的新物理账本。"""
    interference, _noise, occupancy = _validate_ledger_view(view)
    result = np.empty(205, dtype=np.float32)
    result[:5] = (2. * (float(view["latitude_deg"]) - 29.) / 50.,
                  2. * (float(view["longitude_deg"]) - 104.) / 62.,
                  float(view["demand_bps"]) / 500_000_000.,
                  np.clip(float(view["ground_diameter_deg"]) / 2., 0., 1.),
                  float(view["remaining_power_w"]) / 6000.)
    interference_db = 10. * np.log10(interference + 1e-30)
    result[5:105] = np.clip((interference_db + 300.) / 200., 0., 1.)
    result[105:] = occupancy
    return result


def adapt_observation(adapter, view):
    if adapter == "modern205":
        return modern205(view), MODERN_VERSION
    if adapter == "legacy_scaled205":
        return legacy_scaled205(view), LEGACY_SCALED_VERSION
    raise ValueError(f"未知观察适配器：{adapter}")
