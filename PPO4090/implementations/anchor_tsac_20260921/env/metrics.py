"""全场景正需求分母；无样本指标返回 None。"""
import numpy as np


def satisfaction(scenario, rate_bps):
    out = np.zeros(len(scenario.beam_id), np.float64)
    np.divide(rate_bps, scenario.demand_bps, out=out, where=scenario.demand_mask)
    return np.minimum(out, 1.)


def compute_metrics(scenario, ledger, evaluation, config):
    mask = scenario.demand_mask
    n = scenario.n_demand
    rates, demand = evaluation.rate_bps[mask], scenario.demand_bps[mask]
    utility = satisfaction(scenario, evaluation.rate_bps)[mask]
    entries = list(ledger.values()) if isinstance(ledger, dict) else list(ledger)
    get = lambda entry, key: entry.get(key) if isinstance(entry, dict) else getattr(entry, key)
    allocated = int(np.sum(evaluation.slot_power_w[mask].sum(axis=1) > 0))
    demand_ids = set(scenario.beam_id[mask].tolist())
    skipped = len({get(entry, "beam_id") for entry in entries if get(entry, "beam_id") in demand_ids
                   and get(entry, "status") == "skipped"
                   and (get(entry, "start"), get(entry, "length"), get(entry, "power_total_w")) == (-1, 0, 0)})
    slots = evaluation.slot_power_w > 0
    sinr = evaluation.sinr_linear[slots]
    db = 10*np.log10(np.maximum(sinr, np.finfo(float).tiny))
    beta = float(demand.mean()) if n else None
    # 新版辅助 SGM：零速率极限为0，beta每景正需求均值。
    sgm = np.zeros(n)
    nonzero = rates > 0
    if n and nonzero.any():
        ratio = rates[nonzero]/demand[nonzero]
        real = np.where(ratio >= 1, ratio-1, 1-1/ratio)
        imag = (rates[nonzero]-demand[nonzero])/beta
        magnitude = np.hypot(real, imag)
        sgm[nonzero] = 1-(-np.expm1(-magnitude))**3
    return {
        "n_entities": scenario.n_entities, "n_demand": n, "empty_demand": n == 0,
        "mean_satisfaction": float(utility.mean()) if n else None,
        "fully_satisfied_fraction": float(np.mean(rates >= demand*(1-config.env.satisfaction_tolerance))) if n else None,
        "served_fraction": allocated/n if n else None, "skip_fraction": skipped/n if n else None,
        "zero_rate_fraction": float(np.mean(rates == 0)) if n else None,
        "delivered_bps": float(np.minimum(rates, demand).sum()), "raw_throughput_bps": float(rates.sum()),
        "unmet_bps": float(np.maximum(demand-rates, 0).sum()), "total_demand_bps": float(demand.sum()),
        "power_used_w": float(evaluation.slot_power_w.sum()),
        "remaining_power_w": float(config.env.power_budget_w-evaluation.slot_power_w.sum()),
        "power_budget_w": config.env.power_budget_w,
        "pool_occupancy": float(evaluation.occupancy.mean()),
        "physical_frequency_coverage": float(evaluation.occupancy.any(axis=0).mean()),
        "sinr_sample_count": int(sinr.size), "mean_slot_sinr_db": float(db.mean()) if sinr.size else None,
        "p05_slot_sinr_db": float(np.quantile(db, .05)) if sinr.size else None,
        "linear_mean_sinr_to_db": float(10*np.log10(sinr.mean())) if sinr.size and sinr.mean() > 0 else None,
        "jain_clipped_satisfaction": float(utility.sum()**2/(n*np.square(utility).sum())) if n and np.square(utility).sum()>0 else None,
        "sgm_mean_positive_demand": float(sgm.mean()) if n else None, "sgm_beta_bps": beta,
        "constraint_violation_count": len(evaluation.constraint_violations),
        "constraint_violations": evaluation.constraint_violations,
        "end_to_end_metric_available": False, "j_root": None,
    }

