"""完整场景、单 cursor、硬约束逐束构造环境。"""
from copy import deepcopy
import numpy as np
from ..config import Config
from .action import Action, ActionSpec
from .state import Allocation, IDLE, PENDING, ALLOCATED, SKIPPED, STATUS_NAMES
from .physics import evaluate_allocation, channel_geometry, coverage_polygon
from .metrics import satisfaction
from .reward import delta_reward
from .observation import adapt_observation


class Environment:
    def __init__(self, config=None):
        self.config = deepcopy(config or Config())
        self.config.validate()
        if self.config.env.reward_version != "delta_mean_satisfaction_v1":
            raise ValueError("首版只启用已验收的全局满足度增量奖励")
        self.action_spec = ActionSpec(self.config)
        self.scenario = None

    def reset(self, scenario, seed=None):
        # 自持副本，外部修改 metadata 不改变正在运行的回合。
        self.scenario = type(scenario).from_dict(scenario.to_dict())
        self.scenario.validate(self.config.physics.num_groups)
        if self.config.physics.group_profile != "shared_grid_parity_isolation_v1":
            raise ValueError("不支持未冻结的 group 物理 profile")
        if np.any(self.scenario.polarization_id != self.scenario.group_id % 2):
            raise ValueError("当前物理 profile 要求 polarization_id=group_id%2")
        self.rng = np.random.default_rng(seed)
        self.cursor, self.ledger = 0, {}
        self._index = {int(b): i for i, b in enumerate(scenario.beam_id)}
        self._channel = channel_geometry(self.scenario, self.config.physics)
        self._evaluation = evaluate_allocation(self.scenario, self.ledger, self.config, channel_cache=self._channel)
        self._polygons = {int(scenario.beam_id[i]): coverage_polygon(scenario.latitude_deg[i], scenario.longitude_deg[i], scenario.ground_diameter_deg[i])
                          for i in np.flatnonzero(scenario.entity_mask)}
        obs = self.observe()
        return obs, {"scenario_id": scenario.scenario_id, "empty_demand": scenario.n_demand == 0, "metrics": deepcopy(self._evaluation.metrics),
                     "valid_action_count": int(obs["valid_action_mask"].sum()), "terminated_reason": "empty_demand" if self.terminal else None}

    @property
    def terminal(self):
        return self.scenario is not None and self.cursor >= self.scenario.n_demand

    @property
    def current_beam_id(self):
        return None if self.terminal else int(self.scenario.service_order[self.cursor])

    @property
    def remaining_power_w(self):
        return float(self._evaluation.metrics["remaining_power_w"])

    def observe(self):
        if self.scenario is None:
            raise RuntimeError("先 reset 再获取观察")
        s, e, p = self.scenario, self.config.env, self.config.physics
        n = len(s.beam_id)
        status = np.where(s.demand_mask, PENDING, IDLE).astype(np.int64)
        starts, lengths, powers = np.full(n, -1, np.int64), np.zeros(n, np.int64), np.zeros(n)
        order_rank = np.full(n, -1, np.int64)
        for rank, beam_id in enumerate(s.service_order):
            order_rank[self._index[int(beam_id)]] = rank
        for beam_id, allocation in self.ledger.items():
            i = self._index[beam_id]
            status[i] = ALLOCATED if allocation.status == "allocated" else SKIPPED
            starts[i], lengths[i], powers[i] = allocation.start, allocation.length, allocation.power_total_w
        sat = satisfaction(s, self._evaluation.rate_bps)
        static = np.column_stack((s.latitude_deg/90, s.longitude_deg/180, np.log1p(s.demand_bps/1e8), s.ground_diameter_deg,
                                  s.tx_gain_peak_dbi/50, s.rx_gain_peak_dbi/50, s.noise_temperature_k/290, self._channel["slant_km"]/42164)).astype(np.float32)
        dynamic = np.column_stack((np.log1p(self._evaluation.rate_bps/1e8), sat, starts/p.num_slots,
                                   lengths/e.max_block_length, powers/max(e.power_levels_w))).astype(np.float32)
        current = -1 if self.terminal else self._index[self.current_beam_id]
        group = -1 if self.terminal else int(s.group_id[current])
        row = np.zeros(p.num_slots, bool) if self.terminal else self._evaluation.occupancy[group]
        interference = np.zeros(p.num_slots) if self.terminal else self._evaluation.interference_w[current]
        noise = np.ones(p.num_slots)*1e-13 if self.terminal else self._evaluation.noise_w[current]
        slots = np.column_stack((np.arange(p.num_slots)/max(p.num_slots-1, 1), row, np.log1p(interference/noise), np.log1p(noise/1e-13))).astype(np.float32)
        pending = status == PENDING
        pending_sum = float(s.demand_bps[pending].sum())
        global_features = np.array([e.power_budget_w/6000, self.remaining_power_w/6000, s.n_demand/200, self.cursor/max(s.n_demand, 1),
            pending.sum()/200, np.log1p(pending_sum/1e8), np.log1p(pending_sum/max(int(pending.sum()), 1)/1e8), np.log1p(s.demand_bps.sum()/1e8),
            p.num_slots/100, max(e.power_levels_w)/50, p.slot_bandwidth_hz/25e6, p.frequency_start_hz/1e10,
            p.sinr_margin_db/10, p.satellite_radius_km/42164, p.earth_radius_km/6371, p.satellite_longitude_deg/180,
            p.satellite_latitude_deg/90, *np.mean(~self._evaluation.occupancy, axis=1)], dtype=np.float32)
        anchor205 = np.zeros(5+2*p.num_slots, np.float32)
        adapter_version = ("modern205.v1" if e.observation_adapter == "modern205"
                           else "legacy_scaled205_new_physics.v1")
        if not self.terminal:
            anchor205, adapter_version = adapt_observation(e.observation_adapter, {
                "latitude_deg": s.latitude_deg[current], "longitude_deg": s.longitude_deg[current],
                "demand_bps": s.demand_bps[current], "ground_diameter_deg": s.ground_diameter_deg[current],
                "remaining_power_w": self.remaining_power_w, "interference_w": interference,
                "noise_w": noise, "occupancy": row})
        obs = {"beam_id": s.beam_id.copy(), "beam_static": static, "beam_dynamic": dynamic, "group_id": s.group_id.copy(),
            "order_rank": order_rank, "entity_mask": s.entity_mask.copy(), "demand_mask": s.demand_mask.copy(), "status": status,
            "pending_mask": pending, "allocated_mask": status == ALLOCATED, "skipped_mask": status == SKIPPED,
            "allocation_start": starts, "allocation_length": lengths, "allocation_power_w": powers.astype(np.float32),
            "occupancy": self._evaluation.occupancy.copy(), "current_slot_features": slots, "global_features": global_features,
            "current_beam_index": current, "current_group": group, "remaining_power_w": self.remaining_power_w,
            "terminal": self.terminal, "anchor205": anchor205, "observation_version": e.observation_version,
            "observation_adapter": e.observation_adapter, "observation_adapter_version": adapter_version}
        obs["action_spec_signature"] = self.action_spec.signature()
        obs["valid_action_mask"] = self.action_spec.valid_actions(obs)
        return obs

    def step(self, action):
        if self.scenario is None or self.terminal:
            raise RuntimeError("空需求/已终止场景不能 step；请 reset")
        # 所有校验发生在任何账本、cursor 或 RNG 修改之前。
        if isinstance(action, (int, np.integer)):
            action = self.action_spec.decode(action)
        action_id = self.action_spec.encode(action)
        before_obs = self.observe()
        if not before_obs["valid_action_mask"][action_id]:
            raise ValueError("动作违反占用或功率约束；环境未改变")
        beam_id, s = self.current_beam_id, self.scenario
        acted_index = self._index[beam_id]
        old_evaluation = self._evaluation
        before_sat = satisfaction(s, old_evaluation.rate_bps)
        power = float(self.action_spec.power_levels_w[action.power_index]) if action.kind == "ALLOC" else 0.
        allocation = Allocation(beam_id, "allocated" if action.kind == "ALLOC" else "skipped", action.start, action.length, power)
        # 先计算候选账本；物理失败也不会半提交。
        candidate = dict(self.ledger)
        candidate[beam_id] = allocation
        evaluation = evaluate_allocation(s, candidate, self.config, channel_cache=self._channel)
        if evaluation.constraint_violations:
            raise RuntimeError(f"内部约束不一致: {evaluation.constraint_violations}")
        after_sat = satisfaction(s, evaluation.rate_bps)
        reward, terms = delta_reward(before_sat, after_sat, acted_index, s.demand_mask, self.config.env.reward_scale)
        changed = np.flatnonzero(evaluation.rate_bps != old_evaluation.rate_bps)
        affected = [{"beam_id": int(s.beam_id[i]), "rate_before_bps": float(old_evaluation.rate_bps[i]),
                     "rate_after_bps": float(evaluation.rate_bps[i]), "satisfaction_before": float(before_sat[i]),
                     "satisfaction_after": float(after_sat[i])} for i in changed]
        self.ledger, self._evaluation = candidate, evaluation
        self.cursor += 1
        obs = self.observe()
        info = {"scenario_id": s.scenario_id, "beam_id": beam_id, "acted_beam_id": beam_id, "next_beam_id": self.current_beam_id,
                "action_id": action_id, "requested_action": action.to_dict(), "executed_action": action.to_dict(),
                "reward_terms": terms, "metrics": deepcopy(evaluation.metrics), "remaining_power_w": self.remaining_power_w,
                "valid_action_count": int(before_obs["valid_action_mask"].sum()), "forced_skip": int(before_obs["valid_action_mask"].sum()) == 1,
                "terminated_reason": "all_demand_processed" if self.terminal else None,
                "constraint_violations": [], "affected_beams": affected, "step_index": self.cursor}
        return obs, reward, self.terminal, False, info

    def evaluate(self):
        return evaluate_allocation(self.scenario, self.ledger, self.config)

    def snapshot(self):
        obs, s = self.observe(), self.scenario
        sat = satisfaction(s, self._evaluation.rate_bps)
        beams = []
        for i in np.flatnonzero(s.entity_mask):
            beam_id = int(s.beam_id[i])
            a = self.ledger.get(beam_id)
            active = a is not None and a.status == "allocated"
            interval = slice(a.start, a.start+a.length) if active else slice(0, 0)
            beams.append({"beam_id": beam_id, "latitude_deg": float(s.latitude_deg[i]), "longitude_deg": float(s.longitude_deg[i]),
                "ground_diameter_deg": float(s.ground_diameter_deg[i]), "beamwidth_kind": self.config.physics.beamwidth_kind,
                "coverage_polygon": self._polygons[beam_id], "group_id": int(s.group_id[i]), "polarization_id": int(s.polarization_id[i]),
                "status": STATUS_NAMES[int(obs["status"][i])], "demand_bps": float(s.demand_bps[i]), "rate_bps": float(self._evaluation.rate_bps[i]),
                "satisfaction": float(sat[i]) if s.demand_mask[i] else None, "start": a.start if a else -1,
                "length": a.length if a else 0, "power_total_w": a.power_total_w if a else 0.,
                "slot_power_w": a.power_total_w/a.length if active else None,
                "sinr_db": (10*np.log10(np.maximum(self._evaluation.sinr_linear[i, interval], np.finfo(float).tiny))).tolist() if active else None,
                "interference_w": self._evaluation.interference_w[i, interval].tolist() if active else None})
        owners = [[int(b) if used else None for b, used in zip(row, occupied)]
                  for row, occupied in zip(self._evaluation.beam_ids, self._evaluation.occupancy)]
        p = self.config.physics
        return {"scenario_id": s.scenario_id, "source_hash": s.source_hash, "step_index": self.cursor, "terminal": self.terminal,
                "current_beam_id": self.current_beam_id, "beams": beams,
                "resources": {"num_groups": p.num_groups, "num_slots": p.num_slots, "occupancy": self._evaluation.occupancy.tolist(), "beam_ids": owners,
                    "remaining_power_w": self.remaining_power_w, "power_budget_w": self.config.env.power_budget_w,
                    "slot_bandwidth_hz": p.slot_bandwidth_hz, "frequency_start_hz": p.frequency_start_hz},
                "metrics": deepcopy(self._evaluation.metrics)}

    def state_dict(self):
        return {"scenario": self.scenario.to_dict(), "cursor": self.cursor, "ledger": [a.to_dict() for a in self.ledger.values()],
                "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        from ..data.schema import CoverageScenario
        scenario = CoverageScenario.from_dict(state["scenario"])
        self.reset(scenario)
        # 逐步重放严格核验保存前缀，不信任已缓存指标。
        for entry in state["ledger"]:
            if entry["beam_id"] != self.current_beam_id:
                raise ValueError("保存账本与服务顺序不一致")
            if entry["status"] == "skipped":
                self.step(Action())
            elif entry["status"] == "allocated":
                matches = np.flatnonzero(self.action_spec.power_levels_w == entry["power_total_w"])
                if not len(matches):
                    raise ValueError("保存功率与 ActionSpec 不兼容")
                self.step(Action("ALLOC", entry["start"], entry["length"], int(matches[0])))
            else:
                raise ValueError("保存账本含非法分配状态")
        if self.cursor != state["cursor"]:
            raise ValueError("保存 cursor 与账本不一致")
        self.rng.bit_generator.state = state["rng_state"]
