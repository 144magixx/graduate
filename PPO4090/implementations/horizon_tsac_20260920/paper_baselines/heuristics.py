"""论文 IV 节的 Fixed、Greedy、Random 在当前硬约束环境中的适配。

这是同一物理环境、同一 ActionSpec 下的算法复现，并非旧论文数值的逐项重放。
论文未公布 Fixed 的固定长度/功率、Greedy 的长度/需求阈值，因此这些参数
必须在比较前冻结，或仅用 validation 选择；本模块不自动调参。

来源 group 和 service_order 均保持不变。Fixed 依当前服务顺序，在来源 group
内按频率起点递增分配；论文中可顺序选择 FG 的自由度在此环境中不可用。
Greedy 的每槽占用波束数等于该槽跨 group 的 occupancy 之和；固定长度窗口
的总占用数最低者优先，同分按稳定候选 ID 选择，不调用物理评价或预测奖励。
Random 均匀抽样完整合法 ALLOC 组合，故边界处不同长度的边缘概率无需相同。
"""
from copy import deepcopy

import numpy as np

from ..env.action import Action, ActionSpec


# 供调用方显式冻结/validation 调参使用；不在 act() 内搜索这些超参数。
SUGGESTED_LENGTH_GRID = (1, 3, 5, 10)
GREEDY_POWER_LEVELS_W = (20.0, 25.0, 30.0)


class HeuristicPolicy:
    """无学习参数的论文基线，实例绑定一个场景及独立随机数生成器。

    默认 Greedy 阈值是当前输入场景的正需求三分位数（linear 插值），属于
    场景已知需求的决策规则，不从测试场景集合或测试结果拟合参数。低需求
    ``d <= q1`` 使用 20 W，``q1 < d <= q2`` 使用 25 W，其余使用 30 W。
    可传入事先冻结的 ``greedy_demand_thresholds_bps=(q1, q2)`` 替换此规则。
    固定/贪心所声明长度或功率没有可行动作时 SKIP，绝不截短或降功率。

    Random 的 deterministic 参数为接口兼容保留；其语义始终为均匀抽样。
    ``random_include_skip=True`` 将算法名明确改为 ``uniform_with_skip`` 诊断，
    不能以论文 Random 的名称报告该变体。
    """

    def __init__(self, name, action_spec, scenario, config=None, *, fixed_length=5,
                 fixed_power_w=30.0, greedy_length=5,
                 greedy_demand_thresholds_bps=None, seed=42,
                 random_include_skip=False):
        if not isinstance(action_spec, ActionSpec):
            raise TypeError("action_spec 必须是当前环境的 ActionSpec")
        name = str(name).strip().lower()
        if name not in ("fixed", "greedy", "random", "uniform_with_skip"):
            raise ValueError(f"未知论文启发式基线: {name}")
        if not isinstance(random_include_skip, (bool, np.bool_)):
            raise ValueError("random_include_skip 必须是布尔值")
        if random_include_skip and name not in ("random", "uniform_with_skip"):
            raise ValueError("只有随机诊断算法可启用 random_include_skip")
        self.name = "uniform_with_skip" if random_include_skip else name
        self.action_spec = action_spec
        if config is not None and ActionSpec(config).signature() != action_spec.signature():
            raise ValueError("config 与 action_spec 的候选动作定义不一致")
        scenario.validate(config.physics.num_groups if config is not None else None)
        self._beam_id = scenario.beam_id.copy()
        self._group_id = scenario.group_id.copy()
        self._demand_bps = scenario.demand_bps.copy()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.fixed_length = fixed_length
        self.fixed_power_w = fixed_power_w
        self.greedy_length = greedy_length
        self.greedy_demand_thresholds_bps = None
        self.greedy_threshold_source = None
        self._fixed_power_index = None
        self._greedy_power_indices = None

        if self.name == "fixed":
            self.fixed_length = self._validate_length(fixed_length, "fixed_length")
            self._fixed_power_index = self._power_index(fixed_power_w, "fixed_power_w")
            self.fixed_power_w = float(fixed_power_w)
        elif self.name == "greedy":
            self.greedy_length = self._validate_length(greedy_length, "greedy_length")
            self._greedy_power_indices = tuple(
                self._power_index(p, "greedy_power_levels_w") for p in GREEDY_POWER_LEVELS_W)
            if greedy_demand_thresholds_bps is None:
                positive = self._demand_bps[np.asarray(scenario.demand_mask, dtype=bool)]
                thresholds = np.quantile(positive, [1 / 3, 2 / 3], method="linear") if positive.size else np.zeros(2)
                self.greedy_threshold_source = "scenario_positive_demand_tertiles_linear"
            else:
                thresholds = np.asarray(greedy_demand_thresholds_bps, dtype=np.float64)
                self.greedy_threshold_source = "explicit_frozen_thresholds_bps"
            if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or np.any(thresholds < 0) or thresholds[0] > thresholds[1]:
                raise ValueError("greedy_demand_thresholds_bps 须是两个非负有限递增阈值（允许相等）")
            self.greedy_demand_thresholds_bps = tuple(float(x) for x in thresholds)

    def _validate_length(self, value, field):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= min(self.action_spec.max_block_length, self.action_spec.num_slots):
            raise ValueError(f"{field} 必须是 ActionSpec 范围内的正整数，禁止静默截短")
        return int(value)

    def _power_index(self, value, field):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{field} 必须是 ActionSpec 中的功率档")
        matches = np.flatnonzero(self.action_spec.power_levels_w == value)
        if matches.size != 1:
            raise ValueError(f"{field}={value} 不在 ActionSpec 中；禁止功率取整或裁剪")
        return int(matches[0])

    def act(self, obs, deterministic=True):
        """返回当前真实合法动作；terminal 无合法动作，调用方应结束回合。"""
        if bool(obs["terminal"]):
            raise ValueError("terminal 观察没有合法动作")
        if not np.array_equal(obs["beam_id"], self._beam_id):
            raise ValueError("策略绑定场景与观察的 beam_id 不一致；每个场景须重建策略")
        current = int(obs["current_beam_index"])
        if not 0 <= current < len(self._beam_id) or int(obs["current_group"]) != self._group_id[current]:
            raise ValueError("当前波束/source group 与策略绑定场景不一致")
        signature = obs.get("action_spec_signature")
        if signature is not None and signature != self.action_spec.signature():
            raise ValueError("观察的 ActionSpec 与策略不一致")
        # 使用唯一动作约束函数，不通过裁剪、临时改变 group 或模拟 step 使动作可行。
        mask = self.action_spec.valid_actions(obs)
        ids = np.flatnonzero(mask[1:]) + 1
        if not ids.size:
            return Action()
        if self.name in ("random", "uniform_with_skip"):
            if self.name == "uniform_with_skip":
                ids = np.r_[0, ids]
            return self.action_spec.decode(int(self.rng.choice(ids)))

        candidates = self.action_spec.candidates[ids]
        if self.name == "fixed":
            keep = (candidates[:, 1] == self.fixed_length) & (candidates[:, 2] == self._fixed_power_index)
            feasible = ids[keep]
            return self.action_spec.decode(int(feasible[0])) if feasible.size else Action()

        level = int(np.searchsorted(self.greedy_demand_thresholds_bps, self._demand_bps[current], side="left"))
        keep = (candidates[:, 1] == self.greedy_length) & (candidates[:, 2] == self._greedy_power_indices[level])
        ids, candidates = ids[keep], candidates[keep]
        if not ids.size:
            return Action()
        beam_count_per_slot = np.asarray(obs["occupancy"], dtype=bool).sum(axis=0, dtype=np.int64)
        prefix = np.r_[0, np.cumsum(beam_count_per_slot, dtype=np.int64)]
        scores = prefix[candidates[:, 0] + candidates[:, 1]] - prefix[candidates[:, 0]]
        # ids 本来已按 ActionSpec 排序，argmin 的首个最小值即稳定候选 ID tie-break。
        return self.action_spec.decode(int(ids[int(np.argmin(scores))]))

    def metadata(self):
        """可原样记录到运行清单的规则及显式复现假设。"""
        parameters = {}
        if self.name == "fixed":
            parameters = {"fixed_length": self.fixed_length, "fixed_power_w": self.fixed_power_w,
                          "start_rule": "first_feasible_start_in_source_group"}
        elif self.name == "greedy":
            parameters = {"greedy_length": self.greedy_length, "greedy_power_levels_w": list(GREEDY_POWER_LEVELS_W),
                          "greedy_demand_thresholds_bps": list(self.greedy_demand_thresholds_bps),
                          "threshold_source": self.greedy_threshold_source, "threshold_boundary_rule": "d<=q1:20;q1<d<=q2:25;d>q2:30",
                          "slot_score": "sum_of_occupied_beam_slots_across_all_groups",
                          "tie_break": "lowest_action_id"}
        else:
            parameters = {"sampling": "uniform_joint_legal_allocations_with_skip" if self.name == "uniform_with_skip" else "uniform_joint_legal_allocations",
                          "include_skip_when_alloc_feasible": self.name == "uniform_with_skip", "seed": self.seed}
        return deepcopy({"algorithm": self.name, "implementation_version": "paper_heuristics_masked.v1",
                         "parameters": parameters, "action_spec_signature": self.action_spec.signature(),
                         "group_adaptation": "source_group_and_service_order_unchanged",
                         "infeasible_rule": "skip_without_length_or_power_clipping",
                         "comparison_scope": "same_environment_masked_reproduction_not_original_numeric_replication"})
