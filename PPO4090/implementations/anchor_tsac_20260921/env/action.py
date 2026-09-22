"""冻结动作字典：SKIP=0，其后start/length/power，共9551项。"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Action:
    kind: str = "SKIP"
    start: int = -1
    length: int = 0
    power_index: int = -1

    def to_dict(self):
        return asdict(self)


class ActionSpec:
    def __init__(self, config):
        config.validate()
        self.num_slots = config.physics.num_slots
        self.max_block_length = config.env.max_block_length
        self.power_levels_w = np.asarray(config.env.power_levels_w, dtype=np.float64)
        self.budget_tolerance_w = config.env.budget_tolerance_w
        self.actions = (Action(),) + tuple(
            Action("ALLOC", start, length, power)
            for start in range(self.num_slots)
            for length in range(1, min(self.max_block_length, self.num_slots - start) + 1)
            for power in range(len(self.power_levels_w)))
        self.candidates = np.asarray([(a.start, a.length, a.power_index) for a in self.actions], dtype=np.int64)
        self._ids = {action: index for index, action in enumerate(self.actions)}
        if len(self.actions) != 9551 or self.actions[0].kind != "SKIP":
            raise AssertionError("动作字典必须保持SKIP=0与9551候选")

    def __len__(self):
        return len(self.actions)

    def signature(self):
        return {"version": "contiguous_total_power_skip0.v1", "num_actions": len(self),
                "num_slots": self.num_slots, "max_block_length": self.max_block_length,
                "power_levels_w": self.power_levels_w.tolist(), "skip_id": 0}

    def decode(self, action_id):
        if isinstance(action_id, (bool, np.bool_)) or not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < len(self):
            raise ValueError("action_id不在候选字典内")
        return self.actions[int(action_id)]

    def encode(self, action):
        if not isinstance(action, Action):
            raise ValueError("动作类型不符")
        try:
            return self._ids[action]
        except KeyError as exc:
            raise ValueError("非法动作；禁止裁剪") from exc

    def valid_actions(self, observation):
        mask = np.zeros(len(self), dtype=bool)
        if bool(observation["terminal"]):
            return mask
        mask[0] = True
        occupied = np.asarray(observation["occupancy"], dtype=bool)[int(observation["current_group"])]
        prefix = np.r_[0, np.cumsum(occupied, dtype=np.int64)]
        candidates = self.candidates[1:]
        free = prefix[candidates[:, 0] + candidates[:, 1]] == prefix[candidates[:, 0]]
        fits = self.power_levels_w[candidates[:, 2]] <= float(observation["remaining_power_w"]) + self.budget_tolerance_w
        mask[1:] = free & fits
        return mask
