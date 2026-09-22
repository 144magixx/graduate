"""唯一合法动作定义。SKIP 为 0，候选按 start/length/power 排序。"""
from dataclasses import dataclass, asdict
import numpy as np
from ..config import Config


@dataclass(frozen=True)
class Action:
    kind: str = "SKIP"
    start: int = -1
    length: int = 0
    power_index: int = -1

    def to_dict(self):
        return asdict(self)


class ActionSpec:
    def __init__(self, config=None):
        config = config or Config()
        config.validate()
        self.num_slots = config.physics.num_slots
        self.max_block_length = config.env.max_block_length
        self.power_levels_w = np.asarray(config.env.power_levels_w, dtype=np.float64)
        self.budget_tolerance_w = config.env.budget_tolerance_w
        self.actions = (Action(),) + tuple(Action("ALLOC", s, length, p) for s in range(self.num_slots)
                 for length in range(1, min(self.max_block_length, self.num_slots-s)+1) for p in range(len(self.power_levels_w)))
        self.candidates = np.asarray([(a.start, a.length, a.power_index) for a in self.actions], dtype=np.int64)
        self.num_actions = len(self.actions)
        self._ids = {a: i for i, a in enumerate(self.actions)}

    def __len__(self):
        return self.num_actions

    def signature(self):
        return {"num_slots": self.num_slots, "max_block_length": self.max_block_length,
                "power_levels_w": self.power_levels_w.tolist()}

    def decode(self, action_id):
        if isinstance(action_id, (bool, np.bool_)) or not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < len(self):
            raise ValueError("action_id 不在候选字典内")
        return self.actions[int(action_id)]

    def encode(self, action):
        if not isinstance(action, Action) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (action.start, action.length, action.power_index)):
            raise ValueError("动作类型/索引必须严格符合 ActionSpec")
        try:
            return self._ids[action]
        except KeyError as exc:
            raise ValueError("非法动作；禁止越界和静默裁剪") from exc

    def valid_actions(self, obs):
        return valid_actions(obs, self)

    def conditional_masks(self, obs):
        mask = valid_actions(obs, self)
        full = np.zeros((self.num_slots, self.max_block_length, len(self.power_levels_w)), bool)
        ids = np.flatnonzero(mask[1:]) + 1
        c = self.candidates[ids]
        full[c[:, 0], c[:, 1]-1, c[:, 2]] = True
        return {"gate": np.array([mask[0], full.any()]), "start": full.any(axis=(1, 2)), "length": full.any(axis=2), "power": full}


def valid_actions(obs, action_spec):
    mask = np.zeros(len(action_spec), dtype=bool)
    if bool(obs["terminal"]):
        return mask
    mask[0] = True
    occupied = np.asarray(obs["occupancy"], dtype=bool)[int(obs["current_group"])]
    prefix = np.r_[0, np.cumsum(occupied, dtype=np.int64)]
    c = action_spec.candidates[1:]
    free = prefix[c[:, 0]+c[:, 1]] == prefix[c[:, 0]]
    fits = action_spec.power_levels_w[c[:, 2]] <= float(obs["remaining_power_w"]) + action_spec.budget_tolerance_w
    mask[1:] = free & fits
    return mask
