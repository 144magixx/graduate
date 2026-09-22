"""可恢复的transition-uniform回放；所有语义在提交前逐条核验。"""
from dataclasses import dataclass
import copy
import random
import numpy as np


@dataclass
class Transition:
    observation: dict
    action_id: int
    reward: float
    next_observation: dict
    terminated: bool
    versions: dict

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls(**value)


class ReplayBuffer:
    FORMAT = "anchor_replay.v2"
    MODE = "transition_uniform"

    def __init__(self, capacity, semantic_versions, action_spec_signature, seed=42):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Replay容量必须为正整数")
        self.capacity = capacity
        self.semantic_versions = copy.deepcopy(dict(semantic_versions))
        self.action_spec_signature = copy.deepcopy(dict(action_spec_signature))
        self.rng = random.Random(seed)
        self._values = []

    def _validate(self, value):
        transition = Transition.coerce(value)
        if dict(transition.versions) != self.semantic_versions:
            raise ValueError("经验语义版本不兼容")
        if isinstance(transition.action_id, bool) or not isinstance(transition.action_id, (int, np.integer)):
            raise ValueError("经验action_id必须为整数")
        if not np.isfinite(transition.reward):
            raise ValueError("经验reward必须有限")
        if bool(transition.terminated) != bool(transition.next_observation.get("terminal", False)):
            raise ValueError("terminated必须与next_observation终局标志一致")
        expected_adapter = self.semantic_versions["observation_adapter_version"]
        for observation in (transition.observation, transition.next_observation):
            if observation.get("observation_adapter_version") != expected_adapter:
                raise ValueError("经验观察适配器不兼容")
            if observation.get("action_spec_signature") != self.action_spec_signature:
                raise ValueError("经验动作字典签名不兼容")
            mask = np.asarray(observation.get("valid_action_mask"), dtype=bool)
            if mask.shape != (self.action_spec_signature["num_actions"],):
                raise ValueError("经验动作mask形状不兼容")
        mask = np.asarray(transition.observation["valid_action_mask"], dtype=bool)
        if not 0 <= int(transition.action_id) < len(mask) or not mask[int(transition.action_id)]:
            raise ValueError("经验实际动作不合法")
        return transition

    def add(self, transition):
        staged = copy.deepcopy(self._validate(transition))
        if len(self._values) == self.capacity:
            self._values.pop(0)
        self._values.append(staged)

    def sample(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 0 < batch_size <= len(self._values):
            raise ValueError("Replay采样量必须位于1到当前容量")
        return copy.deepcopy(self.rng.sample(self._values, batch_size))

    def __len__(self):
        return len(self._values)

    def diagnostics(self):
        return {"mode": self.MODE, "size": len(self), "capacity": self.capacity,
                "sampling": "uniform_without_replacement_over_transitions"}

    def state_dict(self):
        return {"format": self.FORMAT, "mode": self.MODE, "capacity": self.capacity,
                "semantic_versions": copy.deepcopy(self.semantic_versions),
                "action_spec_signature": copy.deepcopy(self.action_spec_signature),
                "rng_state": copy.deepcopy(self.rng.getstate()), "values": copy.deepcopy(self._values)}

    def load_state_dict(self, state):
        if state.get("format") != self.FORMAT or state.get("mode") != self.MODE:
            raise ValueError("Replay格式或采样模式不兼容")
        if state.get("semantic_versions") != self.semantic_versions:
            raise ValueError("Replay语义版本不兼容")
        if state.get("action_spec_signature") != self.action_spec_signature:
            raise ValueError("Replay动作字典签名不兼容")
        if state.get("capacity") != self.capacity or len(state.get("values", [])) > self.capacity:
            raise ValueError("Replay容量不兼容")
        staged_values = [copy.deepcopy(self._validate(value)) for value in state["values"]]
        staged_rng = random.Random()
        try:
            staged_rng.setstate(copy.deepcopy(state["rng_state"]))
        except Exception as exc:
            raise ValueError("Replay RNG状态不兼容") from exc
        self._values = staged_values
        self.rng.setstate(staged_rng.getstate())
