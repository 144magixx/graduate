"""完整回合回放：版本隔离、不可变快照、可恢复分层随机采样。"""
from collections import deque
from dataclasses import dataclass, field
import bisect
import copy
import random
import numpy as np


class FrozenDict(dict):
    def _deny(self, *args, **kwargs):
        raise TypeError("回放快照不可修改")
    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _deny

    def __reduce__(self):
        return (freeze, (dict(self),))

    def __deepcopy__(self, memo):
        return freeze(dict(self))


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def normalize_signature(signature):
    if signature is None:
        return None
    return dict(signature, power_levels_w=list(signature["power_levels_w"]))


@dataclass(frozen=True)
class Transition:
    observation: dict
    action_id: int
    reward: float
    next_observation: dict
    terminated: bool
    truncated: bool = False
    scenario_id: str = ""
    versions: dict = field(default_factory=dict)
    episode_id: str = ""
    step_index: int | None = None
    env_step: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.reward):
            raise FloatingPointError("reward必须有限")
        if isinstance(self.action_id, (bool, np.bool_)) or not isinstance(self.action_id, (int, np.integer)):
            raise ValueError("回放action_id必须为整数")
        if bool(self.observation.get("terminal", False)):
            raise ValueError("回放动作前观察不能为terminal")
        if self.terminated != bool(self.next_observation.get("terminal", False)):
            raise ValueError("terminated与实际next_observation必须一致")
        for name in ("step_index", "env_step"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f"回放{name}必须是非负整数或None")
        for name in ("observation", "next_observation", "versions"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "action_id", int(self.action_id))

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls(**value)


class ReplayBuffer:
    def __init__(self, capacity=20000, versions=None, action_spec_signature=None,
                 mode="episode_balanced", seed=42, bucket_boundaries=(80, 120, 168, 220, 256)):
        if capacity < 1 or mode not in ("episode_balanced", "uniform_transition"):
            raise ValueError("回放容量/采样模式无效")
        self.capacity, self.mode = int(capacity), mode
        self.versions = dict(versions or {})
        self.action_spec_signature = normalize_signature(action_spec_signature)
        self.bucket_boundaries = tuple(bucket_boundaries)
        if sorted(set(self.bucket_boundaries)) != list(self.bucket_boundaries):
            raise ValueError("规模桶边界必须严格递增")
        self.episodes = deque()
        self.rng = random.Random(seed)
        self.transition_count = 0
        self.evicted_episodes = 0
        self.sampled_by_bucket = {}
        self.sampled_by_domain_cell = {}
        self.sampled_by_domain_cell_membership = {}

    def __len__(self):
        return self.transition_count

    def size(self):
        return len(self)

    def add_episode(self, transitions, scenario_id=None, n_demand=None, domain_cell=None, fragment=False):
        values = tuple(Transition.coerce(t) for t in transitions)
        if not values:
            raise ValueError("空回合不进入经验池")
        if len(values) > self.capacity:
            raise ValueError("单回合超过经验池容量；拒绝截掉头部，请增加容量")
        if not values[-1].terminated and not (fragment and values[-1].truncated):
            raise ValueError("只接收已完成回合；截断片段必须显式fragment=True")
        if any(t.terminated or t.truncated for t in values[:-1]):
            raise ValueError("回合中间不能包含终止/截断标记")
        actual_scenario = values[0].scenario_id
        expected_signature = self.action_spec_signature or normalize_signature(values[0].observation.get("action_spec_signature"))
        if expected_signature is None:
            raise ValueError("回放观察缺少ActionSpec签名")
        for t in values:
            if dict(t.versions) != self.versions:
                raise ValueError("经验语义版本不兼容，禁止混池")
            if t.scenario_id != actual_scenario:
                raise ValueError("完整回合不能跨场景")
            for observation in (t.observation, t.next_observation):
                if normalize_signature(observation.get("action_spec_signature")) != expected_signature:
                    raise ValueError("ActionSpec签名不兼容，禁止混池")
        if scenario_id is not None and scenario_id != actual_scenario:
            raise ValueError("回合scenario_id与transition不一致")
        observed_count = int(np.asarray(values[0].observation["demand_mask"]).sum())
        if n_demand is None:
            n_demand = observed_count
        elif int(n_demand) != observed_count:
            raise ValueError("回放规模桶必须来自真实观察的正需求数")
        bucket = bisect.bisect_left(self.bucket_boundaries, n_demand)
        item = dict(transitions=values, scenario_id=actual_scenario, n_demand=int(n_demand),
                    bucket=bucket, domain_cell=domain_cell, fragment=bool(fragment))
        self.action_spec_signature = copy.deepcopy(expected_signature)
        while self.transition_count + len(values) > self.capacity:
            self.transition_count -= len(self.episodes.popleft()["transitions"])
            self.evicted_episodes += 1
        self.episodes.append(item)
        self.transition_count += len(values)

    def sample(self, batch_size):
        if batch_size < 1 or not self.episodes:
            raise ValueError("需要正批量与非空回放")
        by_bucket = {}
        for episode in self.episodes:
            by_bucket.setdefault(episode["bucket"], []).append(episode)
        buckets = sorted(by_bucket)
        # 采样有放回，容量不足batch时也能完成一个明确有效batch。
        flat = [(episode, t) for episode in self.episodes for t in episode["transitions"]] if self.mode == "uniform_transition" else None
        result = []
        for _ in range(batch_size):
            if flat is None:
                episode = self.rng.choice(by_bucket[self.rng.choice(buckets)])
                transition = self.rng.choice(episode["transitions"])
            else:
                episode, transition = self.rng.choice(flat)
            bucket = str(episode["bucket"])
            self.sampled_by_bucket[bucket] = self.sampled_by_bucket.get(bucket, 0) + 1
            cell = str(episode["domain_cell"])
            self.sampled_by_domain_cell[cell] = self.sampled_by_domain_cell.get(cell, 0) + 1
            memberships=episode["domain_cell"] if isinstance(episode["domain_cell"],(list,tuple)) else [episode["domain_cell"] or "unknown"]
            for membership in memberships or ["unknown"]:
                membership=str(membership)
                self.sampled_by_domain_cell_membership[membership]=self.sampled_by_domain_cell_membership.get(membership,0)+1
            result.append(transition)
        return result

    def diagnostics(self):
        counts = {}
        for episode in self.episodes:
            key = str(episode["bucket"])
            counts.setdefault(key, dict(episodes=0, transitions=0))
            counts[key]["episodes"] += 1
            counts[key]["transitions"] += len(episode["transitions"])
        return dict(transitions=len(self), episodes=len(self.episodes), buckets=counts,
                    evicted_episodes=self.evicted_episodes, sampled_by_bucket=dict(self.sampled_by_bucket),
                    sampled_by_domain_cell=dict(self.sampled_by_domain_cell),
                    sampled_by_domain_cell_membership=dict(self.sampled_by_domain_cell_membership),sampling_mode=self.mode)

    def state_dict(self):
        return copy.deepcopy(dict(format_version="episode_replay.v1", capacity=self.capacity, mode=self.mode,
            versions=self.versions, action_spec_signature=self.action_spec_signature,
            bucket_boundaries=self.bucket_boundaries, episodes=list(self.episodes), rng_state=self.rng.getstate(),
            transition_count=self.transition_count, evicted_episodes=self.evicted_episodes,
            sampled_by_bucket=self.sampled_by_bucket, sampled_by_domain_cell=self.sampled_by_domain_cell,
            sampled_by_domain_cell_membership=self.sampled_by_domain_cell_membership))

    def load_state_dict(self, state):
        for key in ("capacity", "mode", "versions", "bucket_boundaries"):
            if state[key] != getattr(self, key):
                raise ValueError(f"回放恢复不兼容：{key}")
        expected_signature = normalize_signature(state["action_spec_signature"])
        if self.action_spec_signature is not None and expected_signature != self.action_spec_signature:
            raise ValueError("回放恢复不兼容：action_spec_signature")
        if state["format_version"] != "episode_replay.v1":
            raise ValueError("未知回放格式")
        episodes = copy.deepcopy(state["episodes"])
        count = sum(len(ep["transitions"]) for ep in episodes)
        if count != state["transition_count"] or count > self.capacity:
            raise ValueError("回放恢复容量/计数错误")
        if any(dict(t.versions) != self.versions for ep in episodes for t in ep["transitions"]):
            raise ValueError("恢复经验版本不一致")
        if any(normalize_signature(obs.get("action_spec_signature")) != expected_signature
               for ep in episodes for t in ep["transitions"] for obs in (t.observation, t.next_observation)):
            raise ValueError("恢复经验ActionSpec不一致")
        self.action_spec_signature = copy.deepcopy(expected_signature)
        self.episodes, self.transition_count = deque(episodes), count
        self.rng.setstate(state["rng_state"])
        self.evicted_episodes = state["evicted_episodes"]
        self.sampled_by_bucket = dict(state["sampled_by_bucket"])
        self.sampled_by_domain_cell = dict(state["sampled_by_domain_cell"])
        self.sampled_by_domain_cell_membership = dict(state.get("sampled_by_domain_cell_membership",{}))
