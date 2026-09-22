"""Anchor 的版本化配置；默认只做预检，预算为零。"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class DataConfig:
    source_schema: str = "legacy_lon_in_lat"
    rate_unit: str = "Mbps"
    traffic_scale: float = 0.25
    beamwidth_kind: str = "ground_angular_diameter_deg"
    group_assignment: str = "canonical_demand_rank_mod8"
    service_order: str = "demand_desc"
    augment: bool = False
    width_augmentation: bool = False
    split_seed: int = 20260920
    dataset_version: str = "legacy_aggregate.v1"


@dataclass
class PhysicsConfig:
    physics_version: str = "uniform_slot_shannon.v1"
    group_profile: str = "shared_grid_parity_isolation_v1"
    beamwidth_kind: str = "ground_angular_diameter_deg"
    num_groups: int = 8
    num_slots: int = 100
    slot_bandwidth_hz: float = 25_000_000.0
    frequency_start_hz: float = 17_700_000_000.0
    earth_radius_km: float = 6371.0
    satellite_radius_km: float = 42164.0
    satellite_longitude_deg: float = 122.2
    satellite_latitude_deg: float = 0.0
    sinr_margin_db: float = 5.0
    tx_gain_peak_dbi: float = 50.0
    rx_gain_peak_dbi: float = 40.0
    noise_temperature_k: float = 290.0
    rate_model: str = "shannon_with_sinr_margin"
    power_semantics: str = "beam_total_uniform_slots"


@dataclass
class EnvConfig:
    power_budget_w: float = 6000.0
    power_levels_w: tuple = (5., 10., 15., 20., 25., 30., 35., 40., 45., 50.)
    max_block_length: int = 10
    reward_scale: float = 100.0
    reward_version: str = "delta_mean_satisfaction_v1"
    metric_version: str = "positive_demand_metrics.v1"
    observation_version: str = "anchor205.v1"
    observation_adapter: str = "legacy_scaled205"
    action_version: str = "contiguous_total_power_skip0.v1"
    budget_tolerance_w: float = 1e-8
    satisfaction_tolerance: float = 1e-8

    def __post_init__(self):
        self.power_levels_w = tuple(self.power_levels_w)


@dataclass
class ModelConfig:
    model_version: str = "anchor_pure_transformer14.v1"
    encoder: str = "pure_transformer_14"
    actor: str = "independent"
    critic: str = "additive"
    d_model: int = 128
    attention_heads: int = 8
    encoder_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    feedforward_multiplier: int = 4
    norm_first: bool = False
    position_initialization: str = "zeros"
    candidate_chunk_size: int = 512


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    episodes: int = 0
    max_env_steps: int = 0
    batch_size: int = 256
    microbatch_size: int = 32
    replay_capacity: int = 20000
    replay_mode: str = "transition_uniform"
    warmup_steps: int = 2000
    update_schedule: str = "env_step"
    updates_per_step: int = 1
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    initial_alpha: float = 0.01
    target_entropy_ratio: float = 0.5
    gamma: float = 0.99
    tau: float = 0.005
    gradient_clip_norm: float = 10.0
    torch_num_threads: int = 2
    deterministic: bool = True
    checkpoint_every: int = 1
    max_wall_seconds: float = 0.0
    checkpoint_compression: bool = False


@dataclass
class TelemetryConfig:
    enabled: bool = True
    trace_level: str = "basic_all"
    keyframe_interval: int = 20
    update_window: int = 50
    queue_max_messages: int = 512
    queue_max_bytes: int = 64 * 1024 * 1024
    flush_interval: float = 1.0
    enqueue_timeout: float = 10.0


@dataclass
class Config:
    schema_version: str = "coverage.v2"
    algorithm_id: str = "anchor_tsac"
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key, typ in (("data", DataConfig), ("physics", PhysicsConfig), ("env", EnvConfig),
                         ("model", ModelConfig), ("train", TrainConfig), ("telemetry", TelemetryConfig)):
            if key in value:
                value[key] = typ(**value[key])
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        p, e, m, t = self.physics, self.env, self.model, self.train
        if self.algorithm_id != "anchor_tsac":
            raise ValueError("algorithm_id 必须为 anchor_tsac")
        expected_versions = {
            "schema_version": (self.schema_version, "coverage.v2"),
            "physics_version": (p.physics_version, "uniform_slot_shannon.v1"),
            "reward_version": (e.reward_version, "delta_mean_satisfaction_v1"),
            "metric_version": (e.metric_version, "positive_demand_metrics.v1"),
            "observation_version": (e.observation_version, "anchor205.v1"),
        }
        unknown = [name for name, (actual, expected) in expected_versions.items() if actual != expected]
        if unknown:
            raise ValueError(f"未知或不兼容的语义版本：{unknown}")
        if e.observation_adapter not in ("modern205", "legacy_scaled205"):
            raise ValueError("观察适配器必须为 modern205 或 legacy_scaled205")
        if (p.num_slots, p.num_groups, e.max_block_length) != (100, 8, 10):
            raise ValueError("Anchor v1 固定100槽、8组和最大连续10槽")
        if e.power_levels_w != (5., 10., 15., 20., 25., 30., 35., 40., 45., 50.):
            raise ValueError("Anchor v1冻结旧三头词典：功率档必须为5..50W、步长5W")
        if m.encoder != "pure_transformer_14" or m.actor != "independent" or m.critic != "additive":
            raise ValueError("Anchor v1 只允许纯Transformer、独立Actor、可加Critic")
        if (m.d_model, m.attention_heads, m.encoder_layers, m.activation,
                m.feedforward_multiplier, m.norm_first, m.position_initialization) != (128, 8, 2, "relu", 4, False, "zeros"):
            raise ValueError("模型必须匹配冻结旧core：d128/8头/2层/ReLU/FF4d/post-LN/零位置")
        if not math.isfinite(m.dropout) or not 0 <= m.dropout < 1:
            raise ValueError("dropout必须位于[0,1)")
        for name in ("episodes", "max_env_steps", "warmup_steps", "updates_per_step"):
            value = getattr(t, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"train.{name}必须为非负整数")
        for name in ("batch_size", "microbatch_size", "replay_capacity", "torch_num_threads", "checkpoint_every"):
            value = getattr(t, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"train.{name}必须为正整数")
        if t.update_schedule != "env_step":
            raise ValueError("Anchor v1 更新调度固定为env_step")
        if t.replay_mode != "transition_uniform":
            raise ValueError("Anchor v1只实现可恢复的transition_uniform回放")
        if not 0 <= t.gamma <= 1 or not 0 < t.tau <= 1 or not 0 <= t.target_entropy_ratio <= 1:
            raise ValueError("gamma/tau/熵比例越界")
        if p.power_semantics != "beam_total_uniform_slots" or p.rate_model != "shannon_with_sinr_margin":
            raise ValueError("只支持已验收的新物理语义")
        if self.data.width_augmentation:
            raise ValueError("Anchor v1 禁止宽度增强")
        if e.action_version != "contiguous_total_power_skip0.v1":
            raise ValueError("动作协议必须保留SKIP=0和现有9551候选顺序")
        for name, value in (("candidate_chunk_size", m.candidate_chunk_size),
                            ("num_groups", p.num_groups), ("num_slots", p.num_slots),
                            ("max_block_length", e.max_block_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}必须为正整数")
        for name, value in (("traffic_scale", self.data.traffic_scale),
                            ("slot_bandwidth_hz", p.slot_bandwidth_hz),
                            ("frequency_start_hz", p.frequency_start_hz),
                            ("earth_radius_km", p.earth_radius_km),
                            ("satellite_radius_km", p.satellite_radius_km),
                            ("noise_temperature_k", p.noise_temperature_k),
                            ("reward_scale", e.reward_scale), ("actor_lr", t.actor_lr),
                            ("critic_lr", t.critic_lr), ("alpha_lr", t.alpha_lr),
                            ("initial_alpha", t.initial_alpha), ("gradient_clip_norm", t.gradient_clip_norm)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}必须为有限正数")
        if p.satellite_radius_km <= p.earth_radius_km:
            raise ValueError("卫星轨道半径必须大于地球半径")
        if isinstance(e.power_budget_w, bool) or not isinstance(e.power_budget_w, (int, float)) or not math.isfinite(e.power_budget_w) or e.power_budget_w < 0:
            raise ValueError("power_budget_w必须为非负有限数")
        for name in ("sinr_margin_db", "tx_gain_peak_dbi", "rx_gain_peak_dbi",
                     "satellite_latitude_deg", "satellite_longitude_deg"):
            if not math.isfinite(getattr(p, name)):
                raise ValueError(f"physics.{name}必须为有限数")
        if abs(p.satellite_latitude_deg) > 90 or abs(p.satellite_longitude_deg) > 180:
            raise ValueError("卫星经纬度超物理范围")
        if self.data.rate_unit != "Mbps" or self.data.source_schema != "legacy_lon_in_lat":
            raise ValueError("Anchor v1只接受已冻结的CSV字段与Mbps语义")
        if p.beamwidth_kind != "ground_angular_diameter_deg" or self.data.beamwidth_kind != p.beamwidth_kind:
            raise ValueError("波束宽度语义不兼容")
        if p.group_profile != "shared_grid_parity_isolation_v1" or self.data.group_assignment != "canonical_demand_rank_mod8":
            raise ValueError("group/polarization规则不兼容")
        if self.data.service_order not in ("demand_desc", "raw", "random"):
            raise ValueError("未知服务顺序")
        if any(not isinstance(name, str) or not name for name in (
                self.schema_version, self.data.dataset_version, p.physics_version,
                e.reward_version, e.metric_version, e.observation_version, m.model_version)):
            raise ValueError("所有语义版本字符串必须非空")
        if not 0 <= e.budget_tolerance_w <= 1e-6 or not 0 <= e.satisfaction_tolerance <= 1e-4:
            raise ValueError("约束容差越界")
        if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in e.power_levels_w):
            raise ValueError("功率档必须为有限正数")
        if not math.isfinite(t.max_wall_seconds) or t.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds必须为非负有限数")
        if not isinstance(t.checkpoint_compression, bool):
            raise ValueError("checkpoint_compression必须为布尔值")
        if t.checkpoint_compression:
            raise ValueError("Anchor v1检查点仅支持未压缩原子格式")
        telemetry = self.telemetry
        for name in ("keyframe_interval", "update_window", "queue_max_messages", "queue_max_bytes"):
            value = getattr(telemetry, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"telemetry.{name}必须为正整数")
        if telemetry.trace_level != "basic_all" or not math.isfinite(telemetry.flush_interval) or telemetry.flush_interval <= 0:
            raise ValueError("telemetry协议或刷新周期不合法")
        if not math.isfinite(telemetry.enqueue_timeout) or telemetry.enqueue_timeout <= 0:
            raise ValueError("telemetry.enqueue_timeout必须为有限正数")

    def semantic_versions(self):
        adapter = "modern205.v1" if self.env.observation_adapter == "modern205" else "legacy_scaled205_new_physics.v1"
        return {"schema_version": self.schema_version, "physics_version": self.physics.physics_version,
                "observation_version": self.env.observation_version, "observation_adapter_version": adapter,
                "action_version": self.env.action_version, "reward_version": self.env.reward_version,
                "metric_version": self.env.metric_version, "model_version": self.model.model_version}


def load_config(path=None):
    if path is None:
        result = Config()
    else:
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        result = Config.from_dict(value)
    result.validate()
    return result


ASSUMPTIONS = [
    "legacy_scaled205仅恢复旧输入数值尺度，底层仍为新物理账本，不等价于旧环境",
    "旧权重可能见过现有CSV，迁移结果只能作为回归诊断",
    "8个attention head与dropout=0.1来自冻结旧源码默认值；state_dict不含这些超参，不能证明历史运行实际配置",
    "traffic_scale、group规则、宽度与margin来源仍待上游确认",
]
