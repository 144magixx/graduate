"""版本化配置；所有文件系统根目录由 project_paths 管理。"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class DataConfig:
    source_schema: str = "legacy_lon_in_lat"
    rate_unit: str = "Mbps"
    traffic_scale: float = 0.25
    beamwidth_kind: str = "ground_angular_diameter_deg"
    group_assignment: str = "canonical_demand_rank_mod8"
    service_order: str = "demand_desc"
    augment: bool = True
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
    observation_version: str = "obs_full_v2"
    action_version: str = "contiguous_total_power.v1"
    budget_tolerance_w: float = 1e-8
    satisfaction_tolerance: float = 1e-8

    def __post_init__(self):
        self.power_levels_w = tuple(self.power_levels_w)


@dataclass
class ModelConfig:
    model_version: str = "spectrum_tsac.v1"
    encoder: str = "cnn_attention_residual"
    actor: str = "independent"
    critic: str = "additive"
    d_model: int = 128
    attention_heads: int = 4
    encoder_layers: int = 2
    context_layers: int = 1
    context_residual_scale: float = 0.25
    dropout: float = 0.0
    spectrum_tokens: str = "slots"
    candidate_chunk_size: int = 512


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    episodes: int = 10
    max_env_steps: int = 0
    batch_size: int = 256
    microbatch_size: int = 32
    replay_capacity: int = 20000
    replay_mode: str = "episode_balanced"
    warmup_steps: int = 2000
    update_schedule: str = "episode"
    updates_per_step: int = 1
    updates_per_episode: int = 1
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    initial_alpha: float = 0.01
    target_entropy_ratio: float = 0.5
    gamma: float = 1.0
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
        for key, typ in (("data", DataConfig), ("physics", PhysicsConfig),
                         ("env", EnvConfig), ("model", ModelConfig),
                         ("train", TrainConfig), ("telemetry", TelemetryConfig)):
            if key in value:
                value[key] = typ(**value[key])
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        import math
        p, e, m, t = self.physics, self.env, self.model, self.train
        for name, value in (("num_groups",p.num_groups),("num_slots",p.num_slots),("max_block_length",e.max_block_length),
                            ("d_model",m.d_model),("attention_heads",m.attention_heads),("encoder_layers",m.encoder_layers),("context_layers",m.context_layers),
                            ("batch_size",t.batch_size),("microbatch_size",t.microbatch_size),("replay_capacity",t.replay_capacity),
                            ("candidate_chunk_size",m.candidate_chunk_size),("torch_num_threads",t.torch_num_threads)):
            if isinstance(value,bool) or not isinstance(value,int) or value < 1:
                raise ValueError(f"{name}必须为正整数")
        for name in ("episodes","max_env_steps","warmup_steps","updates_per_episode","updates_per_step"):
            value=getattr(t,name)
            if isinstance(value,bool) or not isinstance(value,int) or value < 0:
                raise ValueError(f"{name}必须为非负整数")
        if t.update_schedule not in ("episode", "env_step"):
            raise ValueError("update_schedule 必须为 episode 或 env_step")
        if isinstance(t.checkpoint_every,bool) or not isinstance(t.checkpoint_every,int) or t.checkpoint_every<1:
            raise ValueError("checkpoint_every必须为正整数")
        if not math.isfinite(t.max_wall_seconds) or t.max_wall_seconds<0:
            raise ValueError("max_wall_seconds必须为非负有限数")
        if not isinstance(t.checkpoint_compression,bool):raise ValueError('checkpoint_compression必须为布尔值')
        for name, value in (("slot_bandwidth_hz",p.slot_bandwidth_hz),("frequency_start_hz",p.frequency_start_hz),
                            ("earth_radius_km",p.earth_radius_km),("satellite_radius_km",p.satellite_radius_km),
                            ("noise_temperature_k",p.noise_temperature_k),("traffic_scale",self.data.traffic_scale),
                            ("actor_lr",t.actor_lr),("critic_lr",t.critic_lr),("alpha_lr",t.alpha_lr),
                            ("initial_alpha",t.initial_alpha),("gradient_clip_norm",t.gradient_clip_norm)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}必须为有限正数")
        if p.satellite_radius_km <= p.earth_radius_km:
            raise ValueError("卫星轨道半径必须大于地球半径")
        for name in ("sinr_margin_db","tx_gain_peak_dbi","rx_gain_peak_dbi","satellite_latitude_deg","satellite_longitude_deg"):
            if not math.isfinite(getattr(p,name)):
                raise ValueError(f"{name}必须为有限数")
        if abs(p.satellite_latitude_deg)>90 or abs(p.satellite_longitude_deg)>180:
            raise ValueError("卫星角坐标越界")
        if p.beamwidth_kind != "ground_angular_diameter_deg" or self.data.beamwidth_kind != p.beamwidth_kind:
            raise ValueError("首版只支持地面角直径语义")
        if p.group_profile != "shared_grid_parity_isolation_v1" or self.data.group_assignment != "canonical_demand_rank_mod8":
            raise ValueError("不支持未验收的group规则")
        if self.data.width_augmentation:
            raise ValueError("宽度增强须有上游几何/归属验证，首版默认禁止")
        if self.data.service_order not in ("demand_desc","raw","random"):
            raise ValueError("未定义服务顺序")
        if m.encoder not in ("legacy14","summary","full_pool","full_attention","mlp_local","cnn_local","cnn_attention_residual") or m.actor not in ("independent","conditional") or m.critic not in ("additive","joint"):
            raise ValueError("未知模型消融配置")
        if isinstance(m.context_residual_scale, bool) or not isinstance(m.context_residual_scale, (int, float)) or not math.isfinite(m.context_residual_scale) or not 0 <= m.context_residual_scale <= 1:
            raise ValueError("context_residual_scale必须为0到1之间的有限数")
        if m.spectrum_tokens not in ("slots","blocks10"):
            raise ValueError("未知频谱token配置")
        if not math.isfinite(e.reward_scale) or e.reward_scale <= 0 or not 0 <= e.budget_tolerance_w <= 1e-6 or not 0 <= e.satisfaction_tolerance <= 1e-4:
            raise ValueError("奖励尺度或约束容差不合法")
        if p.num_slots < 1 or p.num_groups < 1 or e.max_block_length < 1:
            raise ValueError("槽数、组数、连续块长度必须为正整数")
        if not e.power_levels_w or any(isinstance(x,bool) or not math.isfinite(x) or x <= 0 for x in e.power_levels_w):
            raise ValueError("功率档必须为有限正数")
        if sorted(set(e.power_levels_w)) != list(e.power_levels_w):
            raise ValueError("功率档必须严格递增")
        if not math.isfinite(e.power_budget_w) or e.power_budget_w < 0:
            raise ValueError("总功率预算必须非负有限")
        if m.d_model % m.attention_heads or m.dropout != 0:
            raise ValueError("模型宽度需整除注意力头数，首版 dropout 必须为 0")
        if min(t.batch_size, t.microbatch_size, m.candidate_chunk_size) < 1:
            raise ValueError("批量及候选块必须为正")
        if not 0 <= t.gamma <= 1 or not 0 < t.tau <= 1 or not 0 <= t.target_entropy_ratio <= 1:
            raise ValueError("gamma/tau/熵目标比例越界")
        if p.power_semantics != "beam_total_uniform_slots" or p.rate_model != "shannon_with_sinr_margin":
            raise ValueError("此实现仅支持已验收的总功率均分 Shannon 语义")
        for name in ("keyframe_interval","update_window","queue_max_messages","queue_max_bytes"):
            value=getattr(self.telemetry,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"telemetry.{name}必须为正整数")
        if self.telemetry.trace_level != "basic_all":
            raise ValueError("首版必须保留全部回合基本轨迹")
        if not math.isfinite(self.telemetry.enqueue_timeout) or self.telemetry.enqueue_timeout<=0:
            raise ValueError("日志背压超时必须有限且为正")

    def semantic_versions(self):
        return dict(schema_version=self.schema_version, physics_version=self.physics.physics_version,
                    observation_version=self.env.observation_version, action_version=self.env.action_version,
                    reward_version=self.env.reward_version, metric_version=self.env.metric_version,
                    model_version=self.model.model_version)


def load_config(path=None):
    if path is None:
        result = Config()
    else:
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
        result = Config.from_dict(value)
    result.validate()
    return result


ASSUMPTIONS = [
    "legacy_lon_in_lat 坐标适配来自旧代码，未获上游来源确认",
    "traffic_scale=0.25 的业务解释未获上游确认",
    "波束宽度按地面角直径处理，方向图继承旧近似，未获载荷确认",
    "8逻辑池共享频率网格、group奇偶极化理想隔离，未获载荷确认",
    "5 dB 按 SINR margin 处理，未获损耗来源确认",
]
