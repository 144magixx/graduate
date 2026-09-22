"""旧14-token纯Transformer core与新9551动作桥接。"""
import numpy as np
import torch
from torch import nn


def masked_distribution(logits, mask, dim=-1):
    mask = mask.bool()
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(any_valid, safe, torch.zeros_like(safe))
    logp = torch.log_softmax(safe, dim=dim)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    return torch.where(mask, logp.exp(), torch.zeros_like(logp)), logp


def collate_observations(observations, device="cpu"):
    if not observations:
        raise ValueError("观察批不能为空")
    adapter_versions = {o.get("observation_adapter_version") for o in observations}
    if len(adapter_versions) != 1 or None in adapter_versions:
        raise ValueError("观察适配器版本缺失或混批")
    features = np.asarray([o["anchor205"] for o in observations], dtype=np.float32)
    masks = np.asarray([o["valid_action_mask"] for o in observations], dtype=bool)
    if features.ndim != 2 or features.shape[1] != 205:
        raise ValueError("Anchor输入必须是205维")
    if not np.isfinite(features).all():
        raise FloatingPointError("观察存在非有限值")
    return {"anchor205": torch.as_tensor(features, device=device),
            "valid_action_mask": torch.as_tensor(masks, device=device),
            "terminal": torch.as_tensor([o["terminal"] for o in observations], device=device)}


class PureTransformer14(nn.Module):
    """与冻结旧PolicyNet/ValueNet相同的token化与Transformer core。"""
    def __init__(self, config):
        super().__init__()
        d = config.model.d_model
        self.geo_embed = nn.Linear(2, d)
        self.rate_embed = nn.Linear(1, d)
        self.bw_embed = nn.Linear(1, d)
        self.power_embed = nn.Linear(1, d)
        self.slot_embed = nn.Linear(20, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.model.attention_heads,
            dim_feedforward=config.model.feedforward_multiplier * d,
            dropout=config.model.dropout, activation=config.model.activation,
            batch_first=False, norm_first=config.model.norm_first)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.model.encoder_layers)

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != 205:
            raise ValueError("PureTransformer14需要[B,205]输入")
        batch = values.shape[0]
        row = values[:, 5:105].reshape(batch, 10, 10)
        occupancy = values[:, 105:205].reshape(batch, 10, 10)
        tokens = torch.cat((
            self.geo_embed(values[:, :2]).unsqueeze(1),
            self.rate_embed(values[:, 2:3]).unsqueeze(1),
            self.bw_embed(values[:, 3:4]).unsqueeze(1),
            self.power_embed(values[:, 4:5]).unsqueeze(1),
            self.slot_embed(torch.cat((row, occupancy), dim=-1))), dim=1)
        hidden = self.encoder((tokens + self.pos_embed).permute(1, 0, 2)).permute(1, 0, 2)
        return hidden[:, 0]


class Actor(nn.Module):
    """旧三分量头加显式SKIP头；联合mask后形成新策略。"""
    def __init__(self, config, action_spec, _example=None):
        super().__init__()
        self.core = PureTransformer14(config)
        d = config.model.d_model
        self.freq_head = nn.Linear(d, action_spec.num_slots)
        self.slots_head = nn.Linear(d, action_spec.max_block_length)
        self.power_head = nn.Linear(d, len(action_spec.power_levels_w))
        self.skip_head = nn.Linear(d, 1)
        self.register_buffer("candidates", torch.as_tensor(action_spec.candidates, dtype=torch.long))

    def component_logits(self, observation):
        hidden = self.core(observation["anchor205"])
        return hidden, self.freq_head(hidden), self.slots_head(hidden), self.power_head(hidden)

    def forward(self, observation):
        hidden, freq, slots, power = self.component_logits(observation)
        candidates = self.candidates[1:]
        scores = freq[:, candidates[:, 0]] + slots[:, candidates[:, 1] - 1] + power[:, candidates[:, 2]]
        logits = torch.cat((self.skip_head(hidden), scores), dim=-1)
        return masked_distribution(logits, observation["valid_action_mask"])


class Critic(nn.Module):
    """旧三分量Q相加；SKIP使用独立新头。"""
    def __init__(self, config, action_spec, _example=None):
        super().__init__()
        self.core = PureTransformer14(config)
        d = config.model.d_model
        self.q_freq = nn.Linear(d, action_spec.num_slots)
        self.q_slots = nn.Linear(d, action_spec.max_block_length)
        self.q_power = nn.Linear(d, len(action_spec.power_levels_w))
        self.q_skip = nn.Linear(d, 1)
        self.register_buffer("candidates", torch.as_tensor(action_spec.candidates, dtype=torch.long))

    def component_values(self, observation):
        hidden = self.core(observation["anchor205"])
        return hidden, self.q_freq(hidden), self.q_slots(hidden), self.q_power(hidden)

    def values_from_encoded(self, hidden, action_ids, paired=False):
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=hidden.device)
        candidate = self.candidates[ids]
        skip = ids == 0
        starts = candidate[..., 0].clamp_min(0)
        lengths = (candidate[..., 1] - 1).clamp_min(0)
        powers = candidate[..., 2].clamp_min(0)
        fq, sl, pw = self.q_freq(hidden), self.q_slots(hidden), self.q_power(hidden)
        if paired:
            rows = torch.arange(len(hidden), device=hidden.device)
            value = fq[rows, starts] + sl[rows, lengths] + pw[rows, powers]
            return torch.where(skip, self.q_skip(hidden)[:, 0], value)
        value = fq[:, starts] + sl[:, lengths] + pw[:, powers]
        return torch.where(skip.unsqueeze(0), self.q_skip(hidden), value)

    def forward(self, observation, action_ids):
        hidden = self.core(observation["anchor205"])
        return self.values_from_encoded(hidden, action_ids, paired=True)
