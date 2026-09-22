"""完整合法动作上的模型；模型构造不重播种、不执行文件系统操作。"""
import numpy as np
import torch
from torch import nn


def masked_distribution(logits, mask, dim=-1):
    """空前缀返回全零；非法项的 probability/log_probability 都用有限零。"""
    mask = mask.bool()
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(any_valid, safe, torch.zeros_like(safe))
    logp = torch.log_softmax(safe, dim=dim)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    return torch.where(mask, logp.exp(), torch.zeros_like(logp)), logp


def collate_observations(observations, device="cpu"):
    """可变实体数仅对实体轴补零；padding mask 的 True 明确表示有效实体。"""
    if not observations:
        raise ValueError("观察批不能为空")
    entity_keys = ("beam_static", "beam_dynamic", "group_id", "order_rank", "entity_mask",
                   "demand_mask", "status", "allocation_start", "allocation_length", "allocation_power_w")
    lengths = [len(o["entity_mask"]) for o in observations]
    max_entities = max(1, max(lengths))
    result = {}
    for key in entity_keys:
        example = np.asarray(observations[0][key])
        array = np.zeros((len(observations), max_entities) + example.shape[1:], dtype=example.dtype)
        for i, obs in enumerate(observations):
            array[i, :lengths[i]] = obs[key]
        result[key] = torch.as_tensor(array, device=device)
    for key in ("current_slot_features", "global_features", "current_beam_index", "valid_action_mask", "terminal"):
        result[key] = torch.as_tensor(np.asarray([o[key] for o in observations]), device=device)
    if "legacy205" in observations[0]:
        result["legacy205"] = torch.as_tensor(np.asarray([o["legacy205"] for o in observations]), device=device)
    for key in ("beam_static", "beam_dynamic", "current_slot_features", "global_features", "legacy205"):
        if key in result:
            result[key] = result[key].float()
            if not torch.isfinite(result[key]).all():
                raise FloatingPointError(f"观察字段 {key} 存在非有限值")
    return result


def _transformer(width, heads, layers):
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0., batch_first=True,
                                      activation="gelu", norm_first=False)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ObservationEncoder(nn.Module):
    """共用输出维度；新增CNN上下文残差是受控修复假设，非性能提升证明。"""
    def __init__(self, config, example):
        super().__init__()
        m, p = config.model, config.physics
        self.mode, self.width, self.num_slots = m.encoder, m.d_model, p.num_slots
        self.slot_mode = m.spectrum_tokens
        if self.mode not in ("legacy14", "summary", "full_pool", "full_attention", "mlp_local", "cnn_local", "cnn_attention_residual"):
            raise ValueError(f"未知编码器 {self.mode}")
        if self.slot_mode not in ("slots", "blocks10"):
            raise ValueError("spectrum_tokens 必须为 slots 或 blocks10")
        if self.mode in ("mlp_local", "cnn_local", "cnn_attention_residual"):
            local_size=len(np.asarray(example["legacy205"]))
            if local_size != 5+2*p.num_slots:
                raise ValueError("局部MLP/CNN基线输入必须是5个属性与两条N槽频谱向量")
            if self.mode=="mlp_local":
                self.local_network=nn.Sequential(nn.Linear(local_size,m.d_model),nn.ReLU(),nn.Linear(m.d_model,m.d_model),nn.ReLU())
            else:
                self.local_convolutions=nn.Sequential(nn.Conv1d(2,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU())
                self.local_network=nn.Sequential(nn.Linear(5+64*p.num_slots,m.d_model),nn.ReLU(),nn.Linear(m.d_model,m.d_model),nn.ReLU())
            if self.mode == "cnn_attention_residual":
                # 保持CNN参数名和逐槽flatten路径；只有新增参数使用context_前缀。
                features = np.asarray(example["beam_static"]).shape[-1] + np.asarray(example["beam_dynamic"]).shape[-1] + 3
                self.context_residual_scale = m.context_residual_scale
                self.context_beam_mlp = nn.Sequential(nn.Linear(features, m.d_model), nn.GELU(), nn.Linear(m.d_model, m.d_model))
                self.context_group_embedding = nn.Embedding(p.num_groups, m.d_model)
                self.context_status_embedding = nn.Embedding(4, m.d_model)
                self.context_attention = _transformer(m.d_model, m.attention_heads, m.context_layers)
                self.context_global_mlp = nn.Sequential(nn.Linear(len(example["global_features"]), m.d_model), nn.GELU())
                self.context_query = nn.Linear(m.d_model * 3, m.d_model)
                self.context_read_beams = nn.MultiheadAttention(m.d_model, m.attention_heads, dropout=0., batch_first=True)
                self.context_out = nn.Linear(m.d_model * 3, m.d_model)
                nn.init.zeros_(self.context_out.weight)
                nn.init.zeros_(self.context_out.bias)
            return
        if self.mode in ("legacy14", "summary"):
            if p.num_slots != 100 or np.asarray(example.get("legacy205", [])).shape != (205,):
                raise ValueError("legacy14/summary 需要100槽与205维兼容观察")
            self.attributes = nn.ModuleList([nn.Linear(n, m.d_model) for n in (2, 1, 1, 1)])
            self.legacy_slot = nn.Linear(20 if self.slot_mode == "blocks10" else 2, m.d_model)
            count = 14 if self.slot_mode == "blocks10" else 104
            self.legacy_positions = nn.Parameter(torch.zeros(1, count, m.d_model))
            nn.init.normal_(self.legacy_positions, std=.02)
            self.legacy_attention = _transformer(m.d_model, m.attention_heads, m.encoder_layers)
            if self.mode == "summary":
                self.summary_fusion = nn.Sequential(nn.Linear(m.d_model + len(example["global_features"]), m.d_model), nn.GELU())
            return
        features = np.asarray(example["beam_static"]).shape[-1] + np.asarray(example["beam_dynamic"]).shape[-1] + 3
        self.beam_mlp = nn.Sequential(nn.Linear(features, m.d_model), nn.GELU(), nn.Linear(m.d_model, m.d_model))
        self.group_embedding = nn.Embedding(p.num_groups, m.d_model)
        self.status_embedding = nn.Embedding(4, m.d_model)
        self.beam_attention = _transformer(m.d_model, m.attention_heads, m.encoder_layers) if self.mode == "full_attention" else None
        self.global_mlp = nn.Sequential(nn.Linear(len(example["global_features"]), m.d_model), nn.GELU())
        slot_features = np.asarray(example["current_slot_features"]).shape[-1]
        self.block_size = 10 if self.slot_mode == "blocks10" else 1
        self.slot_count = (p.num_slots + self.block_size - 1) // self.block_size
        self.slot_projection = nn.Linear(slot_features * self.block_size, m.d_model)
        self.slot_positions = nn.Parameter(torch.zeros(1, self.slot_count, m.d_model))
        nn.init.normal_(self.slot_positions, std=.02)
        self.slot_attention = _transformer(m.d_model, m.attention_heads, m.encoder_layers)
        self.query = nn.Linear(m.d_model * 2, m.d_model)
        self.read_beams = nn.MultiheadAttention(m.d_model, m.attention_heads, dropout=0., batch_first=True)
        self.read_slots = nn.MultiheadAttention(m.d_model, m.attention_heads, dropout=0., batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(m.d_model * 4, m.d_model), nn.GELU(), nn.LayerNorm(m.d_model))

    def _context_residual(self, obs, local_z):
        valid = obs["entity_mask"].bool()
        count = valid.sum(1).clamp_min(1).float().unsqueeze(-1)
        extra = torch.stack((obs["order_rank"].float() / count,
                             obs["demand_mask"].float(), valid.float()), -1)
        features = torch.cat((obs["beam_static"], obs["beam_dynamic"], extra), -1)
        beams = (self.context_beam_mlp(features)
                 + self.context_group_embedding(obs["group_id"].long())
                 + self.context_status_embedding(obs["status"].long()))
        beams = beams.masked_fill(~valid.unsqueeze(-1), 0.)
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(1), 0] = True
        beams = self.context_attention(beams, src_key_padding_mask=~safe_valid)
        beams = beams.masked_fill(~valid.unsqueeze(-1), 0.)
        indices = obs["current_beam_index"].long()
        current = beams[torch.arange(len(beams), device=beams.device), indices.clamp_min(0)]
        current = torch.where((indices >= 0).unsqueeze(-1), current, torch.zeros_like(current))
        global_z = self.context_global_mlp(obs["global_features"])
        query = self.context_query(torch.cat((local_z, current, global_z), -1)).unsqueeze(1)
        context = self.context_read_beams(query, beams, beams, key_padding_mask=~safe_valid, need_weights=False)[0][:, 0]
        # 第一更新只训练零初始化出口；出口非零后梯度才能进入其余上下文参数。
        projected = self.context_out(torch.cat((current, global_z, context), -1))
        magnitude = local_z.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1.)
        return self.context_residual_scale * magnitude * torch.tanh(projected)

    def forward(self, obs):
        if self.mode in ("mlp_local","cnn_local","cnn_attention_residual"):
            x=obs["legacy205"]
            if self.mode in ("cnn_local", "cnn_attention_residual"):
                spectrum=torch.stack((x[:,5:5+self.num_slots],x[:,5+self.num_slots:]),dim=1)
                x=torch.cat((x[:,:5],self.local_convolutions(spectrum).flatten(1)),dim=1)
            local_z = self.local_network(x)
            if self.mode == "cnn_attention_residual":
                # 不在相加后做归一化，保证迁移CNN权重且出口为零时逐张量等价。
                return local_z + self._context_residual(obs, local_z)
            return local_z
        if self.mode in ("legacy14", "summary"):
            x = obs["legacy205"]
            attrs = [layer(part).unsqueeze(1) for layer, part in zip(self.attributes, (x[:, :2], x[:, 2:3], x[:, 3:4], x[:, 4:5]))]
            if self.slot_mode == "blocks10":
                slot = torch.cat((x[:, 5:105].reshape(-1, 10, 10), x[:, 105:205].reshape(-1, 10, 10)), -1)
            else:
                slot = torch.stack((x[:, 5:105], x[:, 105:205]), -1)
            tokens = torch.cat(attrs + [self.legacy_slot(slot)], 1) + self.legacy_positions
            z = self.legacy_attention(tokens)[:, 0]
            return self.summary_fusion(torch.cat((z, obs["global_features"]), -1)) if self.mode == "summary" else z
        valid = obs["entity_mask"].bool()
        # 分母基于实际实体数，padding的加入不能改变order rank语义。
        count = valid.sum(1).clamp_min(1).float().unsqueeze(-1)
        rank = obs["order_rank"].float() / count
        extra = torch.stack((rank, obs["demand_mask"].float(), valid.float()), -1)
        x = torch.cat((obs["beam_static"], obs["beam_dynamic"], extra), -1)
        beams = self.beam_mlp(x) + self.group_embedding(obs["group_id"].long()) + self.status_embedding(obs["status"].long())
        beams = beams.masked_fill(~valid.unsqueeze(-1), 0.)
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(1), 0] = True
        if self.beam_attention is not None:
            beams = self.beam_attention(beams, src_key_padding_mask=~safe_valid)
        beams = beams.masked_fill(~valid.unsqueeze(-1), 0.)
        current_index = obs["current_beam_index"].long().clamp_min(0)
        current = beams[torch.arange(len(beams), device=beams.device), current_index]
        current = torch.where((obs["current_beam_index"] >= 0).unsqueeze(-1), current, torch.zeros_like(current))
        global_z = self.global_mlp(obs["global_features"])
        slots = obs["current_slot_features"]
        padding = self.slot_count * self.block_size - slots.shape[1]
        if padding:
            slots = torch.nn.functional.pad(slots, (0, 0, 0, padding))
        slots = slots.reshape(len(slots), self.slot_count, -1)
        slots = self.slot_attention(self.slot_projection(slots) + self.slot_positions)
        query = self.query(torch.cat((current, global_z), -1)).unsqueeze(1)
        if self.mode == "full_pool":
            beam_context = beams.sum(1) / count
        else:
            beam_context = self.read_beams(query, beams, beams, key_padding_mask=~safe_valid, need_weights=False)[0][:, 0]
        slot_context = self.read_slots(query, slots, slots, need_weights=False)[0][:, 0]
        return self.fusion(torch.cat((current, global_z, beam_context, slot_context), -1))


class Actor(nn.Module):
    def __init__(self, config, action_spec, example):
        super().__init__()
        self.encoder = ObservationEncoder(config, example)
        self.kind = config.model.actor
        self.n, self.l, self.k = action_spec.num_slots, action_spec.max_block_length, len(action_spec.power_levels_w)
        d = config.model.d_model
        self.register_buffer("candidates", torch.as_tensor(np.array(action_spec.candidates), dtype=torch.long))
        self.gate = nn.Linear(d, 2)
        self.start = nn.Linear(d, self.n)
        if self.kind == "independent":
            self.length = nn.Linear(d, self.l)
            self.power = nn.Linear(d, self.k)
        elif self.kind == "conditional":
            self.start_embedding = nn.Embedding(self.n, d)
            self.length_embedding = nn.Embedding(self.l, d)
            self.length = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.l))
            self.power = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.k))
        else:
            raise ValueError(f"未知 Actor {self.kind}")

    def distribution_from_encoded(self, z, mask):
        if not mask[:, 0].all():
            raise ValueError("策略仅接受非terminal状态，SKIP必须合法")
        c = self.candidates[1:]
        f, l, p = c[:, 0], c[:, 1] - 1, c[:, 2]
        legal = mask[:, 1:].bool()
        gate_mask = torch.stack((mask[:, 0], legal.any(-1)), -1)
        gp, gl = masked_distribution(self.gate(z), gate_mask)
        if self.kind == "independent":
            scores = self.start(z)[:, f] + self.length(z)[:, l] + self.power(z)[:, p]
            ap, al = masked_distribution(scores, legal)
        else:
            dense = torch.zeros((len(z), self.n, self.l, self.k), dtype=torch.bool, device=z.device)
            dense[:, f, l, p] = legal
            length_mask = dense.any(-1)
            start_mask = length_mask.any(-1)
            _, sl = masked_distribution(self.start(z), start_mask)
            starts = z[:, None, :] + self.start_embedding.weight[None, :, :]
            _, ll = masked_distribution(self.length(starts), length_mask)
            pairs = starts[:, :, None, :] + self.length_embedding.weight[None, None, :, :]
            _, pl = masked_distribution(self.power(pairs), dense)
            al = sl[:, f] + ll[:, f, l] + pl[:, f, l, p]
            ap = torch.where(legal, al.exp(), torch.zeros_like(al))
        logp = torch.cat((gl[:, :1], gl[:, 1:2] + al), -1)
        probability = torch.cat((gp[:, :1], gp[:, 1:2] * ap), -1)
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        return probability, logp

    def forward(self, obs):
        return self.distribution_from_encoded(self.encoder(obs), obs["valid_action_mask"])


class Critic(nn.Module):
    def __init__(self, config, action_spec, example):
        super().__init__()
        self.encoder = ObservationEncoder(config, example)
        self.kind = config.model.critic
        d = config.model.d_model
        self.register_buffer("candidates", torch.as_tensor(np.array(action_spec.candidates), dtype=torch.long))
        if self.kind == "additive":
            self.start = nn.Linear(d, action_spec.num_slots)
            self.length = nn.Linear(d, action_spec.max_block_length)
            self.power = nn.Linear(d, len(action_spec.power_levels_w))
            self.skip = nn.Linear(d, 1)
        elif self.kind == "joint":
            self.kind_embedding = nn.Embedding(2, d)
            self.start_embedding = nn.Embedding(action_spec.num_slots, d)
            self.length_embedding = nn.Embedding(action_spec.max_block_length, d)
            self.power_embedding = nn.Embedding(len(action_spec.power_levels_w), d)
            self.q = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            raise ValueError(f"未知 Critic {self.kind}")

    def values_from_encoded(self, z, action_ids, paired=False):
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=z.device)
        c = self.candidates[ids]
        skip = ids == 0
        f, l, p = c[..., 0].clamp_min(0), (c[..., 1] - 1).clamp_min(0), c[..., 2].clamp_min(0)
        if self.kind == "additive":
            if paired:
                rows = torch.arange(len(z), device=z.device)
                value = self.start(z)[rows, f] + self.length(z)[rows, l] + self.power(z)[rows, p]
                return torch.where(skip, self.skip(z)[:, 0], value)
            value = self.start(z)[:, f] + self.length(z)[:, l] + self.power(z)[:, p]
            return torch.where(skip[None, :], self.skip(z), value)
        embedding = self.kind_embedding((~skip).long())
        alloc = self.start_embedding(f) + self.length_embedding(l) + self.power_embedding(p)
        embedding = embedding + torch.where(skip.unsqueeze(-1), torch.zeros_like(alloc), alloc)
        if paired:
            return self.q(torch.cat((z, embedding), -1)).squeeze(-1)
        states = z[:, None, :].expand(-1, len(ids), -1)
        actions = embedding[None, :, :].expand(len(z), -1, -1)
        return self.q(torch.cat((states, actions), -1)).squeeze(-1)

    def forward(self, obs, action_ids):
        return self.values_from_encoded(self.encoder(obs), action_ids, paired=True)
