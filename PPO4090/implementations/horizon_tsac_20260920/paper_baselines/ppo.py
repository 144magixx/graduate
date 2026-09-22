"""论文第IV节MLP-PPO的合法动作适配；未刊训练参数显式保存。

论文仅说明三独立离散Actor头与状态价值Critic。这里使用SKIP门控、
完整合法ALLOC集合上的三头logit加和归一化，采样和PPO比率共用此联合分布。
本模块没有经验回放或文件I/O；调用方必须采集当前策略rollout并仅提交一次。
"""
import copy
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
from torch import nn

from ..models.networks import masked_distribution
from ..rl.replay import Transition


@dataclass(frozen=True)
class PPOHyperparameters:
    gae_lambda: float = .95
    clip_epsilon: float = .2
    epochs: int = 10
    minibatch_size: int = 64
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    normalize_advantages: bool = True

    def validate(self):
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_epsilon < 1:
            raise ValueError("PPO的GAE lambda或clip epsilon越界")
        for name in ("epochs", "minibatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"PPO {name}必须为正整数")
        for name in ("value_coefficient", "entropy_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"PPO {name}必须为有限非负数")


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           gamma=1., gae_lambda=.95, episode_boundaries=None):
    """外部截断继续bootstrap，但终止/截断/采集边界都切断GAE跨段递推。"""
    rewards, values, next_values = [np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values)]
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    shape = rewards.shape
    if len(shape) != 1 or any(x.shape != shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE各字段必须是等长一维数组")
    if not all(np.isfinite(x).all() for x in (rewards, values, next_values)):
        raise FloatingPointError("GAE输入包含非有限数")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE gamma/lambda越界")
    boundaries = np.zeros(shape, bool) if episode_boundaries is None else np.asarray(episode_boundaries, bool)
    if boundaries.shape != shape:
        raise ValueError("GAE回合边界数量错误")
    advantages = np.empty(shape, dtype=np.float64)
    running = 0.
    for index in range(len(rewards)-1, -1, -1):
        bootstrap = 0. if terminated[index] else next_values[index]
        delta = rewards[index]+gamma*bootstrap-values[index]
        continuation = not (terminated[index] or truncated[index] or boundaries[index])
        running = delta+gamma*gae_lambda*running*continuation
        advantages[index] = running
    return advantages, advantages+values


def clipped_policy_loss(new_log_probability, old_log_probability, advantages, epsilon=.2):
    ratio = (new_log_probability-old_log_probability).exp()
    clipped = ratio.clamp(1.-epsilon, 1.+epsilon)
    return -torch.minimum(ratio*advantages, clipped*advantages).mean(), ratio


def _mlp(input_dim, width):
    return nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())


class _Actor(nn.Module):
    def __init__(self, input_dim, width, action_spec):
        super().__init__()
        self.encoder = _mlp(input_dim, width)
        self.gate = nn.Linear(width, 2)
        self.start = nn.Linear(width, action_spec.num_slots)
        self.length = nn.Linear(width, action_spec.max_block_length)
        self.power = nn.Linear(width, len(action_spec.power_levels_w))
        self.register_buffer("candidates", torch.as_tensor(np.array(action_spec.candidates), dtype=torch.long))

    def forward(self, features, mask):
        if not mask[:, 0].all():
            raise ValueError("PPO策略仅接受SKIP合法的非终止状态")
        encoded = self.encoder(features)
        legal = mask[:, 1:]
        gate_mask = torch.stack((mask[:, 0], legal.any(-1)), -1)
        gate_probability, gate_log_probability = masked_distribution(self.gate(encoded), gate_mask)
        candidates = self.candidates[1:]
        scores = (self.start(encoded)[:, candidates[:, 0]]
                  +self.length(encoded)[:, candidates[:, 1]-1]
                  +self.power(encoded)[:, candidates[:, 2]])
        allocation_probability, allocation_log_probability = masked_distribution(scores, legal)
        probability = torch.cat((gate_probability[:, :1], gate_probability[:, 1:2]*allocation_probability), -1)
        log_probability = torch.cat((gate_log_probability[:, :1], gate_log_probability[:, 1:2]+allocation_log_probability), -1)
        return probability, torch.where(mask, log_probability, torch.zeros_like(log_probability))


class _Critic(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.encoder = _mlp(input_dim, width)
        self.value = nn.Linear(width, 1)

    def forward(self, features):
        return self.value(self.encoder(features)).squeeze(-1)


class BaselinePPO:
    def __init__(self, config, action_spec, observation_example, *, hyperparameters=None):
        config.validate()
        self.config = copy.deepcopy(config)
        self.action_spec = action_spec
        expected_signature = dict(num_slots=config.physics.num_slots, max_block_length=config.env.max_block_length,
                                  power_levels_w=list(config.env.power_levels_w))
        if self.action_signature() != expected_signature:
            raise ValueError("PPO配置与ActionSpec不一致")
        self.device = torch.device(config.train.device)
        self.hyperparameters = (hyperparameters if isinstance(hyperparameters, PPOHyperparameters)
                                else PPOHyperparameters(**(hyperparameters or {})))
        self.hyperparameters.validate()
        example = np.asarray(observation_example["legacy205"])
        if example.ndim != 1 or len(example) == 0:
            raise ValueError("PPO需要由example确定的一维legacy205输入")
        self.input_dim, self.width = len(example), config.model.d_model
        self.actor = _Actor(self.input_dim, self.width, action_spec).to(self.device)
        self.critic = _Critic(self.input_dim, self.width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.train.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.train.critic_lr)
        self.action_rng = torch.Generator(device=self.device)
        self.action_rng.manual_seed(config.train.seed)
        self.minibatch_rng = torch.Generator(device="cpu")
        self.minibatch_rng.manual_seed(config.train.seed+101)
        self.update_step = 0
        self.rollout_updates = 0

    def train(self, mode=True):
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def action_signature(self):
        return self.action_spec.signature()

    def _batch(self, observations, allow_terminal=False):
        features, masks = [], []
        for observation in observations:
            signature = observation.get("action_spec_signature")
            if signature is None or dict(signature, power_levels_w=list(signature["power_levels_w"])) != self.action_spec.signature():
                raise ValueError("PPO观察ActionSpec签名不兼容")
            if observation["terminal"] and not allow_terminal:
                raise ValueError("终止观察不能送入PPO策略")
            feature = np.asarray(observation["legacy205"], dtype=np.float32)
            if feature.shape != (self.input_dim,):
                raise ValueError("PPO观察维度与模型输入不一致")
            if not np.isfinite(feature).all():
                raise FloatingPointError("PPO观察包含非有限数")
            mask = self.action_spec.valid_actions(observation)
            if not np.array_equal(mask, observation["valid_action_mask"]):
                raise ValueError("PPO观察动作mask与纯函数重建不一致")
            features.append(feature)
            masks.append(mask)
        if not features:
            raise ValueError("PPO观察批不能为空")
        return (torch.as_tensor(np.stack(features), device=self.device),
                torch.as_tensor(np.stack(masks), device=self.device))

    @torch.no_grad()
    def probabilities(self, observation):
        features, mask = self._batch([observation])
        probability, _ = self.actor(features, mask)
        if not torch.isfinite(probability).all():
            raise FloatingPointError("PPO策略分布非有限")
        return probability[0].cpu().numpy()

    @torch.no_grad()
    def collect(self, observation):
        features, mask = self._batch([observation])
        probability, log_probability = self.actor(features, mask)
        action_id = torch.multinomial(probability[0], 1, generator=self.action_rng)[0]
        value = self.critic(features)[0]
        return (self.action_spec.decode(int(action_id.item())),
                float(log_probability[0, action_id].item()), float(value.item()))

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        if not deterministic:
            return self.collect(observation)[0]
        probability = self.probabilities(observation)
        return self.action_spec.decode(int(probability.argmax()))

    @torch.no_grad()
    def value(self, observation):
        if observation["terminal"]:
            return 0.
        features, _ = self._batch([observation])
        return float(self.critic(features)[0].item())

    def update_rollout(self, transitions, old_log_probs=None, old_values=None):
        """提交一次当前策略rollout；必须提供collect实际保存的旧logp与value。"""
        started = time.perf_counter()
        if old_log_probs is None or old_values is None:
            raise ValueError("必须同时提供collect采集的旧行为logp与value，禁止从当前网络重算")
        samples = [Transition.coerce(item) for item in transitions]
        if not samples:
            raise ValueError("PPO不能更新空rollout")
        for sample in samples:
            if dict(sample.versions) != self.config.semantic_versions():
                raise ValueError("PPO rollout语义版本不一致")
            if not 0 <= sample.action_id < len(self.action_spec) or not sample.observation["valid_action_mask"][sample.action_id]:
                raise ValueError("PPO rollout包含非法动作")
        self.train(True)
        features, masks = self._batch([sample.observation for sample in samples])
        next_features, _ = self._batch([sample.next_observation for sample in samples], allow_terminal=True)
        ids = torch.as_tensor([sample.action_id for sample in samples], dtype=torch.long, device=self.device)
        row_ids = torch.arange(len(samples), device=self.device)
        with torch.no_grad():
            _, current_logp = self.actor(features, masks)
            current_logp = current_logp[row_ids, ids]
            current_values = self.critic(features)
            next_values = self.critic(next_features).cpu().numpy()
        old_logp = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device).detach()
        old_value = torch.as_tensor(old_values, dtype=torch.float32, device=self.device).detach()
        if old_logp.shape != current_logp.shape or old_value.shape != current_values.shape:
            raise ValueError("PPO行为logp/value数量与rollout不一致")
        if not torch.isfinite(old_logp).all() or not torch.isfinite(old_value).all():
            raise FloatingPointError("PPO旧行为记录包含非有限数")
        if not torch.allclose(old_logp, current_logp, rtol=1e-5, atol=1e-4) or not torch.allclose(old_value, current_values, rtol=1e-5, atol=1e-4):
            raise ValueError("PPO拒绝陈旧或不同策略的rollout；采集期间不得更新网络，不能使用replay")
        behavior_source = "recorded_during_collection_verified_against_current_policy"
        boundaries = np.zeros(len(samples), dtype=bool)
        for index in range(len(samples)-1):
            left, right = samples[index], samples[index+1]
            # 缺少任一连续性元数据时，只用该转移real next state bootstrap，不跨样本递推。
            metadata_complete = (bool(left.episode_id and right.episode_id)
                                 and left.step_index is not None and right.step_index is not None
                                 and left.env_step is not None and right.env_step is not None)
            continuous = (metadata_complete and left.scenario_id == right.scenario_id
                          and left.episode_id == right.episode_id and right.step_index == left.step_index+1
                          and right.env_step == left.env_step+1)
            boundaries[index] = not continuous
        hp = self.hyperparameters
        advantage_array, returns_array = generalized_advantages(
            [sample.reward for sample in samples], old_value.cpu().numpy(), next_values,
            [sample.terminated for sample in samples], [sample.truncated for sample in samples],
            self.config.train.gamma, hp.gae_lambda, boundaries)
        raw_mean, raw_std = float(advantage_array.mean()), float(advantage_array.std())
        normalized = bool(hp.normalize_advantages and raw_std > 1e-8)
        if normalized:
            advantage_array = (advantage_array-raw_mean)/(raw_std+1e-8)
        advantages = torch.as_tensor(advantage_array, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_array, dtype=torch.float32, device=self.device)
        totals = {key: 0. for key in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                                    "actor_gradient_norm", "critic_gradient_norm")}
        initial_ratio_error = float(((current_logp-old_logp).exp()-1.).abs().max().item())
        optimizer_steps = 0
        mini_updates = []
        total_weight = len(samples)*hp.epochs
        for epoch in range(hp.epochs):
            permutation = torch.randperm(len(samples), generator=self.minibatch_rng)
            for offset in range(0, len(samples), hp.minibatch_size):
                minibatch_started = time.perf_counter()
                indices = permutation[offset:offset+hp.minibatch_size].to(self.device)
                probability, log_probability = self.actor(features[indices], masks[indices])
                selected_logp = log_probability.gather(1, ids[indices, None]).squeeze(1)
                actor_loss, ratio = clipped_policy_loss(selected_logp, old_logp[indices], advantages[indices], hp.clip_epsilon)
                predicted_value = self.critic(features[indices])
                value_loss = torch.mean((predicted_value-returns[indices])**2)
                entropy = -(probability*log_probability).sum(-1).mean()
                loss = actor_loss+hp.value_coefficient*value_loss-hp.entropy_coefficient*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO损失非有限")
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.train.gradient_clip_norm, error_if_nonfinite=True)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.train.gradient_clip_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                log_ratio = selected_logp-old_logp[indices]
                values = dict(actor_loss=actor_loss, value_loss=value_loss, entropy=entropy,
                              approx_kl=((ratio-1)-log_ratio).mean(),
                              clip_fraction=((ratio-1).abs()>hp.clip_epsilon).float().mean(),
                              actor_gradient_norm=actor_norm, critic_gradient_norm=critic_norm)
                weight = len(indices)/total_weight
                actual_values = {key: float(value.detach().cpu()) for key, value in values.items()}
                for key, value in actual_values.items():
                    totals[key] += value*weight
                optimizer_steps += 1
                self.update_step += 1
                mini_updates.append(dict(actual_values, update_step=self.update_step, epoch=epoch+1,
                    minibatch=offset//hp.minibatch_size+1, batch_size=len(indices),
                    rollout_indices=permutation[offset:offset+hp.minibatch_size].tolist(),
                    update_seconds=time.perf_counter()-minibatch_started,
                    actor_lr=self.actor_optimizer.param_groups[0]["lr"],
                    critic_lr=self.critic_optimizer.param_groups[0]["lr"]))
        self.rollout_updates += 1
        totals.update(batch_size=len(samples), optimizer_steps=optimizer_steps, update_step=self.update_step,
                      rollout_updates=self.rollout_updates, update_seconds=time.perf_counter()-started,
                      initial_ratio_max_error=initial_ratio_error, advantage_mean=raw_mean, advantage_std=raw_std,
                      normalized_advantages=normalized, behavior_source=behavior_source,
                      actor_lr=self.actor_optimizer.param_groups[0]["lr"], critic_lr=self.critic_optimizer.param_groups[0]["lr"],
                      rollout_policy="on_policy_no_replay", hyperparameters=asdict(hp), mini_updates=mini_updates,
                      aggregation={"kind": "rollout_weighted_minibatch_mean", "count": optimizer_steps,
                                   "first_update_step": mini_updates[0]["update_step"],
                                   "last_update_step": mini_updates[-1]["update_step"]})
        return totals

    def state_dict(self):
        return dict(format_version="paper_mlp_ppo.v1", config=self.config.to_dict(),
                    versions=self.config.semantic_versions(), action_signature=self.action_spec.signature(),
                    architecture={"input_dim": self.input_dim, "hidden_width": self.width, "hidden_layers": 2,
                                  "activation": "relu", "actor": "skip_gate_masked_three_head_joint", "critic": "state_value"},
                    hyperparameters=asdict(self.hyperparameters), discount_factor=self.config.train.gamma,
                    publication_status="第IV节只明确MLP和三头Actor/状态价值Critic；隐藏层、PPO超参及合法mask/gate为显式未刊实现细节",
                    actor=copy.deepcopy(self.actor.state_dict()), critic=copy.deepcopy(self.critic.state_dict()),
                    actor_optimizer=copy.deepcopy(self.actor_optimizer.state_dict()),
                    critic_optimizer=copy.deepcopy(self.critic_optimizer.state_dict()),
                    action_rng=self.action_rng.get_state().clone(), minibatch_rng=self.minibatch_rng.get_state().clone(),
                    update_step=self.update_step, rollout_updates=self.rollout_updates)

    def load_state_dict(self, state, restore_rng=True):
        if state.get("format_version") != "paper_mlp_ppo.v1" or state["versions"] != self.config.semantic_versions():
            raise ValueError("PPO checkpoint格式或语义版本不兼容")
        if state["action_signature"] != self.action_spec.signature():
            raise ValueError("PPO checkpoint动作字典不兼容")
        if (state["architecture"]["input_dim"], state["architecture"]["hidden_width"]) != (self.input_dim, self.width):
            raise ValueError("PPO checkpoint输入或模型宽度不兼容")
        if state["hyperparameters"] != asdict(self.hyperparameters) or state["discount_factor"] != self.config.train.gamma:
            raise ValueError("PPO checkpoint训练超参数不兼容")
        if restore_rng and torch.device(state["config"]["train"]["device"]).type != self.device.type:
            raise ValueError("PPO精确续训不支持跨设备类型；纯推理使用restore_rng=False")
        if restore_rng:
            current = self.config.to_dict()
            stored = state["config"]
            for key in ("schema_version", "data", "physics", "env", "model"):
                if stored[key] != current[key]:
                    raise ValueError(f"PPO精确续训的{key}配置不兼容")
            runtime_changes = {"episodes", "max_env_steps", "max_wall_seconds", "device", "torch_num_threads", "checkpoint_every", "checkpoint_compression"}
            if {key: value for key, value in stored["train"].items() if key not in runtime_changes} != {
                    key: value for key, value in current["train"].items() if key not in runtime_changes}:
                raise ValueError("PPO精确续训的训练配置不兼容；只能改变显式运行预算/设备/线程/保存间隔")
        for name in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.update_step = state["update_step"]
        self.rollout_updates = state["rollout_updates"]
        if restore_rng:
            self.action_rng.set_state(state["action_rng"].cpu())
            self.minibatch_rng.set_state(state["minibatch_rng"].cpu())
