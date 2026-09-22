"""精确离散SAC：状态微批、无梯度候选Q分块、完整有效batch一次优化。"""
import copy
import math
import random
import time
import numpy as np
import torch
from ..models import Actor, Critic, collate_observations
from .replay import Transition


def exact_expectations(probability, log_probability, q1, q2, alpha):
    """完整动作先 min 再求期望；调用者决定Q/alpha的梯度隔离。"""
    qmin = torch.minimum(q1, q2)
    entropy = -(probability * log_probability).sum(-1)
    value = (probability * (qmin - alpha * log_probability)).sum(-1)
    actor_loss = (probability * (alpha * log_probability - qmin)).sum(-1)
    return value, actor_loss, entropy


def temperature_loss(log_alpha, entropy, legal_count, ratio):
    useful = legal_count > 1
    target = ratio * legal_count.float().log()
    if not useful.any():
        return None, target
    return (log_alpha * (entropy[useful] - target[useful]).detach()).mean(), target


def _finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name}存在非有限值，停止此次更新")


class DiscreteSAC:
    def __init__(self, config, action_spec, observation_example):
        config.validate()
        self.config = copy.deepcopy(config)
        self.action_spec = action_spec
        self.device = torch.device(config.train.device)
        self.actor = Actor(config, action_spec, observation_example).to(self.device)
        self.critic_1 = Critic(config, action_spec, observation_example).to(self.device)
        self.critic_2 = Critic(config, action_spec, observation_example).to(self.device)
        self.target_critic_1, self.target_critic_2 = copy.deepcopy(self.critic_1), copy.deepcopy(self.critic_2)
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.train.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.train.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.train.critic_lr)
        self.log_alpha = torch.tensor(math.log(config.train.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.train.alpha_lr)
        self.action_rng = torch.Generator(device=self.device)
        self.action_rng.manual_seed(config.train.seed)
        self.update_step = 0

    def train(self, mode=True):
        """训练网络的模式明确切换；目标网络始终评估模式。"""
        for model in (self.actor, self.critic_1, self.critic_2):
            model.train(mode)
        return self

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action_signature(self):
        return self.action_spec.signature()

    def _batch(self, observations):
        for obs in observations:
            signature = obs.get("action_spec_signature")
            if signature is None or dict(signature, power_levels_w=list(signature["power_levels_w"])) != self.action_signature():
                raise ValueError("观察ActionSpec签名不兼容，不能混合功率档或候选语义")
            reconstructed = self.action_spec.valid_actions(obs)
            if not np.array_equal(reconstructed, obs["valid_action_mask"]):
                raise ValueError("观察中保存的action mask与纯函数重建结果不一致")
        return collate_observations(observations, self.device)

    @torch.no_grad()
    def probabilities(self, observation):
        if observation["terminal"]:
            raise ValueError("terminal观察不调用策略")
        probability, _ = self.actor(self._batch([observation]))
        _finite("policy", probability)
        return probability[0].cpu().numpy()

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        if observation["terminal"]:
            raise ValueError("terminal观察不调用策略")
        probability, _ = self.actor(self._batch([observation]))
        _finite("policy", probability)
        # torch.argmax在并列时返回第一个候选ID，是完整联合分布的全局argmax。
        index = probability[0].argmax() if deterministic else torch.multinomial(probability[0], 1, generator=self.action_rng)[0]
        return self.action_spec.decode(int(index.item()))

    @torch.no_grad()
    def _candidate_q(self, critic, obs):
        z = critic.encoder(obs)
        pieces = []
        for start in range(0, len(self.action_spec), self.config.model.candidate_chunk_size):
            ids = torch.arange(start, min(start + self.config.model.candidate_chunk_size, len(self.action_spec)), device=self.device)
            pieces.append(critic.values_from_encoded(z, ids))
        result = torch.cat(pieces, -1)
        _finite("candidate Q", result)
        return result

    @torch.no_grad()
    def targets(self, transitions):
        transitions = [Transition.coerce(t) for t in transitions]
        values = torch.tensor([t.reward for t in transitions], device=self.device, dtype=torch.float32)
        alive = [i for i, t in enumerate(transitions) if not t.terminated]
        if alive:
            if any(transitions[i].next_observation["terminal"] for i in alive):
                raise ValueError("非terminated经验需要可bootstrap的真实final observation")
            obs = self._batch([transitions[i].next_observation for i in alive])
            p, lp = self.actor(obs)
            q1, q2 = self._candidate_q(self.target_critic_1, obs), self._candidate_q(self.target_critic_2, obs)
            v, _, _ = exact_expectations(p, lp, q1, q2, self.alpha.detach())
            values[alive] += self.config.train.gamma * v
        _finite("TD target", values)
        return values

    def _check_gradients(self, model, name):
        parameters = [p for p in model.parameters() if p.grad is not None]
        for p in parameters:
            _finite(name + "梯度", p.grad)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.train.gradient_clip_norm)
        _finite(name + "梯度范数", norm)
        return float(norm.detach().cpu())

    @torch.no_grad()
    def soft_update(self):
        tau = self.config.train.tau
        for source, target in ((self.critic_1, self.target_critic_1), (self.critic_2, self.target_critic_2)):
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.lerp_(src, tau)
            # 未来增加运行统计时也不会遗漏；整数候选字典原样同步。
            for src, dst in zip(source.buffers(), target.buffers()):
                if dst.is_floating_point():
                    dst.lerp_(src, tau)
                else:
                    dst.copy_(src)

    def update(self, transitions):
        started = time.perf_counter()
        self.train(True)
        samples = [Transition.coerce(t) for t in transitions]
        if not samples:
            raise ValueError("更新需要非空固定transition集合")
        for t in samples:
            if dict(t.versions) != self.config.semantic_versions():
                raise ValueError("SAC经验语义版本不兼容")
            # terminal不送进策略，也必须检验其动作语义，防止错误reset观察混入。
            for observation in (t.observation, t.next_observation):
                signature = observation.get("action_spec_signature")
                if signature is None or dict(signature, power_levels_w=list(signature["power_levels_w"])) != self.action_signature():
                    raise ValueError("经验ActionSpec签名不兼容")
            if t.action_id < 0 or t.action_id >= len(self.action_spec) or not t.observation["valid_action_mask"][t.action_id]:
                raise ValueError("回放实际动作不合法")
        n = len(samples)
        micro = self.config.train.microbatch_size
        metrics = dict(critic_1_loss=0., critic_2_loss=0., actor_loss=0., entropy=0., target_entropy=0.,
                       td_abs_mean=0., q_mean=0., target_mean=0., batch_size=n)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        for offset in range(0, n, micro):
            subset = samples[offset:offset + micro]
            obs = self._batch([t.observation for t in subset])
            ids = torch.tensor([t.action_id for t in subset], device=self.device)
            y = self.targets(subset)
            q1, q2 = self.critic_1(obs, ids), self.critic_2(obs, ids)
            loss1, loss2 = ((q1 - y) ** 2).sum() / n, ((q2 - y) ** 2).sum() / n
            _finite("critic loss", loss1 + loss2)
            (loss1 + loss2).backward()
            metrics["critic_1_loss"] += float(loss1.detach().cpu())
            metrics["critic_2_loss"] += float(loss2.detach().cpu())
            metrics["td_abs_mean"] += float(((q1 - y).abs() + (q2 - y).abs()).detach().sum().cpu()) / (2 * n)
            metrics["q_mean"] += float((q1 + q2).detach().sum().cpu()) / (2 * n)
            metrics["target_mean"] += float(y.sum().cpu()) / n
        metrics["critic_1_gradient_norm"] = self._check_gradients(self.critic_1, "critic1")
        metrics["critic_2_gradient_norm"] = self._check_gradients(self.critic_2, "critic2")
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        entropy_errors, useful_count = [], 0
        for offset in range(0, n, micro):
            subset = samples[offset:offset + micro]
            obs = self._batch([t.observation for t in subset])
            p, lp = self.actor(obs)
            q1, q2 = self._candidate_q(self.critic_1, obs), self._candidate_q(self.critic_2, obs)
            _, losses, entropy = exact_expectations(p, lp, q1, q2, self.alpha.detach())
            loss = losses.sum() / n
            _finite("actor loss", loss)
            loss.backward()
            legal_count = obs["valid_action_mask"].sum(-1)
            target_entropy = self.config.train.target_entropy_ratio * legal_count.float().log()
            useful = legal_count > 1
            useful_count += int(useful.sum().item())
            entropy_errors.append((entropy - target_entropy)[useful].detach().sum())
            metrics["actor_loss"] += float(loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().sum().cpu()) / n
            metrics["target_entropy"] += float(target_entropy.sum().cpu()) / n
        metrics["actor_gradient_norm"] = self._check_gradients(self.actor, "actor")
        self.actor_optimizer.step()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        metrics["alpha_loss"] = None
        if useful_count:
            alpha_loss = self.log_alpha * torch.stack(entropy_errors).sum() / useful_count
            _finite("alpha loss", alpha_loss)
            alpha_loss.backward()
            _finite("alpha梯度", self.log_alpha.grad)
            self.alpha_optimizer.step()
            metrics["alpha_loss"] = float(alpha_loss.detach().cpu())
        _finite("alpha", self.alpha)
        self.soft_update()
        self.update_step += 1
        metrics.update(alpha=float(self.alpha.detach().cpu()), nonforced_count=useful_count,
                       update_step=self.update_step, update_seconds=time.perf_counter() - started,
                       actor_lr=self.actor_optimizer.param_groups[0]["lr"],
                       critic_lr=self.critic_1_optimizer.param_groups[0]["lr"],
                       alpha_lr=self.alpha_optimizer.param_groups[0]["lr"])
        return metrics

    def state_dict(self):
        result = dict(format_version="discrete_sac.v1", config=self.config.to_dict(), versions=self.config.semantic_versions(),
                      action_signature=self.action_signature(), update_step=self.update_step,
                      log_alpha=self.log_alpha.detach().cpu().clone(), action_rng=self.action_rng.get_state(),
                      python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
                     "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
            result[name] = copy.deepcopy(getattr(self, name).state_dict())
        return result

    def load_state_dict(self, state, restore_rng=True):
        if state["format_version"] != "discrete_sac.v1" or state["versions"] != self.config.semantic_versions():
            raise ValueError("checkpoint语义/格式版本不兼容")
        if state["action_signature"] != self.action_signature():
            raise ValueError("checkpoint ActionSpec不兼容")
        stored_model = state["config"]["model"].copy()
        current_model = self.config.to_dict()["model"].copy()
        stored_model.pop("candidate_chunk_size")
        current_model.pop("candidate_chunk_size")
        if stored_model != current_model:
            raise ValueError("checkpoint模型结构不兼容")
        saved_device_type = torch.device(state["config"]["train"]["device"]).type
        if restore_rng and saved_device_type != self.device.type:
            raise ValueError("精确续训不支持跨设备类型恢复随机状态："
                             f"{saved_device_type} -> {self.device.type}；只读评估请使用restore_rng=False")
        for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
                     "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.update_step = state["update_step"]
        if restore_rng:
            # CPU与CUDA Generator采用不同状态格式；纯推理不恢复任何随机源。
            self.action_rng.set_state(state["action_rng"].cpu())
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if torch.cuda.is_available() and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
