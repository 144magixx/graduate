"""完整联合动作上的离散SAC；正式RL路径显式关闭dropout。"""
import copy
import math
import os
import random
import numpy as np
import torch
from ..models import Actor, Critic, collate_observations
from .replay import Transition


def exact_expectations(probability, log_probability, q1, q2, alpha):
    qmin = torch.minimum(q1, q2)
    entropy = -(probability * log_probability).sum(-1)
    value = (probability * (qmin - alpha * log_probability)).sum(-1)
    actor_loss = (probability * (alpha * log_probability - qmin)).sum(-1)
    return value, actor_loss, entropy


def _finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name}存在非有限值")


class DiscreteSAC:
    FORMAT = "anchor_discrete_sac.v1"
    RL_FORWARD_MODE = "eval_dropout_disabled"

    def __init__(self, config, action_spec, observation_example):
        config.validate()
        self.config = copy.deepcopy(config)
        self.action_spec = action_spec
        self.device = torch.device(config.train.device)
        self.actor = Actor(config, action_spec, observation_example).to(self.device)
        self.critic_1 = Critic(config, action_spec, observation_example).to(self.device)
        self.critic_2 = Critic(config, action_spec, observation_example).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).requires_grad_(False)
        self.target_critic_2 = copy.deepcopy(self.critic_2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.train.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.train.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.train.critic_lr)
        self.log_alpha = torch.tensor(math.log(config.train.initial_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.train.alpha_lr)
        self.action_rng = torch.Generator(device=self.device).manual_seed(config.train.seed)
        self.update_step = 0
        self._set_rl_mode()

    def _set_rl_mode(self):
        # eval模式不会阻断梯度；它只让结构中保留的dropout在RL路径保持确定。
        for network in (self.actor, self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2):
            network.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action_signature(self):
        return self.action_spec.signature()

    def _batch(self, observations):
        expected_adapter = self.config.semantic_versions()["observation_adapter_version"]
        for observation in observations:
            if observation.get("action_spec_signature") != self.action_signature():
                raise ValueError("观察ActionSpec签名不兼容")
            if observation.get("observation_adapter_version") != expected_adapter:
                raise ValueError("观察适配器版本不兼容")
            if not np.array_equal(self.action_spec.valid_actions(observation), observation["valid_action_mask"]):
                raise ValueError("保存的合法动作mask无法重建")
        return collate_observations(observations, self.device)

    @torch.no_grad()
    def probabilities(self, observation):
        self._set_rl_mode()
        probability, _ = self.actor(self._batch([observation]))
        _finite("policy", probability)
        return probability[0].cpu().numpy()

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        if observation["terminal"]:
            raise ValueError("terminal观察不能调用策略")
        probability = torch.as_tensor(self.probabilities(observation), device=self.device)
        action_id = probability.argmax() if deterministic else torch.multinomial(probability, 1, generator=self.action_rng)[0]
        return self.action_spec.decode(int(action_id))

    @torch.no_grad()
    def _candidate_q(self, critic, observation):
        hidden = critic.core(observation["anchor205"])
        parts = []
        for start in range(0, len(self.action_spec), self.config.model.candidate_chunk_size):
            ids = torch.arange(start, min(start + self.config.model.candidate_chunk_size, len(self.action_spec)), device=self.device)
            parts.append(critic.values_from_encoded(hidden, ids))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def targets(self, transitions):
        self._set_rl_mode()
        transitions = [Transition.coerce(t) for t in transitions]
        values = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        alive = [i for i, transition in enumerate(transitions) if not transition.terminated]
        if alive:
            observation = self._batch([transitions[i].next_observation for i in alive])
            probability, log_probability = self.actor(observation)
            q1 = self._candidate_q(self.target_critic_1, observation)
            q2 = self._candidate_q(self.target_critic_2, observation)
            bootstrap, _, _ = exact_expectations(probability, log_probability, q1, q2, self.alpha.detach())
            values[alive] += self.config.train.gamma * bootstrap
        return values

    def _clip(self, network, name):
        parameters = [p for p in network.parameters() if p.grad is not None]
        for parameter in parameters:
            _finite(name + "梯度", parameter.grad)
        return float(torch.nn.utils.clip_grad_norm_(parameters, self.config.train.gradient_clip_norm))

    def update(self, transitions):
        self._set_rl_mode()
        samples = [Transition.coerce(t) for t in transitions]
        if not samples:
            raise ValueError("更新需要非空合成transition")
        for transition in samples:
            if transition.versions != self.config.semantic_versions():
                raise ValueError("经验语义版本不兼容")
            if not transition.observation["valid_action_mask"][transition.action_id]:
                raise ValueError("经验动作不合法")
        observation = self._batch([t.observation for t in samples])
        action_ids = torch.tensor([t.action_id for t in samples], device=self.device)
        target = self.targets(samples)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        q1, q2 = self.critic_1(observation, action_ids), self.critic_2(observation, action_ids)
        loss1, loss2 = torch.mean((q1 - target) ** 2), torch.mean((q2 - target) ** 2)
        (loss1 + loss2).backward()
        norm1, norm2 = self._clip(self.critic_1, "critic1"), self._clip(self.critic_2, "critic2")
        self.critic_1_optimizer.step(); self.critic_2_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        probability, log_probability = self.actor(observation)
        candidate_q1 = self._candidate_q(self.critic_1, observation)
        candidate_q2 = self._candidate_q(self.critic_2, observation)
        _, actor_terms, entropy = exact_expectations(probability, log_probability, candidate_q1, candidate_q2, self.alpha.detach())
        actor_loss = actor_terms.mean()
        actor_loss.backward()
        actor_norm = self._clip(self.actor, "actor")
        self.actor_optimizer.step()
        legal_count = observation["valid_action_mask"].sum(-1)
        useful = legal_count > 1
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = None
        if useful.any():
            target_entropy = self.config.train.target_entropy_ratio * legal_count.float().log()
            alpha_loss = self.log_alpha * (entropy[useful] - target_entropy[useful]).detach().mean()
            alpha_loss.backward(); self.alpha_optimizer.step()
        self.soft_update()
        self.update_step += 1
        return {"critic_1_loss": float(loss1.detach()), "critic_2_loss": float(loss2.detach()),
                "actor_loss": float(actor_loss.detach()), "alpha_loss": None if alpha_loss is None else float(alpha_loss.detach()),
                "critic_1_gradient_norm": norm1, "critic_2_gradient_norm": norm2,
                "actor_gradient_norm": actor_norm, "alpha": float(self.alpha.detach()), "update_step": self.update_step,
                "rl_forward_mode": self.RL_FORWARD_MODE}

    @torch.no_grad()
    def soft_update(self):
        for source, target in ((self.critic_1, self.target_critic_1), (self.critic_2, self.target_critic_2)):
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.lerp_(src, self.config.train.tau)
            for src, dst in zip(source.buffers(), target.buffers()):
                dst.copy_(src)

    def state_dict(self):
        result = {"format_version": self.FORMAT, "config": self.config.to_dict(),
                  "versions": self.config.semantic_versions(), "action_signature": self.action_signature(),
                  "update_step": self.update_step, "log_alpha": self.log_alpha.detach().cpu().clone(),
                  "rl_forward_mode": self.RL_FORWARD_MODE, "device_type": self.device.type,
                  "deterministic_algorithms": self.config.train.deterministic,
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "action_rng": self.action_rng.get_state(), "python_rng": random.getstate(),
                  "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
                     "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
            result[name] = copy.deepcopy(getattr(self, name).state_dict())
        return result

    def load_state_dict(self, state, restore_rng=True):
        if state.get("format_version") != self.FORMAT or state.get("versions") != self.config.semantic_versions():
            raise ValueError("checkpoint格式/语义不兼容")
        if state.get("action_signature") != self.action_signature() or state.get("rl_forward_mode") != self.RL_FORWARD_MODE:
            raise ValueError("checkpoint动作或dropout运行协议不兼容")
        if copy.deepcopy(state.get("config", {}).get("model")) != self.config.to_dict()["model"]:
            raise ValueError("checkpoint模型配置不兼容")
        if restore_rng and state.get("device_type") != self.device.type:
            raise ValueError(f"精确RNG恢复不支持跨设备类型：{state.get('device_type')} -> {self.device.type}")
        if restore_rng and bool(state.get("deterministic_algorithms")) != bool(self.config.train.deterministic):
            raise ValueError("精确RNG恢复的确定性算法设置不兼容")
        if restore_rng and self.device.type == "cuda":
            if state.get("cuda_rng") is None or not torch.cuda.is_available():
                raise ValueError("CUDA精确恢复缺少可用CUDA RNG状态")
            if self.config.train.deterministic and state.get("cublas_workspace_config") not in (":4096:8", ":16:8"):
                raise ValueError("CUDA确定性恢复缺少冻结CUBLAS_WORKSPACE_CONFIG")
        staged = copy.deepcopy(self.state_dict())
        try:
            for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
                         "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
                getattr(self, name).load_state_dict(state[name])
            with torch.no_grad(): self.log_alpha.copy_(state["log_alpha"].to(self.device))
            self.update_step = int(state["update_step"])
            if restore_rng:
                self.action_rng.set_state(state["action_rng"])
                random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
                torch.set_rng_state(state["torch_rng"])
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
        except Exception:
            for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
                         "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"):
                getattr(self, name).load_state_dict(staged[name])
            with torch.no_grad(): self.log_alpha.copy_(staged["log_alpha"].to(self.device))
            self.update_step = staged["update_step"]
            raise
        self._set_rl_mode()
