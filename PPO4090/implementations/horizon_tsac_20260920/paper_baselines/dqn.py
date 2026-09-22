"""论文 MLP-DQN 的统一环境适配：三分量 Q 求和，完整合法动作选优。

MLP 为两层 128 维 ReLU；新增 SKIP 独立 head 是硬约束环境适配。
epsilon 日程、Adam/MSE 和 Polyak target 更新为显式工程配置，不冒称论文给值。
"""
import copy
import math
import random
import time
import numpy as np
import torch
from torch import nn

from ..rl.replay import Transition, normalize_signature


class AdditiveQNetwork(nn.Module):
    def __init__(self, input_dim, action_spec):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.start_head = nn.Linear(128, action_spec.num_slots)
        self.length_head = nn.Linear(128, action_spec.max_block_length)
        self.power_head = nn.Linear(128, len(action_spec.power_levels_w))
        self.skip_head = nn.Linear(128, 1)
        self.register_buffer("candidates", torch.as_tensor(np.array(action_spec.candidates), dtype=torch.long))

    def forward(self, observation):
        hidden = self.encoder(observation)
        candidates = self.candidates[1:]
        allocation_q = (self.start_head(hidden)[:, candidates[:, 0]]
                        + self.length_head(hidden)[:, candidates[:, 1]-1]
                        + self.power_head(hidden)[:, candidates[:, 2]])
        return torch.cat((self.skip_head(hidden), allocation_q), dim=-1)


class BaselineDQN:
    """纯内存 DQN；调用者管理完整回合回放、日志、文件与数据划分。"""
    FORMAT = "paper_mlp_dqn.v1"

    def __init__(self, config, action_spec, observation_example, *, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay_steps=10000):
        config.validate()
        if not math.isfinite(epsilon_start) or not math.isfinite(epsilon_end) or not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("epsilon必须满足0<=end<=start<=1")
        if isinstance(epsilon_decay_steps, bool) or not isinstance(epsilon_decay_steps, int) or epsilon_decay_steps < 1:
            raise ValueError("epsilon_decay_steps必须为正整数")
        self.config = copy.deepcopy(config)
        self.action_spec = copy.deepcopy(action_spec)
        expected = {"num_slots": config.physics.num_slots, "max_block_length": config.env.max_block_length,
                    "power_levels_w": list(config.env.power_levels_w)}
        if self.action_signature() != expected:
            raise ValueError("ActionSpec与DQN配置不兼容")
        self.device = torch.device(config.train.device)
        self.input_dim = 5+2*action_spec.num_slots
        self.epsilon_start, self.epsilon_end = float(epsilon_start), float(epsilon_end)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.interaction_step, self.update_step = 0, 0
        self.rng = np.random.default_rng(config.train.seed)
        self._validate_observation(observation_example)
        self.online = AdditiveQNetwork(self.input_dim, action_spec).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.train.critic_lr)

    def action_signature(self):
        return self.action_spec.signature()

    def agent_spec(self):
        return {"agent_type": self.FORMAT, "input": "legacy205", "input_dim": self.input_dim,
                "hidden_layers": [128, 128], "activation": "relu", "double_dqn": False,
                "q_decomposition": "start_plus_length_plus_power_with_independent_skip",
                "epsilon_start": self.epsilon_start, "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps, "epsilon_axis": "training_environment_interactions",
                "optimizer": "adam", "loss": "mse", "target_update": "polyak", "tau": self.config.train.tau,
                "gamma": self.config.train.gamma, "learning_rate": self.config.train.critic_lr,
                "gradient_clip_norm": self.config.train.gradient_clip_norm}

    @property
    def epsilon(self):
        fraction = min(self.interaction_step/self.epsilon_decay_steps, 1.0)
        return self.epsilon_start+(self.epsilon_end-self.epsilon_start)*fraction

    def set_interaction_step(self, value):
        """控制器如绕过act做外部采集，必须显式同步已发生的环境交互计数。"""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("interaction_step必须为非负整数")
        self.interaction_step = int(value)

    def train(self, mode=True):
        self.online.train(mode)
        self.target.eval()
        return self

    @staticmethod
    def _finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"DQN {name}含非有限值")

    def _validate_observation(self, observation):
        if normalize_signature(observation.get("action_spec_signature")) != self.action_signature():
            raise ValueError("DQN观察ActionSpec签名不兼容")
        values = np.asarray(observation.get("legacy205"), dtype=np.float32)
        if values.shape != (self.input_dim,) or not np.isfinite(values).all():
            raise ValueError(f"DQN局部输入必须为有限的{self.input_dim}维向量")
        saved = np.asarray(observation.get("valid_action_mask"))
        if saved.dtype != np.bool_ or saved.shape != (len(self.action_spec),):
            raise ValueError("DQN合法动作mask必须为正确长度的bool数组")
        reconstructed = self.action_spec.valid_actions(observation)
        if not np.array_equal(saved, reconstructed):
            raise ValueError("DQN保存mask与纯函数重建结果不一致")
        return values, saved

    def _batch(self, observations):
        validated = [self._validate_observation(o) for o in observations]
        return (torch.as_tensor(np.stack([x[0] for x in validated]), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack([x[1] for x in validated]), dtype=torch.bool, device=self.device))

    @torch.no_grad()
    def q_values(self, observation):
        values, _ = self._batch([observation])
        result = self.online(values)
        self._finite("Q", result)
        return result[0].cpu().numpy()

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        if bool(observation["terminal"]):
            raise ValueError("DQN不在terminal观察调用策略")
        values, mask = self._validate_observation(observation)
        legal = np.flatnonzero(mask)
        if legal.size == 1:
            index = int(legal[0])
        elif not deterministic and self.rng.random() < self.epsilon:
            index = int(self.rng.choice(legal))  # 完整合法候选均匀，而非逐head采样。
        else:
            q = self.online(torch.as_tensor(values, device=self.device).unsqueeze(0))[0]
            self._finite("Q", q)
            legal_tensor = torch.as_tensor(mask, device=self.device)
            index = int(q.masked_fill(~legal_tensor, -torch.inf).argmax().item())
        if not deterministic:
            self.interaction_step += 1
        return self.action_spec.decode(index)

    def _validate_transitions(self, transitions):
        samples = [Transition.coerce(t) for t in transitions]
        if not samples:
            raise ValueError("DQN更新需要非空转移集合")
        for t in samples:
            if dict(t.versions) != self.config.semantic_versions():
                raise ValueError("DQN经验语义版本不兼容")
            _, mask = self._validate_observation(t.observation)
            self._validate_observation(t.next_observation)  # terminal同样校验，不能跨语义混池。
            if not 0 <= t.action_id < len(mask) or not mask[t.action_id]:
                raise ValueError("DQN回放动作不合法")
        return samples

    @torch.no_grad()
    def targets(self, transitions):
        samples = self._validate_transitions(transitions)
        targets = torch.tensor([t.reward for t in samples], dtype=torch.float32, device=self.device)
        alive = [i for i, t in enumerate(samples) if not t.terminated]
        if alive:
            values, mask = self._batch([samples[i].next_observation for i in alive])
            q = self.target(values)
            self._finite("target Q", q)
            # 标准DQN：target网络自身在合法全集取max；不是Double DQN。
            bootstrap = q.masked_fill(~mask, -torch.inf).max(dim=-1).values
            targets[alive] += self.config.train.gamma*bootstrap
        self._finite("TD target", targets)
        return targets

    @torch.no_grad()
    def soft_update(self):
        for source, target in zip(self.online.parameters(), self.target.parameters()):
            target.lerp_(source, self.config.train.tau)
        for source, target in zip(self.online.buffers(), self.target.buffers()):
            target.copy_(source)

    def update(self, transitions):
        started = time.perf_counter()
        samples = self._validate_transitions(transitions)
        self.train(True)
        size = len(samples)
        diagnostics = {"dqn_loss": 0., "td_abs_mean": 0., "q_mean": 0., "target_mean": 0., "batch_size": size}
        self.optimizer.zero_grad(set_to_none=True)
        for offset in range(0, size, self.config.train.microbatch_size):
            subset = samples[offset:offset+self.config.train.microbatch_size]
            values, _ = self._batch([t.observation for t in subset])
            indices = torch.tensor([t.action_id for t in subset], dtype=torch.long, device=self.device)
            target = self.targets(subset)
            chosen_q = self.online(values).gather(1, indices[:, None]).squeeze(1)
            loss = ((chosen_q-target)**2).sum()/size
            self._finite("loss", loss)
            loss.backward()
            diagnostics["dqn_loss"] += float(loss.detach().cpu())
            diagnostics["td_abs_mean"] += float((chosen_q-target).detach().abs().sum().cpu())/size
            diagnostics["q_mean"] += float(chosen_q.detach().sum().cpu())/size
            diagnostics["target_mean"] += float(target.sum().cpu())/size
        parameters = [p for p in self.online.parameters() if p.grad is not None]
        for p in parameters:
            self._finite("gradient", p.grad)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.train.gradient_clip_norm)
        self._finite("gradient norm", norm)
        self.optimizer.step()
        self.soft_update()
        self.update_step += 1
        diagnostics.update(critic_1_loss=diagnostics["dqn_loss"], gradient_norm=float(norm.detach().cpu()),
                           epsilon=self.epsilon, interaction_step=self.interaction_step, update_step=self.update_step,
                           update_seconds=time.perf_counter()-started, critic_lr=self.optimizer.param_groups[0]["lr"])
        return diagnostics

    def state_dict(self):
        return {"format_version": self.FORMAT, "config": self.config.to_dict(), "versions": self.config.semantic_versions(),
                "agent_spec": self.agent_spec(), "action_signature": self.action_signature(),
                "online": copy.deepcopy(self.online.state_dict()), "target": copy.deepcopy(self.target.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "interaction_step": self.interaction_step,
                "update_step": self.update_step, "training": self.online.training, "training_device_type": self.device.type,
                "action_rng": copy.deepcopy(self.rng.bit_generator.state), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def load_state_dict(self, state, restore_rng=True):
        if state.get("format_version") != self.FORMAT or state.get("versions") != self.config.semantic_versions():
            raise ValueError("DQN checkpoint格式或语义版本不兼容")
        if normalize_signature(state.get("action_signature")) != self.action_signature() or state.get("agent_spec") != self.agent_spec():
            raise ValueError("DQN checkpoint动作/网络/epsilon/更新配置不兼容")
        saved_device_type = state.get("training_device_type", torch.device(state["config"]["train"]["device"]).type)
        if restore_rng and saved_device_type != self.device.type:
            raise ValueError("DQN精确恢复不能跨CPU/CUDA设备类型；仅评估须显式restore_rng=False")
        if restore_rng:
            from ..config import Config
            stored = Config.from_dict(state["config"]).to_dict()
            current = self.config.to_dict()
            for key in ("schema_version", "physics", "env", "data", "model"):
                if stored[key] != current[key]:
                    raise ValueError(f"DQN精确恢复的{key}配置不兼容")
            runtime_changes = {"episodes", "max_env_steps", "max_wall_seconds", "device", "torch_num_threads", "checkpoint_every", "checkpoint_compression"}
            if {key: value for key, value in stored["train"].items() if key not in runtime_changes} != {
                    key: value for key, value in current["train"].items() if key not in runtime_changes}:
                raise ValueError("DQN精确恢复的train配置不兼容；只能改变运行预算/同类型设备/线程/保存间隔")
        for name in ("interaction_step", "update_step"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DQN checkpoint {name}非法")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.interaction_step, self.update_step = state["interaction_step"], state["update_step"]
        self.rng.bit_generator.state = copy.deepcopy(state["action_rng"])
        self.train(state.get("training", True))
        if restore_rng:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if torch.cuda.is_available() and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
