# 处理离散问题的模型
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
from datetime import datetime
import os
import csv

from project_paths import get_output_dir


# ----------------------------------------- #
# 经验回放池
# ----------------------------------------- #
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池容量
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    # 经验池增加
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch组
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 取出这batch组数据
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 当前时刻的经验池容量
    def size(self):
        return len(self.buffer)


# ----------------------------------------- #
# 策略网络
# ----------------------------------------- #

class PolicyNet(nn.Module):
    """
    Transformer-based Actor
    输入向量 x 维度约定：
        · 0-3   : 4 维局部波束特征  (rate, lat, lon, group_idx 等)
        · 4-103 : 100 维 row_norm  (干扰功率或其他连续量)
        · 104-203: 100 维 occ_flag (0/1 占用标志)
        共 204 维（如维度有所调整，请同步修改切片）。
    """

    def __init__(self, n_states, n_hiddens, freq_dim, slots_dim, power_dim,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()

        # -------- 嵌入层 --------
        self.local_embed = nn.Linear(1, d_model)   # 4 维 → d_model
        self.power_embed = nn.Linear(1, d_model)
        self.slot_embed  = nn.Linear(2, d_model)   # 2 维 → d_model

        # 可学习位置编码 (101 = 1 cls + 100 slots)
        self.pos_embed = nn.Parameter(torch.zeros(1, 102, d_model))

        # -------- Transformer 编码器 --------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,   # keep
            # norm_first ❶  删
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # -------- 三个动作头 --------
        self.freq_head  = nn.Linear(d_model, freq_dim)   # 起始频率
        self.slots_head = nn.Linear(d_model, slots_dim)  # 时隙数量
        self.power_head = nn.Linear(d_model, power_dim)  # 功率等级

    def forward(self, x: torch.Tensor, freq_mask):
        """
        x         : (B, 204) float32
        freq_mask : (B, freq_dim) 0/1 可行动作掩码
        返回       : 三个动作概率分布 tuple
        """
        B = x.size(0)

        # -------- 1) 切片 & 嵌入 --------
        local = x[:, :1]                      # (B,4)
        power_pool = x[:, 1: 2]
        row   = x[:, 2:102]                  # (B,100)
        occ   = x[:, 102:202]                # (B,100)

        slot_feats = torch.stack((row, occ), dim=-1)      # (B,100,2)

        tok0 = self.local_embed(local).unsqueeze(1)       # (B,1,d)
        tok1 = self.power_embed(power_pool).unsqueeze(1)
        tokN = self.slot_embed(slot_feats)                # (B,100,d)

        tokens = torch.cat([tok0, tok1, tokN], dim=1) + self.pos_embed  # (B,101,d)

        # -------- 2) Transformer 编码 --------
        h = self.encoder(tokens.permute(1, 0, 2))                         # (B,101,d)
        h = h.permute(1, 0, 2)
        pooled = h[:, 0]                                  # 取 cls token

        # -------- 3) 三头输出 --------
        log_mask = torch.log(torch.FloatTensor(freq_mask).to(x.device))
        freq_logits  = self.freq_head(pooled) + log_mask
        slots_logits = self.slots_head(pooled)
        power_logits = self.power_head(pooled)

        return (torch.softmax(freq_logits, dim=-1),
                torch.softmax(slots_logits, dim=-1),
                torch.softmax(power_logits, dim=-1))


class ValueNet(nn.Module):
    """Transformer-based Critic，输出三组 Q-values"""
    def __init__(self, n_states, n_hiddens, freq_dim, slots_dim, power_dim,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()

        self.local_embed = nn.Linear(1, d_model)
        self.power_embed = nn.Linear(1, d_model)
        self.slot_embed  = nn.Linear(2, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 102, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.q_freq  = nn.Linear(d_model, freq_dim)
        self.q_slots = nn.Linear(d_model, slots_dim)
        self.q_power = nn.Linear(d_model, power_dim)

    def forward(self, x: torch.Tensor):
        local = x[:, :1]                      # (B,4)
        power_pool = x[:, 1: 2]
        row   = x[:, 2:102]                  # (B,100)
        occ   = x[:, 102:202]                # (B,100)

        slot_feats = torch.stack((row, occ), dim=-1)      # (B,100,2)

        tok0 = self.local_embed(local).unsqueeze(1)       # (B,1,d)
        tok1 = self.power_embed(power_pool).unsqueeze(1)
        tokN = self.slot_embed(slot_feats)                # (B,100,d)
        tokens = torch.cat([tok0, tok1, tokN], dim=1) + self.pos_embed  # (B,101,d)

        h = self.encoder(tokens.permute(1, 0, 2))
        h = h.permute(1, 0, 2)            # 再转回 [B, 101, d]
        pooled = h[:, 0]    # cls

        return (
            self.q_freq(pooled),
            self.q_slots(pooled),
            self.q_power(pooled)
        )


# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class SAC:
    def __init__(self, n_states, n_hiddens, freq_dim, slots_dim, power_dim,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, freq_dim, slots_dim, power_dim).to(device)
        # 实例化第一个价值网络--预测
        self.critic_1 = ValueNet(n_states, n_hiddens, freq_dim, slots_dim, power_dim).to(device)
        # 实例化第二个价值网络--预测
        self.critic_2 = ValueNet(n_states, n_hiddens, freq_dim, slots_dim, power_dim).to(device)
        # 实例化价值网络1--目标
        self.target_critic_1 = ValueNet(n_states, n_hiddens, freq_dim, slots_dim, power_dim).to(device)
        # 实例化价值网络2--目标
        self.target_critic_2 = ValueNet(n_states, n_hiddens, freq_dim, slots_dim, power_dim).to(device)

        # 预测和目标的价值网络的参数初始化一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 目标网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # 属性分配
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.init_logging()


    def init_logging(self):
        # 使用正斜杠定义路径，并确保目录存在
        log_dir = get_output_dir("sac_city_power", "logs")
        self.log_file = os.path.join(log_dir, f'sac_transformer_log_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
        os.makedirs(log_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

        self.log_fields = [
            'episode',
            'beam_id',
            'satisfaction',
            'satisfaction_increment',
            'total_satisfaction',
            'success_slots',
            "spectral_efficiency_avg",
            "power",
            "is_interfere",
            "out",
            "occupy",
            "required_rate",
            "location",
            "group",
            "freq",
            "origin_slots",
            "sgm_rate",
            "interfere_in_range",
            "power_gap"
        ]

        # 写入CSV头
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fields)
            writer.writeheader()
            
    def log_step(self, episode, info):
        """记录单步日志"""


        log_data = {
            'episode': episode,
            'beam_id': info['beam_id'],
            'satisfaction': float(f"{info['current_satisfaction']:.2f}"),
            'satisfaction_increment': float(f"{info['satisfaction_increment']:.2f}"),
            'total_satisfaction': float(f"{info['total_satisfaction']:.2f}"),
            'success_slots': info['success_slots'],
            "spectral_efficiency_avg": float(f"{info['spectral_efficiency_avg']:.2f}"),
            "power": float(f"{info['power']:.2f}"),
            "is_interfere": info['is_interfere'],
            "out": info['out'],
            "occupy": info['occupy'],
            "required_rate": info['required_rate'],
            'location': info['location'],
            'group': info['group'],
            'freq': info['freq'],
            'origin_slots': info['origin_slots'],
            'sgm_rate': float(f"{info['sgm_rate'].item():.2f}"),
            'interfere_in_range': info['interfere_in_range'],
            'power_gap': info['power_gap']
        }

        # 使用追加模式写入
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fields)
            writer.writerow(log_data)
    
    # 动作选择
    def take_action(self, state, mask):  # 输入当前状态 [n_states]
        # 维度变换 numpy[n_states]-->tensor[1,n_states]
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze_(0)
        #state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)  # 转换为张量并传输到设备

        freq_logits, slots_logits, power_logits = self.actor(state, mask)  # 获取动作概率分布
        # 构造与输出动作概率相同的概率分布
        freq_dist = torch.distributions.Categorical(freq_logits)
        slots_dist = torch.distributions.Categorical(slots_logits)
        power_dist = torch.distributions.Categorical(power_logits)

        freq = freq_dist.sample()
        slots = slots_dist.sample()
        power = power_dist.sample()
        # 从当前概率分布中随机采样tensor-->int

        action = [
            freq.cpu().item(),
            slots.cpu().item(),
            power.cpu().item()
        ]
        return action

    # 计算目标，当前状态下的state_value
    def calc_target(self, rewards, next_states, dones, ns_masks):
        # 策略网络预测下一时刻的state_value  [b,n_states]-->[b,n_actions]
        probs_freq, probs_slots, probs_power = self.actor(next_states, ns_masks)
        # 对每个动作的概率计算ln  [b,n_actions]
        logp_freq = torch.log(probs_freq + 1e-8)
        logp_slots = torch.log(probs_slots + 1e-8)
        logp_power = torch.log(probs_power + 1e-8)

        # 计算熵 [b,1]
        entropy = (
                -(probs_freq * logp_freq).sum(1, keepdim=True)
                - (probs_slots * logp_slots).sum(1, keepdim=True)
                - (probs_power * logp_power).sum(1, keepdim=True)
        )
        # 目标价值网络，下一时刻的state_value [b,n_actions]
        q1_freq, q1_slots, q1_power = self.target_critic_1(next_states)
        q2_freq, q2_slots, q2_power = self.target_critic_2(next_states)
        # 取出最小的q值  [b, 1]
        min_freq = torch.min(q1_freq, q2_freq)
        min_slots = torch.min(q1_slots, q2_slots)
        min_power = torch.min(q1_power, q2_power)
        exp_min_q = (
                (probs_freq * min_freq).sum(1, keepdim=True) +
                (probs_slots * min_slots).sum(1, keepdim=True) +
                (probs_power * min_power).sum(1, keepdim=True)
        )
        # 下个时刻的state_value  [b, 1]
        next_value = exp_min_q + self.log_alpha.exp() * entropy

        # 时序差分，目标网络输出当前时刻的state_value  [b, n_actions]
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    # 软更新，每次训练更新部分参数
    def soft_update(self, net, target_net):
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 预测网络的参数赋给目标网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def _joint_q(self, q_freq, q_slots, q_power, actions):
        """
        q_freq / q_slots / q_power : (B,10)
        actions                    : (B,3)  = [af, as, ap]
        ----------------------------------------------------------
        return : Tensor (B,1)
        """
        a_f = actions[:, 0:1]  # (B,1)
        a_s = actions[:, 1:2]
        a_p = actions[:, 2:3]
        return (
                q_freq.gather(1, a_f) +  # 频率起点 Q
                q_slots.gather(1, a_s) +  # 槽数       Q
                q_power.gather(1, a_p)  # 功率       Q
        )
        # 模型训练

    def update(self, transition_dict):
        """one gradient step"""
        device = self.device
        # ---------- 1. 取 batch ----------
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float32, device=device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long, device=device)  # (B,3)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float32, device=device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float32, device=device).unsqueeze(1)
        s_masks = torch.tensor(transition_dict['s_mask'],
                              dtype=torch.float32, device=device).cpu()
        ns_masks = torch.tensor(transition_dict['ns_mask'],
                              dtype=torch.float32, device=device).cpu()
        # ---------- 2. TD 目标 ----------
        td_target = self.calc_target(rewards, next_states, dones, ns_masks)  # (B,1)

        # ---------- 3. Critic 更新 ----------
        #       Q1
        q1_f, q1_s, q1_p = self.critic_1(states)
        q1_joint = self._joint_q(q1_f, q1_s, q1_p, actions)
        loss_q1 = F.mse_loss(q1_joint, td_target.detach())

        #       Q2
        q2_f, q2_s, q2_p = self.critic_2(states)
        q2_joint = self._joint_q(q2_f, q2_s, q2_p, actions)
        loss_q2 = F.mse_loss(q2_joint, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        loss_q1.backward()
        loss_q2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ---------- 4. Actor 更新 ----------
        # B = states.size(0)
        # 这里如果有动作可行性掩码，就传进去；没有则直接用 all-ones
        #freq_mask = np.ones((B, self.actor.freq_head.out_features), dtype=np.float32)
        pf, ps, pp = self.actor(states, s_masks)  # (B,10) × 3
        log_pf = torch.log(pf + 1e-8)
        log_ps = torch.log(ps + 1e-8)
        log_pp = torch.log(pp + 1e-8)

        # 熵 H(s) = -Σ π log π
        entropy = (- torch.sum(pf * log_pf, dim=1, keepdim=True)
                   - torch.sum(ps * log_ps, dim=1, keepdim=True)
                   - torch.sum(pp * log_pp, dim=1, keepdim=True)
        )  # (B,1)

        # 重新取最小 Q(s,·)  并求 Eπ[·]
        q1_f, q1_s, q1_p = self.critic_1(states)
        q2_f, q2_s, q2_p = self.critic_2(states)
        min_f = torch.min(q1_f, q2_f)
        min_s = torch.min(q1_s, q2_s)
        min_p = torch.min(q1_p, q2_p)
        exp_min_q = (
                torch.sum(pf * min_f, dim=1, keepdim=True) +
                torch.sum(ps * min_s, dim=1, keepdim=True) +
                torch.sum(pp * min_p, dim=1, keepdim=True)
        )  # (B,1)

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - exp_min_q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- 5. α 温度系数 ----------
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        #alpha_loss = torch.mean(-alpha * (entropy + self.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ---------- 6. 软更新目标 Q ----------
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
    def save_model(self):
        model_dir = get_output_dir("sac_city_power", "models")
        os.makedirs(model_dir, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        actor_path = os.path.join(model_dir, f"sac_transformer_actor_{timestamp}.pth")
        critic_1_path = os.path.join(model_dir, f"sac_transformer_critic_1_{timestamp}.pth")
        critic_2_path = os.path.join(model_dir, f"sac_transformer_critic_2_{timestamp}.pth")
        target_critic_1_path = os.path.join(model_dir, f"sac_transformer_target_critic_1_{timestamp}.pth")
        target_critic_2_path = os.path.join(model_dir, f"sac_transformer_target_critic_2_{timestamp}.pth")

        # 保存完整模型参数（包含网络结构和参数）
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1':self.critic_1.state_dict(),
            'critic_2':self.critic_2.state_dict(),
            'target_critic_1':self.target_critic_1.state_dict(),
            'target_critic_2':self.target_critic_2.state_dict()

        }, os.path.join(model_dir, f"sac_transformer_full_model_{timestamp}.pth"))

        # 同时单独保存参数方便后续加载
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)
        torch.save(self.target_critic_1.state_dict(), target_critic_1_path)
        torch.save(self.target_critic_2.state_dict(), target_critic_2_path)

        print(f"Models saved to {model_dir} with timestamp {timestamp}")
