import torch  # 导入 PyTorch，作为深度学习的核心框架
import torch.nn as nn
import torch.nn.functional as F  # 导入 PyTorch 的常用函数库，包括激活函数、损失函数等
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.legacy_ppo.state_1603_power import rl_utils_1603power as rl_utils
from implementations.legacy_ppo.state_1603_power.Environment_1603_power import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num
from project_paths import get_output_dir
import csv
from datetime import datetime
import os

beam_info = BeamInfo()
satellite_freq_state = SatelliteFreqState()
current_beam_state = CurrentBeamState(beam_info.beam_info)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 定义策略网络（PolicyNet），用于生成动作分布
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(PolicyNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.freq_head = nn.Linear(128, 10)  # 100个频率起点
        self.slots_head = nn.Linear(128, 10)  # 1-10个时隙
        self.power_head = nn.Linear(128, 10)  # 25-34 dB

    def forward(self, x, freq_mask):
        x = self.shared(x)
        freq_mask = torch.log(torch.FloatTensor(freq_mask).to(device))

        return (torch.softmax(self.freq_head(x) + freq_mask, dim=-1),
                torch.softmax(self.slots_head(x), dim=-1),
                torch.softmax(self.power_head(x), dim=-1))  # 输出动作概率分布，使用 softmax 激活


# 定义价值网络（ValueNet），用于评估状态的价值
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.net(x)  # 使用 ReLU 激活函数
        return x  # 输出状态价值


# 定义 PPO 算法，采用截断（Clipping）方式
class PPO:
    ''' PPO 算法,采用截断方式 '''

    def __init__(self, state_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim).to(device)  # 策略网络
        self.critic = ValueNet(state_dim).to(device)  # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE 参数
        self.epochs = epochs  # 每次更新训练的轮数
        self.eps = eps  # PPO 中截断范围参数
        self.device = device  # 使用的设备（CPU 或 GPU）
        self.init_logging()
        print(self.device)

    def init_logging(self):
        # 使用正斜杠定义路径，并确保目录存在
        log_dir = get_output_dir("legacy_state_1603_power", "logs")
        self.log_file = os.path.join(log_dir, f'ppo_log_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
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
            "interfere_in_range"
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
            'interfere_in_range': info['interfere_in_range']
        }

        # 使用追加模式写入
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fields)
            writer.writerow(log_data)

    def take_action(self, state, mask):
        # 根据当前策略网络对状态 state 进行采样，生成一个动作
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze_(0)
        # state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)  # 转换为张量并传输到设备
        freq_logits, slots_logits, power_logits = self.actor(state, mask)  # 获取动作概率分布
        freq_dist = torch.distributions.Categorical(freq_logits)
        slots_dist = torch.distributions.Categorical(slots_logits)
        power_dist = torch.distributions.Categorical(power_logits)

        freq = freq_dist.sample()
        slots = slots_dist.sample()
        power = power_dist.sample()

        log_prob = (freq_dist.log_prob(freq) +
                    slots_dist.log_prob(slots) +
                    power_dist.log_prob(power))

        return [freq.cpu().item(),
                slots.cpu().item(),
                power.cpu().item()], log_prob  # 返回动作（整数）

    def update(self, transition_dict, entropy_coef):
        # 更新策略和价值网络
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)  # 状态
        freq_actions = torch.tensor([a[0] for a in transition_dict['actions']], dtype=torch.long).to(self.device)
        slots_actions = torch.tensor([a[1] for a in transition_dict['actions']], dtype=torch.long).to(self.device)
        power_actions = torch.tensor([a[2] for a in transition_dict['actions']], dtype=torch.long).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # 奖励
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)  # 下一状态
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # 是否结束
        old_log_probs = torch.tensor(transition_dict['old_log_probs'], dtype=torch.float).to(self.device)
        masks = np.array(transition_dict['masks'])
        # 计算 TD 目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # TD 残差
        # 使用工具函数计算优势函数
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        for _ in range(self.epochs):
            freq_logits, slots_logits, power_logits = self.actor(states, masks)  # 获取动作概率分布
            freq_dists = torch.distributions.Categorical(freq_logits)
            slots_dists = torch.distributions.Categorical(slots_logits)
            power_dists = torch.distributions.Categorical(power_logits)
            new_log_probs = (
                    freq_dists.log_prob(freq_actions) +
                    slots_dists.log_prob(slots_actions) +
                    power_dists.log_prob(power_actions)
            )

            # entropys = torch.stack(entropys).to(self.device)
            ratio = torch.exp(new_log_probs - old_log_probs).view(-1, 1)  # 比例因子 r_theta
            surr1 = ratio * advantage  # 未裁剪项
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断项
            entropy = (
                    #group_dists.entropy().mean() +
                    freq_dists.entropy() +
                    slots_dists.entropy() +
                    power_dists.entropy()
            ).mean()
            actor_loss = torch.mean(-torch.min(surr1,
                                               surr2))   #+ entropy_coef * entropy  # 策略损失，因为我们要最大化策略目标，所以取负号，将其转换为损失函数（梯度下降算法需要最小化损失）
            # print(actor_loss)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # 价值网络损失
            # print(critic_loss)
            # 更新参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 反向传播更新策略网络
            critic_loss.backward()  # 反向传播更新价值网络
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            return actor_loss, critic_loss


actor_lr = 3e-4  # 策略网络学习率
critic_lr = 1e-3  # 价值网络学习率
num_episodes = 100000  # 训练的总回合数
gamma = 0.9  # 折扣因子
lmbda = 0.95  # GAE 参数
epochs = 20  # 每次更新的轮数
eps = 0.2  # PPO 截断范围
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
# 创建环境

env = Env(satellite_freq_state, beam_info, current_beam_state)
env.reset(beam_info)  # 设置随机种子
state_dim = 1603
agent = PPO(state_dim=state_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            gamma=gamma,
            device=device)  # 初始化 PPO 算法
# 开始训练
return_list, satisfaction_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)  # 使用工具函数训练
rl_utils.plot_reward(return_list)
rl_utils.plot_satisfaction(satisfaction_list)
print("Training completed!")
model_dir = get_output_dir("legacy_state_1603_power", "models")
os.makedirs(model_dir, exist_ok=True)

# 生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d%H%M")
actor_path = os.path.join(model_dir, f"ppo_group_actor_{timestamp}.pth")
critic_path = os.path.join(model_dir, f"ppo_group_critic_{timestamp}.pth")

# 保存完整模型参数（包含网络结构和参数）
torch.save({
    'actor_state_dict': agent.actor.state_dict(),
    'critic_state_dict': agent.critic.state_dict(),

}, os.path.join(model_dir, f"full_group_model_{timestamp}.pth"))

# 同时单独保存参数方便后续加载
torch.save(agent.actor.state_dict(), actor_path)
torch.save(agent.critic.state_dict(), critic_path)

print(f"Models saved to {model_dir} with timestamp {timestamp}")
