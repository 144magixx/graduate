from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.sac_city_shared.Environment_sac_p import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from implementations.sac_city_power.sac_transformer_p import ReplayBuffer, SAC
from project_paths import get_output_dir

def plot_reward(rewards, window_size=20):
    """绘制训练结果曲线

    Args:
        rewards (list): 每轮的奖励值
        window_size (int): 滑动平均窗口大小
    """
    plt.figure(figsize=(14, 6))

    # ------------------ 奖励曲线 ------------------

    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')

    # 计算滑动平均
    smooth_rewards = [np.mean(rewards[max(0, i - window_size):i + 1])
                      for i in range(len(rewards))]
    plt.plot(smooth_rewards, color='red', linewidth=2,
             label=f'Smooth Reward(window={window_size})')

    plt.title('SAC Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # 保存和显示图像
    plt.tight_layout()
    filename = f'reward_group_{datetime.now().strftime("%Y%m%d%H%M")}.png'
    save_dir = get_output_dir("sac_city_power", "figures")

    # 自动创建目录（跨平台兼容）
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    # 使用路径拼接（自动适配操作系统）
    filepath = os.path.join(save_dir, filename)
    # 保存图像
    plt.savefig(filepath, dpi=300)  # 推荐添加 bbox_inches 避免边框截断
    plt.show()


def plot_satisfaction(satisfactions, window_size=20):
    """绘制训练结果曲线

    Args:
        satisfactions: 满足率
        window_size (int): 滑动平均窗口大小
    """
    plt.figure(figsize=(14, 6))

    plt.plot(satisfactions, alpha=0.9, color='yellow', label='Raw Satisfaction')

    smooth_satisfactions = [np.mean(satisfactions[max(0, i - window_size):i + 1])
                            for i in range(len(satisfactions))]
    plt.plot(smooth_satisfactions, color='green', linewidth=2,
             label=f'Smooth Satisfactions(window={window_size})')

    plt.title('SAC Episode Satisfactions')
    plt.xlabel('Episode')
    plt.ylabel('Satisfactions')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存和显示图像
    plt.tight_layout()
    filename = f'satisfaction_group_{datetime.now().strftime("%Y%m%d%H%M")}.png'
    save_dir = get_output_dir("sac_city_power", "figures")

    # 自动创建目录（跨平台兼容）
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    # 使用路径拼接（自动适配操作系统）
    filepath = os.path.join(save_dir, filename)
    # 保存图像
    plt.savefig(filepath, dpi=300)  # 推荐添加 bbox_inches 避免边框截断
    plt.show()


# -------------------------------------- #
# 参数设置
# -------------------------------------- #

num_epochs = 4000  # 训练回合数
capacity = 5000  # 经验池容量
min_size = 2000  # 经验池训练容量
batch_size = 64
n_hiddens = 128
actor_lr = 1e-4  # 策略网络学习率
critic_lr = 1e-3  # 价值网络学习率
alpha_lr = 1e-3  # 课训练变量的学习率
target_entropy = -6.9
tau = 0.005  # 软更新参数
gamma = 0.99  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
print(device)
beam_info = BeamInfo()
satellite_freq_state = SatelliteFreqState()
current_beam_state = CurrentBeamState(beam_info.beam_info)

# -------------------------------------- #
# 环境加载
# -------------------------------------- #


env = Env(satellite_freq_state, beam_info, current_beam_state)
env.reset(beam_info)  # 设置随机种子
n_states = 202 # 状态数 4
freq_dim = 10
slots_dim = 10
power_dim = 10

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

agent = SAC(n_states=n_states,
            n_hiddens=n_hiddens,
            freq_dim=freq_dim,
            slots_dim=slots_dim,
            power_dim=power_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            device=device,
            )

# -------------------------------------- #
# 经验回放池
# -------------------------------------- #

buffer = ReplayBuffer(capacity=capacity)

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

return_list = []  # 保存每回合的return
satisfaction_list = []
for i in range(num_epochs):
    episode_satisfaction = 0
    beam_info = BeamInfo()
    state = env.reset(beam_info)
    epochs_return = 0  # 累计每个时刻的reward
    done = False  # 回合结束标志

    while not done:
        # 动作选择
        mask = env.get_action_mask(state)
        action = agent.take_action(state, mask)

        actual_action = action
        env_action = [
            env.beam_idx % 8,  # group: 0-7
            env.get_first_free_slot(actual_action),  # freq: 0-99
            action[1] + 1,  # slots: 1-10
            10 * np.log10((action[2] + 1) * 5)  # power: 0-50W to dB
        ]
        # 环境更新
        next_state, reward, done, info = env.step(env_action)
        # 将数据添加到经验池
        buffer.add(state, actual_action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计回合奖励
        epochs_return += reward
        agent.log_step(i, info)
        episode_satisfaction = info['total_satisfaction']

        # 经验池超过要求容量，就开始训练
        if buffer.size() > min_size:
            s, a, r, ns, d = buffer.sample(batch_size)  # 每次取出batch组数据
            # 构造数据集
            transition_dict = {'states': s,
                               'actions': a,
                               'rewards': r,
                               'next_states': ns,
                               'dones': d,
                               's_mask': env.get_action_mask(np.array(s)),
                               'ns_mask': env.get_action_mask(np.array(ns))}
            # 模型训练
            agent.update(transition_dict)
    # 保存每个回合return
    return_list.append(epochs_return)
    satisfaction_list.append(episode_satisfaction)
    util = np.count_nonzero(env.satellite_freq_state.freq_pool_real)
    occupy = env.get_occupy()
    print(
        f"Episode {i}, Reward: {epochs_return:.2f}, Satisfaction: {episode_satisfaction:.2f}, util{util},occupy{occupy},interfere{env.interfere_num},inrange{env.interfere_in_range_num}")

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plot_reward(return_list)
plot_satisfaction(satisfaction_list)
print(f"Training completed at {datetime.now().strftime('%Y%m%d%H%M')}")

agent.save_model()
