from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.sac_fyh_io.Environment_fyh_IO import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
import re
from implementations.sac_fyh_io.sac_fyh_IO import ReplayBuffer, SAC
from project_paths import COVER_OUTPUT_DIR, get_output_dir
# 顶部 import 补充
import pandas as pd
import random


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
    save_dir = get_output_dir("sac_fyh_io", "figures")

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
    save_dir = get_output_dir("sac_fyh_io", "figures")

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
capacity = 20000  # 经验池容量
min_size = 2000  # 经验池训练容量
batch_size = 256
n_hiddens = 128
actor_lr = 1e-5  # 策略网络学习率
critic_lr = 1e-4  # 价值网络学习率
alpha_lr = 1e-4  # 课训练变量的学习率
target_entropy = -6.9
tau = 0.005  # 软更新参数
gamma = 0.99  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
print(device)
# 收集 cover_output_*.csv 文件并按数字顺序排序
def _num_key(p):
    m = re.search(r'(\d+)', str(p))
    return int(m.group(1)) if m else -1

csv_paths = sorted(COVER_OUTPUT_DIR.glob('cover_output_*.csv'), key=_num_key)
assert len(csv_paths) > 0, "未找到 cover_output_*.csv 文件，请确认路径。"

# 用第一份 CSV 初始化一次（仅用于构建 env）
csv_dfs = [pd.read_csv(p) for p in csv_paths]

# 用第一份 df 初始化一次（仅用于构建 env）
beam_info = BeamInfo(data_df=csv_dfs[0])
satellite_freq_state = SatelliteFreqState()
current_beam_state = CurrentBeamState(beam_info.beam_info)
env = Env(satellite_freq_state, beam_info, current_beam_state)
env.reset(beam_info)


# -------------------------------------- #
# 环境加载
# -------------------------------------- #



n_states = 205 # 状态数 4
freq_dim = 100
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
def augment_df(df: pd.DataFrame,
               rate_scale_range=(0.7, 1.3),
               rate_noise_frac=0.03,
               bw_scale_range=(0.85, 1.15)):
    d = df.copy()

    # -------- 随机参数记录 --------
    aug_info = {}

    # 1) rate scale
    rate_scale = np.random.uniform(*rate_scale_range)
    aug_info["rate_scale"] = rate_scale
    d["rate"] = d["rate"].astype(float) * rate_scale

    # 2) rate noise
    rate_noise = np.random.normal(0.0, rate_noise_frac, size=len(d))
    aug_info["rate_noise_std"] = rate_noise_frac
    d["rate"] = d["rate"] * (1.0 + rate_noise)

    # 3) beamwidth scale（如果有）
    if "beamwidth" in d.columns:
        bw_scale = np.random.uniform(*bw_scale_range)
        aug_info["beamwidth_scale"] = bw_scale
        d["beamwidth"] = d["beamwidth"].astype(float) * bw_scale
        d["beamwidth"] = np.clip(d["beamwidth"], 0.05, 2.0)
    else:
        aug_info["beamwidth_scale"] = None

    return d, aug_info


for i in range(num_epochs):
    episode_satisfaction = 0
    # 80% 常规场景 + 20% 强扰动“hard”场景（尾部分布）
    csv_idx = random.randrange(len(csv_dfs))
    base_df = csv_dfs[csv_idx]
    csv_name = f"csv_{csv_idx}"


    if random.random() < 0.2:
        df, aug_info = augment_df(
            base_df,
            rate_scale_range=(0.5, 1.6),
            rate_noise_frac=0.06,
            bw_scale_range=(0.8, 1.25),
        )
        aug_level = "hard"
    else:
        df, aug_info = augment_df(
            base_df,
            rate_scale_range=(0.8, 1.2),
            rate_noise_frac=0.02,
            bw_scale_range=(0.9, 1.1),
        )
        aug_level = "normal"


    beam_info = BeamInfo(data_df=df)
    state = env.reset(beam_info)
    epochs_return = 0  # 累计每个时刻的reward
    done = False  # 回合结束标志

    while not done:
        # 动作选择
        action = agent.take_action(state)

        actual_action = action
        env_action = [
            env.beam_idx % 8,  # group: 0-7
            action[0],  # freq: 0-99
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
                               'dones': d}
            # 模型训练
            agent.update(transition_dict)
    # 保存每个回合return
    return_list.append(epochs_return)
    satisfaction_list.append(episode_satisfaction)
    util = np.count_nonzero(env.satellite_freq_state.freq_pool_real)
    occupy = env.get_occupy()
    print(
        f"{csv_name} | Episode {i} | aug={aug_level} | "
        f"rate_scale={aug_info['rate_scale']:.3f} | "
        f"rate_noise_std={aug_info['rate_noise_std']:.3f} | "
        f"bw_scale={aug_info['beamwidth_scale']} | "
        f"Reward={epochs_return:.2f} | Satisfaction={episode_satisfaction:.2f} | "
        f"util={util} | occupy={occupy} | interfere={env.interfere_num} | "
        f"inrange={env.interfere_in_range_num}"
    )

    agent.flush_log()
# -------------------------------------- #
# 绘图
# -------------------------------------- #

plot_reward(return_list)
plot_satisfaction(satisfaction_list)
print(f"Training completed at {datetime.now().strftime('%Y%m%d%H%M')}")
metrics_dir = get_output_dir("sac_fyh_io", "metrics")
stamp = datetime.now().strftime('%Y%m%d%H%M')
metrics_path = metrics_dir / f'metrics_{stamp}.npz'
np.savez(metrics_path, returns=np.array(return_list), satisfactions=np.array(satisfaction_list))
print(f"Saved metrics to {metrics_path}")

agent.save_model()
