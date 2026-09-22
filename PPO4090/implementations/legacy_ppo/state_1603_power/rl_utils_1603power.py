import torch  # 导入 PyTorch，作为深度学习的核心框架
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from implementations.legacy_ppo.state_1603_power.Environment_1603_power import BeamInfo
from project_paths import get_output_dir


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []  # 存储每回合的总回报
    satisfaction_list = []
    for episode in range(num_episodes):  # 将训练分为 10 个阶段
        episode_return = 0  # 初始化当前回合的总回报
        episode_satisfaction = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],
                           'old_log_probs': [], 'masks': []}  # 记录当前回合的数据
        for _ in range(1):
            beam_info = BeamInfo()
            beam_info.shuffle()  # 生成新的随机排列
            beam_info.apply_shuffle()  # 应用排列
            # print(beam_info.beam_info)
            state = env.reset(beam_info)  # 重置环境
            mask = env.get_action_mask()

            init_entropy_coef = 0.01
            end_entropy_coef = 0.001
            entropy_coef = init_entropy_coef - episode * (init_entropy_coef - end_entropy_coef) / (num_episodes / 4)
            done = False
            while not done:  # 游戏未结束时继续
                action, old_log_probs = agent.take_action(state, mask)  # 选择动作
                actual_action = action
                env_action = [
                    env.beam_idx % 8,  # group: 0-7
                    env.get_first_free_slot(actual_action),  # freq: 0-99
                    action[1] + 1,  # slots: 1-10
                    10 * np.log10((action[2] + 1) * 5)  # power: 0-50W to dB
                ]

                next_state, reward, done, info = env.step(env_action)  # 执行动作，获取环境反馈
                transition_dict['states'].append(state)  # 记录状态
                transition_dict['actions'].append(actual_action)  # 记录动作
                transition_dict['next_states'].append(next_state)  # 记录下一状态
                transition_dict['rewards'].append(reward)  # 记录奖励
                transition_dict['dones'].append(done)  # 记录是否结束
                transition_dict['old_log_probs'].append(old_log_probs.item())
                transition_dict['masks'].append(mask)
                mask = env.get_action_mask()
                state = next_state  # 更新当前状态
                episode_return += reward  # 累加回报
                agent.log_step(episode, info)
                episode_satisfaction = info['total_satisfaction']
        return_list.append(episode_return)  # 记录当前回合总回报
        satisfaction_list.append(episode_satisfaction)
        actor_loss, critic_loss = agent.update(transition_dict, entropy_coef)  # 使用当前回合数据更新策略
        util = np.count_nonzero(env.satellite_freq_state.freq_pool_real)
        occupy = env.get_occupy()
        print(
            f"Episode {episode}, Reward: {episode_return:.2f}, Satisfaction: {episode_satisfaction:.2f}, actor_loss{actor_loss}, critic_loss{critic_loss},util{util},occupy{occupy},interfere{env.interfere_num},inrange{env.interfere_in_range_num}")
    return return_list, satisfaction_list  # 返回所有回合的总回报


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()  # 将 TD-误差转换为 NumPy 数组
    advantage_list = []  # 初始化优势值列表
    advantage = 0.0  # 初始化递归变量
    for delta in td_delta[::-1]:  # 从后往前遍历 TD-误差
        advantage = gamma * lmbda * advantage + delta  # 递归计算优势值
        advantage_list.append(advantage)  # 存储优势值
    advantage_list.reverse()  # 恢复正序
    return torch.tensor(np.array(advantage_list), dtype=torch.float)  # 返回优势值张量


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

    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # 保存和显示图像
    plt.tight_layout()
    filename = f'reward_group_{datetime.now().strftime("%Y%m%d%H%M")}.png'
    save_dir = get_output_dir("legacy_state_1603_power", "figures")

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

    plt.title('Episode Satisfactions')
    plt.xlabel('Episode')
    plt.ylabel('Satisfactions')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存和显示图像
    plt.tight_layout()
    filename = f'satisfaction_group_{datetime.now().strftime("%Y%m%d%H%M")}.png'
    save_dir = get_output_dir("legacy_state_1603_power", "figures")

    # 自动创建目录（跨平台兼容）
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    # 使用路径拼接（自动适配操作系统）
    filepath = os.path.join(save_dir, filename)
    # 保存图像
    plt.savefig(filepath, dpi=300)  # 推荐添加 bbox_inches 避免边框截断
    plt.show()
