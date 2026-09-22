# 论文基线范围与 MLP-DQN 实现验收

日期：2026-09-21。本轮以《Transformer-Enhanced SAC for Joint.pdf》第5页 IV 实验部分的真实比较名单为范围。以下记录区分来源明确的基线结构、统一环境适配和论文未给出的工程参数。此文件不宣称已完成远程基线训练。

## 本轮七个基线

论文 IV 节明确列出 Fixed、Greedy、Random、MLP-SAC、CNN-SAC、MLP-DQN 和 MLP-PPO。Greedy 选择最少占用的 FS，并按需求选择 20/25/30 W；MLP-DQN 使用 start/length/power 三个 Q head 求和；MLP-PPO 使用三个独立动作分布和状态价值网络。本次 DQN 代码位于 `PPO4090/implementations/horizon_tsac_20260920/paper_baselines/dqn.py`，其余基线由相应模块实现与验收。

开题报告提供的是更广泛的研究计划。定位方式为 `word/document.xml` 中所有 `w:p` 的1起始索引，包含空段和表格内段落，不是页码：

- P246，表4，3.3「实验方法和步骤」第(7)项，明确写“与传统启发式算法、凸优化算法、经典 DRL 方法（DQN、DDPG、PPO）进行对比分析，包括容量、SINR、干扰水平、公平性、收敛速度等指标”。DQN/DDPG/PPO 有名称，但启发式和凸优化只给类别。
- P263 只说明作者已有 DQN/PPO/DDPG 的复现经验，没有给出本次实验参数。DQN/DDPG/PPO 全文仅出现在 P246 和 P263。
- P113 的 GA[9]、SA[10]、PSO[11] 属于1.2相关工作；对应 P125/P126/P127 的参考文献，不自动扩展为本轮三个必做对照。
- P214、P227、P241–242 将 SCA/WMMSE/凸松弛放在第三阶段功率精细优化，并以 DRL 输出为初值；不能直接当作已完成的第二阶段联合离散分配基线。

因此，本轮七个基线以 PDF 实验名单为准。DDPG 保留为开题报告与论文的范围差异；目前没有给出连续 DDPG 输出到完整合法离散动作的映射，不能把它冒称为已实现。开题报告源文件 SHA256 为 `209d9ca5ecf24b2a3f21da72be43ed66265f09990165748c09ba239e43c81148`，只读核查前后未变。

## DQN 的实际数学与接口

输入使用当前环境的局部 `legacy205`，100槽时为205维，小型诊断场景为 `5+2N` 维。网络采用两层128维 ReLU MLP；ALLOC 动作值为三个分支输出之和。硬约束环境新增独立 SKIP head，避免将 −1 当作最后一个槽。确定性选择在全部合法完整候选上取 argmax，并列时选择最小候选 ID；不会逐 head 取最大值后修补越界动作。

更新使用标准 DQN：`y = r + gamma * max_legal Q_target(s',a')`。目标网络自己选择合法最大值，默认不是 Double DQN。只有 terminated 关闭 bootstrap；外部 truncated 使用实际 final observation 继续 bootstrap。当前/下一观察（包括 terminal）的 ActionSpec、布尔 mask、维度与有限值均验证，保存 mask 必须与环境纯函数重建一致。

回放由外层训练控制器管理，本类不创建文件、不读取数据、不维护自己的经验池。有效 batch 可分微批累计，按整个 batch 的样本数归一化，只做一次 Adam 更新和一次 target 软更新；每次有效更新令 update_step 加1。

## 明示的工程选择

本轮固定以下可审计配置，不将其冒称为论文已经给出的全部原始超参数：

- epsilon 默认从 1.0 随 **训练环境交互数**线性下降到 0.05，10000 步完成；构造器可显式配置 start/end/decay_steps，`agent_spec()` 和 checkpoint 保存实际值。
- 每次非确定性 `act` 代表一次采集交互，interaction_step 加1，包括强制 SKIP；确定性评估既不推进 epsilon，也不消耗探索 RNG。训练控制器所有采集步调用 act，warmup 只控制更新门槛。若其他控制器绕过 act 采集，须调用 `set_interaction_step` 同步真实计数。
- epsilon 探索在完整合法候选集合均匀抽样。无 ALLOC 时只选择 SKIP。
- Adam 学习率取 `config.train.critic_lr`，MSE TD loss，gradient_clip_norm、gamma、Polyak tau 均取本 run 的统一 Config；与 SAC 比较时报告真实更新次数和交互预算。
- 记录训练设备类型。精确恢复（restore_rng=True）拒绝 CPU/CUDA 类型切换；CPU 评估可以 restore_rng=False 加载 CUDA 来源权重，完全跳过 CUDA RNG 恢复。本 DQN 的探索使用独立 NumPy RNG，没有 CUDA 动作采样 generator。
- DQN 自身的精确恢复入口也校验 schema、physics、env（含总预算）、data、model 和非运行期 train 字段，不依赖外层 artifacts 才能拒绝配置漂移。仅允许回合/步数/墙钟上限、同类型设备、线程数、保存间隔变化；仅评估的 restore_rng=False 不代表精确续训。

## 本地验收

```powershell
cd D:\graduate\PPO4090
.\.venv-horizon\Scripts\python.exe -m unittest tests.horizon_tsac_20260920.test_paper_dqn -v
```

12 项测试通过（最近完整运行2.743秒）。覆盖：

1. 逐 head 最大组合越界的反例，完整合法三分量求和 argmax 正确；SKIP head 独立。
2. 手算 TD 目标 `[7,7,3]`，同时验证非法高 Q 功率档屏蔽、标准而非 Double DQN、truncated bootstrap、terminal 无 bootstrap。
3. epsilon 线性日程、完整合法集合探索、无预算强制 SKIP、评估不动计数/RNG、终局拒绝动作。
4. 两动作诊断 oracle 中学习更高回报动作、loss 下降、目标网络无梯度并按 tau 软更新。
5. 微批与整批损失/参数更新一致，optimizer 计数不被微批倍增。
6. checkpoint 恢复后下一批探索动作及下一次更新的 online/target 参数逐位一致；保存快照不别名引用后续参数。
7. 错误 mask、功率签名、terminal 下一观察签名、不同 epsilon 配置均在更新前拒绝。
8. 100槽/205维/9551动作接口，CPU评估跳过CUDA RNG与跨设备精确恢复拒绝。
9. 绕过外层检查点工具时，改变预算、物理噪声、业务比例、model配置、batch/microbatch、更新日程或seed仍拒绝精确恢复；明确允许的运行限制变化可加载。

诊断 oracle 只用于更新公式验证，不是生产训练样本。此实现已完成本地数学与恢复测试，远程训练结果、正式多 seed 结论和各基线统一预算比较需另行记录。
