# PPO4090 第二部分已完成成果（T-SAC）代码只读审查与重构方案

## 1. 审查边界与结论

本次审查只读取 `D:\graduate\PPO4090`、论文/开题报告的文本提取件及 `问题清单.md`，没有运行训练，没有导入项目模块，也没有用 `torch.load`、`pickle` 或其他方式反序列化 checkpoint。模型产物仅通过文件名、文件大小、修改时间和源码保存逻辑核验；NPZ 指标使用 NumPy `allow_pickle=False` 读取。除本报告外未修改任何项目文件。

最重要的结论是：目录中并存至少两代 T-SAC 实验，不能把它们当成一套实现来解释。

1. **当前主线（2025-12-12 产物）**：`sac_train_fyh_IO.py` → `Environment_fyh_IO.py` + `sac_fyh_IO.py`。它读取 `cover_output_list` 的 100 份覆盖方案，每份 200 行，状态 205 维，频率起点 100 类；当前工作区源码设置 `beam_num=200`、`Ptot=6000 W`。用户已确认论文实验主要使用 2025 年 12 月的 fyh_IO 覆盖输入版本，唯一的 `metrics_202512120428.npz`、同分钟模型和 `sac_full_fig` 图也与此归属一致。但论文表格写的是 100 波束、3000 W，说明论文运行时配置/源码快照与当前文件至少在规模参数上不同；没有当次 manifest 或版本记录，不能凭当前源码断言论文实验实际跑了 200 波束。
2. **旧的 token/城市版（2025-07-15 产物候选）**：`sac_train_tok.py` → `Environment_sac_p.py` + `sac_transformer_tok.py`。它直接读取 `cities_china.csv`，随机生成 100 个需求，状态 202 维，频率动作 10 类（十槽分组），波束数 100。
3. `sac_train_p.py` + `sac_transformer_p.py`、`Environment_103.py` 等是并列实验分支；文件名相似不等于运行关系。项目不是 Git 仓库，无法用提交历史证明哪一份源码生成了哪个 checkpoint。

在当前主线上，最需要先处理的是四个会改变结果可信度的问题：零需求填充被计作满满意度、总功率越界动作仍被执行且惩罚未进入奖励、温度目标的符号约定使 α 单向趋零、单独保存的 critic-2 覆盖 critic-1 文件。其次是动作可行性、随机种子/运行清单、指标定义和模型结构的可解释性。

## 2. 真实调用链、数据流与动作语义

### 2.1 当前主线

`sac_train_fyh_IO.py:1-9` 明确导入 `Environment_fyh_IO` 和 `sac_fyh_IO`。训练脚本在 `107-116` 搜集并一次性读取全部 `cover_output_*.csv`，在 `198-225` 每回合随机选择一份并增强 rate/beamwidth，再构造 `BeamInfo`。环境、模型和训练循环的关系是：

```text
cover_output_*.csv
  → BeamInfo（截断/补齐为 200 行，按 rate 降序）
  → Env.reset / Env.step（顺序处理 200 个波束）
  → 205 维状态
  → PolicyNet（三个独立 categorical 头）
  → [起始 FS 0..99, 连续槽数 1..10, 功率档 5..50 W]
  → Env 将 group 固定为 beam_idx % 8
  → ReplayBuffer → 双 critic / actor / α 更新
```

状态的实际布局由 `Environment_fyh_IO.py:462-486` 和 `sac_fyh_IO.py:88-104` 共同确定：`[geo(2), rate(1), beamwidth(1), remaining_power(1), interference(100), occupancy(100)]`，共 205 维。模型把前四类标量编码成 4 个 token，再把 100 个槽 reshape 成 `10×10`，每一行把 10 个干扰值和 10 个占用值拼成一个 token，因此一共是 14 个 token；`pos_embed` 的 `(1,14,d_model)` 是一致的。源码注释所写的“101 token”和“204 维”已经过时，不应据此判定 shape bug。

动作在 `sac_train_fyh_IO.py:234-239` 被映射为：频组 `beam_idx % 8`（不是策略动作）、起始 FS `0..99`、连续槽数 `1..10`、离散发射功率 `5,10,...,50 W` 转为 dBW。当前主线确实是“固定频组 + 起始槽/带宽/功率联合决策”，不是对 8 个频组做联合优化。

### 2.2 旧 token 版

`sac_train_tok.py:1-8` 的真实依赖是 `Environment_sac_p.py` 和 `sac_transformer_tok.py`。其频率头只有 10 类（`sac_train_tok.py:113-116`）；动作先选一个十槽分组，再由 `get_first_free_slot` 在该分组内找首个空槽（`sac_train_tok.py:161-165`、`Environment_sac_p.py:732-742`）。所以旧版的 `freq_dim=10` 与当前主线的 `freq_dim=100` 是不同动作定义，并非同一模型参数前后矛盾。

## 3. 数据与产物核验

- `cover_output_list` 实际有 **100 个 CSV**（`cover_output_0.csv` 至 `cover_output_99.csv`），不是旧清单所称 56 份。每份恰好 200 个数据行，表头均为 `lat,lon,rate,beamwidth`，合计 20,000 行。
- CSV 实际数值显示名为 `lat` 的列范围为 73.7–134.51，名为 `lon` 的列范围为 12.2–56.04，列名与通常地理含义相反。`Environment_fyh_IO.py:321` 以 `df[['lon','lat']]` 读入，随后按 `[lat,lon]` 使用，因此当前代码恰好补偿了上游错名。应在接口层显式改正，不能只改代码列顺序或只改 CSV 表头，否则会引入新错。
- 20,000 行中有 **5,870 行（29.35%）同时出现 `rate=0`、`beamwidth=0`**。它们显然像覆盖方案的空位/填充项，而环境把它们当作“满意度 1”的波束，见发现 F1。
- 保存目录有 3 组时间戳：`202507150301`、`202507150308`、`202512120428`。每组都有 actor、critic_1、target_critic_1/2、full_model，但均无独立 `critic_2`，与源码的覆盖错误完全吻合（发现 F4）。
- `metrics_202512120428.npz` 修改时间为 2025-12-12 04:27，使用 NumPy `allow_pickle=False` 读取后确认：`returns` 与 `satisfactions` 均为 `(4000,) float64`。`returns` 最小值 −266.708、最大值 187.873、均值 82.209、末值 66.249；`satisfactions` 最小值 73.385、最大值 198.730、均值 176.783、末值 195.714。后者是约 200 个波束的满意度**总和**而非 `[0,1]` 均值，且受 F1 的零需求填充计分影响，不能直接解释成“平均满意度 0.98”。它和 `202512120428` 模型同一运行的可能性很高，但因保存时各自重新取分钟时间、没有 run ID/manifest/hash，只能标为**高可信推断，不能证明**。
- 目录不是 Git 仓库，产物没有源码 commit、配置、种子、数据 hash 或环境版本。现有 checkpoint 的精确来历不可审计。

## 4. 高价值发现

### F1（确定 bug，严重）：零需求填充被人为计为满满意度

`Environment_fyh_IO.py:927-960` 对 `required_rate <= 0` 的行返回 reward 0，却执行 `self.total_satisfaction += 1`。这会使约 29.35% 的输入行自动贡献满分，且日志中的 `info['total_satisfaction']` 在加一之前构造（`931-959`），产生一步滞后的内部/外部指标。训练脚本最后取每回合最后一次 `info`（`sac_train_fyh_IO.py:249,264`），因此终局满意度甚至少计最后一个零行，但总体仍被大量填充项抬高。修复应使用有效波束 mask：零行不进入 episode，或分母只统计有效波束，绝不能用“填充=已满足”混合业务指标。

### F2（确定 bug，严重）：功率越界动作先执行，终止惩罚却不进 reward

功率先在 `Environment_fyh_IO.py:979-981` 加到 `total_power`，超过 `Ptot=beam_num×30` 后在 `991-995` 设 `done=True`；但 `power_exceed_penalty` 在奖励中被注释掉（`1072-1088`）。越界波束仍会进入链路计算、`history_beam` 和满意度。这不是严格满足论文总功率约束的 MDP。应在动作进入环境前屏蔽超预算档位，或明确使用“截断到可行功率”的投影；若研究软约束，应保留越界转移并定义拉格朗日/罚项，而不是无惩罚终止。

### F3（确定 bug，严重）：α 的目标熵符号约定不一致，温度会被持续压低

训练设置 `target_entropy=-6.9`（`sac_train_fyh_IO.py:101`），模型计算的 `entropy` 是正的 Shannon entropy；更新式为 `(entropy - target_entropy) * exp(log_alpha)`（`sac_fyh_IO.py:445`）。因此括号恒为正，梯度下降会持续降低 `log_alpha`，α 趋向 0，而不是围绕目标熵调节。若保留正熵定义，目标也应为正并使用与其一致的 loss；若使用标准 `log π` 约定，则应整体按该约定重写并做方向单元测试。

### F4（确定 bug，高）：critic-2 的独立 checkpoint 覆盖 critic-1

`sac_fyh_IO.py:462-463` 把 `critic_1_path` 和 `critic_2_path` 都命名为 `sac_transformer_critic_1_<timestamp>.pth`，随后 `479-480` 依次写入，第二次覆盖第一次。三组历史产物均只有 critic_1 文件，实际内容按控制流应是 critic-2。`full_model` 字典仍同时含两个 critic（`468-476`），所以完整包可能可用，但未经反序列化不作保证。

### F5（确定 bug，高）：终止步不写频谱状态，最终环境状态与已执行动作不一致

正常分支仅在 `not done` 时调用 `update_state`（`Environment_fyh_IO.py:997-1002`），但无论是否 done 都把动作加入历史并计算速率（`1004-1032`）。第 200 个波束以及功率越界步的频谱占用不进入 `freq_pool_real`，历史/吞吐/功率却已更新。terminal `next_state` 因 `beam_idx==beam_num` 被重置成零频谱（`462-475`）。虽然 `done` 正确阻止 bootstrap，但日志、终局约束验证和离线评估会看到互相矛盾的状态。

### F6（确定问题，高）：当前主线完全没有动作 mask，碰撞后由环境静默缩短为零或更少槽

环境定义了 `get_action_mask`（`Environment_fyh_IO.py:898-907`），但训练在 `sac_train_fyh_IO.py:229-261` 从不调用，`sac_fyh_IO.PolicyNet.forward` 也无 mask 参数（`80-118`）。若起始槽已占用，环境把实际槽数改成 0；若中途遇占用，则截短连续块（`Environment_fyh_IO.py:968-976`）。actor/critic 存储和学习的是原始动作索引，环境执行的是被修改动作，形成 action aliasing。应定义纯函数 `project_action` 并把“提出动作”和“执行动作”都记录下来，最好直接生成联合可行 mask。

### F7（确定表述/接口问题，高）：论文公式、解释文字与代码的 group 语义未对齐

频组由 `beam_idx % 8` 固定（`sac_train_fyh_IO.py:235`），不由策略选择；相同槽号在不同 group 可以复用。论文式 (7a) `Σ_b y_gb x_bs ≤ 1` 本身约束的是**同一频组内**每槽至多一个波束，并非全局 FS 唯一，因此不能据此断言代码的跨组复用违反 (7a)。当前环境在同一 group 内通过占用检查截断动作（`Environment_fyh_IO.py:968-976`），通常实现了同组排他。真正需要校正的是：论文解释文字若写成“每个 FS 最多一个波束”会与公式的 group 条件产生歧义；同时代码的 group 是固定轮询而非优化变量。应在 schema 和论文中统一 `group` 的物理含义（频带、颜色、极化组合）、`y_gb` 的来源，以及策略究竟优化哪些变量。

### F8（建模选择，高风险）：三个动作头和 Q 的加法分解排除了动作交互

actor 独立输出 frequency/slots/power 三个 categorical 分布（`sac_fyh_IO.py:112-118`）；critic 也输出三组边际 Q，并在 `_joint_q` 中相加（对应 `sac_transformer_tok.py:336-350`，当前文件同构）。频率起点、槽数、功率在物理上显著耦合，真实 `Q(s,f,n,p)` 未必可写为三项之和。这不是语法 bug，而是强建模假设；需要用“自回归 actor + joint/conditional critic”作消融，不能把现实现直接称为完整联合动作 SAC。

### F9（确定算法偏差，中高）：双 critic 的 clipped target 是逐头取 min 后相加

当前主线在 `sac_fyh_IO.py:337-346` 计算 `Σ_h E_{a_h}[min(Q1_h,Q2_h)]`。它一般不等于标准联合动作目标 `E_{a~π}[min(Q1(s,a),Q2(s,a))]`，会把不同 critic 在不同 head 上的低值拼成一个不存在的“混合 critic”，产生额外悲观偏差。即便先分别求两个 critic 的期望再取 `min(Σ_h E[Q1_h], Σ_h E[Q2_h])`，仍通常不等于“逐联合动作取 min 后再求期望”。若暂时保留 additive Q 和 factorized policy，应对 100×10×10=10,000 个联合动作精确枚举 `Q_i(s,f,n,p)=Q_i^f+Q_i^n+Q_i^p` 后计算乘积策略下的 `E[min(Q1_joint,Q2_joint)]`，或用联合动作抽样得到一致估计；actor 目标也应按同一联合分布定义。长期方案是 joint/conditional critic。

### F10（建模选择，中高）：按需求降序是顺序策略，不是自动的信息泄漏

`Environment_fyh_IO.py:337-350` 始终按 rate 降序处理；`shuffle=False` 并不会保持原顺序，而是仅决定是否再随机打乱排序索引。排序本身使用的是当前调度器可观测的需求时，不构成训练/测试标签泄漏；它会造成先后次序优势和分布偏置。应报告“降序策略”并与随机序、多序评估、原始覆盖顺序做对照。旧清单把它直接定性为“泄漏”是不成立的。

### F11（确定可复现性缺陷，高）：随机源无统一 seed，且环境主动取消 NumPy 可复现性

训练同时使用 Python `random`（场景选择）和 NumPy（增强），模型使用 PyTorch；没有统一 seed。`BeamInfo.__init__` 还在每回合执行 `np.random.seed(None)`（`Environment_fyh_IO.py:306-307`），会破坏调用方设置的 NumPy seed。没有保存 RNG state、CUDA deterministic 设置、配置或数据 hash，同一 checkpoint 无法复跑。

### F12（确定问题，中高）：数据单位和字段契约模糊，rate 被额外乘 0.25

`Environment_fyh_IO.py:320-323` 注释称 CSV rate 为 Mbps，却实际乘 `0.25e6`，不是通常的 `1e6`。这可能是把覆盖阶段总需求折算为 25% 负载的有意选择，也可能是单位 bug；源码和论文材料未给依据。它直接改变满意度与最优槽数，应提升为配置项 `traffic_scale=0.25` 并写入运行清单。

### F13（确定问题，中）：有效波束数可变，却固定为 200 步并混入 padding

每份 CSV 固定 200 行，但有效项数量不同；代码在 `315-318` 还会复制不足 200 行的数据。固定长度便于 Transformer/批处理，却不应改变业务总体。建议 CSV 增加 `valid` 和稳定 `beam_id`，episode 只遍历有效项；若必须 padding，环境、attention 和指标都使用同一个 mask。

### F14（建模选择，中高）：奖励主要是阈值符号 + SGM，直接干扰项和功率项未启用

`Environment_fyh_IO.py:1055-1061` 将总满意度增量是否达到 1 映射为 ±1；奖励在 `1072-1088` 使用 `sat + sgm_rate - sgm_penalty - occupy_penalty`，干扰损失、显式干扰惩罚、功率 gap、越界惩罚都被注释。干扰仍会通过 SINR→速率→SGM 间接进入奖励，所以“奖励完全没有干扰机制”也不准确；但论文若把 SINR 提升归因于显式干扰抑制，证据不足。应把奖励各项、权重和间接因果链写清，并保存逐项 episode 指标。

### F15（待确认，中高）：地理列名反转被代码偶然补偿，跨阶段接口极脆弱

CSV 的 `lat` 实际像经度、`lon` 实际像纬度；代码反序读取后得到正确数值顺序。当前链路可能数值正确，但任何“清理列名”的单边修改都会把坐标再次颠倒。应在覆盖阶段输出时改成标准 WGS84 schema，并用范围断言与一个已知城市的 golden case 验证。由于缺少覆盖生成代码，无法确认这是导出约定还是历史错误。

### F16（建模选择/待确认，中）：功率上限并非“完全不存在”

策略动作本身限制为 5–50 W 十档（`sac_train_fyh_IO.py:238`），因此每波束有一个隐式硬上限 50 W。旧清单说“没有 Pmax”过强；准确说法是：上限由动作映射隐式实现，没有命名为 `Pmax`、没有环境断言，且论文参数表/约束与代码配置没有单一来源。总功率 `Ptot` 则存在，但实施方式有 F2 的问题。

### F17（确定语义冲突，严重）：一次计入总预算的“波束功率”被每个槽重复用于链路预算

环境在 `Environment_fyh_IO.py:979-981` 只把动作功率加到 `total_power` 一次；`receiving_power` 在 `591-617` 也不除以实际槽数，得到一个 `P_rx`。随后 `current_rate_calculate` 对每个已分配槽都使用同一个 `P_rx` 计算 SINR 并累加速率（该函数由 `1027-1031` 调用）。因此若动作的 `P_b` 是论文式 (7f) 总预算中的“每波束总功率”，代码等价于在每槽重复发射 `P_b`，实际总功率应随槽数增加；若动作其实是“每槽功率”，`total_power += power` 又少乘槽数。两种解释无法同时成立。这必须作为改变结果的语义修复：在 schema 中选定 `p_beam_total` 或 `p_per_slot`，前者明确分配为 `p_bs`（例如均分 `p_beam_total/|S_b|`，或另设槽级变量），后者按 `Σ_s p_bs` 计总预算，并相应重算干扰。不能在保持语义重构中悄悄改掉。

### F18（易错自我识别与非法输入边界，中）：用动作相等代替 beam ID 排除当前记录

当前动作在调用 `interfere_beams` 前已经追加到 `history_beam`（`Environment_fyh_IO.py:1004-1011`），所以 `639-645` 的 `(hist_group == group and hist_freq == freq)` 主要是在排除刚追加的“自身”。正常可达路径下，同 group 已占用起点会在 `968-976` 把当前实际槽数截为 0，因此不能据此断言“有效解中的完全重叠干扰被漏算”，前一版报告对此定性过强。风险在于代码用“动作字段相等”猜测身份：非法/手工输入、terminal 步未写占用（F5）、未来允许同组空间复用后，历史波束可能被一并误排。应明确写成 `if hist_beam_id == self.beam_idx: continue`，再单独处理异极化与频谱重叠；测试覆盖自身、合法历史项和非法边界输入。

### F19（论文建模问题，确定）：式 (7c) 本身不能严格表达“恰好一个连续频率块”

论文式 (7c) 使用 `0 < Σ_{s=1}^{Ns-1}|x[s+1]-x[s]| ≤ 2`。它允许 `[1,1,0,0,1,1]`（两次跳变、两个离散块），同时排除全 1（零次跳变）这一合法连续块。因此“代码实现连续、论文式也保证连续”不能一起成立。代码的 start+length 参数化天然给出单连续块；论文应改用起点/长度变量，或加入边界项/连通性约束准确计数 0→1 与 1→0 跳变。

## 5. 对旧问题清单的关键纠偏

1. **SAC bootstrap 并未缺失。** `sac_fyh_IO.py:349` 明确使用 `r + γ V(next) (1-done)`；逐步即时奖励不等于“没有 bootstrap”。真正的问题是 terminal 状态一致性、奖励定义和 Q 分解。
2. **频槽连续约束已实现。** 动作是起始槽加连续长度，环境用连续 `range(start,start+slots)`，越界/占用时缩短尾部仍保持连续。问题是执行动作被静默改写，以及未保证请求长度全部可行，而不是“不连续”。
3. **单波束功率有隐式 50 W 上限。** 5–50 W 的动作映射是硬边界；需改成显式配置/断言。总功率约束也存在，但目前是越界后终止且无奖励惩罚，语义不等于论文的始终可行约束。
4. **按需求排序不是自动泄漏。** 若需求在决策时可观测，它是调度规则；风险是顺序偏置与评估不充分。
5. **早停不会必然“高估总满意度”。** `calculate_total_satisfaction` 对已处理前缀求和而非平均；少处理波束通常让总和变小，不能直接称高估。真正会抬高指标的是零需求填充加一；若再把前缀总和除以前缀长度，则才有选择性偏差问题。
6. **“mask<8”属于旧 token 版，不是当前 100-way 主线实际策略。** 当前主线虽保留同名函数，却根本未调用。两者应分别审查。
7. **同极化、不同 group 的干扰不能只凭 group 不同就排除。** 当前 group 映射同时编码频率/极化的物理含义不清；是否应排除必须回到频谱重叠与极化定义，不能依据旧清单一句话判断。
8. **干扰筛选实际同时排除异极化和“同组同起点”。** 后者在当前调用顺序中主要用于排除刚写入 history 的自身，不能直接断言影响合法主路径；问题是应按 `hist_beam_id` 明确识别自身，见 F18。A10 对条件的转述不准确。
9. **连续槽代码与论文连续性公式要分别审查。** 代码以 start+length 保证执行块连续；论文式 (7c) 却不能严格保证单块，见 F19。

## 6. 两条重构轨道

### 轨道 A：保持语义的工程重构（目标是同输入、同 RNG、同数值结果）

建议先冻结一个 reference run 的配置与少量确定性转移，再做以下拆分：

```text
tsac/
  config.py                 # EnvConfig / ModelConfig / TrainConfig
  data/schema.py            # CoverageScenario, BeamRecord, 单位与范围校验
  data/loader.py            # CSV 读取、列名兼容、有效 mask、hash
  env/state.py              # EnvState / ObservationSpec
  env/action.py             # ProposedAction / ExecutedAction / ActionSpec
  env/link_budget.py        # FSPL、天线增益、MCS、SINR（纯函数）
  env/interference.py       # 冲突图与反向影响更新
  env/environment.py        # reset/step 生命周期
  models/encoder.py         # 14-token encoder
  models/policy.py          # 三头 actor（先保持现语义）
  models/critic.py          # 现有 additive critic（先保持现语义）
  rl/replay.py
  rl/discrete_sac.py
  train.py                  # 只有 main()，无 import-time 训练
  evaluate.py
  artifacts.py              # 原子保存、manifest、load 安全策略
```

核心接口建议：

```python
scenario = load_coverage(path, schema_version="coverage.v1")
obs, info = env.reset(scenario=scenario, seed=seed)
mask = env.action_mask()
proposed = policy.sample(obs, mask, generator)
obs2, reward, terminated, truncated, info = env.step(proposed)
# info 必含 executed_action、reward_terms、constraint_violations、valid_beam_count
```

实施顺序：

1. 建立 schema、单位、有效行和坐标 golden tests；记录当前 CSV hash。
2. 将模块级常量搬入不可变 config；由 observation/action spec 推导维度，删除 200/205/100/8 的散落硬编码。
3. 把链路预算、冲突判定和状态编码抽成无副作用纯函数，用当前代码快照生成 golden vectors。
4. 引入 `ProposedAction`/`ExecutedAction`，先保持现有截短语义，但显式记录差异。
5. 拆训练入口，统一 RNG 对象；保存 run manifest。
6. 修复 checkpoint 文件命名属于工程正确性修复，但会改变产物文件集合，不改变训练数值；单独提交。

轨道 A 验收标准：固定输入和固定 RNG 下，前 N 步 observation、executed action、reward 分项、SINR、beam_bps、replay 样本逐项一致（浮点容差明确）；参数数量和初始 state_dict key 一致；4000 回合训练不是验收前提。

### 轨道 B：改变结果的科学修复（每项独立消融）

按以下顺序，每次只改变一类语义并重新建立基线：

1. **有效波束 mask 与指标**：移除零需求 padding 的奖励/满意度贡献，所有指标报告总和与有效波束均值。
2. **可行功率动作**：根据剩余预算屏蔽功率档，区分 `terminated`（完成）与 `truncated`（外部截断）；任何执行动作满足 `sum P≤Ptot`、`Pmin≤Pb≤Pmax`。
3. **联合频率可行性**：对 `(start,length)` 生成二维 mask，禁止跨界和穿过占用；决定是否允许空间复用后再定义占用约束。
4. **统一功率语义**：明确动作是波束总功率还是每槽功率，按 F17 修正预算、信号和干扰；这是单独实验版本，不能与其他环境修复捆绑。
5. **明确干扰记录身份**：用 `hist_beam_id == current_beam_id` 排除自身，再独立判断极化和频谱重叠；用合法路径及非法/手工动作覆盖边界。
6. **修复温度更新**：统一正熵或 log-prob 约定；验证 α 在目标两侧梯度方向相反。
7. **修复 clipped double-Q 聚合顺序**，再比较 additive critic 与 joint/conditional critic。
8. **自回归 actor**：`p(start|s) p(length|s,start) p(power|s,start,length)`，自然表达动作耦合；与参数量匹配的独立三头版本对照。
9. **顺序鲁棒性**：训练时随机顺序或使用显式 priority；测试报告降序、随机多序和原始顺序的均值/方差。
10. **奖励重新标定**：优先用可解释的容量匹配、SGM、功耗和约束违约；直接 SINR 项是否加入由研究目标决定，所有权重做敏感性分析。

轨道 B 的验收不是“曲线看起来更好”，而是：约束违约率为 0（硬约束方案）；有效波束满意度、SGM、吞吐、总功率、频谱利用率、平均/5%分位 SINR 均有明确定义；至少 10 个固定测试场景、多个种子、置信区间；与原实现及贪心/随机/等功率基线在同一场景同一预算下比较。

## 7. 必要的测试与验收矩阵

### 环境单元/性质测试

- `reset` 后维度、dtype、范围、剩余功率均正确；输入 CSV 坐标通过中国范围 golden case。
- `step` 不修改传入 action；提出动作和执行动作可追踪。
- 连续块始终满足边界和占用规则；零可行槽有明确 no-op/拒绝语义。
- 每一步和 terminal 步的 `freq_pool`、history、beam_bps、total_power 一致。
- 增加一个干扰源后，受影响槽的 SINR 不上升；无重叠槽不受影响。
- 当前 history 记录只按 beam ID 排除自身；构造非法/手工输入时，同组同起点的其他 beam 不得被错误当作自身。
- 功率守恒测试同时核对动作预算、每槽发射功率之和及接收功率；槽数变化不能凭空增加发射能量。
- 有效 mask 下 padding 不改变 reward、episode 长度或任何均值指标。
- 硬功率约束在随机动作序列下始终成立。

### SAC 算法测试

- terminal transition 的 target 等于 reward；非 terminal 有 bootstrap（纠正旧误判）。
- α 在 `H>Htarget` 与 `H<Htarget` 时更新方向相反。
- joint Q 的双 critic min 顺序与定义一致。
- mask 后非法动作概率严格为 0；全非法状态有显式处理，不回退为全合法。
- save/load round-trip 覆盖 actor、两个 critic、两个 target、三个 optimizer、log_alpha、step、RNG state；文件名不覆盖。

### 端到端验收

- 小型 4 波束/8 槽确定性场景可手算，检查速率、冲突、功率和奖励。
- 固定 10 份 coverage 文件作为 immutable test split；训练增强只用于 train split。
- 报告环境 steps、wall-clock、GPU 型号/小时及达到 baseline 95% 的样本数。
- 所有图和表可由一个 manifest 驱动的 evaluate 命令重建，不读取训练日志中的最后一行冒充终局评估。

## 8. 复现检查清单

每次运行至少保存：

- `run_id`（UUID，而非仅分钟时间）、UTC 时间、源码 commit 或源码 tarball SHA-256；当前目录应先纳入版本控制。
- Python、PyTorch、CUDA/cuDNN、NumPy、pandas 版本，OS、CPU/GPU 型号。
- 完整配置：beam/group/slot 数、槽带宽、功率档、Ptot/Pmax、γ/τ/学习率/target entropy、网络层数/head/d_model。
- Python/NumPy/PyTorch CPU/CUDA seed 和 RNG state；明确 deterministic 算法策略。
- train/validation/test 文件列表及每个 CSV hash；增强参数和每回合实际抽到的 scenario ID。
- observation/action/reward schema 版本，坐标/速率/功率/带宽单位。
- 每回合有效波束数、terminated/truncated 原因、约束违约、reward 分项。
- 原子 checkpoint：模型、optimizer、log_alpha、replay（若需续训）、global step、manifest；加载时优先安全的 weights-only 格式并验证 hash。
- 评估使用确定性/随机策略的明确规则，至少多个 seed 和置信区间；训练集与测试集严格分开。

## 9. 建议的落地优先级

第一批只做轨道 A：锁定当前主线、建立 schema/manifest、拆入口和纯函数、统一随机源、修复保存命名。第二批逐项修复 F1/F2/F3/F5/F6，每项用固定测试集做独立 ablation。第三批再研究自回归联合策略与 joint critic，避免同时改环境、奖励和模型后无法归因。论文结果在完成 F1（padding 指标）、F2（功率约束）和 F3（α）复核前，不宜继续把现有满意度、功率可行性和最大熵探索作为已验证结论。
