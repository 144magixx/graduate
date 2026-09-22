# 第二阶段 T-SAC 重构执行规范（面向重构 Agent）

版本：设计 v3，2026-09-20。目标代码来源：`D:/graduate/PPO4090/implementations/sac_fyh_io/`。

本文件是后续实施任务书。本次只编写方案，没有实施源码重构、训练、性能测试或 checkpoint 加载。研究目标与取舍见 [面向研究者的方案](<D:/graduate/第二阶段T-SAC重构方案/第二阶段重构方案（面向研究者）.md>)；本文件给出可直接落实的契约、公式、迁移顺序和验收条件。

## 1. 执行任务与边界

本轮用户明确要求：**当前阶段只更新方案，不改代码**。已确认加入日志重构、波束分布/训练曲线/频谱分配/单步回放/实验对比五模块，同时支持实时与训练后查看；训练数据需尽可能覆盖第一阶段输出场景。第18–22节为新增执行规范，须与前述数据、环境、算法要求共同实现。本轮不创建前端、数据库、场景数据或训练任务。

以当前 `sac_fyh_io` 为事实来源，构建一个可复现、物理口径一致、可变有效波束数量、满足硬约束的离散 T-SAC。最终交付应包含新实现、兼容基线、测试、独立评估、实验清单及第三阶段数据接口。

“第二阶段”是覆盖之后的频谱/带宽/离散功率分配阶段。单景开始时覆盖、几何位置、需求已知；策略逐束构造分配。首版不学习下一束、不学习 group、不加入连续功率求解器或真实时间业务队列。

执行前必须按顺序阅读：

1. [AGENTS.md](<D:/graduate/AGENTS.md>)、[项目说明.md](<D:/graduate/PPO4090/项目说明.md>)、[project_paths.py](<D:/graduate/PPO4090/project_paths.py>)。
2. [sac_train_fyh_IO.py](<D:/graduate/PPO4090/implementations/sac_fyh_io/sac_train_fyh_IO.py>)。
3. [Environment_fyh_IO.py](<D:/graduate/PPO4090/implementations/sac_fyh_io/Environment_fyh_IO.py>)。
4. [sac_fyh_IO.py](<D:/graduate/PPO4090/implementations/sac_fyh_io/sac_fyh_IO.py>)。
5. 覆盖 CSV、当前实现的产物命名和已有研究说明；旧说明只作上下文，事实以当前源码和数据为准。

不可导入旧训练入口做验证：它会立即启动长训练。先做 AST 与只读检查，再使用环境/模型模块。不得覆盖历史模型或改写原始 CSV，不将其他 SAC/PPO 的环境混接到本链路。

本目录、后续说明、报告、图表等用户交付物使用中文文件名。技术模块名、schema key 和 run_id 保留稳定技术标识；训练产物的描述性文件名使用中文。

## 2. 基线事实与变更分类

### 2.1 当前文件指纹

执行前重新计算；如不一致，说明源码已经变化，重新核验差异后调整本任务书，不盲用行号。

```text
sac_train_fyh_IO.py
SHA256 4ab10ed147c2f4ebd40195b07c478d2fb7ca1c4f236b13a883c943a43fb220f2
299 行

Environment_fyh_IO.py
SHA256 5b1cd4131463147429a194bd7ec83814b9684f280ec819094b8be6fa6f540307
1179 行

sac_fyh_IO.py
SHA256 58b0be8efa43f5f17e0eb4e75afa9a1dfee92d5249ad9d55527d078346767b74
486 行
```

### 2.2 已确认现状

- 100 个覆盖文件×200 行；正需求且正宽度共 14,130 行，双零行 5,870；每景正需求数最小 52、最大 168，平均 141.3。原始数据没有 200 个有效需求的场景。
- 环境常量：8 个 group、100 个槽、单槽 25 MHz，频率 17.7–20.2 GHz，模板 200 束，预算 6000 W。动作十档 5–50 W。
- 输入：`[geo2, demand1, width1, power_left1, interference100, occupancy100]`，205 维。14-token 由 4 属性 token 与 10×20 维频谱块组成。
- Actor 三头独立；Critic 为三个分支相加；双 Q 先分支取最小；α 使用正实际熵和负目标熵。
- 回放容量 20,000、开始更新阈值大于 2000、batch 256、4000 回合、actor_lr=1e-5、critic_lr=1e-4、alpha_lr=1e-4、τ=0.005、γ=0.99。新实验不要把回合数与有效环境步数混同。
- 当前 critic-2 独立文件名已经修正，保留回归测试即可。当前有 bootstrap、连续区间与单束功率档上限，不能把它们写成缺失。

### 2.3 区分三类提交/实验

**工程整理**：模块拆分、路径、配置、入口 main、run manifest、日志与保存。兼容模式固定输入和随机流后应数值等价。

**科学/算法正确性修复**：有效需求统计、功率守恒、干扰公式、终止账本、硬合法动作、温度符号、完整动作双 Q、奖励目标。必然可能改变结果，不要求与旧数字等价。

**待验证模型改进**：新状态信息、自回归策略、联合动作 Critic、集合/逐槽编码、规模课程。必须在相同可信环境下消融。

禁止把三类改动一次合并后宣称“Transformer 提升了性能”。可以分批累积实现，但要保存能单独运行的配置与对照。

## 3. 目标代码与产物布局

默认新增同源实现 `PPO4090/implementations/sac_fyh_io_v2/`，保留原实现只读作为历史参考，避免直接改坏旧 checkpoint 所依赖的代码。后续如决定原地替换，先建立可运行的旧版本快照。

```text
PPO4090/
  project_paths.py                         # 唯一根路径定义
  implementations/sac_fyh_io/             # 原链路
  implementations/sac_fyh_io_v2/
    __init__.py
    config.py                             # 数据/物理/环境/模型/训练配置
    data/
      schema.py
      loader.py
      split.py
      augment.py
    env/
      state.py                            # 场景、分配账本、观察
      action.py                           # 动作/候选索引/合法性
      physics.py                          # 纯函数参考计算与可选缓存
      reward.py
      metrics.py
      environment.py                      # reset/step
    models/
      encoders.py                         # legacy14、摘要、集合/频谱
      policy.py                           # 独立评分基线/条件策略
      critic.py                           # 可加基线/联合标量Q
    rl/
      replay.py
      discrete_sac.py
    telemetry/
      schema.py                           # 事件/指标/轨迹版本
      recorder.py                         # 有界队列、独立writer
      store.py                            # SQLite事务/查询
      traces.py                           # 稀疏增量与关键帧
      import_legacy.py                     # 旧产物只读导入
      export_run.py                       # 一致性运行包
    dashboard/
      app.py                              # 独立FastAPI查看服务
      queries.py
      schemas.py
    artifacts.py
    train.py
    evaluate.py
    export.py
  tests/sac_fyh_io_v2/
  frontend/sac_fyh_io_v2/                 # React/TypeScript前端工程
    src/modules/                          # 五模块与公用日志侧栏
    src/api/
    src/components/
    public/                               # 离线底图等静态资源
  outputs/sac_fyh_io_v2/<run_id>/
    运行清单.json
    配置快照.json
    数据划分.json
    运行记录.sqlite                       # 事件/指标/轨迹索引主来源
    诊断日志.jsonl                        # 轮转诊断镜像/降级记录
    场景快照/                             # 实际增强后的输入
    轨迹快照/                             # 关键帧和可选详细数组
    导出/                                 # CSV、图表和离线运行包
    检查点.pt
    第二阶段分配结果.npz
    评估报告.md
```

代码文件名为建议技术标识，按职责合并小模块可以接受，不要求为了目录树创建空壳。所有数据根路径和输出根路径从 `project_paths.py` 取得；run 子目录在实现输出目录下，不散落裸相对路径。

新增链路时同步更新 `PPO4090/项目说明.md` 和根 `AGENTS.md` 的路由说明。原路由仍保留。若发生移动或重命名，完成全项目 AST 语法检查与非训练模块导入检查。

以下为实施后拟提供的命令，不是当前已经存在的功能：

```powershell
# 工作目录 D:\graduate\PPO4090
python -m implementations.sac_fyh_io_v2.train --config <配置路径> --mode smoke
python -m implementations.sac_fyh_io_v2.train --config <配置路径>
python -m implementations.sac_fyh_io_v2.evaluate --manifest <运行清单路径> --split test
python -m implementations.sac_fyh_io_v2.export --manifest <运行清单路径>
python -m implementations.sac_fyh_io_v2.dashboard.app --run-root <运行根目录>
python -m implementations.sac_fyh_io_v2.telemetry.export_run --run-dir <运行目录>
```

## 4. 首版默认配置及科学假设

以下是推荐实施起点，尚未通过训练调优：

```yaml
schema_version: coverage.v2
source_schema: legacy_lon_in_lat
rate_unit: Mbps
traffic_scale: 0.25
beamwidth_kind: ground_angular_diameter_deg
group_profile: shared_grid_parity_isolation_v1
group_assignment: canonical_demand_rank_mod8
service_order: demand_desc
num_groups: 8
num_slots: 100
slot_bandwidth_hz: 25000000
frequency_start_hz: 17700000000
power_budget_w: 6000
power_levels_w: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
power_semantics: beam_total_uniform_slots
max_block_length: 10
rate_model: shannon_with_sinr_margin
sinr_margin_db: 5
reward_version: delta_mean_satisfaction_v1
reward_scale: 100
gamma: 1.0
target_entropy_ratio: 0.5
d_model: 128
attention_heads: 4
encoder_layers: 2
dropout: 0.0
candidate_chunk_size: 512
```

以上 `traffic_scale`、宽度含义、group 物理含义和损耗位置继承/明确化当前行为，必须在 manifest 的 `assumptions` 中记录“未获上游/载荷来源确认”。它们支持工程实施和具名实验，不构成已验证卫星物理模型。

默认总预算固定 6000 W，不随有效需求数变化。小场景测试显式覆盖其他预算。100 束/3000 W 作为独立 profile；不要凭当前数据去反推历史论文配置。

配置应区分 `physics_version、observation_version、action_version、reward_version、model_version、metric_version`。不同环境语义不得混用 replay；checkpoint 加载验证版本兼容性。

## 5. 数据契约、mask 和顺序

### 5.1 Scenario 最小字段

统一记号：`B_entity` 为真实实体数（可以含零需求实体），`N_demand` 为正需求数，`N` 为频槽数。下面张量中的 `B` 表示实体轴容量，batch可再padding；`D` 为该景的 `N_demand`。本文170/200/220束研究默认指正需求数，同时记录真实实体数。旧CSV的200是模板行数，三者不能混用。

```python
CoverageScenario:
    scenario_id: str
    source_hash: str
    source_schema: str
    beam_id: int64[B]                 # 原文件稳定行ID；不是当前处理序号
    latitude_deg: float64[B]
    longitude_deg: float64[B]
    demand_bps: float64[B]
    ground_diameter_deg: float64[B]
    tx_gain_peak_dbi: float64[B]
    rx_gain_peak_dbi: float64[B]
    noise_temperature_k: float64[B]
    group_id: int64[B]
    polarization_id: int64[B]
    entity_mask: bool[B]
    demand_mask: bool[B]
    service_order: int64[D]           # 恰好遍历demand_mask为真的ID
    metadata: dict                   # 生成族、增强、单位、映射版本
```

新版第一阶段数据另提供`root_scene_id、coverage_candidate_id、generator_version、candidate_seed、split_group_id、generator_family_id、dataset_version`；其中scenario_id指向coverage_candidate_id，episode_id由训练采样时另生成。split_group_id表示共享母场景/业务布局的派生谱系，严格隔离；generator_family_id表示热点等生成类型，范围内训练/测试可共享，OOD才整类留出。旧source_family_id若存在只作为split_group_id兼容别名，不混用两种语义。具备源业务时保留需求ID、归属/分摊和未覆盖需求，见第22节；未知谱系不从文件名推造。

物理层使用 float64；送入模型前转换 float32。张量 batch 的 padding mask 与业务 mask 单独保存。

### 5.2 旧数据映射

- `source_schema=legacy_lon_in_lat`：`latitude_deg=原lon`，`longitude_deg=原lat`。标准新 schema 按正确名称读，禁止再次交换。不要仅凭数值范围自动猜 schema。
- `demand_bps=raw_rate×1e6×traffic_scale`，把单位换算与业务比例分开。
- 当前原始 `rate==0 && beamwidth==0` 作为旧模板 padding，先识别，再做增强。原始文件保持不变。
- 新 schema 允许 `entity_mask=True、demand=0` 的真实闲置波束，其 `demand_mask=False`；不要把真实无业务实体与不存在的 padding 合并。
- 正需求但无合法宽度、负需求、非有限数值、重复业务 ID、坐标超物理范围均报错；宽度上限须来自 schema/profile。现有有效值达到 2.8，不直接套 2.0 上限。
- 不复制场景行，不截断正需求。模型批量按当前 batch 的最大 B padding；若配置容量上限不足，给清晰错误，不能偷偷丢数据。

### 5.3 各 mask 的职责

`entity_mask` 标识真实实体；`demand_mask` 标识评价分母中的正需求；`pending_mask` 标识尚未决策；`allocated_mask` 标识已经分配正资源；`skipped_mask` 标识已处理但未分配；`batch_padding_mask` 只用于 attention/batch。

满足：`allocated_mask ∩ skipped_mask = ∅`；每个已处理正需求属于二者之一；所有正需求都参与最终满足度分母。不得用 `allocated_mask.sum()` 作平均满足度分母。

### 5.4 group 与服务顺序解耦

旧实现先按增强后的需求降序，再按处理序号 `%8` 分组，排序变化会同时改变 group，不能把这种实验叫纯顺序消融。

新兼容定义：每个原始场景先按未增强的需求降序、beam_id 作稳定 tie-break，得到 canonical rank，将正需求 canonical rank `%8` 固定到 beam_id；随后增强、改变服务顺序都不改变映射。该映射在原始需求无并列歧义时尽量接近旧基线，但需求噪声、并列处理和删除 padding 仍需记录为语义变更。

若源数据提供权威硬件 group，则直接使用源映射并更换 profile。改变为 raw beam_id `%8`、学习 group 或其他分组均为独立变体。

服务顺序默认需求降序；随机顺序、原始顺序作为独立实验。完整 service_order 或每束顺序 rank 进入观察，不能将影响转移的未观察排列藏在环境内部。

### 5.5 增强、拆分与随机源

仅增强训练集中的真实正需求行；先识别 mask，后增强。需求统一缩放与噪声可沿旧 normal/hard 方案建立兼容版本；正需求必须维持有效，非法样本重采样或按具名规则处理。宽度增强默认关闭，直到明确几何含义；如果启用，不用统一 clip 到 2.0 掩盖变化，也不能把 padding 变为有效。

上述直接修改波束聚合rate的增强仅用于legacy聚合CSV兼容实验。有root/assignment的新主数据必须在原始业务d_k层增强，同步重算D_b、未覆盖量和共同分母；同一增强root的多个候选共享完全相同的业务实现并记录augmented_root_id。改变几何/宽度/归属时通过第一阶段重新生成或验证候选，不能保留过时assignment。此规则优先于旧normal/hard逐波束独立噪声方案。

按场景来源族分组再拆分约 70/15/15；若存在已有可靠冻结清单则沿用。先去完全重复；来源谱系缺失时报告泛化范围限制，不能宣称场景族独立。标准化参数只在训练集拟合，验证/测试冻结。随机性分为场景选择、增强、顺序、探索、回放采样等局部 RNG，移除 `np.random.seed(None)`。

按第22节在root业务场景层面拆分；同root下第一阶段不同初始化、波束数或宽度生成的所有候选都留在同一split。数据扩展不止覆盖文件数量，须满足支持域、组合覆盖和采样到达率要求。

## 6. 完整状态与观察接口

### 6.1 环境状态

内部状态由以下信息构成：固定 Scenario 与物理配置；每束分配 `(status,start,length,power_total_w)`；固定服务顺序与 cursor；完整占用账本。其他速率、SINR、干扰、剩余功率均从该状态确定或作为可校验缓存。

只有一个 cursor 管理当前决策，不再让 Env 与 CurrentBeamState 各自推进。history 通过稳定 beam_id 访问，不把 beam_id 当 Python 列表位置。

### 6.2 模型观察

```python
Observation:
    beam_static: float32[B, Fs]
    beam_dynamic: float32[B, Fd]
    group_id: int64[B]
    order_rank: int64[B]
    entity_mask: bool[B]
    demand_mask: bool[B]
    status: int64[B]                  # idle/pending/allocated/skipped
    allocation_start: int64[B]        # 无分配=-1
    allocation_length: int64[B]       # 无分配=0
    allocation_power_w: float32[B]    # 无分配=0
    occupancy: bool[G, N]
    current_slot_features: float32[N, Fslot]
    global_features: float32[Fg]
    current_beam_index: int64         # 终止=-1
    terminal: bool
```

静态特征至少包含坐标、需求、宽度、发射/接收增益、噪声温度、距离或可确定距离的几何；动态特征包含当前速率/满足度及分配状态。全局特征包含预算与余额、有效需求数、cursor/剩余步数、剩余需求总量/均值、每组剩余槽数、总需求。槽特征包含位置、占用、当前束入射干扰、噪声或干噪比。

从完整分配及静态字段可重建其他组占用和双向干扰；`occupancy` 是显式校验和便捷输入，不是唯一历史记录。若模型只接局部/汇总观察，该配置叫近似观察，不声称严格 Markov。全状态版本保留完整数据，神经压缩是否保留决策信息通过消融判断。

标准化由 ObservationSpec 决定，维度由字段推导。功率用明确的 W 比例，需求可用 `log1p(D/D_ref)`，角度使用具名参考尺度；不因超训练范围就静默 clipping。分配缺省值必须伴随 status，防止“槽0”与“未分配”混淆。

增量接入顺序：`obs_legacy205` → `obs_205_plus_summary` → `obs_full_v2`。新版功率与干扰内核下生成的 205 维兼容观察也要有新 physics_version，不假装与原始观察数值一致。

## 7. 动作、合法性与环境生命周期

### 7.1 ActionSpec

```python
Action:
    kind: Literal['SKIP', 'ALLOC']
    start: int        # ALLOC时0..N-1；SKIP时-1
    length: int       # ALLOC时1..Lmax；SKIP时0
    power_index: int  # ALLOC时0..K-1；SKIP时-1
```

环境内通过 power_levels 得到 `power_total_w`；不把 dBW 塞入动作，再在不同位置反复转换。group 从 Scenario 当前束读取，不能从外部动作越权指定。

合法 `ALLOC` 满足：当前束有正需求且 pending；`0≤start`、`start+length≤N`；完整区间均未被同组占用；功率为配置档位并满足 `power≤remaining_budget+eps`。不以是否达到业务需求作为合法性规则，弱链路/部分满足也允许尝试。不同组同极化的复用保留为软性能影响，不误当资源冲突全部屏蔽。

`SKIP` 在每个非终止决策状态合法。无其他动作时只允许 SKIP；合法动作集合永不为空。终止观察不再调用策略，目标值直接为 0。

### 7.2 候选索引与条件 mask

候选字典按稳定 `(start,length,power_index)` 顺序编号，SKIP 单独一个 ID。基准 `N=100、Lmax=10、K=10` 时边界内共有 9550 个 ALLOC，加 SKIP 共 9551。一般数量为 `K×Σ_{l=1}^{min(Lmax,N)}(N-l+1)+1`，不要散落硬编码。

`valid_actions(obs)` 是纯函数，依据观察中占用、预算和配置生成 mask；环境调用相同规则再验证，防止模型与环境各维护一套不同逻辑。

条件 actor 的 gate、start、length、power mask 由完整合法集合向前缀投影：只有存在合法后续的前缀可被选中。无效分支概率严格为 0。不要全 mask 后回退全合法，不要先独立采样再无限重试。

### 7.3 reset/step 契约

```python
obs, info = env.reset(scenario=scenario, seed=seed)
action = policy.sample(obs, action_spec)
next_obs, reward, terminated, truncated, info = env.step(action)
```

`reset` 清空全部分配与缓存，重新计算场景需求统计。当前旧 `beta` 只在初始化设置，重构必须在新场景刷新。无正需求场景 reset 返回 `terminal=True` 和 `empty_demand=True`，主循环不调用 step；业务指标为 null/不适用，不能返回满意度1，评估报告空场景数量。

每步严格顺序：

1. 只读校验动作，非法时抛明确异常；账本、cursor 和 RNG 状态不因非法动作改变。
2. 保存当前全局效用，执行完整 ALLOC 或 SKIP，写入以 beam_id 索引的账本。
3. 更新资源/预算，并通过统一物理内核重算当前及受影响历史束；参考版本全量重算。
4. 计算全局效用、奖励分项和约束校验；包括最后一步。
5. 推进唯一 cursor，所有正需求已处理则 `terminated=True`。
6. 构造下一观察及与账本一致的 info，终止时 current_beam_index=-1、pending=0，保留完整终局资源信息。

预算用尽后继续 SKIP 到回合终点。不产生无惩罚超预算转移，不用提前停止隐藏未服务需求。

外部训练截断用 `truncated=True`，保留实际 final_observation，不能把下一场景 reset 观察当成 s'。任务固有终点关闭 bootstrap；外部截断在最终有效观察上继续 bootstrap。[终止/截断依据](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

`info` 至少包含：scenario_id、beam_id、requested_action、executed_action、reward_terms、final/current metrics、remaining_power_w、valid_action_count、terminated_reason、constraint_violations。正常新版本 requested=executed；两者分开保留用于审计，不允许静默投影。

## 8. 物理内核与独立评价器

### 8.1 唯一功率口径

主 profile 使用每束总功率。对已分配束 i，`L_i=Σ_s x_i,s`：

\[
p_{i,s}=x_{i,s}p_i/L_i,
\qquad\sum_s p_{i,s}=p_i,
\qquad\sum_i p_i\le P_{budget}.
\]

未分配束 `L=0,p=0`，直接返回零，不做除零。同一份槽功率同时用于有用信号和对所有受害接收端的干扰，不能只修正当前速率而保留旧干扰功率。

### 8.2 group、频率、极化

首版兼容物理假设：8 个逻辑调度池共享 `f_s=f0+(s+0.5)Bslot` 网格；`polarization_id=group%2`；异极化理想隔离；同池不可重叠，同极化不同池复用时干扰。没有时域 duty cycle，不能自行解释为4时隙×2极化。

如果未来证明 group 表示独立频带，必须定义组内到绝对频率的映射，再以真实重叠决定干扰，并独立重跑实验。不得把这个变化藏在性能优化提交中。

### 8.3 统一信道与速率公式

定义线性功率增益 `h[j,i,s]`，表示发射波束 j 在接收位置 i 的增益；包含发射方向图、**接收端 i** 的增益与对应路径损耗。i≠j 时不能误取发射束 j 自己接收机的增益。采用相同源宽度/指向与受害位置计算角度，所有角度和距离接口均具名。

\[
S_{i,s}=p_{i,s}h_{i,i,s},\quad
I_{i,s}=\sum_{j\ne i,\,pol_j=pol_i}p_{j,s}h_{j,i,s},\quad
N_{i,s}=k_BT_iB_{slot}.
\]

首版明确把旧 5 dB 当作 **SINR gap/margin**，令 `L=10^(margin_db/10)`：

\[
\mathrm{SINR}^{eff}_{i,s}=S_{i,s}/[L(N_{i,s}+I_{i,s})],\qquad
R_i=\sum_{s:x_{i,s}=1}B_{slot}\log_2(1+\mathrm{SINR}^{eff}_{i,s}).
\]

该语义是显式兼容选择，不等于已确认的馈线损耗或信号衰减。若后来确认为射频损耗，应移动到对应信号/干扰链路并升级 physics_version。

当前历史速率用已经减 5 dB 的 SINR 反推分母再加新干扰，等价于给新旧干扰不同权重。新内核从线性 S/N/I 统一重算，禁止从旧 dB SINR 反推新噪声。也删除角度与公里直接比较的筛选；若需要邻居裁剪，使用同单位条件并对照全量误差。

Shannon 与 MCS 是具名互斥 profile。主版不再调用 MCS 表计算所谓理想功率来混合惩罚。方向图、宽度转换先封装为可测试纯函数并明确其近似，不能把旧公式当经实测校准的真实天线。

### 8.4 实现策略

先提供无增量历史的 `evaluate_allocation(scenario, ledger, physics_config)`，返回每束每槽 SINR、速率、预算与约束。环境调用它产生真值。随后才增加耦合系数缓存、向量化、受影响集合更新；每项优化与参考实现对齐。

“独立评价器”指不依赖训练日志、history_cinr 的递推结果、缓存累计满意度；它从最终方案重新计算。可以共享经测试的物理纯函数，但至少建立手算 oracle、随机全量对照，避免两个入口共同复用同一个错误缓存。

同一最终分配集合、固定 group/几何/功率，改变合法动作的执行顺序，最终物理速率必须相同。策略因顺序不同选择了不同方案是另一件事，不能用于该性质测试。

## 9. 奖励与指标契约

### 9.1 主奖励

`D` 是 reset 时确定的正需求集合，整个 episode 分母固定。未分配和 SKIP 的速率为0：

\[
U_t=|D|^{-1}\sum_{i\in D}\min(R_i(t)/d_i,1),\quad
r_t=100(U_{t+1}-U_t),\quad \gamma=1.
\]

先更新所有受影响束再算 `U_next`，奖励可以为负，不截成 ±1，不额外裁剪负增量。`reward_terms` 至少记录 current_beam_utility_gain、historical_utility_loss、delta_U、reward_scale。首版无其他 reward 项；全量验证 `sum(reward)=100(U_terminal-U_reset)`。

有限回合每步推进，允许 γ=1；cursor/剩余步骤必须可观察。若另测 γ=.99，不再宣称原始增量目标等价于终局 U。SAC 带熵正则的训练目标也需与环境原始回报分开记录。

### 9.2 可选奖励版本

能耗变体：`r=100ΔU−ΔP/Pbudget`。SGM 变体：先定义全场景终局 SGM 效用，再取同样全局增量；不要只加当前束 SGM 而忽略对历史束的破坏。

主报告不能把旧 SGM、满足度比例、平均吞吐统称“满意度”。对于 SGM 的 β：历史审计模式显式复现旧初始化来源；新指标按每景正需求均值重算并命名为新 metric_version，不能与旧 200 行分母的 SGM 直接合并。

### 9.3 最终指标定义

- `mean_satisfaction=mean_{i∈D} min(R_i/d_i,1)`，主指标。
- `fully_satisfied_fraction=mean(R_i≥d_i×(1−tol))`，容差写入 metric config。
- `served_fraction=分配正资源的正需求数/|D|`；`skip_fraction` 同理。另给零速率比例，分配资源不等于成功服务。
- `delivered_bps=Σ min(R_i,d_i)`，`raw_throughput_bps=Σ R_i`，`unmet_bps=Σ max(d_i−R_i,0)`。
- `power_used_w=Σp_i`，预算/单束违约次数与幅度，槽连续/越界/同池冲突违约次数。
- `pool_occupancy=Σ_{g,s}occupied[g,s]/(G×N)`；`physical_frequency_coverage=Σ_s any_g occupied[g,s]/N`。二者不互换。
- SINR 在已分配的有效束槽对上统计，给样本数、平均值、5%分位和 served_fraction；无样本返回 null。dB 均值与线性均值转 dB 分别命名，默认报告逐槽 dB 分布统计。
- 公平性如 Jain 需声明作用于 clipped satisfaction 还是吞吐；全零样本返回不适用，不把它当满分。
- 训练环境步数、update 数、墙钟、峰值显存、整场景推理 P50/P95 和硬件版本。

场景间主统计默认先算每景均值，再对场景等权；另给按需求数加权微平均。bootstrap 区间按场景/训练 seed 的层级抽样，策略差值使用同场景配对。

## 10. 模型实现与数学更新

### 10.1 编码器与观察消融

保留 `legacy14` 编码器：4 属性 token+10频谱块，不把它误写成10-way频率策略。与它比较逐槽100-token时，其他输入信息、动作、奖励、优化器与预算相同；参数量不完全相同则实报并加近似匹配版本。

完整主编码器：

1. 对全部实体使用共享 MLP+集合注意力，实体 padding 不参与 key/value 和池化。无任意 beam_id 位置嵌入；group 为类别嵌入，service_order rank 是任务特征。
2. 当前组 N 个槽使用逐槽特征+位置编码+浅层 Transformer。频率位置不能去掉。
3. 当前束向量和全局向量构成 query，经交叉注意力读取波束和频谱分支，融合为 z(s)。全局保留需求总量与波束数，不能只用 mean pooling。
4. 初版 actor、critic1、critic2 各自编码，目标网络为完整 critic 的副本；不要先做 actor/critic encoder 梯度共享优化。

参考 `d_model=128、4 heads、2层、dropout=0`，明确 `batch_first`、训练/评估模式、padding mask 方向。模型对实体张量置换应等变/汇聚不变，current index、group、order rank 与记录一起置换。[集合模型依据](https://proceedings.mlr.press/v97/lee19d.html)

### 10.2 Actor

gate 输出 `SKIP/ALLOC`；分配分支依次输出：

\[
\pi(ALLOC,f,l,p|s)
=\pi(ALLOC|s)\pi(f|s)\pi(l|s,f)\pi(p|s,f,l).
\]

以 embedding 或候选区间特征条件化后两个头。联合 log-prob 是沿路径 log-prob 之和，SKIP 仅取 gate 的 log-prob。概率总和为1，mask后合法动作期望才是学习所用分布。[mask 研究依据](https://arxiv.org/abs/2006.14171)

确定性评估使用**完整合法动作联合概率 argmax**，并固定候选 ID tie-break；不能把逐层 argmax 误称全局最高概率动作。随机评估另列种子。环境采样可以逐层条件采样，不必枚举所有动作。

### 10.3 联合 Critic

每个 `Q_i(s,a)` 对完整动作输出一个标量：将独立状态编码 `z_i(s)`、kind/start/length/power embedding 和可选区间特征送入 MLP。区间特征只能从当前状态推导，例如入射干扰统计；不能偷偷调用未来真实奖励作为模型输入。

SKIP 使用独立 action embedding，不访问 start=-1 对应的最后一个槽。先形成完整 `Q1(s,a)、Q2(s,a)` 再取 min。

可加 Critic 只作消融：ALLOC 的 `Q_i=Qif+Qil+Qip`，SKIP 用独立标量 head `Q_i_skip(s)`，不得以索引-1代入分支。也必须**先相加后 min**，不能保留旧分支 min 错误。其带 mask 的期望仍在完整合法集合上计算。

### 10.4 精确离散 SAC 主方案

定义合法集合 A(s)、`Q_min=min(Q1,Q2)`，`alpha=exp(log_alpha)`：

\[
V_{target}(s')=\sum_{a'\in A(s')}\pi(a'|s')
[\min_i\bar Q_i(s',a')-\alpha\log\pi(a'|s')],
\]

\[
y=r+\gamma(1-terminated)V_{target}(s'),\qquad
L_{Qi}=E[(Q_i(s,a)-y)^2],
\]

\[
L_\pi=E_s\sum_{a\in A(s)}\pi(a|s)
[\operatorname{stopgrad}(\alpha)\log\pi(a|s)
-\operatorname{stopgrad}(Q_{min}(s,a))].
\]

目标全部 `no_grad`；真正 terminal 直接令 y=r，不必给 terminal 人造可行动作。actor 更新只回传 actor 参数；critic 参数冻结或 Q 值 detach；alpha 单独更新。目标网络初始与在线相等，每次按 τ=.005 软更新完整参数。[离散 SAC 原论文](https://arxiv.org/abs/1910.07207)

精确求和保留概率对 actor 的梯度；**禁止对 Categorical.sample 得到的离散整数直接套连续 SAC 重参数化 loss**。那样 `Q(s,a_sample)` 无法把动作收益的梯度传给 actor。

### 10.5 温度

\[
H(s)=-\sum_{a\in A(s)}\pi(a|s)\log\pi(a|s),
\quad H_*(s)=\eta\log |A(s)|,
\]

\[
L_{log\alpha}=E[log\alpha\cdot\operatorname{stopgrad}(H-H_*)].
\]

梯度下降必须满足：H低于目标→α增大，H高于目标→α减小。只有SKIP时 H=H*=0，不参与温度批均值，以免大量强制动作稀释温度更新；整批均为单动作则跳过此次α更新。

η=.5 仅为起点，验证比较 .2/.5/.8 等候选并记录调参预算。初始α和学习率写入配置，不能沿负目标熵。所有 log 使用自然对数，非法项避免 `0×(-inf)`，使用安全的 masked sum；不能为非法动作加 epsilon 概率。

### 10.6 候选计算与显存

Transformer 每个状态编码一次。动作头分块评估最多9551候选；目标Q、actor所用Q均无梯度，critic反向只对回放实际动作。候选特征能用前缀和取得区间统计时避免重复物理求解。

单个 `[256,9551,128]` FP32 张量约1.25 GB（十进制）；多个中间层和梯度图会更大，不能承诺现有 batch256 一定适合4090。

起步用微批32、候选块512，按状态微批累计梯度至有效batch256；候选块循环必须控制计算图生命周期。仅将 forward 分块、却一直保留全部有梯度图，不能保证省显存；需要分块梯度累计、checkpoint/recompute 或受控微批，并与小空间未分块梯度对齐。

候选分块只分计算，不分概率分布：条件softmax覆盖该前缀的全部合法后继；独立评分基线的归一化覆盖全部合法ALLOC。禁止每个候选块独立softmax后拼接。可使用全局log-sum-exp、两遍计算或预先计算条件概率，保持精确联合概率。

一个有效batch内先固定采样transition集合。完成全部微批critic梯度累计后，各critic optimizer只step一次；再使用更新后的critic计算全部微批actor梯度，actor optimizer只step一次；alpha按该有效batch非强制状态的总数加权而非每微批各自均值，alpha optimizer只step一次，最后target软更新一次。普通loss按状态样本数加权，候选期望按概率求和，不能再除以候选块数。必要的分块重算须保持同一次更新内参数不变。

先测候选求和的正确性、峰值显存、更新吞吐，才决定批量。只有确认精确枚举成为瓶颈才做采样近似；它是单独加速实验，必须用正确 score-function 梯度及小动作空间对照，不是首版交付前提。

## 11. 回放、训练与 checkpoint

回放至少保存：不可变 observation、action ID/结构、reward、next_observation、terminated、truncated、scenario_id、各语义版本。mask可存储也可重建，但必须验证两者一致；不可依赖采样当时已经被更新的 Env 对象。

规模变化时明确回放分布。主研究配置先按预登记的正需求规模桶抽样，再在桶内均匀抽已完成episode，最后均匀抽该episode的transition；基础原数据实验也可只设一个桶，得到episode均衡采样。空桶仅在现存非空桶之间重分配，记录实际episode/transition占比及丢弃规则。按完整episode管理约20,000 transition容量，淘汰时淘汰最早整episode，避免只保留长回合尾部。未完成episode暂存，完成后入研究回放池；外部截断片段显式标记其片段属性。

原始均匀transition回放保留作具名对照。两种分布都是训练设计选择，不能声称某一种必然无偏或更优；所有结构消融必须共用同一采样协议，避免长回合自然占比变化被误归因为规模泛化能力。

静态 Scenario 可以按 hash 去重存储，transition 存引用与动态状态；序列化时保证引用完整。当前/下一分配账本必须深复制或使用不可变快照，避免后来 step 修改历史经验。新旧schema不得混池。

新入口使用 `if __name__ == '__main__': main()`，导入不得训练、读全数据、创建输出目录或弹图。绘图在评估阶段生成文件，训练不调用阻塞 `plt.show()` 或 `input()`。

起步沿旧学习率与回放容量建立可比配置，但用 environment steps、updates 和实际有效 batch 作为比较预算。热身阶段从合法完整动作近似/精确均匀采样，说明分层均匀不一定等于完整动作均匀。正式探索用条件策略。

checkpoint 为同一run_id原子写入，至少包含 actor、双critic、双target、三个网络optimizer、alpha optimizer、log_alpha、训练步/更新步、配置、spec版本、数据split/hash、归一化统计、Python/NumPy/torch CPU/CUDA RNG 状态。需精确续训时同时保存 replay、当前环境/采样器状态；若仅回合边界续训，明确检查点只在边界保存。设备确定性限制写入清单。

不要声称只有模型权重的旧 full_model 可以精确续训。新模型结构不同默认从头训练；旧权重通过 legacy adapter 做单独审计。加载仅处理可信本地产物并校验格式/哈希，不为了本次方案反序列化旧模型。

## 12. 实验协议与防止错误归因

### 12.1 环境修复轨迹

保存 E0 原样参考→E1 工程拆分等价→E2 有效数据/生命周期→E3 功率/统一干扰→E4 硬动作与新奖励等版本。每次变更先用相同动作序列查看差异，区分数据、物理和奖励造成的变化。中间版本仅用于定位，不要求都进行4000回合训练。

模型比较统一使用验收后的物理内核、预算、mask、奖励与数据split。最终环境与旧环境的分数不能合并为模型优越性证据。

### 12.2 模型实验组

- B0：合法随机。
- B1：共享新环境的贪心，每步选最大即时全局效用增量；大空间时若用受限候选，报告候选规则与成本。另做等功率频谱搜索，功率档和预算相同。
- B2：旧205/14-token观察编码的合法适配基线，修正温度、终止、Qmin等。对ALLOC令 `score(f,l,p)=logit_f[f]+logit_l[l]+logit_p[p]`，在全部合法ALLOC上做masked softmax，等价于各头概率乘积后重新归一化；不得直接相乘logits。再乘gate的ALLOC概率，SKIP取gate的SKIP概率；无合法ALLOC时gate强制SKIP。联合mask会产生统计依赖，不能仍叫严格独立采样。
- S1：B2加全局摘要，其他不变，测摘要信息收益。
- S2：使用完整波束信息，先用共享MLP+集合池化；与S1比较时如同时改变结构，报告信息与结构的组合收益。若要单独归因新增信息，使用同一个完整模型，把新增字段遮蔽作对照；记录容量及实际输入差别。
- A1/C1：在同一完整观察与同一编码器上，分别比较独立评分/条件Actor、可加/联合Critic，必要时做2×2组合，测动作耦合收益。
- T1：冻结完整观察和动作/Q形式，分别做两个单变量对照：频谱分支的10个十槽块（对应旧编码器总14-token中的频谱部分）与100逐槽token；波束分支的集合池化与集合注意力。不要将两个变化合成一个对照后分别归功。报告参数量/算力。
- R1：默认Δ满足度与功耗变体，单独报告；G1：单规模/混合规模训练，单独报告。

历史贪心城市环境不能直接作为B1；可移植其选动作思路，但最终动作合法性和评价都由新内核负责。不得基于测试集挑η、reward权重或最优checkpoint。

### 12.3 预算、种子与结论

烟雾测试用少量场景/步数，测试逻辑与收敛趋势，不用于论文显著性。正式核心比较至少5个训练seed，固定验证选择规则，统一环境交互步数并同时报告墙钟/GPU成本；结构更重时补算力匹配对照。

测试每个原始场景使用同一预算、group、服务顺序和评价规则；多顺序鲁棒性单列。确定性策略与随机策略结果分开。报告场景配对差值及95%区间，区间方法写进脚本。没有稳定正收益也应交付如实结论，不能以预设提升百分比倒推指标。

### 12.4 可变规模

原始有效需求仅52–168，不将200模板行当200束训练。N_demand=170/200/220先用于合成shape/容量测试；正式泛化需要上游覆盖生成器产生物理一致且独立的新场景。不能复制相同位置/需求凑220，不得把截取/重采样称为天然新覆盖。训练域不止规模一轴，须按第22节覆盖第一阶段候选的空间、业务、几何、重叠和资源压力组合。

三族规模实验分开：固定预算/单束需求分布；固定预算/总需求；预算与总需求随规模缩放。零样本冻结权重、α、标准化及可行动作规则；微调、目标规模重训单列。缺失有效生成器时如实标记该研究实验未完成，不伪造数据；已经可做的接口和原数据实验继续完成。

生成器到位后先冻结 `train_sizes、validation_sizes、test_sizes` 与各规模场景族清单，所有算法共用；未见测试规模不得参与checkpoint选择、η调参或标准化。170只有被训练规模两侧包围且未用于训练时才叫未见插值；如果训练最大正需求数仍是168，170与220都属于外推。规模只是一个轴，来自同一生成母场景的派生版本仍必须遵守族隔离。

## 13. 必须通过的验收测试

所有测试围绕科学语义和危险边界，不写只复述实现的形式测试。

### 13.1 数据与观察

- 两种坐标schema对同一已知地点产生一致内部坐标；标准schema不再二次交换。
- 当前100份文件重现20,000行、14,130需求、5,870双零；新增padding不改变有效业务、奖励或最终指标。
- `N_demand=1/4/52/168/170/200/220`通过接口；另加真实零需求实体验证`B_entity≠N_demand`，大规模合成仅为fixture并标明来源。
- 零需求真实实体、全padding、空文件、正需求零宽、NaN、重复ID按契约处理。
- 同seed得到相同采样/增强/顺序，不被任何构造器全局重播种破坏。
- attention padding不进入汇聚；整体置换记录并同步current index/order rank后，决策分布仅按候选定义保持一致。
- 用两个局部205维观察相同而历史受害业务不同的场景展示观察混叠；完整观察能区分它们。

### 13.2 动作与生命周期

- N=100时只受边界约束的最大候选数为9551；N=8/Lmax=3/K=2时与穷举逐项一致。
- 起点99长度1可行，长度2非法；中间占用洞不能穿过；预算不足5 W只剩SKIP。
- 非法动作拒绝后环境快照不变；正常 requested=executed=replay_action。
- SKIP不占槽、不扣功率、推进一次、该需求仍在分母；最后一个ALLOC完整写账本。
- 终止后step报错；外部truncated保存的是原场景final observation。
- 随机合法动作序列中，无越界、同池冲突、预算负数、重复处理或漏处理。

### 13.3 物理与奖励

- 单束多槽的发射功率和始终等于该束总功率；增加槽数不能凭空增加发射能量。
- 无干扰、单干扰、两极化、同频跨group小例子与手算匹配；干扰源和受害者使用不同增益时可发现端点误用。
- 对其他条件固定的受害链路，增加同极化干扰源功率不能提高其SINR；不重叠槽不受该源影响。
- 固定同一最终分配，调换合法执行顺序后终局SINR/速率/效用相同；专门覆盖5 dB margin问题。
- 参考全量重算与增量缓存一致；float64关键物理量建议 `rtol=1e-8`，近零量使用与单位相称的atol，不能一个大绝对误差掩盖极小接收功率差。
- `Σr=100(U_T−U_0)`，双向干扰可产生负reward；padding不加分；未服务需求计0。
- 预算阈值使用约1e-8 W数值容差，所有业务约束仍严格满足，不以容差放宽一个功率档。

### 13.4 SAC 数学与梯度

- 在2–3个候选的小空间手算概率、log-prob、熵、目标V、actor loss和TD target，和实现一致。
- 检查mask非法动作概率为0、合法概率和为1、无NaN；全不可分配时SKIP概率1。
- conditional联合概率之和为1；确定性完整argmax能区别于逐层贪心的反例。
- terminal y=r；非terminal有bootstrap；truncated若非terminal也有bootstrap。
- 反例Q1分量(10,0)、Q2分量(0,10)时完整min=10，禁止旧分支min之和0。
- 两动作不同Q、相同初始概率，在合适α下actor更新朝高Q动作提升概率；验证完整actor梯度与有限差分/小空间精确参考。
- H高/低于目标时α更新方向相反；单候选不更新α。
- actor更新不改变critic/α；critic更新不改变actor/target；target只被软更新改变。
- 分块/微批loss与梯度和不分块小参考一致，防止分块平均权重错误。
- checkpoint续训在确定性小任务中复现下一次采样与更新；纯评估加载结果一致。

### 13.5 端到端与实测

最小穷举oracle固定为3个正需求、4槽、长度≤2、2档功率，每束最多15个含SKIP动作，完整序列上界3375；穷举合法方案得到真最优效用。另设4束8槽fixture做物理/端到端手算检查，不要求对其所有配置都全量穷举。大规模贪心不是数学上界。短训练不要求每seed必然找到精确最优，但需和随机比较、能在过拟合小fixture中学习明确有收益动作，失败时先定位环境/更新。

端到端完成reset→policy→step→replay→update→save/load→evaluate→export，记录实际运行命令、版本和结果。长训练不是前述正确性测试的替代。

## 14. 实施工单、依赖与完成定义

### W0：冻结现状（所有后续依赖）

保存源码/配置/数据hash、当前数据统计、模块依赖图和小型动作序列；保留旧链路。输出中文《基线冻结记录.md》。只读源码即可完成，不加载来源不明checkpoint。

### W1：工程可运行骨架（依赖W0）

同步执行L0/L1：冻结事件、指标、轨迹和API契约，建立异步写入与检索；建立第22节场景覆盖规格与root场景族划分。

建立配置、数据schema、入口main、输出run_id与保存；提取物理纯函数的旧兼容参考。先检查无训练副作用与路径。更新项目文档路由。完成AST与非训练导入检查。

### W2：正确环境（依赖W1）

每步产出全部基本回放信息；环境/数据契约冻结后，前端fixture、查询服务和覆盖审计可并行开发，真实页面必须接入验收后的环境。

按数据有效性→完整账本→功率守恒→统一干扰→可行动作/SKIP→新reward顺序落地，保存各步差异。运行13.1–13.3；全部通过才允许主环境用于算法结论。可同时由另一agent开发独立评价器，但共用接口先冻结。

### W3：数学正确的离散SAC（依赖W2）

同步记录优化诊断并接通实时曲线；短训练测量日志、前端同时开启的吞吐和存储成本。

先用小状态编码器/旧14-token适配器实现精确联合期望、正确温度/目标/双Q、replay与恢复；用短训练测显存和吞吐。运行13.4–13.5。之后接入条件Actor与联合Critic的对照配置。

### W4：状态与编码器研究（依赖W3）

按摘要→完整波束表→集合/频谱注意力增加信息和结构，落实12.2消融。不得同时改物理、需求比例、group或reward来掩盖结果。

### W5：正式评估与第三阶段交接（依赖W4）

交付五模块实时/历史联动、一致性运行包和实验对比；执行第22节场景覆盖验收，不以已有100份CSV替代广覆盖数据建设。

完成多seed与场景配对评估、原始数据可支持的规模实验、导出/重载复算。新的物理一致规模数据不足时，把对应研究任务标为待数据，其他交付继续完成。整理可复现命令、实际结果与局限。

每个工单交付说明包括：改了什么、影响何种语义、运行了哪些必要检查、实际结果、尚未解决事项。工程正确性完成与研究性能提升分开判断；不因新模型分数没有变高就伪造成功，也不因模型尚未长训而省略已可完成的正确性检查。

## 15. 第三阶段输出契约

输出用具名schema，包含：scenario_id/hash、beam_id、entity/demand/status mask、固定group/极化、每束连续块X[B,N]、p_total_w[B]、p_slot_w[B,N]、需求、宽度、噪声、频率网格、物理配置/增益来源、最终指标和run_id。另带 `reward_version、metric_version、U_definition、SGM_beta、SGM_beta_rule` 与第三阶段拟优化目标。

数组保存为无需pickle的NPZ，附中文JSON清单；复杂元数据不塞object数组。矩阵行通过稳定beam_id映射，不能只靠当前排序位置。

验收：导出再载入，在同一物理内核下重算与环境终局一致；`Σ_s p_slot=p_total`、零槽则零功率、硬约束全部通过。第三阶段冻结X优化连续每束总功率，默认沿用槽内均分及本物理版本；改变物理或槽功率定义必须形成新的联动实验。

既有第三阶段方案侧重SGM，本方案主目标是clipped mean satisfaction U，两者不是同一目标。第三阶段精修后至少同时回算U与SGM；允许二者存在权衡，但不能仅凭SGM上升宣称U也改善。如要求精修保证U不下降，应在第三阶段显式加入接受规则/约束，不能默认为求解器已保证。

## 16. 不确定信息的处理规则

本方案已经给出能开始工作的默认profile。优先查询项目内已有论文参数、覆盖生成器和用户既有说明；得到新证据后更新具名配置与文档，不反复要求用户为常规实现选择确认。

没有可靠依据时，显式保留 `traffic_scale=.25`、旧格式坐标适配、group共享网格解释、地面角直径宽度、5dB SINR margin 等假设。不能自动声称这就是真实载荷设计。涉及最终论文物理口径、缺失上游数据等确实影响研究结论的问题，记录具体缺口并继续与其无关的测试/实现。

不要为了完成当前第二阶段，顺手替换为城市SAC、增加PPO主线、训练内接SCA、迁移全部历史目录、删除旧产物或重写所有原始数据。后续实施以本方案界定的目标和项目指令为准。

## 17. 当前事实的源码定位

- `Environment_fyh_IO.py:25–29`：200模板与6000W；`:307–342`：随机源/重复/截断/schema/需求与排序；`:464–488`：205维观察。
- `:490–505`：reset；`:593–619`：接收功率；`:621–719`：干扰身份、极化与双向计算；`:721–824`：速率与历史/观察干扰；`:853–868`：SGM。
- `:929–962`：零需求计分；`:965–1006`：裁剪、功率、终止账本；`:1057–1094`：奖励。
- `sac_fyh_IO.py:20–33`：回放；`:59–120`：14-token与独立Actor；`:321–352`：目标/bootstrap；`:361–375`：可加Q；`:428–447`：分支min与温度；`:457–484`：保存。
- `sac_train_fyh_IO.py:101–112`：主要超参数；`:120–130`：全量数据；`:171–201`：增强；`:204–281`：场景抽样、动作映射、回放与训练。

行号只定位编制日版本；真正执行前先核验第2节hash。旧审查报告的历史发现与当前源码不一致时，保留纠偏说明，以新证据为准。

## 18. 日志、指标与轨迹模块规范

### 18.1 当前问题与职责划分

`sac_fyh_IO.py:223–294`在SAC类内创建分钟级命名CSV，以w模式写文件头，多处先保留两位小数，每回合结束才flush。同分钟多运行可能覆盖，异常退出可能丢掉回合内缓冲；优化诊断、统一场景身份、堆栈和恢复谱系缺失。

新算法返回标量诊断，环境返回业务与状态变化，由训练控制器通过Recorder关联。**每run的SQLite数据库和其引用快照为查询主来源**；JSONL为轮转诊断镜像/失败后备，CSV为派生导出，不分别维护多套指标真值。Python标准logging可用于级别、堆栈与队列机制，但需自行定义队列满策略，不能依赖默认Handler行为保证核心记录可靠。[Python日志处理器](https://docs.python.org/3/library/logging.handlers.html)

### 18.2 进程与存储边界

```text
训练/环境/优化器
  → 有界CPU消息队列 → 唯一Recorder写入进程
                       → SQLite事务 + 场景/轨迹快照
                                 ↓ 已提交数据
                       独立FastAPI查询服务
                                 ↓ REST + SSE
                       React五模块 + 共用日志侧栏
```

浏览器或Web服务关闭不改变训练。仅查看历史数据时服务不导入训练入口、不加载Actor，不要求可用GPU。数据库与writer置于同机本地磁盘，WAL模式、单writer批量事务、reader短事务分页；不将活跃WAL数据库放网络共享盘供多机挂载。[SQLite WAL说明](https://www.sqlite.org/wal.html)

建议表为`runs、episodes、events、metrics、transitions、beam_deltas、artifacts`。动态回放常用标量存数据库，大数组使用无object/pickle的NPZ。按episode/step、event_seq、级别/类型、beam_id、metric/native_axis建索引；全文查询有界分页，若用FTS5先验证运行时能力。

快照发布顺序：同卷临时文件→flush并执行平台支持的fsync/FlushFileBuffers→关闭/hash→原子改名并按平台能力同步元数据→DB事务提交引用和事件。确认工件发布后才提交引用，UI只读已提交引用；启动时检测孤儿文件、缺失引用和hash损坏。SQLite与NPZ没有跨文件原子事务，掉电保证受文件同步协议和硬件能力约束，不能将DB提交等同所有工件绝对安全，也不能用新内核静默重算替代损坏原记录。活跃run导出使用数据库一致性backup和同一提交水位的工件清单，不能只复制`.sqlite`遗漏WAL。[SQLite备份接口](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup)

### 18.3 身份、恢复与快照关联

事件公共字段：`schema_version、event_id、run_id、trial_id、attempt_id、event_seq、occurred_at_utc、committed_at_utc、elapsed_seconds、severity、component、event_type、phase、episode_id、scenario_id、step_index、env_step、update_step、beam_id、payload`。不适用字段为null。

- event_id由producer生成，重试去重；event_seq由writer分配，在该run数据库内只增，以它排序，不依赖墙钟。
- episode_id是实际增强后的场景运行实例，scenario_id是原始候选身份；同一CSV重复采样仍生成不同episode_id。
- step_index=0是reset后状态，t表示第t个动作完成后的状态；acted_beam_id与next_beam_id分开。
- update_step只计真实optimizer更新，微批和梯度累计不冒充多次update。耗时采用单调计时累计，各恢复段独立保存。
- **恢复默认创建新run_id**，通过parent_run_id、parent_checkpoint_id、resume_env_step/update_step和稳定trial_id关联；每进程生成attempt_id。checkpoint记录训练状态对应的durable_event_seq。父run水位之后的旧尾部保留为历史分支，主曲线不与新段重复连线，也不计成独立seed。此规则补充第11节保存要求。
- 共享snapshot键为`(run_id,episode_id,step_index,physics_version)`。每次状态响应给出snapshot_id和committed_event_seq，地图、频谱、回放、指标必须引用同一版本，不能各取一次“最新”导致混合半步。

回合中途恢复时child run另写resume anchor：实际Scenario、checkpoint对应step=k的完整UI状态、parent episode/step和checkpoint引用。child新episode实例保留step_index=k作为可独立回放起点，后续delta接k+1；前缀0..k-1从父轨迹取，缺父工件时显示范围不足。仅child离线包也能从k查看，不要求查看服务加载torch checkpoint或重跑环境。入队记录使用不可变CPU副本，不能将持续变化的Env/list/tensor引用交给writer。

### 18.4 记录内容与采样

**每回合**保存实际增强后的Scenario快照/hash、顺序/group、需求/规模/预算、开始结束步、环境回报、U、业务指标、耗时、终止原因与trace完整性。不仅保存原CSV路径和seed。

**每个动作**保存完整动作、合法动作数、forced_skip、预算/U前后、reward_terms；保存当前束及所有速率发生变化历史束的稀疏before/after，包括rate与clipped satisfaction。每个完整step的transition、beam_deltas和该步指标同事务提交。

**优化诊断**包括Actor/双Critic loss、TD error统计、Q/target统计、实际/目标熵、α/alpha_loss、实际batch/非强制样本数、学习率、梯度范数、更新耗时、回放容量及规模桶占比。普通模式每50次update记录窗口count/mean/min/max/last及首末update编号；首个update和异常立即记录，debug可逐update。不能把窗口均值标成单次update原值。

**事件**包括开始/完成/恢复/中断、数据校验失败、非法动作、NaN/Inf、约束违约、异常堆栈、checkpoint、写入失败、队列拥塞和诊断降级。有限性检查仍按必要的每次更新执行，不能因50次采样间隔漏掉首次NaN；无效梯度不进入optimizer更新。JSON非有限数写null并附nonfinite_fields和异常类型，不写非法JSON NaN。

**性能**默认每5秒采集steps/s、updates/s、CPU/内存、GPU显存与可用利用率；不可用字段为null。避免逐标量反复同步GPU。数值保留原精度，仅显示时格式化。

loss来自回放批次，保存必要的样本transition引用，不把当前正在服务的波束伪装为该loss唯一来源。完整候选Q/logits、注意力和梯度数组为可选diagnostic，默认不全量写入。

### 18.5 所有回合的基本回放

主配置`trace_level=basic_all`：全部回合保留真实初始输入、每步动作、全部受影响束的稀疏业务变化、资源和全局指标；每20步及终局记录关键帧，reset为第0帧。指定step由最近关键帧＋已记录delta恢复，不重新采样Actor、不用新环境代替原始结果。动作轨迹、UI回放和模型精确续训是三个不同概念。

逐槽I/SINR、概率等detail按指定场景/回合或异常采集。CPU有界诊断缓冲默认覆盖异常前20步，不保留GPU张量；缓冲不替代全量基本轨迹。未记录detail返回unavailable；若使用匹配物理版本另行复算，标recomputed及配置hash，不混同recorded。

图形降采样不删除基本轨迹。可选detail轮转/归档须保留工件状态与可用范围；运行包记录实际空间和每百万步存储成本，压缩率以实测为准。

### 18.6 持久化与故障

建议flush_interval=1秒、batch_events=256，队列同时限制消息数和CPU字节数。episode结束、checkpoint和正常退出执行flush确认屏障。页面展示最近已提交步骤，不声称RAM中尚未提交的步骤已保存。

队列拥塞先合并心跳/高频诊断并减少可选detail；核心transition、回合指标与错误不静默丢弃。核心持续无法入队时有限背压，超时或writer死亡则明确失败、受控停止，尽力保存有效checkpoint；磁盘满时不能保证后备文件还能落盘，stderr需给出原因。Web断线不触发训练停止。

强杀/掉电可能丢失未提交尾部，只保证已持久化前缀可查，活动episode显示未完成；记录最后水位。正式可追溯配置优先SQLite FULL同步，实际同步设置/SQLite版本写入manifest；单writer负责写事务与数据库checkpoint，短读事务避免WAL无限增长。

### 18.7 旧日志与导出

旧CSV/NPZ只读导入，保留hash、source=legacy_import和真实字段能力。两位小数不能恢复精度，缺失loss/α/场景/历史变化不能补造，旧满意度不自动当新版U，只有旧曲线时不宣称精确逐步回放。

派生导出按粒度分为`训练回合指标.csv、优化更新指标.csv、分配步骤.csv、波束状态变化.csv、评估场景指标.csv`，保留关联键/单位/版本/截止水位。前端使用API查询数据库，不扫描不断增长的CSV。原始文件不改写。

## 19. 前端、API与实时协议

### 19.1 技术与运行边界

推荐React＋TypeScript；ECharts用于曲线/热图，Leaflet用于地图与GeoJSON，FastAPI提供版本化只读API和静态页面。实施时核验并锁定依赖版本。五模块为已确认范围，日志为共用侧栏，数据覆盖摘要属于实验对比视图。

训练CLI与看板CLI独立，默认监听本机。服务从project_paths提供的运行根目录索引manifest，前端传run/artifact ID而非任意文件路径。远端查看通过适当转发访问服务，数据库仍在训练机；首版不提供训练启动、暂停或改参接口。播放器暂停只暂停页面时间轴。

### 19.2 API最小契约

```text
GET /api/v1/runs
GET /api/v1/runs/{id}/manifest
GET /api/v1/runs/{id}/events             # cursor+结构/文本过滤
GET /api/v1/runs/{id}/metrics            # 指标/原生轴/范围/分辨率
GET /api/v1/runs/{id}/episodes           # 可回放范围
GET /api/v1/runs/{id}/episodes/{ep}/state?step=t
GET /api/v1/runs/{id}/episodes/{ep}/transitions?from=t&limit=n
GET /api/v1/runs/{id}/stream             # SSE已提交水位
GET /api/v1/comparisons?...              # 兼容性/配对统计
GET /api/v1/datasets/{id}/coverage       # 第一阶段输出域覆盖摘要
GET /api/v1/artifacts/{id}
```

定义单位、null含义、phase、metric_version、recorded/recomputed来源。分页默认200、最大1000；曲线按视口限点，响应给原始范围、聚合方法和count，放大可取原始点，异常标记单独保留。

`state`返回同一snapshot的波束、资源、指标与detail能力；分开发送大数组时共享snapshot_id。缺失step返回可用范围和明确状态，不拿最近一帧冒充指定步骤。

### 19.3 实时、重连和离线

REST负责历史查询，SSE只发已提交事件/水位的轻量通知；每run一条共享连接，前端默认2Hz合并刷新。SSE id=event_seq，Last-Event-ID/cursor用于补发，客户端按event_id去重；cursor过期返回resync并重新获取快照，不伪装无缺口连续流。[SSE机制](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

查询捕获一致提交水位，再补发其后事件，覆盖“取快照与订阅之间”的竞态。UI分别显示数据更新时间、连接状态、训练心跳和运行终态，不把Web断线等同训练失败。

历史模式固定episode/step，后台更新不抢占；返回实时才恢复跟随。运行包含一致数据库备份、所引用工件、manifest/hash与静态资源。训练进程关闭、外网断开后，通过本地服务仍能浏览已有结果，无须模型/GPU；恢复谱系缺父段时明确标出缺失范围。

## 20. 五模块与共用日志侧栏

### 20.1 公用上下文

页首选择run、phase、episode/scenario、step和beam_id，显示实时/历史模式与记录完整性。共用日志侧栏支持级别/组件/类型/场景/波束/时间与全文筛选、异常堆栈、上下文和snapshot跳转。

原始业务点可定位回合，平滑/聚合点先展开原记录；优化点关联update与回放样本。原始、复算和演示数据分别标记。缺失值显示未记录/不适用，不补0；单位/颜色范围与导出一致。

这里的“原记录”仅指实际持久化内容：普通优化曲线每50次update只有窗口统计，不能展开不存在的逐update值；debug、首个及异常update才有对应原值。API返回raw_available与aggregation/count/range，窗口详情据此展示统计，不给出虚假的逐update下载按钮。

### 20.2 波束分布

显示标准经纬度中心、模型覆盖轮廓、status，支持需求/满足度/总功率着色及group/极化/status筛选。GeoJSON是经度、纬度，Leaflet LatLng是纬度、经度，在适配层转换并测试。[Leaflet文档](https://leafletjs.com/reference.html)

后台根据beamwidth_kind与地球模型生成地理多边形；地面角直径用球面大圆半径构造测地轮廓并标“模型覆盖范围”。不能将经纬度角直接当平面公里/像素；含义未确认时只显示中心和原值。需求点大小与覆盖轮廓图例分开。

隐藏padding，真实零需求可筛选；点击稳定beam_id联动频谱/日志并展示需求、实际速率、满足度、频槽、总功率/槽功率及已有SINR。没有用户归属数据时不制造用户点或覆盖率。

### 20.3 训练曲线

业务曲线含return/U/吞吐/未满足量/功率/SKIP/违约；优化曲线含loss/TD error/熵/目标熵/α/梯度；性能曲线含速度/耗时/显存。轴来自真实env_step、update_step、episode或elapsed_seconds，不用点序号替代。train/validation/test分系列，评估关联checkpoint。

原始值可查，平滑方法/窗口可见，实时平滑只用过去点；窗口降采样保留min/max与异常标记，缺段断线。多seed区间由后台统计，不拿单run波动冒充。[ECharts数据管理](https://echarts.apache.org/handbook/en/concepts/dataset/)

业务曲线原始值与优化窗口统计区分；存储阶段的窗口聚合不等于前端可逆降采样。离线地图随包提供许可可用的本地底图，缺底图时使用坐标网格/纯背景仍能看波束，不依赖在线瓦片或CDN字体脚本。

点击曲线点定位日志/回合，检查主reward下return与100ΔU一致。loss不单调本身不直接判断失败；非有限数和硬约束违约有独立事件。

### 20.4 频谱分配

显示G×N逻辑池，按beam_id或槽功率着色，列可切换槽索引/绝对频率。完整连续块悬停显示起止/长度/group/极化及总功率与p_total/length。选中波束可查看它自己的逐槽干扰/SINR；同频复用束的SINR不同，不能画成一个公共槽SINR。

同group重复占用为错误，跨group同极化复用单列；SKIP无资源块，末束分配可见。池占用率与物理频率覆盖率分开，选中/高亮使用共同snapshot和beam_id。

### 20.5 单步回放

支持前后一步、播放/暂停/倍速、任意step/异常跳转、reset/终局和动作前后对照。最近关键帧+delta恢复状态，后退采用重新应用或经过校验的逆delta。

展示acted_beam与next_beam、动作/合法数/forced_skip、资源与U前后、reward_terms、全部受影响历史束。无合法ALLOC才叫强制SKIP；主动SKIP不推测未知动机。中断显示最后完整step和缺段范围。

不调用策略采样，不干扰训练；detail缺失不影响基本回放，另行物理复算明确标记，不替换历史记录。

### 20.6 实验对比

显示模型/token/Actor/Critic、seed/trial、split/hash、N_demand、预算、group/顺序、成本和physics/reward/metric版本。兼容条件下计算场景配对差值、多seed区间；不兼容时显示差异/分面，不自动生成同条件排名。

不同reward的return不可排名；不同physics需要共同参考环境独立复评，结果另建evaluation记录。区分零样本/微调/重训，展示失败/中断/缺seed，恢复段不重复算独立样本。可对比同场景两套分配，并显示第22节训练域覆盖摘要和缺口。图表/中文报告导出附来源、筛选和聚合口径。

## 21. 日志与前端工单及验收

- L0（依赖W0/W1）：冻结事件/指标/轨迹/API契约、身份和小型真实fixture，审计旧日志字段能力。
- L1（依赖L0）：独立Recorder、单writer、日志检索、故障恢复和一致性运行包。
- V0（依赖L0）：五模块布局、具名演示fixture、公用选择器和日志侧栏；演示不冒充训练结果。
- V1（依赖L1/W2/W3）：真实step/update、实时曲线、波束/频谱/回放联动及SSE。
- V2（依赖V1/W5/数据清单）：实验对比、恢复谱系/多seed、覆盖摘要、离线包和导出。

后续验收至少覆盖：

1. 全精度、中文/多行堆栈、null/非有限数、同分钟多run、重复event去重、按各关联键检索。
2. 关闭Web/浏览器、SSE断线重连、重复通知、快照订阅竞态：训练继续，已提交记录不漏不重。
3. writer退出、队列满、磁盘失败、step事务中断、孤儿快照、活跃DB导出：明确行为和可用范围。
4. 末束、SKIP、空需求、旧束受新束干扰、checkpoint恢复后重复env_step：各视图一致，前后回放一致。
5. 220正需求、经纬度转换、地图缩放保持物理轮廓、padding隐藏、pending的未定义SINR不显示0dB。
6. 原生横轴/phase/聚合/平滑正确，降采样保留异常，loss不错误归因当前束。
7. 不同物理/预算/指标、旧CSV、失败/缺seed/恢复谱系不制造公平排名或不存在的详情。
8. 训练关闭、外网断开时运行包可查看，导出和API/独立评价一致。

建议基准为220正需求、100槽、10万条日志与10万曲线原始记录，记录硬件/磁盘/浏览器。待实测目标：实时端到端P95≤2秒、选中/筛选P95≤200毫秒、常用分页P95≤500毫秒、固定短训练的日志吞吐开销≤5%。未达标先定位GPU同步/序列化/查询/渲染瓶颈，调整可选detail与刷新率，不删基本轨迹伪装通过。

验证包括后端事务/API契约和前端真实浏览器跨模块流程，不仅截图检查。日志开关数值一致性使用独立RNG和确定性fixture，时间戳不参与策略决策。本轮仅补充这些验收要求，不执行实现或训练。

## 22. 广覆盖场景集与第一阶段输出接口

### 22.1 目标、当前缺口与定义

用户要求训练集尽可能覆盖第一阶段波束覆盖模块可能产生的场景。目标是声明并覆盖可支持的部署域，不声称有限数据包含所有可能场景，也不通过将验证/测试全部用于训练实现“全覆盖”。

现有100个CSV只覆盖52–168个正需求，谱系和原始业务归属不完整。它们可作为兼容/种子数据，新增广覆盖主数据必须由第一阶段或其经验证的离线生成器产生。先做只读来源/生成器能力审计，缺失生成器或业务原始数据时列为明确依赖；不得凭空随机坐标凑170/200/220或复制正需求行。

首轮试生成建议声明N_demand=48–256的能力范围，实际范围以第一阶段物理可行域和目标部署规格冻结，超出部分作为另列压力测试。48–256是建设候选而非已验证事实。当前阶段只把审计和生成任务写入方案，不执行生成或修改第一阶段代码。

### 22.2 数据层级与需求守恒

```python
RootScene:
    root_scene_id: str
    split_group_id: str          # 共享母场景/业务布局的派生谱系
    generator_family_id: str     # 业务生成机制/热点类型
    demand_id: int64[K]
    demand_location: float64[K, 2]
    offered_demand_bps: float64[K]
    service_area: geometry_ref
    satellite_geometry: config_ref
    demand_generator_version: str
    seed: int

CoverageCandidate:
    root_scene_id: str
    coverage_candidate_id: str
    coverage_generator_version: str
    parameters_and_seed: dict
    beam_table: CoverageScenario
    assignment: sparse[K, B_entity]
    uncovered_demand_bps: float
    source_quality_metrics: dict
    physics_profile_id: str
    provenance_status: str
```

assignment中A[k,b]≥0、Σ_b A[k,b]≤1：硬归属为0/1，允许显式分摊时记录策略且不得重复计数。所有需求使用同一单位/traffic_scale：D_b=Σ_k A[k,b]d_k，D_uncovered=Σ_k(1−Σ_bA[k,b])d_k，校验Σ_bD_b+D_uncovered=Σ_kd_k。原始业务缩放时两侧一起更新。

beam_id仅在候选内稳定，不能用它直接把不同候选的第7束当同一物理波束。跨候选比较按共同root业务ID/归属和地理关系对齐；同候选不同episode才可直接按beam_id对齐。

每个root保留若干合法的优质/普通/较差覆盖候选，不能只收第一阶段最优解。候选可来自不同初始化、波束数、合法参数和合法搜索中间状态；几何不合法、重复业务、非有限量属于数据错误，资源不足但覆盖合法属于训练难例。

### 22.3 覆盖规格文件

后续新增中文`训练场景覆盖规格.yaml`、`训练场景清单.json`、`训练场景覆盖审计.md`、`场景缺口与补样清单.json`，数据根路径仍在project_paths定义。覆盖规格至少包含：

```yaml
dataset_version: coverage_domain.v1
root_definition: upstream_business_instance
support_domain:                   # 来自部署规格/生成器可行范围
  positive_beam_count: [48, 256]   # 初始建议，试生成后确认
  geometry_profile_ids: []         # 审计后填真实具名profile
  main_power_budget_w: 6000
axes_and_bins: {}                  # 定义、单位、边界、数据来源
feasible_combination_rules: []
mandatory_cells: []                # 单轴/二阶/重点三阶
independent_roots_min:
  train_per_cell: 20
  validation_per_in_domain_cell: 10
  test_per_in_domain_cell: 10
required_cell_coverage: 1.0
split_unit: split_group_id
sampling_protocol_version: domain_sampling.v1
```

这些是验收目标，尚未落实的数据范围不能显示为已通过。可行性排除规则在统计覆盖率前冻结并附物理理由，禁止事后删掉难生成单元来抬高覆盖率。同root多候选/多增强不增加独立root计数；同一root可同时贡献多个已满足单元，但报告相关性。

### 22.4 场景轴与联合生成约束

1. **业务空间形态**：近均匀、单热点、多热点、强弱热点混合/长尾；按聚集度、热点占比、区域中心/边缘分层。
2. **覆盖形态**：N_demand、真实实体数、宽度分布/异质性、波束重叠、边缘比例、漏覆盖量。初始规模桶可为48–80、81–120、121–168、169–220、221–256，冻结时按可行域调整。
3. **需求负荷**：原始总需求、正需求均值/离散度、热点偏移、低/中/高负荷。阈值基于业务规格或训练来源分位确定并冻结，不用测试集拟合；业务缩放参数与traffic_scale分别记录。
4. **资源压力**：ρP=Pbudget/(N_demand·Pmax)、Pbudget/N_demand、逻辑池资源压力、可行覆盖下的业务过载。6000W/50W时N_demand≤120不能触发总功率超限，但仍可能有频谱或业务瓶颈。
5. **几何/物理**：服务区域/边缘位置、可见性、斜距、已确认噪声/增益/方向图profile。物理随机化只在有来源的范围内进行，不凭想象改变卫星或宽度与增益关系。
6. **上游变化**：生成算法/版本、候选初始化、质量档位；固定group后的服务顺序另列，需求增强不能隐式改group。

中心、宽度、数量必须由覆盖生成结果共同确定；变化后重新建立业务归属和D_b。宽度与峰值增益若采用旧固定增益近似，明确profile，不混成真实天线规律。固定预算/总需求、固定预算/单业务分布、预算随规模缩放分别成实验族；不同profile不能在未提供上下文的同一MDP中混合。

现有Observation须增加能区分混合profile的数值上下文，例如实际Pbudget、Pmax/槽宽/总槽数、噪声/几何参数，而不只输入归一化余额；或首版每个物理profile分别训练。同一观测隐藏不同预算/物理转移会再次引入观察混叠。单纯喂入无语义profile ID不能支持未见物理泛化。

首版默认不同动作/物理语义分别训练；同一replay仅允许统一语义版本、统一ActionSpec下已声明的条件化数值参数变化，逐样本保存完整context。功率档集合不能仅由Pmax代表，槽数/频率网格变化还会改变候选ID与位置编码；混合这些ActionSpec需要另行设计映射、网络输出与batch契约，不因加几个全局特征就默认兼容。

### 22.5 组合设计、难例与状态覆盖

覆盖所有声明的单轴档位和可行二阶组合；重点三阶包括“大规模×高负荷×紧预算”“密集重叠×同极化复用×弱链路”“需求长尾×少数宽波束×频谱紧张”。连续量在每格内铺开，避免所有样本只落在档位中心；不盲目穷举全笛卡尔积。

区分**初始场景覆盖**与**决策状态覆盖**。频谱碎片、只剩一个功率档、空闲连续块差一槽、新束使历史束降至需求以下、全无ALLOC导致SKIP，取决于动作轨迹。使用合法随机、贪心和当前策略rollout记录这些状态的到达率；诊断fixture须由可重放合法动作前缀产生，不能拼接不可达状态冒充训练经验。

指定边界：Premaining恰好5W与略低于5W，连续块恰好足够与差一槽，低干扰/严重干扰，极端需求不均衡，所有需求不可同时满足但数据合法。错误schema/覆盖不合法单列负向测试，不作为正常训练分布。

### 22.6 拆分与规模实验

先按split_group_id划分root，再扩候选与增强，默认约70/15/15并兼顾范围内各必覆盖单元。原始用户布局相同的不同候选、微扰、生成版本变体不能跨split。generator_family_id可在范围内训练与独立root测试中共同出现，OOD才整类留出；不能把生成机制类型当成必须全部隔离的派生谱系。已有可靠冻结评估集保留；未知谱系做hash/近重复/已知族审计并标记限制，不悄悄混入新严格测试集。

分三种评估：范围内新root、范围内未见规模/组合、范围外或保留地理/业务生成族。全部训练尺寸/holdout尺寸和root族清单预登记。例：训练锚点48/64/80/96/112/128/144/160/192/224/256，验证可留104/184/240，最终未见规模可留120/170/220；具体数值须先通过生成可行性确认，训练不得含保留尺寸。该示例下170/220为插值，不再叫外推，288/320等只有经上游确认可行才作为额外外推压力测试。

104/184/240与120/170/220是**额外的未见尺寸诊断清单**，不是验证/测试全部尺寸。范围内新root验证/测试还须在与训练相同的规模桶和训练锚点上独立生成，以满足每个范围内必覆盖单元的验证/测试配额；未见尺寸结果另列。

同范围测试的场景类型可以与训练相同，但业务实例独立；有意完全留出的类别叫OOD并单列，不把它计作“训练应当全覆盖却已覆盖”的单元。全部算法共用同一数据和评估清单，测试不用于调参、补样、归一化或checkpoint选择。

### 22.7 训练采样与补样闭环

初始训练先均衡覆盖必需场景单元，再向部署分布过渡，同时保留稀有边界样本。可用的起点方案为：短热身按单元均衡，之后70%部署分布＋20%覆盖不足/稀有单元＋10%合法边界难例；这是待验证的采样配置，不是已知最优比例。必须给单元归属重叠、root/候选权重和epoch定义，避免多候选root天然占据更多训练量。

回放仍按第11节分层采样，新增domain_cell/pressure_tag，使采集与更新的场景分布都可审计；不同模型消融冻结相同协议。优先补训练覆盖缺口和验证弱项，验证反馈只调用训练root生成器，不能把验证root本身加入训练；最终测试不进入补样决策。

每次扩数据升级dataset_version，记录新来源、数量、分布和旧版本关系。当前重构规范禁止不同语义版本混池；同schema下增量数据的续训也应新run并声明是否保留旧replay、是否重置采样权重。保存采样器/RNG状态保证可复现。

### 22.8 第一阶段候选评价

第二阶段内部主指标仍为U；不同第一阶段候选改变N_demand及需求聚合后，U不能单独给第一阶段候选排名。相同root、预算和物理profile下另报：

\[
J_{root}=\frac{\sum_b\min(R_b,D_b)}{\sum_k d_k}.
\]

漏覆盖需求仍在共同分母内而贡献0；需求归属/分摊必须保证不重复计数。没有逐用户调度时J_root是聚合可交付业务代理；如果要报告用户达标率，需明确用户级调度规则并使用原始用户信息。原总需求为0时返回不适用。

入口建议`allocate_candidate(root,candidate,policy,physics_config)`，输出方案、U、J_root、漏覆盖量、资源违约和推理成本；同root候选的seed/预算/顺序规则对齐。只有聚合CSV而缺root业务/归属时返回`end_to_end_metric_available=False`，保留阶段二内部指标，不凭空补原始分母。

主reward仍按第9节100ΔU。若研究目标更重视端到端业务量，可新增`delta_root_delivered_v1=100ΔJ_root`作为独立消融，在相同可信环境/数据下比较后决定，不静默替换主目标。此轮不引入第一阶段反向传播、联合训练或在线优化闭环。

### 22.9 数据验收与前端证据

正式广覆盖训练前交付可重建的覆盖规格、数据清单、审计及缺口表，至少验证：

- 必覆盖可行单元100%有样本；训练每单元≥20独立root，范围内验证/测试对应单元各≥10独立root；初始门槛需试生成成本评估，未达到如实记录，不能以增强凑数或自动降低门槛宣称全覆盖。
- root_scene_id与split_group_id跨split交集为0；generator_family_id是否留出按ID/OOD协议判断；exact hash与近重复审计，阈值/例外具名，已知/未知谱系分开计数。
- 几何、单位、实体mask、宽度/需求正值、group和需求守恒通过；float64守恒可用rtol=1e-8与合理bps绝对容差，记录生成失败率/原因。
- 每单元报告root数/候选数、N_demand与实际负荷/预算范围；训练中报告采集次数、更新抽样次数与难状态到达率，补齐“有数据却没训练到”的缺口。
- 独立评估报告每层U、J_root可用性/数值、违约、拒绝率、低分位与推理时间；全局均值不能掩盖稀有层失败。
- 从一个root生成两个分束数不同的候选，确认总需求守恒、无重叠双计、漏覆盖计0、J_root共同分母和跨候选对齐正确。

实验对比页面查询同一审计产物，显示支持域、已覆盖单元、缺失原因、未知来源与实际采样比例；CSV文件数量不能替代覆盖百分比。coverage_spec、数据生成/采样/指标版本都写入run manifest，供日志和前端追踪。

新增数据工单D0（来源/域规格审计）→D1（第一阶段合法候选生成接口与守恒验证）→D2（族拆分、覆盖审计/补样）→D3（分层采样、范围内/OOD评估与前端摘要）。它们属于后续实施范围；当前只更新方案。
