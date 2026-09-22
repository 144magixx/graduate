# 第一阶段接口与实施规范（面向实施 Agent）

版本：设计 v1，2026-09-20。目标：为当前 Horizon T-SAC 提供合法、可追溯、可评价的第一阶段覆盖输入。本文件是后续实施任务书；本轮仅新增方案文档。

研究取舍见 [方案总览](<D:/graduate/第一阶段波束覆盖方案/第一阶段方案总览（面向研究者）.md>)。事实以当前新链路代码为准，旧重构方案中的拟议接口和既有调研中的固定模板不能覆盖已实现语义。

## 1. 阅读入口与范围

实施前读 [AGENTS.md](<D:/graduate/AGENTS.md>)、[项目说明](<D:/graduate/PPO4090/项目说明.md>)、[统一路径](<D:/graduate/PPO4090/project_paths.py>)，沿以下完整链路复核，禁止导入旧训练入口测试新实现：

```text
implementations/horizon_tsac_20260920/
  train.py
  → data/{schema,loader,augment,split,domain}.py
  → env/{environment,action,physics,metrics}.py
  → models/networks.py
  → rl/{sac,replay}.py
  → evaluate.py / export.py
```

首版只做单颗 GEO、静态原始业务点、固定合法地面宽度、给定实体数的覆盖；不把频组选择、频率动作、功率动作、用户排队和多星选择混入第一阶段。

## 2. 当前代码已有能力与仍需增加的部分

已有能力：`CoverageScenario` 可变长实体表；`RootScene` / `CoverageCandidate`；归属和需求守恒；球面圆盘内归属验证；按 root/谱系/hash 划分；root 层增强工具；候选分配与 `J_root` 计算；第二阶段配置和账本独立复算。

需要增加：原始业务来源适配器、覆盖生成器、可行域/硬件验证、归属求解、候选文件包读写、第一阶段候选比较器、第一阶段运行清单和针对这些功能的测试。现有 `CoverageGenerator` 是协议声明，缺省生成器会抛错，不是可直接调用的数据生成实现。

现有训练器支持 manifest 中的 `scenario_path`，读取的是 `CoverageScenario`。它不会自动从磁盘恢复 `RootScene` / `CoverageCandidate`，也不会自动在每回合计算 `J_root`。`evaluate_run` 主要面向单个 scenario；第一阶段联评应显式使用 `allocate_candidate`，或在后续独立评估入口中增加 root/candidate 加载。

## 3. 建议的新目录路由（尚未创建的实现）

```text
PPO4090/
├─ project_paths.py                       # 将来在这里定义新增根目录
├─ implementations/coverage_stage1_20260920/
│  ├─ config.py                           # 物理profile、约束、搜索/训练预算
│  ├─ data/{root_loader,profiles,manifest}.py
│  ├─ geometry/{coverage,feasibility}.py
│  ├─ assignment/{associate,repair}.py
│  ├─ generators/{clustering,graph_search,bilevel}.py
│  ├─ env/{environment,action,reward}.py   # 仅选PPO路线后建设
│  ├─ models/networks.py                  # 仅选学习路线后建设
│  ├─ rl/ppo.py                           # 独立于历史PPO
│  ├─ adapter/horizon.py                  # 显式调用新第二阶段
│  ├─ generate.py / evaluate.py / export.py
│  └─ train.py                            # 仅选学习路线后建设
├─ data/coverage_stage1_20260920/          # 来源卡、root、候选、split
├─ outputs/coverage_stage1_20260920/<run_id>/
└─ tests/coverage_stage1_20260920/
第一阶段波束覆盖方案/                     # 本轮方案
第一阶段波束覆盖实施记录/                 # 后续实际验收再建立
```

技术模块名是拟议代码路由，不是本轮已交付代码。后续数据、输出、实施记录根只在 `project_paths.py` 定义；用户交付的报告、配置、清单、数据文件和图片使用中文文件名。新增入口必须有 main 守卫；训练从 `PPO4090` 根目录用 `python -m implementations...` 启动。

## 4. 物理与任务 profile 必须先冻结

每个 profile 至少记录服务区域、卫星位置/轨道半径、地球模型、可见性、可指向域、允许地面角直径、最大实体数/同时开启数、中心间隔约束（若有）、接收端假设、增益来源、资源配置和版本。

当前默认配置：GEO 经度 122.2°、地心轨道半径 42164 km；8 逻辑池、100 共享频槽、25 MHz/槽；6000 W 总预算、5–50 W 十档总功率、每束连续 1–10 槽。默认 50 dBi 发射峰值、40 dBi 接收增益、290 K 和 5 dB SINR margin 是既有模型假设，不是本轮核实的真实载荷参数。

必须增加上游检查：业务点和中心在服务区/可指向域、卫星可见、宽度来自允许集合、实体数满足上限、需要时满足时间重构约束。当前 `RootScene.validate` 只要求服务区和卫星几何非空，并不证明其与实际配置一致；`CoverageCandidate.validate` 也不检查 profile ID 是否与配置内容一致。适配器应比对实值与 hash，不能只比较字符串标签。

当前宽度是地面角直径；所有覆盖判定用球面大圆角距 `distance_deg ≤ diameter_deg / 2`。局部投影可用于聚类加速，最终校验回到球面。首版固定宽度；未来变宽需要独立标定的增益/旁瓣 profile、边缘用户审计和重新验证的下游策略。禁止把增益当作可任意提高的第一阶段动作。

## 5. root、candidate 与波束实体的精确契约

### 5.1 RootScene：同一次原始业务实现

直接复用 [RootScene](<D:/graduate/PPO4090/implementations/horizon_tsac_20260920/data/domain.py:17>)，不另建同名但字段不一致的结构。

```text
root_scene_id, split_group_id, generator_family_id
demand_id[K]
demand_location[K,2]          # 纬度、经度，degree
offered_demand_bps[K]         # bps，已完成所有业务缩放
service_area, satellite_geometry
demand_generator_version, seed, metadata
```

补充 metadata 保存来源、许可/使用范围、时间戳（如有）、业务单位换算、来源文件 hash、合成/实测标志。可以开展可追溯合成 root 实验，但必须明示合成机制和支持域；测试 fixture 不计生产覆盖，合成研究结果不称真实部署验证。不能从聚合 CSV 伪造真实原始用户。

### 5.2 CoverageCandidate：同一root的一个布局

直接复用 [CoverageCandidate](<D:/graduate/PPO4090/implementations/horizon_tsac_20260920/data/domain.py:50>)：

```text
root_scene_id, coverage_candidate_id, coverage_generator_version
parameters_and_seed
beam_table: CoverageScenario
assignment[K,B]              # 当前实现接收稠密数值数组
uncovered_demand_bps
source_quality_metrics
physics_profile_id, provenance_status
```

首版用硬归属 `A_kb∈{0,1}`，每行最多一个 1；当前校验也允许显式分摊，但必须另声明策略与容量解释。建议磁盘保存稀疏索引降低体积，加载到现有接口前转为稠密数组并检查内存预算；不能把 scipy sparse 直接当现有 `np.asarray` 接口已支持的类型。

归属轴必须绑定稳定的 `demand_id` 与该候选的 `beam_id`。波束排序时要同步排列 assignment 列和全部实体字段；禁止仅排序需求数组。不同候选的同名 beam_id 不自动代表同一个物理波束，跨候选比较按 root 业务对齐。

### 5.3 CoverageScenario：给第二阶段的实体表

必须完整保存 `scenario_id`、`source_hash`、`source_schema`、`metadata` 与以下数组：

```text
beam_id, latitude_deg, longitude_deg, demand_bps, ground_diameter_deg
tx_gain_peak_dbi, rx_gain_peak_dbi, noise_temperature_k
group_id, polarization_id, entity_mask, demand_mask, service_order
```

约定 `scenario_id = coverage_candidate_id`；新数据用 `coverage.v2` 或 `standard`，经纬度字段不互换，bps 不再乘 0.25。`traffic_scale=0.25` 是 legacy 适配口径，不能再次作用在已经缩放的 root 上。首版可直接不带 padding；真实零需求实体保留 `entity_mask=True, demand_mask=False`，不通过删零需求实体掩盖硬件占用。

`service_order` 是正需求 beam_id 的排列，不是数组下标。首版沿用需求降序、ID打破并列；不优化服务顺序。`beam_id` 使用非负、候选内唯一且生成可复现的整数，不能作为优化器操纵 tie-break 的隐藏决策。

### 5.4 group规则及增强

新候选首次生成时，默认按正需求降序的 canonical rank mod 8 分组，极化按 group mod 2；生成一次后显式写入实体表。第一阶段主研究不学习 group。loader 虽能接受源 group，也不代表新增频组设计已获物理与模型验收。

`Environment.reset` 实际强制 `polarization_id == group_id % 2`；这不只是推荐默认。不得以loader单独加载成功为依据传入任意极化表。

同一候选的纯需求增强沿当前 `augment_root_candidates` 保留已有 group/极化、重算业务聚合与服务顺序。几何、宽度、归属、拆分合并改变时重新构造候选，按冻结规则重新产生映射并记录。这样的布局变化可能同时引起 group 重排：主结论属于“布局＋既定映射规则”的系统结果；若单独声称几何收益，再做固定映射的受控消融。

## 6. 守恒、覆盖与状态边界

每个候选需通过：有限非负业务、唯一业务ID、唯一波束ID、归属行和不超过1、padding无业务、正归属落在球面覆盖内、聚合需求与未覆盖量守恒。

```text
D_b = A[:,b]^T d
D_uncovered = (1 − row_sum(A))^T d
sum(D_b) + D_uncovered = sum(d)
```

使用当前验证容差：聚合需求 `rtol=1e-8, atol=1e-5 bps`，球面归属边界容差 `1e-8 degree`；硬件与 profile 限制另有具名容差，不能靠加大容差提高可行率。

区分三种结果：合法且满足任务覆盖约束；几何/守恒合法但未满足全覆盖任务；数据/物理非法。第二类可作为明确标签的困难候选，但全覆盖主结果中列失败，不能混成达标解。对一般启发式用 `search_failed` 表示未找到可行解；仅有证书时使用 `infeasible_proven`。

## 7. 第二阶段调用、返回值与缓存

复用的真实函数为 [allocate_candidate](<D:/graduate/PPO4090/implementations/horizon_tsac_20260920/evaluate.py:40>)。参数名虽然是 `physics_config`，当前实际需要完整 `Config`，不是单独 `PhysicsConfig`。

```python
# 设计示例：限正总需求root；空业务另走返回不适用指标的分支。
# root、candidate、frozen_policy、config由适配器显式加载并核验。
from implementations.horizon_tsac_20260920.evaluate import allocate_candidate
result = allocate_candidate(root, candidate, frozen_policy, physics_config=config)
assert result["end_to_end_metric_available"]
score = result["J_root"]
```

不把字符串 checkpoint 路径当 policy 传入。适配器按 checkpoint 配置、模型/动作签名实例化 `DiscreteSAC`，加载 agent 权重、置评价模式，并核验 checkpoint/profile hash。首版确定性推理；同候选重复调用必须可复现。

当前返回顶层字段含 `allocation`、`U`、`J_root`、`end_to_end_metric_available`、root总需求/已覆盖/未覆盖量、`constraint_violations`、`inference_seconds`、`metrics`。注意大小写：业务共同分母值是顶层 **`J_root`**；嵌套 `metrics` 中原有 `j_root=None`、`end_to_end_metric_available=False` 是阶段内指标占位，不能误读或将其当已经自动填充的联评结果。

`inference_seconds` 从环境 reset 后开始计时，不含全部生成、校验和模型初始化成本。外层必须额外计总耗时，区分冷启动/热启动、缓存命中和真实下游调用。

缓存键至少包含 root业务/坐标内容hash、候选全部物理字段/归属hash、group/极化/服务顺序、完整Config语义hash、checkpoint hash、下游算法及推理模式。不能仅用 candidate_id 或布局坐标缓存，否则需求变化、功率预算变化、重新分组会错误复用结果。

现有受限候选贪心可用于联调与对照，记录 `greedy_candidates` 等实际规则；它不是最优解或性能上界。smoke checkpoint 可验收连通性，不能支持算法性能主结论。

## 8. 文件包与manifest：建议新增的持久化约定

建议每个 root 文件夹包含 `原始业务.json`、`来源与物理规格.json`，每个候选包含 `波束场景.json`、`业务归属.npz`、`覆盖候选清单.json`。这些文件是计划中的输出，并非当前第二阶段自动生成的格式。

`波束场景.json` 直接序列化 `CoverageScenario.to_dict()`，当前 `load_scenario` 可读。其 `source_hash` 定义为不含自身hash字段的规范化内容摘要，避免自引用；manifest 的 `source_hash` 则按当前 `prepare_dataset` 要求记录**实际场景文件字节 SHA256**。现有 `evaluate_run` 会比较两者，因口径不同需后续显式适配/分离内容hash和文件hash。首次串联使用 root-aware 评价器直接调用 `allocate_candidate`，不能承诺未经修改的 `evaluate_run` 可无缝处理该新JSON清单。

若先接入当前训练/CLI评价而暂不调整hash协议，可同时导出标准字段的 `覆盖候选-编号.csv`：完整保存上述逐束字段（服务顺序由配置重算），设置 `data.source_schema="coverage.v2"`、`rate_unit="bps"`、`traffic_scale=1.0`，清单 `source_hash` 为CSV字节SHA256。CSV loader会使用同一文件hash，避免JSON自引用冲突；文件stem、manifest的scenario_id和candidate_id保持一致。root/assignment仍由候选包保留并由外层联评加载，CSV本身不替代这些信息。

候选包另保存 root文件hash、归属hash、profile hash、生成器版本和候选参数。`npz` 使用无 pickle 数值数组。加载端核验所有引用，再恢复 dataclass；JSON 不只凭宣称的ID信任来源。

给训练器的数据清单记录至少包含 `scenario_id, scenario_path, source_hash, root_scene_id, split_group_id, generator_family_id, split, provenance_status`；域审计还要记录 `n_demand`、规模/空间/负荷等实际统计与 `domain_cells`。root与candidate包路径作为新增字段保留。用现有 `manifest_hash` 计算完整清单hash，不手填任意字符串。

当前 `semantic_hash` 面向 CSV，新JSON/候选包需另建规范化内容hash函数。现有 split 校验不是近重复几何检测器，仍需上游谱系与近重复审计。

## 9. 数据和第二阶段训练的先后关系

1. 先冻结物理域与 root split，再为训练root生成多样合法候选。真实数据不足时允许声明合成研究域，不把旧聚合CSV的来源缺口填成真实标签。
2. 用方案一/二在训练split提供候选多样性；按root先均匀再选候选，避免候选多的root占据更多训练权重。
3. 训练/验证第二阶段，选择并冻结 checkpoint 后，再比较方案三/四等第一阶段策略。这样可以减少“上游变化超出下游训练域”的混淆。
4. 同root的候选、需求增强、搜索轨迹、教师标签和时间片共享split；业务增强在root上做一次，各候选重新聚合，不能各自抽不同业务后比较。
5. 先审计单轴，再二阶与重点三阶组合；48–256及每单元20/10/10个独立root沿用现有暂定建设目标，按真实可行性确认，不把这一目标写成已达成结果。
6. 若后续迭代扩展第二阶段训练域，建立新版本checkpoint，并对全部第一阶段方案重评；禁止只让某个方案受益于更新后的下游。

最终测试root不能用于拟合代理/蒸馏/动作超参数。部署时运行已冻结的局部搜索属于推理，其候选和调用成本正常计入；测试结果不能反过来用于修改搜索器。

## 10. 日志与前端衔接

第一阶段至少记录 root/candidate/parent IDs、动作或搜索操作、几何与任务可行性、未覆盖需求、代理分数、真实 `J_root`、下游版本、调用成本、停止原因及失败诊断。

现有五模块前端面向第二阶段单景训练与回放，不能直接宣称已支持root级候选比较。后续可以在波束分布与实验对比中增加root候选视图：同步显示原始需求、归属连线、覆盖轮廓、未覆盖点、分配结果。新增字段/API与只读服务单独验收，训练与Web进程继续独立。

## 11. 分批实施与验收门槛

### A0：规格和来源

输出中文数据卡、物理假设清单、来源缺口、固定/可调变量。验收坐标/单位/profile一致，区分地面角宽与天线角宽；确立实体数量和全覆盖任务范围。

### A1：方案一与共同接口

实现原始业务导入、合法覆盖、唯一归属、失败报告、文件包重载和manifest。至少覆盖重叠用户、孤立点、零需求、空root、边界用户、超出允许宽度、重复ID/归属、拆分合并后的轴对齐等有效测试。

### A2：冻结下游闭环与方案二/三

同root两个不同聚合方式/波束数的候选能守恒并得到共同分母评价；漏覆盖业务仍在分母。验证U与J_root排序不同的反例，验证数据重载前后分配一致；对非法归属、超预算结果拒绝接受。随后开展真实规模推理耗时测试。

### A3：所选学习或鲁棒扩展

PPO检查mask、STOP、请求动作log-prob、确定性转移与奖励累计；蒸馏检查split继承和top-k漏选；鲁棒检查同一需求情景集合的配对评估与信息时序。都要与同成本非学习方法比较。

### A4：正式联评与第三阶段交接

按独立root/谱系报告均值、低分位、失败率、种子波动、调用次数和耗时；保留完整运行清单和全部失败。第二阶段既有第三阶段导出主要携带聚合场景，不完整保留root/assignment，需另外保存关联包才能重算端到端共同分母。

后续改动共享接口时，先跑 `tests/horizon_tsac_20260920/` 及新增第一阶段测试，再做短运行。移动/重命名链路文件时同步项目说明/AGENTS路由并运行全项目AST和非训练模块导入检查。本轮文档交付不启动训练，不把测试计划写成已执行结果。
