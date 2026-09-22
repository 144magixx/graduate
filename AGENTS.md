# 项目交付约定

## Agent模型分工与成本控制

- 主agent使用`gpt-6-astra`的`ultra`档，负责方案设计、重构架构、任务拆分、关键取舍、跨模块集成判断和最终验收。不能把“重构任务”全部编码工作默认留给主agent。
- 调研、资料搜集、代码导航、事实核对、结果整理等明确子任务，优先交给`gpt-5.6-luna`；简单检索用`low`，需要交叉核验用`medium`，只有明确困难时提高档位。
- 编码、测试补齐和常规修复交给子agent：范围清晰、局部改动优先`gpt-5.6-terra`的`medium`；复杂算法、跨模块实现或难排故优先`gpt-5.6-sol`的`high`。可按难度下调或逐级升级，不默认最高档。
- 主agent先给出边界、接口、验收条件和文件归属，再并行派发互不冲突的任务；负责审查结果，避免多个模型重复阅读和重复实现同一工作。
- 创建子agent时显式填写模型和档位，并使用有限上下文或独立任务说明，避免不必要地复制全部长历史。继承了高成本模型的旧子agent不作为低成本任务的默认执行者。
- 只有存在可说明的复杂度、失败或质量风险时才升级模型/档位；记录升级原因。简单问题直接简答，不为形式上的分工增加调度成本。
- 不得把写入约定说成已切换当前主会话模型。主会话实际模型/档位若无法通过工具更改，须如实说明；子agent以工具实际指定的配置为准。

## 中文交付文件名

- 所有新建的用户交付文件均使用中文文件名，包括报告、文档、表格、演示稿、图片、PDF、数据导出和代码生成的说明文件。
- 保留必要的文件扩展名，例如 `.md`、`.docx`、`.xlsx`、`.pptx`、`.pdf`、`.png` 和 `.csv`。
- 技术标识、版本号或日期必须出现在文件名中时，可保留必要英文缩写、数字和连字符；其余描述性文字使用中文。
- 临时文件、缓存、依赖文件和第三方原始文件不视为用户交付文件；除非用户要求，不重命名它们。

示例：`HTS研究调研与落地总方案.md`、`第二部分T-SAC代码审查与重构方案.md`。

## PPO4090 代码路由

### 纯 Transformer 回归实现 Anchor T-SAC（2026-09-22）

- Anchor 独立实现位于 `PPO4090/implementations/anchor_tsac_20260921/`，测试位于 `PPO4090/tests/anchor_tsac_20260921/`，前端位于 `PPO4090/frontend/anchor_tsac_20260921/`；不改写 `sac_fyh_io`、Horizon、Spectrum 或历史产物。
- Anchor 保留旧14-token纯Transformer骨干（4个属性token+10个频谱块token，d128、8头、2层、ReLU、post-LN、零位置初始化），在新环境中使用具名 `legacy_scaled205` 或 `modern205` 观察适配器。适配器只改变输入尺度；`legacy_scaled205`仍来自新物理账本，不代表旧环境精确复原。
- 新动作协议保留合法动作字典和 `SKIP=0`，使用新环境的硬约束、正需求分母和物理账本。旧权重迁移仅为显式SHA校验的weights-only诊断，不迁移alpha、优化器、Replay、RNG或计数；`--resume`与迁移互斥，manifest必须记录谱系。
- 运行入口默认只做preflight；真实CSV rollout、训练、评价和导出必须显式 `--allow-experiment`、显式冻结数据清单和正预算。本轮仅完成代码、合成fixture和前端验收，未运行真实实验。
- 只读看板从 Anchor 的独立输出根读取，空态不导入旧结果；工程demo显著标记且不进入科学比较。默认启动：`python -m implementations.anchor_tsac_20260921.dashboard --host 127.0.0.1 --port 8876`。
- 交付前运行 `python -m pytest tests/anchor_tsac_20260921 -q`；不要把合成fixture、旧训练指标或前端demo当作性能结果。实现记录与验收见 `第二阶段T-SAC实施记录/Anchor代码实施记录.md`、`Anchor验收与代码审查报告.md`。

### 退化修复隔离实现 Spectrum T-SAC（2026-09-21）

- 新实现与测试分别位于 `PPO4090/implementations/spectrum_tsac_20260921/`、`PPO4090/tests/spectrum_tsac_20260921/`；保留 Horizon 源码、配置、测试及历史产物。旧包仅用于可信父模型反序列化与迁移对照，不能用旧训练入口冒充新实现验收。
- 新链路 `train.py` → `data/` → `env/` → `models/networks.py` → `rl/`，日志、评价、导出、监督与打包均使用新包路由；模型修复必须在新包内完成。
- 路径只在 `project_paths.py` 定义：`SPECTRUM_OUTPUT_DIR` 为独立 `outputs/spectrum_tsac_20260921/`；数据、只读前端、实施记录分别通过 `SPECTRUM_DATA_DIR`、`SPECTRUM_FRONTEND_DIR`、`SPECTRUM_REPORT_DIR` 复用现有冻结资源。
- 从 PPO4090 根目录运行 `python -m implementations.spectrum_tsac_20260921...`；新包 `package.py` 打包新源码、新测试、共享冻结数据及只读 UI，并携带冻结Horizon源码以反序列化可信CNN父模型，不携带历史产物或缓存。历史 `sac_fyh_io` 源码仅为只读兼容审计依赖。
- `models/networks.py`新增`cnn_attention_residual`：CNN逐槽路径加零出口、有界完整场景Transformer残差；默认独立Actor/可加Critic。`initialization.py`严格仅复制CNN五网络及alpha，重置优化器/Replay/RNG/计数，不能标为旧Horizon精确续训；`train.py --initialize-from`记录谱系并与`--resume`互斥。
- 本地及GPU验证已通过，2026-09-21 08:44曾在gpu4启动四组同源配对继续训练；仅validation，已见test不再选参。按相同续训seed比较相同新增预算，源100回合单列；两种续训流共享父模型，不冒称从零多seed。`第二阶段T-SAC实施记录/Spectrum远程训练状态.json`保留gpu4历史。
- gpu2恢复批次已于2026-09-21 14:46:55正常结束：SLURM206357.1、新run `20260921T053400_b66b2ea983ba` 从150恢复至累计300回合/42632步/41633更新，全15景validation U=.837377（配对CNN43=.775968）。seed42仅比较双方共同250回合，U=.819597/.804443；不跨预算合并、不宣称独立多seed。状态入口`第二阶段T-SAC实施记录/SpectrumGPU2恢复状态.json`，结果及验收见同目录`Spectrum最终配对验证报告.md`、`Spectrum修复验收与代码审查报告.md`。两CNN300、Residual42墙钟结束271、Residual43恢复后300均不重启；最终工件、报告和15张PNG已同步，自动跟进已暂停。运行代码与旧结果冻结；本次attempt数据库含新增150回合，完整trial计数由checkpoint/训练结果核对。

### 第二阶段新实现 Horizon T-SAC（2026-09-20）

- 当前第二阶段重构新增链路为 `PPO4090/implementations/horizon_tsac_20260920/`，保留以下原 `sac_fyh_io` 路由及历史产物。不得将旧训练入口导入当作新实现测试。
- 入口 `train.py` → `data/{schema,loader,augment,split,domain}.py` → `env/{environment,action,physics,metrics}.py` → `models/networks.py` → `rl/{sac,replay}.py`；沿完整链路核验。
- 配置在新包 `config.py`；数据及输出根目录仍只在 `project_paths.py` 定义。新产物写 `outputs/horizon_tsac_20260920/`；数据接口与覆盖审计在 `data/horizon_tsac_20260920/`。
- 日志写入/回放在 `telemetry/`，只读服务在 `dashboard/`，五模块前端在 `frontend/horizon_tsac_20260920/`。训练与Web进程独立。
- 独立评价及第三阶段交接在 `evaluate.py` / `export.py`。新增入口均有main守卫；本地先通过 `tests/horizon_tsac_20260920/` 后做短训练，正式长训练另行安排。
- 实施记录、验收和代码审查在 `第二阶段T-SAC实施记录/`。未取得真实上游root/assignment前，不把测试fixture计入广覆盖数据验收。
- 论文七基线新增于 `horizon_tsac_20260920/paper_baselines/`：`specs.py`冻结论文范围及适配、`train.py`控制器、`dqn.py`/`ppo.py`学习器、`heuristics.py`静态策略、`suite.py`依赖门控及最终对比。MLP/CNN-SAC共用`models/networks.py`与正确离散SAC；PPO只用on-policy轨迹，不能套用SAC Replay更新。
- 基线与Horizon共用数据、环境和物理评价，局部205输入与完整观察的收益须分开解释；`tsac_205`是额外同信息控制组。远端基线必须使用独立部署，等待当前gpu4夜间批次完成；不热改运行中的源码。

处理 `PPO4090` 前先阅读 `PPO4090/项目说明.md` 和 `PPO4090/project_paths.py`。训练入口、环境、模型、数据和产物必须沿同一条实现链路检查，不能只改入口文件。

- 修改当前论文使用的覆盖输入版 T-SAC：先查 `PPO4090/implementations/sac_fyh_io/sac_train_fyh_IO.py`，再查同目录的 `Environment_fyh_IO.py` 与 `sac_fyh_IO.py`。
- 修改波束数量、功率上限、状态、动作、奖励、干扰或终止条件：查对应实现的 `Environment*.py`；当前论文链路查 `sac_fyh_io/Environment_fyh_IO.py`。
- 修改 Transformer Token、Actor、Critic、经验回放或 SAC 更新公式：查对应实现的模型文件；当前论文链路查 `sac_fyh_io/sac_fyh_IO.py`。
- 修改覆盖 CSV 的加载、抽样、增强或训练轮次：查 `sac_fyh_io/sac_train_fyh_IO.py`；原始覆盖输入在 `PPO4090/data/cover_outputs/`。
- 修改城市 SAC 的公共环境：查 `sac_city_shared/Environment_sac_p.py`，并同时评估 `sac_city_token` 与 `sac_city_power` 两条链路。
- 修改 10 行频谱 Token 方案：查 `sac_city_token/sac_transformer_tok.py`；修改 100 个逐槽 Token 方案：查 `sac_city_power/sac_transformer_p.py`。
- 修改贪心对照算法：入口查 `greedy_baseline/greedy.py`，环境查同目录 `Environment_103.py`。
- 修改历史 PPO：在 `legacy_ppo/` 下选择对应目录，并同时检查其中的 `PPO*.py`、`Environment*.py` 和 `rl_utils*.py`。
- 修改数据位置或输出位置：只在 `PPO4090/project_paths.py` 定义根路径；各实现通过该文件取路径，产物写入 `PPO4090/outputs/` 下的独立目录。
- 移动或重命名链路文件后，同步更新 `PPO4090/项目说明.md` 和本节路由，并运行全项目 AST 语法检查与非训练模块导入检查。

运行训练统一从 `PPO4090` 根目录使用 `python -m implementations...`。训练入口在导入时会立即开始训练，不把导入训练入口作为验证方式。
