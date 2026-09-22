# Anchor T-SAC 验收与代码审查报告

日期：2026-09-22。Anchor 是对存量纯 Transformer 的独立回归实现，用于适配重构后的新环境。本轮已完成代码和只读前端，不运行真实训练、CSV rollout、性能评价、旧 checkpoint 迁移、SSH 或远端部署。

## 交付范围

- 后端：[anchor_tsac_20260921](D:/graduate/PPO4090/implementations/anchor_tsac_20260921/)
- 测试：[tests/anchor_tsac_20260921](D:/graduate/PPO4090/tests/anchor_tsac_20260921/)
- 前端：[anchor_tsac_20260921](D:/graduate/PPO4090/frontend/anchor_tsac_20260921/)
- 设计依据：[纯Transformer回归与渐进重构方案-v4](D:/graduate/第二阶段T-SAC重构方案/纯Transformer回归与渐进重构方案-v4.md)

Anchor 保留旧14-token骨干：4个属性token加10个频谱块token，d=128、8头、2层、ReLU、4d FFN、post-LN和零位置初始化。它使用新环境的完整物理账本、正需求分母、合法动作字典和 `SKIP=0`；这是一条新环境回归适配链路，不声称恢复旧物理环境的精确行为。

提供两个具名205观察适配器：`legacy_scaled205` 恢复旧数值尺度但来源仍为新物理账本，`modern205` 使用新环境尺度。适配器版本写入语义版本，混用会被拒绝。

旧五网络权重迁移使用显式SHA、`weights_only=True`、全量键/shape/dtype/finite预检和事务回滚；不迁移alpha、优化器、Replay、RNG或计数。迁移是新trial的诊断起点，不是精确续训。精确resume保留稳定trial、父checkpoint和原初始化谱系，并拒绝跨语义/跨设备随机状态恢复。

## 工程验收

- Anchor专项合并测试：**34 passed，21条预期警告，24.78秒**，JUnit见[Anchor合并工程测试.xml](D:/graduate/第二阶段T-SAC实施记录/Anchor合并工程测试.xml)。覆盖14-token核心、两种适配器、动作mask、SKIP、SAC合成更新、Replay原子恢复、checkpoint身份、预算边界、日志开关、迁移回滚、CLI授权和只读API。
- 全项目AST：**214个Python模块通过，0语法错误**。
- Anchor安全导入：**32个模块通过**，没有导入训练入口的副作用。
- 旧链路/旧前端冻结文件：**116个文件哈希未变**，见[Anchor静态与隔离验收](D:/graduate/第二阶段T-SAC实施记录/Anchor静态与隔离验收.json)。
- 两份零预算配置均通过preflight；默认预算为0，不创建运行目录。
- 未授权 `train`、`evaluate`、`export` 均在读取数据或清单前拒绝；证据见[Anchor恢复后CLI与指纹核验](D:/graduate/第二阶段T-SAC实施记录/Anchor恢复后CLI与指纹核验.json)。
- 前端 `npm run build` 通过；五模块包含波束分布、训练曲线、频谱分配、单步回放和实验对比。空态不导入历史结果。
- 工程demo只含3个人工波束和3个固定合法动作，明确标记 `engineering_fixture`，不加载CSV、模型、权重或优化器，不进入科学比较。浏览器已核验波束、频谱账本、回放、曲线空态和比较过滤。

## 代码审查结论

1. 路径、输出和前端根独立于Horizon/Spectrum；真实数据必须显式提供冻结清单，不能隐式重划分或借用旧运行结果。
2. 训练和评价入口有显式 `--allow-experiment` 门控；默认preflight/integration只做工程检查。
3. 正式RL前向显式关闭dropout，保留旧dropout结构仅用于受控core回归；这不宣称与旧训练模式完全相同。
4. Replay实际冻结为 `transition_uniform`，与实现一致；动作ID、mask、观察适配器和语义版本逐条校验，恢复失败不部分提交。
5. checkpoint按完整回合边界保存，并保留不可变归档和sidecar；wall/env预算只软停在回合边界，manifest记录累计回合、环境步和更新步。
6. manifest区分 `algorithm_id`、结构化 `algorithm_spec`、`lineage.kind`、初始化来源、`is_demo`和`experiment_authorized`；demo不会冒充from-scratch训练。
7. 评价/导出接口先验证split、limit、策略白名单和授权，再读取数据或创建输出；verify-only仅复算已有交接工件。

## 限制与未完成实验

本轮没有真实CSV rollout、旧权重迁移实验、短训/长训、多seed、验证集性能或远端部署，因此**不能声称 Anchor 已恢复旧Transformer性能，也不能声称它优于CNN/MLP**。旧完整权重缺少历史配置、alpha、优化器、Replay和RNG，只能支持受控结构回归。

上游root/assignment和严格独立广覆盖来源仍缺失。`legacy_scaled205`只保持输入数值尺度，不回滚新环境的物理修正、功率守恒、奖励或SKIP语义。

当前自动化 `gpu4-t-sac` 保持 `PAUSED`；本轮没有重启旧实验、SSH、训练或性能评价。
