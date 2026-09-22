# MLP-PPO低分源码复核

日期：2026-09-21。范围：只读核查冻结PPO、共享SAC、评价入口、单测及既有诊断。本次不改冻结源码、权重、原报告或图。主任务已完成原联合MAP的15景复现及有界机制对照，范围和数值见[低分原因复核](D:/graduate/第二阶段T-SAC实施记录/MLP-PPO低分原因复核.md)。

## 结论

图中低分不是已发现的绘图计算错误，也不能解释为“PPO算法本身无效”。已核实问题是：该版本采用的联合MAP决策，会在具体分配动作概率分散时选择低概率SKIP；原验收确认了数学实现，却没有充分拦截决策性能退化。它是决策设计与性能验收缺陷，不能用“符合冻结协议”代替功能有效性。

## 公式与执行链路

[PPO第69行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:69)使用概率比与裁剪比的最小值；[第60行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:60)自然终止清零bootstrap，终止、截断或边界切断GAE递推。[第235行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:235)核对采集的旧logp/value，未见这些公式直接导致此次低分的证据。

[第101行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:101)构造`P(a)=P(ALLOC)×P(a|ALLOC)`，SKIP单列。[第187行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:187)训练按该分布采样；[第197行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py:197)确定性决策取全部动作的argmax。两者使用同一概率分布，但执行的是不同策略。

既有[实际权重诊断](D:/graduate/第二阶段T-SAC实施记录/PPO确定性解码原始数值.json)在一个validation首状态记录：9550种合法ALLOC总概率99.4518%，SKIP概率0.54818%，最大单ALLOC仅0.22157%，所以MAP选择SKIP。这证明该状态的概率分散机制；仅凭首状态不能量化全部低分归因。

## 共享规则与验收不足

SAC也采用相同门控结构，见[共享网络第178行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/models/networks.py:178)，并取联合argmax，见[SAC第94行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/rl/sac.py:94)。[评价第40行](D:/graduate/PPO4090/implementations/horizon_tsac_20260920/evaluate.py:40)统一传入`deterministic=True`。因此不是仅对PPO使用了不同规则，不能直接单独替换PPO分数后宣称公平提升。

[PPO单测第115行](D:/graduate/PPO4090/tests/horizon_tsac_20260920/test_paper_ppo.py:115)特意构造分配总概率80%、SKIP概率20%，并断言联合MAP必须SKIP。该测试验证定义一致性，没有验证策略服务能力。既有报告已披露训练采样与确定性低分分离；独立图仅标“联合MAP”，不足以解释失败机制。历史数值应保留，但这一栏只能说明该实现与解码组合失败，不能代表PPO算法的一般能力。后续应具名比较解码方式，并补充采样/确定性差异与异常跳过率的质量检查。
