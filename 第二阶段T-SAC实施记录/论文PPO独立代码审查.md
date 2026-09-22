# 论文 MLP-PPO 独立代码审查

审查范围：`PPO4090/implementations/horizon_tsac_20260920/paper_baselines/ppo.py`、`tests/horizon_tsac_20260920/test_paper_ppo.py`，并只读核对 `paper_baselines/train.py` 的实际采集调用，以及共享 `Transition`、`masked_distribution` 契约。本次不修改 PPO 实现、不启动训练、不操作远端。两项初审发现均已由实现负责人修复，并由本审查独立重跑原反例确认闭环。

## 已验证结果

- 初审 PPO 测试为 15 passed、1 skipped。修复后独立复跑 `python -m pytest tests/horizon_tsac_20260920/test_paper_ppo.py tests/horizon_tsac_20260920/test_paper_heuristics.py -q`，结果 35 passed、1 skipped（6.22 秒），其中 PPO 为 21 passed、1 skipped，启发式为 14 passed。跳过项需要真实 CUDA；本次不能声称完成 GPU 恢复验收。
- PPO clipped objective 对正负 advantage 均使用正确的最小值；有利方向超出区间后梯度停止，不利方向继续更新。
- GAE 在自然终止时清零 bootstrap；外部截断时使用真实 next-state value，但切断跨段递推。最后一个采集样本不会递推到不存在的后续样本。有完整 episode/step 元数据时能隔离不同回合和不连续片段。
- Actor 的 SKIP gate 与三个独立头组成同一个合法联合概率；采集 logp、更新 ratio 和熵均来自该联合分布。独立枚举笛卡尔积、施加占用 mask 后计算条件概率，最大绝对误差为 `1.51e-8`。强制 SKIP、非法候选概率为零以及全局 joint argmax 均有测试。
- old logp/value 在单次多 epoch 更新期间不参与梯度，returns 与 advantage 在 minibatch 循环前冻结。提供采集记录时会核对当前网络，拒绝已经变更策略的旧样本。
- 每个真实 minibatch 更新递增一次 update_step，并记录实际 batch_size、样本索引及耗时；汇总按样本数加权。3 条样本、2 epochs、minibatch_size=2 时记录 4 次更新，大小为 2/1/2/1。
- CPU 检查点恢复包括 Actor/Critic、Adam 状态、动作 RNG、minibatch RNG 及更新计数；恢复后的采样和后续参数更新逐项一致。不兼容学习配置会在加载模型前拒绝。

## 发现及影响

### 已修复 P2：省略行为记录可绕过旧 rollout 检查

审查时 `update_rollout` 允许 old_log_probs 和 old_values 同时缺省，并把当前网络输出重新标成 old。复现步骤：采集一次 rollout，带原始 logp/value 更新；再以同样的 transitions 调用 `update_rollout(samples)`。第二次被接受，initial_ratio_max_error 为 0，rollout_updates 增至 2。若第二次仍传原始采集记录，则按预期拒绝陈旧样本。

初审时正式训练入口始终传入采集时记录，因此该入口未触发；公开 API 的缺省重算分支则可绕过“不可拿 replay 更新”的约束。

修复后 `update_rollout` 强制同时提供采集时 logp/value，缺失或 None 会在更新前拒绝。独立重跑原反例：先带原始记录更新，再对同样 samples 省略行为记录；现在立即抛出 ValueError，update_step 保持不变。对应防回放回归测试已加入并通过。

### 已修复 P2：缺省片段身份时可能错误接续 GAE

`Transition` 允许 episode_id 为空、step_index/env_step 为 None。对同场景不连续的第 0、2 步清空这些字段后，`update_rollout` 接受样本并生成 boundaries `[False, False]`，使 GAE 跨越缺失的第 1 步。行为 logp/value 合法不代表时间连续性合法。

初审时训练入口填写上述所有字段，因此实际整回合采集未触发。

修复后只有两端 episode_id、step_index、env_step 都存在，且同场景、同回合、step_index 与 env_step 均递增 1 时才接续 GAE；其余保守切断，仍保留各转移的真实 next-state bootstrap。独立重跑缺省身份的第 0、2 步，得到 boundaries `[True, False]`；advantage 和 returns 与两个独立片段逐项完全相等。完整元数据、单字段缺失及全部缺失的新增测试均通过。

## 启发式评价标识补查

只读核对 `evaluate.py`：非 policy 评价会先移除父运行的 paper_baseline/baseline_settings；论文三种静态方法再写入自身算法名和冻结参数，并标明不需要模型训练、checkpoint 仅作环境与数据参照。每个场景的 metrics.heuristic_spec 使用实际策略实例的 `metadata()`，保留 Greedy 已解析需求阈值、固定长度和随机种子等实际规则。

论文 Random 的 `random_include_skip=True` 在评价入口明确拒绝；套件计划也拒绝这种混名。启发式策略类自身仍将该可选诊断标记为 `uniform_with_skip`。本次 14 项启发式测试通过，未发现上述 metadata 被绕过；完整评价入口的集成测试由主任务统一运行。

## 复现口径

以上是当前统一物理环境、动作 mask、SKIP gate 下的实现审查。论文未刊的隐藏层、PPO 超参数及合法动作适配已在状态清单中标识。本次通过不等于复现论文原始数值，也不替代真实数据和充分训练预算下的算法性能比较。

状态：两项初审问题均已修复并通过独立反例复验。此审查范围内无未关闭发现；真实 CUDA 验收仍未执行。
