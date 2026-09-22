# Spectrum T-SAC 当前网络结构图

对应当前代码 `PPO4090/implementations/spectrum_tsac_20260921/models/networks.py` 与冻结修复配置。这里只说明结构；训练跟进仍按用户要求暂停。

## 1. 单个编码器的结构

CNN保留逐槽信息；Transformer读取整个波束集合，再给CNN特征增加有界修正。下图是编码器模板，Actor、两个Critic各自有一份，参数不共享。

```mermaid
flowchart TB
    subgraph LOCAL[局部路径：保留逐槽位置]
        S["100槽干扰 + 100槽占用<br/>2 × 100"] --> CNN["1D CNN：2 → 32 → 64<br/>卷积核5、3；ReLU"]
        CNN --> FLAT["展平：6400维"]
        A["当前波束5个属性<br/>经纬度、需求、宽度、剩余功率"] --> FC["拼接后6405维<br/>MLP：6405 → 128 → 128"]
        FLAT --> FC
        FC --> ZL["局部特征 zCNN<br/>128维"]
    end
    subgraph GLOBAL[全局路径：学习上下文修正]
        B["全部波束的位置、需求、分配状态<br/>每束16维 + 分组/状态嵌入"] --> T["Transformer<br/>1层 / 4头 / 128维"]
        G["全局资源与分配进度<br/>25维"] --> GM["全局MLP → 128维"]
        T --> READ["提取当前波束特征<br/>用交叉注意力读取全局上下文"]
        GM --> READ
        READ --> OUT["拼接当前波束、全局状态、注意力读出<br/>线性层384 → 128<br/>出口权重与偏置初始化为0"]
        OUT --> DELTA["tanh + 幅度限制<br/>残差 Δz"]
    end
    ZL -. "参与构造注意力查询" .-> READ
    ZL --> ADD((相加))
    DELTA --> ADD
    ADD --> Z["输出特征 z = zCNN + Δz<br/>128维"]
    classDef local fill:#e8f3fb,stroke:#3285aa,color:#183c50;
    classDef context fill:#fff2dc,stroke:#cb8b24,color:#624414;
    classDef fusion fill:#e7f5ec,stroke:#42845b,color:#204d31;
    class S,CNN,FLAT,A,FC,ZL local;
    class B,T,G,GM,READ,OUT,DELTA context;
    class ADD,Z fusion;
```

查询由 `zCNN`、当前波束Token和全局MLP向量拼接后投影生成；Key/Value来自Transformer的全部有效波束Token。padding通过掩码排除。

残差的实际计算为：

\[
\Delta z=0.25\,\max\bigl(\operatorname{RMS}(\operatorname{stopgrad}(z_{CNN})),1\bigr)\,\tanh(W_o[c,g,h]+b_o),\qquad z=z_{CNN}+\Delta z.
\]

这里 `c` 是当前波束特征，`g` 是全局状态向量，`h` 是交叉注意力读出。`W_o,b_o` **仅在初始化时为0，训练中可更新**；残差相加后没有额外LayerNorm。因此迁移同一个CNN父模型时，初始化输出与父CNN精确一致，之后才逐渐学习上下文修正。

## 2. Actor、双Critic与目标网络

```mermaid
flowchart TB
    S["同一观察 s"] --> EA["Actor自己的编码器<br/>CNN + Transformer残差"]
    S --> E1["Critic 1自己的编码器<br/>CNN + Transformer残差"]
    S --> E2["Critic 2自己的编码器<br/>CNN + Transformer残差"]
    EA --> AH["起点100 / 长度10 / 功率10评分头<br/>另有SKIP / ALLOC门"]
    AH --> PI["合法动作掩码 + 联合概率分布<br/>输出分配动作或SKIP"]
    E1 --> Q1["Q1 = 起点Q + 长度Q + 功率Q<br/>SKIP另有一个Q值"]
    E2 --> Q2["Q2 = 起点Q + 长度Q + 功率Q<br/>SKIP另有一个Q值"]
    Q1 --> MIN["逐个完整动作取 min(Q1,Q2)<br/>用于SAC训练"]
    Q2 --> MIN
    Q1 -. "软更新 τ=0.005" .-> T1["目标Critic 1<br/>独立副本，不反传梯度"]
    Q2 -. "软更新 τ=0.005" .-> T2["目标Critic 2<br/>独立副本，不反传梯度"]
    classDef enc fill:#e8f3fb,stroke:#3285aa,color:#183c50;
    classDef head fill:#e7f5ec,stroke:#42845b,color:#204d31;
    classDef target fill:#f1f1f5,stroke:#9292a3,color:#444455;
    class EA,E1,E2 enc;
    class AH,PI,Q1,Q2,MIN head;
    class T1,T2 target;
```

- **不是共享一个编码器**：Actor、Q1、Q2各自独立；加上两个目标Critic，总共5份编码器，没有目标Actor。
- Actor的三个头独立生成评分，但**不独立采样三个动作分量**：评分相加后，对合法ALLOC组合统一做softmax，再与SKIP/ALLOC门组合。默认最多9550个ALLOC加1个SKIP，实际可用数量由掩码决定。
- 两个Critic分别计算完整动作Q，再取较小者；不对三个Q分支分别取min后相加。
- 纯CNN对照使用同一个CNN父检查点，省去全局残差分支；其余冻结设置相同。残差是否带来实际收益，仍需同预算配对验证。

源码：[网络结构](D:/graduate/PPO4090/implementations/spectrum_tsac_20260921/models/networks.py)、[SAC更新](D:/graduate/PPO4090/implementations/spectrum_tsac_20260921/rl/sac.py)、[冻结继续训练计划](D:/graduate/第二阶段T-SAC实施记录/Spectrum修复继续训练计划.json)。
