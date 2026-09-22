"""论文证据、未刊参数和统一环境适配协议。"""
from copy import deepcopy

PAPER_SHA256="3f83777b8eb69d58c1b9d963d2a8b9ea8d68417c3102708e268211b724ea5e69"
PAPER_BASELINES={
    "fixed":"Fixed Scheme", "greedy":"Greedy Scheme", "random":"Random Scheme",
    "mlp_sac":"MLP-SAC", "cnn_sac":"CNN-SAC", "mlp_dqn":"MLP-DQN", "mlp_ppo":"MLP-PPO",
}
LEARNED=("mlp_sac","cnn_sac","mlp_dqn","mlp_ppo","tsac_205")
DISPLAY={**PAPER_BASELINES,"tsac_205":"T-SAC局部观察控制组"}


def adapt_config(config, algorithm):
    if algorithm not in LEARNED:
        raise ValueError("不是可训练的论文基线/控制组")
    result=deepcopy(config)
    result.model.encoder="cnn_local" if algorithm=="cnn_sac" else "legacy14" if algorithm=="tsac_205" else "mlp_local"
    result.model.spectrum_tokens="blocks10" if algorithm=="tsac_205" else "slots"
    result.model.actor="independent"
    result.model.critic="additive"
    result.model.model_version="paper_adapted_"+algorithm+".v1"
    result.env.observation_version="obs_legacy205"
    result.validate()
    return result


def method_metadata(algorithm,settings=None,config=None):
    if algorithm not in DISPLAY:raise ValueError("未知基线")
    return {
        "protocol_version":"paper_baselines_adapted.v1", "algorithm_id":algorithm,
        "display_name":DISPLAY[algorithm],"paper_sha256":PAPER_SHA256,"paper_section":"IV", "paper_page":5,
        "reproduction_scope":"方法级复现与统一修复环境适配，不声称重现原始论文图中数值",
        "comparison_scope":"与Horizon完整观察为系统级比较；tsac_205控制组与MLP/CNN局部观察相同，用于编码器归因",
        "paper_environment":{"beams":100,"budget_w":3000,"width_deg":.5,"gamma":.99,"episodes":4000,"batch_size":256},
        "runtime_profile":{'power_budget_w':config.env.power_budget_w,'gamma':config.train.gamma,'reward_scale':config.env.reward_scale} if config else None,
        "adaptations":["统一使用冻结覆盖CSV及配置声明的预算、v3功率/干扰/终止/奖励口径",
                       "所有方法受同一合法动作集合约束；必要时显式SKIP，不静默裁剪",
                       "局部神经基线输入205维；Horizon完整观察含更多信息，不能把差异全部归因于Transformer",
                       "独立三头在合法联合集合归一化后会产生统计依赖，保留独立评分结构而非无约束独立采样",
                       "相同环境交互预算；PPO为on-policy多epoch，优化次数与SAC/DQN分别报告"],
        "unpublished_choices":deepcopy(settings or {}),
        "source_provenance_limit":"原论文未提供完整基线超参数/seed/生成数据；当前CSV缺root谱系",
        "is_primary_paper_baseline":algorithm in PAPER_BASELINES,
    }


def build_agent(algorithm,config,action_spec,observation_example,settings=None):
    if algorithm in ("mlp_sac","cnn_sac","tsac_205"):
        from ..rl.sac import DiscreteSAC
        return DiscreteSAC(config,action_spec,observation_example)
    if algorithm=="mlp_dqn":
        from .dqn import BaselineDQN
        return BaselineDQN(config,action_spec,observation_example,**(settings or {}))
    if algorithm=="mlp_ppo":
        from .ppo import BaselinePPO
        return BaselinePPO(config,action_spec,observation_example,hyperparameters=settings)
    raise ValueError("未知学习算法")
