"""220正需求/9551动作的有界容量短测，不生成研究训练数据。"""
import argparse
import copy
import ctypes
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from ..config import Config
from ..data.loader import scenario_from_rows
from ..env.environment import Environment
from ..models import Actor, Critic
from .replay import Transition
from .sac import DiscreteSAC


def process_memory():
    """进程累计峰值；不是单次操作独占内存，也不是GPU显存。"""
    try:
        if os.name == "nt":
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (key, ctypes.c_size_t) for key in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            result = Counters()
            result.cb = ctypes.sizeof(result)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(result), result.cb):
                raise OSError("GetProcessMemoryInfo失败")
            return dict(working_set_bytes=result.WorkingSetSize, peak_working_set_bytes=result.PeakWorkingSetSize,
                        source="Windows GetProcessMemoryInfo，进程累计峰值")
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(working_set_bytes=None, peak_working_set_bytes=int(peak if sys.platform == "darwin" else peak*1024),
                    source="resource.ru_maxrss，进程累计峰值")
    except (ImportError, AttributeError, OSError) as error:
        return dict(working_set_bytes=None, peak_working_set_bytes=None, source=None, unavailable_reason=str(error))


def _count(module):
    return sum(p.numel() for p in module.parameters())


def run_benchmark():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Config()
    config.train.batch_size, config.train.microbatch_size = 2, 1
    config.train.device, config.train.seed = "cpu", 42
    config.model.encoder, config.model.d_model = "full_attention", 128
    config.model.attention_heads, config.model.encoder_layers = 4, 2
    config.model.actor, config.model.critic = "conditional", "joint"
    config.model.candidate_chunk_size = 512
    # 可达环境生成观察，行不是第一阶段产物，不进入正式数据集。
    rows = [dict(beam_id=1000+i, latitude_deg=25.+(i//20)*.05, longitude_deg=108.+(i%20)*.05,
                 demand_bps=1e8+i*1e5, ground_diameter_deg=2., group_id=i % 8) for i in range(220)]
    scenario = scenario_from_rows(rows, config, "coverage.v2", "capacity_220_fixture",
        metadata={"purpose": "capacity_shape_fixture_only", "not_training_dataset": True})
    env = Environment(config)
    obs, _ = env.reset(scenario, 42)
    if scenario.n_demand != 220 or len(env.action_spec) != 9551:
        raise AssertionError("主空间容量fixture规格错误")
    memory_before = process_memory()
    started = time.perf_counter()
    agent = DiscreteSAC(config, env.action_spec, obs)
    construction_seconds = time.perf_counter()-started
    started = time.perf_counter()
    probability = agent.probabilities(obs)
    forward_seconds = time.perf_counter()-started
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1., atol=1e-6):
        raise AssertionError("主空间策略概率无效")
    started = time.perf_counter()
    sampled = agent.act(obs)
    sample_seconds = time.perf_counter()-started
    started = time.perf_counter()
    deterministic = agent.act(obs, deterministic=True)
    deterministic_seconds = time.perf_counter()-started
    if env.action_spec.encode(deterministic) != int(probability.argmax()):
        raise AssertionError("确定性决策不等于全局联合概率argmax")
    transitions = []
    current = obs
    for i in range(2):
        action = sampled if i == 0 else agent.act(current)
        following, reward, terminated, _, _ = env.step(action)
        transitions.append(Transition(current, env.action_spec.encode(action), reward, following, terminated,
                                      i == 1, scenario.scenario_id, config.semantic_versions()))
        current = following
    saved_tensor_shapes, q_calls, prefix_shapes = [], [], []
    hooks = []
    def saved_tensor(tensor):
        saved_tensor_shapes.append(dict(shape=list(tensor.shape), numel=tensor.numel(),
                                        bytes=tensor.numel()*tensor.element_size()))
        return tensor
    def q_hook(module, inputs, output):
        if inputs[0].ndim == 3:
            q_calls.append(dict(shape=list(inputs[0].shape), requires_grad=output.requires_grad))
    def prefix_hook(module, inputs, output):
        if torch.is_grad_enabled():
            prefix_shapes.append(list(inputs[0].shape))
    for critic in (agent.critic_1, agent.critic_2, agent.target_critic_1, agent.target_critic_2):
        hooks.append(critic.q.register_forward_hook(q_hook))
    hooks.append(agent.actor.power.register_forward_hook(prefix_hook))
    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(saved_tensor, lambda tensor: tensor):
        diagnostics = agent.update(transitions)
    update_seconds = time.perf_counter()-started
    for hook in hooks:
        hook.remove()
    memory_after_update = process_memory()
    forbidden = [s for s in saved_tensor_shapes if 9551 in s["shape"] and 128 in s["shape"]]
    if forbidden or any(call["requires_grad"] for call in q_calls):
        raise AssertionError("候选Q或完整候选特征持有不应存在的梯度图")
    graph_audit = dict(saved_tensor_count=len(saved_tensor_shapes),
        largest_saved_tensor=max(saved_tensor_shapes, key=lambda value: value["numel"]),
        full_9551_by_128_gradient_tensor_observed=bool(forbidden),
        actor_power_prefix_shapes=prefix_shapes,
        candidate_q_forward_calls=len(q_calls),
        candidate_q_max_chunk=max(call["shape"][1] for call in q_calls),
        candidate_q_any_requires_grad=any(call["requires_grad"] for call in q_calls),
        lifecycle="Actor有梯度前缀特征为[microbatch,100,10,128]，完整候选仅概率/logp [microbatch,9551]；Q候选512分块均no_grad；每个微批backward后释放。")
    del agent
    gc.collect()
    configurations = []
    variants = [("legacy14", "blocks10", "independent", "additive"),
                ("summary", "blocks10", "independent", "additive"),
                ("full_pool", "slots", "conditional", "joint"),
                ("full_attention", "slots", "conditional", "joint")]
    for encoder, tokens, actor_kind, critic_kind in variants:
        variant = copy.deepcopy(config)
        variant.model.encoder, variant.model.spectrum_tokens = encoder, tokens
        variant.model.actor, variant.model.critic = actor_kind, critic_kind
        actor, critic = Actor(variant, env.action_spec, obs), Critic(variant, env.action_spec, obs)
        actor_count, critic_count = _count(actor), _count(critic)
        configurations.append(dict(encoder=encoder, spectrum_tokens=tokens, actor=actor_kind, critic=critic_kind,
            d_model=128, attention_heads=4, encoder_layers=2,
            actor_parameters=actor_count, critic_each_parameters=critic_count,
            actor_encoder_parameters=_count(actor.encoder), critic_encoder_each_parameters=_count(critic.encoder),
            trainable_parameters_including_log_alpha=actor_count+2*critic_count+1,
            parameters_including_two_target_networks_and_log_alpha=actor_count+4*critic_count+1))
        del actor, critic
        gc.collect()
    return dict(purpose="220正需求容量shape fixture，不是第一阶段产物，不进入研究训练数据", passed=True,
        command="python -m implementations.horizon_tsac_20260920.rl.benchmark", versions=config.semantic_versions(),
        runtime=dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                     processor=platform.processor(), torch_num_threads=torch.get_num_threads(), device="cpu",
                     cuda_available=torch.cuda.is_available(), cuda_device=None, cuda_peak_memory_bytes=None),
        actual_fixture=dict(n_demand=220, entity_tensor_shape=list(obs["beam_static"].shape),
            slots=100, action_count=9551, legal_action_count=int(obs["valid_action_mask"].sum()),
            effective_batch=2, microbatch=1, candidate_chunk=512), config=config.to_dict(),
        timings_seconds=dict(model_construction=construction_seconds, first_probability_forward=forward_seconds,
            stochastic_sample=sample_seconds, deterministic_argmax=deterministic_seconds,
            one_effective_batch_update=update_seconds),
        memory_before_model=memory_before, memory_after_effective_batch_update=memory_after_update,
        parameter_configurations=configurations, update_diagnostics=diagnostics, gradient_graph_audit=graph_audit,
        acceptance_limits=["仅测CPU有效batch2/microbatch1一次update，未压测batch256",
            "未验收4090/CUDA显存、有效batch256或GPU吞吐；CUDA指标为null",
            "进程峰值是当前独立进程累计working set，不能解释为update专属分配量",
            "时延为单次有界测量，含当前进程缓存状态，不是P95或稳态吞吐",
            "四组合同时改变输入/模型且参数量不同，只报告容量，不作公平性能排名"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output:
        target = Path(args.output)
    else:
        from project_paths import get_output_dir
        target = get_output_dir("horizon_tsac_20260920", "数学验收") / "性能与容量短测.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    report = run_benchmark()
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "path": str(target), "timings_seconds": report["timings_seconds"],
                      "memory": report["memory_after_effective_batch_update"], "parameters": report["parameter_configurations"],
                      "graph": report["gradient_graph_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
