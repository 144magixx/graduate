"""可重复的小空间穷举与短训练；明确为诊断fixture，不进入研究数据集。"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..config import Config
from ..data.loader import scenario_from_rows
from ..env.action import ActionSpec
from ..env.environment import Environment
from ..env.physics import evaluate_allocation, channel_geometry
from ..env.state import Allocation
from .replay import Transition, ReplayBuffer
from .sac import DiscreteSAC


def oracle_fixture():
    config = Config()
    config.physics.num_groups, config.physics.num_slots = 3, 4
    config.env.max_block_length, config.env.power_levels_w = 2, (5., 10.)
    config.env.power_budget_w = 30.
    config.model.d_model, config.model.attention_heads, config.model.encoder_layers = 16, 2, 1
    config.model.encoder = "full_pool"
    config.model.candidate_chunk_size = 5
    config.train.batch_size, config.train.microbatch_size = 24, 12
    config.train.actor_lr, config.train.critic_lr, config.train.alpha_lr = .003, .003, .0001
    config.train.tau = .05
    rows = [dict(beam_id=10+3*i, latitude_deg=30.+i*.001, longitude_deg=110.+i*.001,
                 demand_bps=1e6, ground_diameter_deg=2., group_id=i) for i in range(3)]
    scenario = scenario_from_rows(rows, config, "coverage.v2", "three_beam_oracle_fixture",
        metadata={"purpose": "diagnostic_fixture_only", "not_training_dataset": True})
    return config, scenario


def enumerate_oracle(config, scenario):
    spec = ActionSpec(config)
    geometry = channel_geometry(scenario, config.physics)
    best, best_ids, legal, utilities = -float("inf"), None, 0, []
    for ids in itertools.product(range(len(spec)), repeat=scenario.n_demand):
        ledger = {}
        for beam, action_id in zip(scenario.service_order, ids):
            action = spec.decode(action_id)
            power = 0. if action_id == 0 else float(spec.power_levels_w[action.power_index])
            ledger[int(beam)] = Allocation(int(beam), "skipped" if action_id == 0 else "allocated",
                                          action.start, action.length, power)
        result = evaluate_allocation(scenario, ledger, config, channel_cache=geometry)
        if result.constraint_violations:
            continue
        utility = result.metrics["mean_satisfaction"]
        legal += 1
        utilities.append(utility)
        if utility > best:
            best, best_ids = utility, ids
    return dict(action_count=len(spec), sequence_upper_bound=len(spec)**scenario.n_demand,
                legal_complete_sequences=legal, optimum_utility=best, optimum_action_ids=list(best_ids),
                uniform_complete_sequence_mean=float(np.mean(utilities)))


def _rollout(agent, env, scenario, deterministic=False, rng=None):
    obs, _ = env.reset(scenario)
    transitions, total, actions = [], 0., []
    while not obs["terminal"]:
        action = env.action_spec.decode(int(rng.choice(np.flatnonzero(obs["valid_action_mask"])))) if rng is not None else agent.act(obs, deterministic)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        action_id = env.action_spec.encode(action)
        transitions.append(Transition(obs, action_id, reward, next_obs, terminated, truncated,
                                      scenario.scenario_id, env.config.semantic_versions()))
        actions.append(action_id)
        total += reward
        obs = next_obs
    utility = env.evaluate().metrics["mean_satisfaction"]
    if not np.isclose(total, env.config.env.reward_scale*utility, atol=1e-10):
        raise AssertionError("真实环境奖励望远镜恒等式失败")
    return transitions, utility, actions


def run_validation(updates=180, seed=42):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config, scenario = oracle_fixture()
    config.train.seed = seed
    oracle = enumerate_oracle(config, scenario)
    if oracle["sequence_upper_bound"] != 3375:
        raise AssertionError("固定oracle空间必须是3束、4槽、2长度、2功率")
    env = Environment(config)
    example, _ = env.reset(scenario)
    agent = DiscreteSAC(config, env.action_spec, example)
    replay = ReplayBuffer(2000, config.semantic_versions(), agent.action_signature(), seed=seed)
    exploration = np.random.default_rng(seed)
    random_utilities = []
    for _ in range(100):
        transitions, utility, _ = _rollout(agent, env, scenario, rng=exploration)
        replay.add_episode(transitions)
        random_utilities.append(utility)
    initial_alloc_probability = 1 - float(agent.probabilities(example)[0])
    before = _rollout(agent, env, scenario, True)[1]
    diagnostics = []
    for update in range(updates):
        if update % 3 == 0:
            replay.add_episode(_rollout(agent, env, scenario)[0])
        diagnostics.append(agent.update(replay.sample(config.train.batch_size)))
    learned_alloc_probability = 1 - float(agent.probabilities(example)[0])
    _, after, chosen = _rollout(agent, env, scenario, True)
    # 完整恢复后下一个随机动作和下一个更新须一致。
    state, replay_state = agent.state_dict(), replay.state_dict()
    restored = DiscreteSAC(config, env.action_spec, example)
    restored.load_state_dict(state)
    restored_replay = ReplayBuffer(2000, config.semantic_versions(), agent.action_signature(), seed=seed)
    restored_replay.load_state_dict(replay_state)
    action_reproduced = agent.act(example) == restored.act(example)
    left, right = agent.update(replay.sample(24)), restored.update(restored_replay.sample(24))
    update_reproduced = left["actor_loss"] == right["actor_loss"]
    report = dict(purpose="合成诊断fixture；不属于广覆盖训练数据，不证明泛化改善", seed=seed,
        oracle=oracle, random_episodes=100, random_mean_utility=float(np.mean(random_utilities)),
        initial_deterministic_utility=before, trained_deterministic_utility=after,
        initial_alloc_probability=initial_alloc_probability, trained_alloc_probability=learned_alloc_probability,
        chosen_action_ids=chosen, updates=updates, diagnostic_update=diagnostics[-1],
        checkpoint_next_action_reproduced=action_reproduced, checkpoint_next_update_reproduced=update_reproduced,
        elapsed_seconds=time.perf_counter()-started,
        passed=bool(after >= np.mean(random_utilities) and learned_alloc_probability > initial_alloc_probability
                    and action_reproduced and update_reproduced))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output:
        target = Path(args.output)
    else:
        from project_paths import get_output_dir
        target = get_output_dir("horizon_tsac_20260920", "数学验收") / "小空间穷举与短训练验收.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    report = run_validation(args.updates, args.seed)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("短训练未达到预登记验收条件；检查报告，不宣称通过")


if __name__ == "__main__":
    main()
