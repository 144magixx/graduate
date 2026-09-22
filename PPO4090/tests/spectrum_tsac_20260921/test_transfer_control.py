"""仅权重初始化的控制器边界、恢复语义与完整validation监督配置。

initializer自身的张量/优化器/RNG隔离由独立迁移测试负责；这里mock其返回
血缘，真实执行短环境、Replay、SAC更新和checkpoint，避免伪造控制器证明。
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sqlite3
import sys
import time

import pytest
import torch

from implementations.spectrum_tsac_20260921 import artifacts, initialization, overnight, train
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.data.split import manifest_hash
from implementations.spectrum_tsac_20260921.rl.sac import DiscreteSAC
from implementations.spectrum_tsac_20260921.telemetry.storage import DB_NAME, create_schema
from tests.spectrum_tsac_20260921.test_data import fixture
from tests.spectrum_tsac_20260921.test_schedule import tiny_config


INITIALIZATION = {
    "kind": "weights_only_fixture",
    "source_checkpoint_id": "cnn-parent-100ep",
    "source_counters": {"episodes": 100, "env_step": 14281, "update_step": 13282},
    "reset": ["optimizer", "replay", "rng", "counters"],
}


def make_dataset(root, config):
    source = root/"三束控制器诊断.json"
    artifacts.atomic_json(source, fixture(3, config).to_dict())
    manifest = {"dataset_version": config.data.dataset_version,
                "records": [{"scenario_id": "diagnostic_fixture", "scenario_path": str(source),
                             "source_hash": sha256(source), "split": "train", "domain_cells": ["diagnostic"]}]}
    manifest["manifest_hash"] = manifest_hash(manifest)
    path = root/"诊断数据清单.json"
    artifacts.atomic_json(path, manifest)
    return path, manifest


def controller_fixture(monkeypatch, tmp_path, *, warmup=5, episodes=2):
    config = tiny_config(warmup=warmup)
    # 此控制器测试使用小型完整观察与明确稳定头，不依赖被复制旧case的默认头。
    config.model.encoder = "full_pool"
    config.model.actor, config.model.critic = "independent", "additive"
    config.train.episodes = episodes
    config.train.checkpoint_every = 1
    dataset, manifest = make_dataset(tmp_path, config)
    monkeypatch.setattr(artifacts, "SPECTRUM_OUTPUT_DIR", tmp_path/"运行")
    calls, ticks, initialized = [], [], []
    original_updates = train.perform_updates

    def act(agent, observation, deterministic=False):
        calls.append(int(observation["current_beam_index"]))
        return agent.action_spec.decode(0)

    def initialize(agent, received_config, received_manifest, path):
        assert agent.update_step == 0
        assert all(not getattr(agent, key).state for key in
                   ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "alpha_optimizer"))
        assert received_config.to_dict() == config.to_dict()
        assert received_manifest == manifest
        initialized.append(str(path))
        return copy.deepcopy(INITIALIZATION)

    def track_updates(cfg, agent, replay, counters, recorder=None, *, trigger):
        count = original_updates(cfg, agent, replay, counters, recorder, trigger=trigger)
        if count:
            ticks.append((counters["env_step"], len(replay), count))
        return count

    monkeypatch.setattr(DiscreteSAC, "act", act)
    monkeypatch.setattr(initialization, "initialize_from_checkpoint", initialize)
    monkeypatch.setattr(train, "perform_updates", track_updates)
    return config, dataset, manifest, calls, ticks, initialized


def test_weights_initialized_warmup_uses_policy_but_waits_for_update_eligibility(tmp_path, monkeypatch):
    config, dataset, manifest, calls, ticks, initialized = controller_fixture(monkeypatch, tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        directory, report = train.run_training(config, dataset=dataset, initialize_from="trusted-parent.pt")
    assert initialized == ["trusted-parent.pt"]
    assert len(calls) == 6
    # 首回合3步全部入池；第5步才达warmup，当前未完成episode仍不进入Replay。
    assert ticks == [(5, 3, 1), (6, 6, 1)]
    assert report["counters"] == {"episodes": 2, "env_step": 6, "update_step": 2}
    assert report["initialization"] == INITIALIZATION
    assert report["update_protocol"]["warmup_collection"] == "initialized_policy"
    saved = artifacts.load_checkpoint(directory/"检查点.pt", config, manifest["manifest_hash"])
    assert saved["initialization"] == INITIALIZATION
    assert saved["replay"]["transition_count"] == 6
    run_manifest = json.loads((directory/"运行清单.json").read_text(encoding="utf-8"))
    assert run_manifest["initialization"] == INITIALIZATION
    assert run_manifest["warmup_collection"] == "initialized_policy"
    assert run_manifest["parent_checkpoint_id"] is None
    assert run_manifest["resume_env_step"] == 0


def test_fresh_uninitialized_warmup_still_uses_uniform_collection(tmp_path, monkeypatch):
    config, dataset, _, calls, ticks, initialized = controller_fixture(monkeypatch, tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        _, report = train.run_training(config, dataset=dataset)
    assert initialized == []
    assert len(calls) == 1  # 前5步随机热身，仅第6步调用策略。
    assert [row[0] for row in ticks] == [5, 6]
    assert report["initialization"] is None
    assert report["update_protocol"]["warmup_collection"] == "uniform_legal"


def test_initialized_incomplete_episode_does_not_update_without_complete_replay(tmp_path, monkeypatch):
    config, dataset, _, calls, ticks, _ = controller_fixture(monkeypatch, tmp_path, warmup=0)
    config.train.max_env_steps = 2
    with contextlib.redirect_stdout(io.StringIO()):
        _, report = train.run_training(config, dataset=dataset, initialize_from="trusted-parent.pt")
    assert len(calls) == 2 and ticks == []
    assert report["counters"] == {"episodes": 1, "env_step": 2, "update_step": 0}
    assert not report["checkpoint_available"]
    assert report["episodes"][0]["truncated"] is True


def test_exact_resume_keeps_initialization_and_policy_warmup_without_reinitializing(tmp_path, monkeypatch):
    config, dataset, manifest, calls, ticks, initialized = controller_fixture(monkeypatch, tmp_path, warmup=8, episodes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        whole, expected_report = train.run_training(config, dataset=dataset, initialize_from="trusted-parent.pt")
    assert len(calls) == 9 and [row[0] for row in ticks] == [8, 9]
    archives = sorted((whole/"检查点存档").glob("*/检查点.pt"),
                      key=lambda p: json.loads((p.parent/"检查点清单.json").read_text(encoding="utf-8"))["counters"]["episodes"])
    first = archives[0]
    before_hash = sha256(first)
    expected = artifacts.load_checkpoint(whole/"检查点.pt")
    calls.clear()
    ticks.clear()

    def forbid_initialize(*args, **kwargs):
        raise AssertionError("精确恢复不能再次执行仅权重初始化")

    monkeypatch.setattr(initialization, "initialize_from_checkpoint", forbid_initialize)
    child_config = copy.deepcopy(config)
    child_config.train.episodes = 2
    with contextlib.redirect_stdout(io.StringIO()):
        child, actual_report = train.run_training(child_config, dataset=dataset, resume=first)
    assert initialized == ["trusted-parent.pt"]
    assert len(calls) == 6  # 恢复后第4–8步仍在热身，必须继续继承策略。
    assert [row[0] for row in ticks] == [8, 9]
    assert actual_report["counters"] == expected_report["counters"]
    assert actual_report["initialization"] == INITIALIZATION
    assert actual_report["update_protocol"]["warmup_collection"] == "initialized_policy"
    actual = artifacts.load_checkpoint(child/"检查点.pt", child_config, manifest["manifest_hash"])
    assert actual["initialization"] == INITIALIZATION
    assert actual["trial_id"] == expected["trial_id"]
    assert actual["counters"] == expected["counters"]
    assert actual["sampler"] == expected["sampler"]
    assert sha256(first) == before_hash
    for network in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        for key, tensor in expected["agent"][network].items():
            torch.testing.assert_close(actual["agent"][network][key], tensor, rtol=0, atol=0)
    child_manifest = json.loads((child/"运行清单.json").read_text(encoding="utf-8"))
    assert child_manifest["resume_env_step"] == 3
    assert child_manifest["initialization"] == INITIALIZATION
    assert child_manifest["warmup_collection"] == "initialized_policy"


def test_resume_and_initialization_conflict_before_dataset_or_output(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("互斥输入必须在读取数据之前拒绝")

    monkeypatch.setattr(train, "prepare_dataset", forbidden)
    with pytest.raises(ValueError, match="不可同时"):
        train.run_training(Config(), resume="resume.pt", initialize_from="source.pt")


def test_train_cli_passes_initialization_path_without_loading_source(monkeypatch, tmp_path):
    config = tiny_config()
    source = tmp_path/"含 空格的父检查点.pt"
    captured = {}

    def fake_run(*args):
        captured["args"] = args
        return tmp_path/"虚拟运行", {"counters": {"episodes": 0, "env_step": 0, "update_step": 0}}

    monkeypatch.setattr(train, "load_config", lambda path: config)
    monkeypatch.setattr(train, "run_training", fake_run)
    monkeypatch.setattr(sys, "argv", ["train", "--mode", "train", "--config", "config.json",
                                     "--initialize-from", str(source), "--seed", "43"])
    with contextlib.redirect_stdout(io.StringIO()):
        train.main()
    args = captured["args"]
    assert args[0].train.seed == 43
    assert args[1:] == ("train", None, None, str(source))


def plan_config():
    config = Config()
    config.train.episodes = 300
    config.train.checkpoint_every = 50
    config.train.warmup_steps = 1000
    config.train.max_wall_seconds = 10
    return config.to_dict()


def test_plan_resolves_initialization_and_rejects_non_main_entry(tmp_path):
    raw = {"quick_evaluation_limit": 0, "quick_evaluation_policies": ["policy"], "quick_backup": False,
           "experiments": [{"name": "迁移诊断", "config": plan_config(), "initialize_from": "父检查点/检查点.pt"}]}
    normalized = overnight.normalize_plan(raw, tmp_path)
    assert normalized["experiments"][0]["initialize_from"] == str((tmp_path/"父检查点"/"检查点.pt").resolve())
    assert normalized["quick_evaluation_limit"] == 0
    assert normalized["quick_evaluation_policies"] == ["policy"] and normalized["quick_backup"] is False
    invalid = copy.deepcopy(raw)
    invalid["experiments"][0]["algorithm"] = "cnn_sac"
    with pytest.raises(ValueError, match="仅主训练入口"):
        overnight.normalize_plan(invalid, tmp_path)


@pytest.mark.parametrize("field,value", [("quick_evaluation_limit", -1), ("quick_evaluation_limit", True),
                                         ("quick_evaluation_policies", []), ("quick_evaluation_policies", ["test"]),
                                         ("quick_backup", "false")])
def test_invalid_quick_evaluation_configuration_rejected(field, value, tmp_path):
    raw = {field: value, "experiments": [{"name": "诊断", "config": plan_config()}]}
    with pytest.raises(ValueError):
        overnight.normalize_plan(raw, tmp_path)


def test_supervisor_forwards_initializer_and_requests_full_validation_without_backup(tmp_path, monkeypatch):
    output, durable = tmp_path/"runs", tmp_path/"durable"
    monkeypatch.setattr(overnight, "SPECTRUM_OUTPUT_DIR", output)
    source = tmp_path/"旧CNN检查点.pt"
    plan = overnight.normalize_plan({"campaign_id": "迁移监督诊断", "poll_seconds": .001,
        "quick_evaluation_limit": 0, "quick_evaluation_policies": ["policy"], "quick_backup": False,
        "experiments": [{"name": "新任务", "config": plan_config(), "initialize_from": str(source),
                         "final_policies": ["policy"]}]}, tmp_path)
    requests, train_commands = [], []

    def archive(run, episode):
        folder = run/"检查点存档"/str(episode)
        folder.mkdir(parents=True)
        (folder/"检查点.pt").write_bytes(b"diagnostic-only-not-a-model")
        artifacts.atomic_json(folder/"检查点清单.json", {"file": "检查点.pt",
                              "counters": {"episodes": episode, "env_step": episode*100, "update_step": episode*100-1000}})

    class TrainChild:
        def __init__(self, run):
            self.run, self.calls, self.finished = run, 0, False

        def poll(self):
            self.calls += 1
            if self.calls < 5:
                return None
            if not self.finished:
                archive(self.run, 300)
                with sqlite3.connect(self.run/DB_NAME) as conn:
                    conn.execute("UPDATE runs SET status='completed'")
                self.finished = True
            return 0

    class DoneChild:
        def poll(self):
            return 0

    def launch(command, log_path, cpu=False):
        if "--config" in command:
            train_commands.append(command)
            assert "implementations.spectrum_tsac_20260921.train" in command
            assert command[command.index("--mode")+1] == "train"
            assert command[command.index("--initialize-from")+1] == str(source)
            cfg = Config.from_dict(overnight.read_json(command[command.index("--config")+1]))
            assert cfg.train.episodes == 300 and cfg.train.checkpoint_every == 50
            run = output/"mock-run"
            artifacts.atomic_json(run/"运行清单.json", {"run_id": run.name, "mode": "train", "config_hash": overnight.config_hash(cfg)})
            conn = sqlite3.connect(run/DB_NAME)
            create_schema(conn)
            conn.execute("INSERT INTO runs VALUES(?,?,?)", (run.name, "{}", "running"))
            conn.commit()
            conn.close()
            archive(run, 50)
            return TrainChild(run)
        if "--evaluation-worker" in command:
            assert cpu is True
            request = overnight.read_json(command[-1])
            requests.append(request)
            assert request["limit"] == 0 and request["policies"] == ["policy"]
            assert "backup_path" not in request and "durable_backup_path" not in request
            artifacts.atomic_json(request["result_path"], {"status": "completed", "kind": request["kind"],
                "checkpoint_counters": request["checkpoint_counters"],
                "report": {"split": "validation", "summary": {"policy": {"mean_U": .5, "scenarios": 15}}}})
            return DoneChild()
        assert "--copy-worker" in command
        assert overnight.copy_worker(command[-1]) == 0
        return DoneChild()

    monkeypatch.setattr(overnight, "launch", launch)
    monkeypatch.setattr(overnight.time, "sleep", lambda seconds: None)
    with contextlib.redirect_stdout(io.StringIO()):
        _, report = overnight.supervise(plan, durable)
    assert len(train_commands) == 1
    assert [(r["kind"], r["checkpoint_counters"]["episodes"]) for r in requests] == [("quick", 50), ("final", 300)]
    assert report["finished"] and report["all_final_evaluations_available"]
    assert report["test_set_used"] is False


def test_full_quick_worker_calls_validation_limit_zero_only(tmp_path, monkeypatch):
    from implementations.spectrum_tsac_20260921 import evaluate
    observed = {}

    def evaluate_run(manifest, **kwargs):
        observed.update(kwargs)
        return {"summary": {"policy": {"scenarios": 15, "mean_U": .5}}}

    monkeypatch.setattr(evaluate, "evaluate_run", evaluate_run)
    request = {"kind": "quick", "run_dir": str(tmp_path/"run"), "checkpoint_path": str(tmp_path/"archive"/"检查点.pt"),
               "checkpoint_counters": {"episodes": 50, "env_step": 5000, "update_step": 4000},
               "limit": 0, "policies": ["policy"], "result_path": str(tmp_path/"结果.json")}
    path = tmp_path/"请求.json"
    artifacts.atomic_json(path, request)
    assert overnight.evaluation_worker(path) == 0
    assert observed["split"] == "validation" and observed["limit"] == 0
    assert observed["policies"] == ("policy",)
    assert observed["checkpoint_path"] == request["checkpoint_path"]


def paired_plan(tmp_path):
    experiments = []
    for seed in (42, 43):
        for encoder in ("cnn_local", "cnn_attention_residual"):
            config = plan_config()
            config["train"]["seed"] = seed
            config["model"]["encoder"] = encoder
            config["model"]["model_version"] = "cnn_control.v1" if encoder == "cnn_local" else "spectrum_residual.v1"
            experiments.append({"name": f"{encoder}-{seed}", "config": config,
                                "initialize_from": str(tmp_path/"共同CNN父检查点.pt"),
                                "final_policies": ["policy"]})
    return {"campaign_id": "同源配对诊断", "paired_transfer": True,
            "quick_evaluation_limit": 0, "quick_evaluation_policies": ["policy"],
            "quick_backup": False, "experiments": experiments}


def paired_evaluation(seed, encoder, phase="final_evaluation"):
    # 同seed内严格同预算；seed43采样场景规模不同，合法产生不同的环境步数。
    episodes = 50 if phase == "quick_evaluation" else 300
    env_steps = episodes*140+(seed-42)*37
    budget = {"episodes": episodes, "env_step": env_steps, "update_step": env_steps-999}
    delta = .1 if seed == 42 else -.02
    scores = [.5+.005*i+(delta if encoder == "cnn_attention_residual" else 0.) for i in range(15)]
    rows = [{"scenario_id": f"validation-{i:02d}", "n_demand": 120+i, "total_demand_bps": 1e10+i*1e8,
             "power_budget_w": 6000., "phase": "validation", "mean_satisfaction": score,
             "constraint_violation_count": 0, "constraint_violations": []} for i, score in enumerate(scores)]
    return {"status": "completed", "checkpoint_counters": budget,
            "report": {"split": "validation", "checkpoint_id": f"{encoder}-{seed}-{episodes}",
                       "evaluated_training_counters": copy.deepcopy(budget),
                       "results": {"policy": rows}, "summary": {"policy": {"scenarios": 15, "mean_U": sum(scores)/15}}}}


def paired_rows(plan, kind="final_evaluation"):
    return [{"name": row["name"], kind: paired_evaluation(row["config"]["train"]["seed"],
             row["config"]["model"]["encoder"], kind)} for row in plan["experiments"]]


def test_four_paired_transfers_accept_only_encoder_and_version_differences(tmp_path):
    plan = overnight.normalize_plan(paired_plan(tmp_path), tmp_path)
    assert len(plan["experiments"]) == 4
    assert len({row["config_hash"] for row in plan["experiments"]}) == 4
    assert len({row["initialize_from"] for row in plan["experiments"]}) == 1
    assert {row["config"]["train"]["seed"] for row in plan["experiments"]} == {42, 43}


@pytest.mark.parametrize("changed", ["parent", "actor_lr", "episodes", "power_budget", "missing_pair", "quick_subset", "final_protocol"])
def test_paired_plan_rejects_different_parent_learning_or_budget(tmp_path, changed):
    plan = paired_plan(tmp_path)
    row = plan["experiments"][1]
    if changed == "parent":
        row["initialize_from"] = str(tmp_path/"另一个父检查点.pt")
    elif changed == "actor_lr":
        row["config"]["train"]["actor_lr"] *= 2
    elif changed == "episodes":
        row["config"]["train"]["episodes"] -= 1
    elif changed == "power_budget":
        row["config"]["env"]["power_budget_w"] -= 100
    elif changed == "missing_pair":
        plan["experiments"].pop()
    elif changed == "quick_subset":
        plan["quick_evaluation_limit"] = 3
    else:
        row["final_policies"] = ["policy", "random"]
    with pytest.raises(ValueError):
        overnight.normalize_plan(plan, tmp_path)


@pytest.mark.parametrize("kind", ["quick_evaluation", "final_evaluation"])
def test_pending_pair_is_reported_without_import_or_missing_result_failure(tmp_path, kind):
    plan = overnight.normalize_plan(paired_plan(tmp_path), tmp_path)
    rows = paired_rows(plan, kind)
    rows[1][kind] = None
    result = overnight.paired_transfer_comparisons(plan, rows, kind)
    assert result["42"] == {"status": "pending"}
    assert result["43"]["status"] == "comparable"


@pytest.mark.parametrize("kind", ["quick_evaluation", "final_evaluation"])
def test_pairing_uses_seed_local_budgets_and_residual_minus_cnn_direction(tmp_path, kind):
    plan = overnight.normalize_plan(paired_plan(tmp_path), tmp_path)
    rows = paired_rows(plan, kind)
    result = overnight.paired_transfer_comparisons(plan, rows, kind)
    assert result["42"]["additional_training_budget"] != result["43"]["additional_training_budget"]
    for seed, expected in (("42", .1), ("43", -.02)):
        row = result[seed]
        assert row["status"] == "comparable"
        assert row["residual_minus_cnn"]["n_pairs"] == 15
        assert row["residual_minus_cnn"]["mean_difference"] == pytest.approx(expected)
        assert row["residual_minus_cnn"]["ci95"] == pytest.approx([expected, expected])
        assert row["shared_pretrained_seed"] is True
        assert row["formal_independent_multiseed"] is False


@pytest.mark.parametrize("changed", ["budget", "non_validation", "row_test_phase", "missing_scene", "duplicate_scene",
                                    "different_demand", "missing_violation_number", "false_violation_number",
                                    "nonempty_violation_list", "missing_satisfaction"])
def test_invalid_evaluation_cannot_claim_fifteen_scene_pair_comparison(tmp_path, changed):
    plan = overnight.normalize_plan(paired_plan(tmp_path), tmp_path)
    rows = paired_rows(plan)
    left, right = rows[0]["final_evaluation"], rows[1]["final_evaluation"]
    values = right["report"]["results"]["policy"]
    if changed == "budget":
        right["checkpoint_counters"]["env_step"] += 1
    elif changed == "non_validation":
        right["report"]["split"] = "test"
    elif changed == "row_test_phase":
        values[0]["phase"] = "test"
    elif changed == "missing_scene":
        values.pop()
    elif changed == "duplicate_scene":
        # 两方法同样出现重复ID，但需求不同，tuple签名仍有15项，不能绕过唯一场景校验。
        for evaluation in (left, right):
            items = evaluation["report"]["results"]["policy"]
            items[1]["scenario_id"] = items[0]["scenario_id"]
    elif changed == "different_demand":
        values[0]["total_demand_bps"] += 1
    elif changed == "missing_violation_number":
        values[0].pop("constraint_violation_count")
    elif changed == "false_violation_number":
        values[0]["constraint_violation_count"] = False
    elif changed == "nonempty_violation_list":
        values[0]["constraint_violations"] = [{"kind": "total_power"}]
    else:
        values[0].pop("mean_satisfaction")
    result = overnight.paired_transfer_comparisons(plan, rows, "final_evaluation")
    assert result["42"]["status"] != "comparable"
    assert "residual_minus_cnn" not in result["42"]
    assert result["43"]["status"] == "comparable"


def test_progress_keeps_both_seed_pairs_comparable_when_global_budgets_differ(tmp_path):
    plan = overnight.normalize_plan(paired_plan(tmp_path), tmp_path)
    states = {row["name"]: dict(row, status="trained", exit_code=0, errors=[], durable_copy="diagnostic-durable-copy",
              quick_result=paired_evaluation(row["config"]["train"]["seed"], row["config"]["model"]["encoder"], "quick_evaluation"),
              final_result=paired_evaluation(row["config"]["train"]["seed"], row["config"]["model"]["encoder"]))
              for row in plan["experiments"]}
    result = overnight.write_progress(tmp_path, plan, states, time.monotonic(), finished=True)
    assert result["matched_final_budget"] is False and result["matched_environment_budget"] is False
    assert result["all_final_evaluations_available"] is True
    assert all(value["status"] == "comparable" for pairs in result["paired_transfer_comparisons"].values() for value in pairs.values())
    assert result["test_set_used"] is False and result["formal_multiseed_complete"] is False
    assert overnight.campaign_successful(result)
