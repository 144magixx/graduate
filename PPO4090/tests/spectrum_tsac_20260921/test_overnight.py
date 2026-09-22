"""夜间监督的运行身份、不可变同步、摘要与并发边界。"""
import json
from pathlib import Path
import sqlite3

import pytest

from implementations.spectrum_tsac_20260921 import overnight
from implementations.spectrum_tsac_20260921.artifacts import atomic_json
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.telemetry.storage import create_schema, DB_NAME


def plan_config(eta=.5):
    config = Config()
    config.train.target_entropy_ratio = eta
    config.train.max_wall_seconds = 10
    return config.to_dict()


def write_run(root, name, digest):
    directory = root/name
    atomic_json(directory/"运行清单.json", {"run_id": name, "mode": "train", "config_hash": digest})
    return directory


def database(directory):
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(directory/DB_NAME)
    create_schema(conn)
    conn.execute("INSERT INTO runs VALUES(?,?,?)", (directory.name, "{}", "running"))
    conn.commit()
    return conn


def test_plan_requires_bounded_unique_config_and_safe_paths(tmp_path):
    value = {"experiments": [{"name": "低熵", "config": plan_config(.2)}, {"name": "高熵", "config": plan_config(.8)}]}
    result = overnight.normalize_plan(value, tmp_path)
    assert len(result["experiments"]) == 2
    assert result["experiments"][0]["config_hash"] != result["experiments"][1]["config_hash"]
    with pytest.raises(ValueError, match="重复配置"):
        overnight.normalize_plan({"experiments": [{"name": "a", "config": plan_config()}, {"name": "b", "config": plan_config()}]}, tmp_path)
    invalid = plan_config()
    invalid["train"]["max_wall_seconds"] = 0
    with pytest.raises(ValueError, match="max_wall_seconds"):
        overnight.normalize_plan([{ "name": "a", "config": invalid}], tmp_path)
    with pytest.raises(ValueError, match="安全文件名"):
        overnight.normalize_plan([{ "name": "../逃逸", "config": plan_config()}], tmp_path)


def test_config_hash_pairing_excludes_old_runs_and_rejects_ambiguity(tmp_path):
    experiments = [{"name": "低熵", "config_hash": "a"}, {"name": "高熵", "config_hash": "b"}]
    write_run(tmp_path, "old", "a")
    a = write_run(tmp_path, "new-a", "a")
    b = write_run(tmp_path, "new-b", "b")
    write_run(tmp_path, "unrelated", "c")
    assert overnight.match_runs(tmp_path, experiments, {"old"}) == {"低熵": a, "高熵": b}
    write_run(tmp_path, "duplicate", "a")
    with pytest.raises(ValueError, match="多个新run"):
        overnight.match_runs(tmp_path, experiments, {"old"})


def test_read_progress_uses_training_events_and_is_readonly(tmp_path):
    run = tmp_path/"运行"
    with database(run) as conn:
        conn.execute("INSERT INTO events(event_id,event_type,phase,env_step,update_step,payload) VALUES(?,?,?,?,?,?)",
                     ("train", "episode_completed", "train", 123, 17, "{}"))
        conn.execute("INSERT INTO events(event_id,event_type,phase,env_step,update_step,payload) VALUES(?,?,?,?,?,?)",
                     ("eval", "episode_completed", "validation", 999, 999, "{}"))
        conn.execute("INSERT INTO episodes(episode_id,phase,status,summary,final_event_seq) VALUES(?,?,?,?,?)",
                     ("ep", "train", "completed", json.dumps({"mean_satisfaction": .7, "env_step": 123, "update_step": 17}), 1))
    before = sha256(run/DB_NAME)
    result = overnight.read_progress(run)
    assert (result["episodes"], result["env_step"], result["update_step"], result["last_U"]) == (1, 123, 17, .7)
    assert sha256(run/DB_NAME) == before


def test_live_sync_copies_only_published_artifacts_and_verifies_hash(tmp_path):
    run, durable = tmp_path/"local", tmp_path/"durable"
    checkpoint = run/"检查点存档"/"cp"/"检查点.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable checkpoint")
    (run/"未发布.txt").write_text("must not copy", encoding="utf-8")
    relative = checkpoint.relative_to(run).as_posix()
    with database(run) as conn:
        conn.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?)", ("cp", 1, relative, sha256(checkpoint), "checkpoint", "available"))
    seen = set()
    overnight.sync_published(run, durable, seen)
    assert (durable/relative).read_bytes() == checkpoint.read_bytes()
    assert not (durable/DB_NAME).exists()
    assert not (durable/"未发布.txt").exists()
    assert len(seen) == 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash漂移"):
        overnight.sync_published(run, durable, set())


def test_closed_copy_keeps_database_and_ignores_wal(tmp_path):
    run, durable = tmp_path/"local", tmp_path/"durable"
    with database(run):
        pass
    (run/"测试.sqlite-wal").write_bytes(b"not a closed database")
    (run/"场景.json").write_text("{}", encoding="utf-8")
    overnight.copy_closed_tree(run, durable)
    assert (durable/DB_NAME).is_file()
    assert (durable/"场景.json").is_file()
    assert not (durable/"测试.sqlite-wal").exists()
    with sqlite3.connect(durable/DB_NAME) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_supervisor_simulated_children_obey_limit_and_report_missing(tmp_path, monkeypatch):
    output, durable = tmp_path/"runs", tmp_path/"durable"
    monkeypatch.setattr(overnight, "SPECTRUM_OUTPUT_DIR", output)
    plan = overnight.normalize_plan({"campaign_id": "模拟夜间", "max_workers": 2, "poll_seconds": .001,
        "experiments": [{"name": str(i), "config": plan_config(eta)} for i, eta in enumerate((.2, .5, .8))]}, tmp_path)
    live, maximum, launched = [], [0], []

    class FakeChild:
        def __init__(self):
            self.polls = 0
            self.done = False
            live.append(self)
            maximum[0] = max(maximum[0], sum(not child.done for child in live))

        def poll(self):
            self.polls += 1
            if self.polls >= 3:
                self.done = True
                return 0
            return None

    def fake_launch(command, log_path, cpu=False):
        launched.append(command)
        return FakeChild()

    monkeypatch.setattr(overnight, "launch", fake_launch)
    monkeypatch.setattr(overnight.time, "sleep", lambda seconds: None)
    folder, result = overnight.supervise(plan, durable)
    assert len(launched) == 3
    assert maximum[0] == 2
    assert result["finished"] is True
    assert not result["all_final_evaluations_available"]
    assert not result["matched_final_budget"]
    assert all(row["final_evaluation"]["status"] == "missing" for row in result["experiments"])
    assert (folder/"夜间最终比较.md").is_file()


def test_comparison_does_not_claim_equal_budget_for_missing_or_unequal_results(tmp_path):
    import time
    states = {}
    for name, steps in (("a", 100), ("b", 200)):
        states[name] = dict(config_hash=name, config=plan_config(), status="trained", errors=[],
                           final_result={"status": "completed", "checkpoint_counters": {"env_step": steps},
                                         "report": {"summary": {"policy": {"mean_U": .5}}}})
    result = overnight.write_progress(tmp_path, {"campaign_id": "预算审查"}, states, time.monotonic(), True)
    assert result["all_final_evaluations_available"]
    assert not result["matched_final_budget"]
    assert not result["test_set_used"]
    assert not result["formal_multiseed_complete"]


def test_evaluation_worker_uses_immutable_checkpoint_and_all_validation(tmp_path, monkeypatch):
    from implementations.spectrum_tsac_20260921 import evaluate
    captured = {}

    def fake_evaluate(manifest, **kwargs):
        captured.update(kwargs)
        return {"summary": {"policy": {"mean_U": .8}}, "output_dir": "fixture"}

    monkeypatch.setattr(evaluate, "evaluate_run", fake_evaluate)
    request = {"run_dir": str(tmp_path/"run"), "result_path": str(tmp_path/"结果.json"), "kind": "final",
               "checkpoint_path": str(tmp_path/"run"/"检查点存档"/"immutable"/"检查点.pt"),
               "checkpoint_counters": {"episodes": 60}, "limit": 0,
               "policies": ["policy", "random", "greedy", "equal_power"]}
    atomic_json(tmp_path/"请求.json", request)
    assert overnight.evaluation_worker(tmp_path/"请求.json") == 0
    assert captured["split"] == "validation"
    assert captured["limit"] == 0
    assert captured["checkpoint_path"] == request["checkpoint_path"]
    assert overnight.read_json(tmp_path/"结果.json")["status"] == "completed"


@pytest.mark.parametrize("different_budgets", [False, True])
def test_simulated_success_includes_final_validation_and_durable_copy(tmp_path, monkeypatch, different_budgets):
    output, durable = tmp_path/"runs", tmp_path/"durable"
    monkeypatch.setattr(overnight, "SPECTRUM_OUTPUT_DIR", output)
    plan = overnight.normalize_plan({"campaign_id": "模拟成功", "max_workers": 3, "poll_seconds": .001, "matched_budget_fallback": True,
        "experiments": [{"name": str(i), "config": plan_config(eta)} for i, eta in enumerate((.2, .5, .8))]}, tmp_path)
    training_count, evaluations = [0], []

    class DoneChild:
        def poll(self):
            return 0

    def fake_launch(command, log_path, cpu=False):
        if "--config" in command:
            config = Config.from_dict(overnight.read_json(command[-1]))
            training_count[0] += 1
            run = write_run(output, "run"+str(training_count[0]), overnight.config_hash(config))
            with database(run):
                pass
            archive = run/"检查点存档"/"immutable"
            archive.mkdir(parents=True)
            checkpoint = archive/"检查点.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            final_episode = 60+training_count[0] if different_budgets else 60
            atomic_json(archive/"检查点清单.json", {"file": checkpoint.name,
                        "counters": {"episodes": final_episode, "env_step": final_episode*100, "update_step": final_episode*100-1000}})
            early = run/"检查点存档"/"early"
            early.mkdir(parents=True)
            (early/"检查点.pt").write_bytes(b"early fixture")
            atomic_json(early/"检查点清单.json", {"file": "检查点.pt",
                        "counters": {"episodes": 25, "env_step": 2500, "update_step": 1500}})
        elif "--copy-worker" in command:
            assert cpu is True
            assert overnight.copy_worker(command[-1]) == 0
        else:
            assert cpu is True
            request = overnight.read_json(command[-1])
            evaluations.append(request)
            atomic_json(request["result_path"], {"status": "completed", "kind": request["kind"],
                        "checkpoint_counters": request["checkpoint_counters"],
                        "report": {"summary": {"policy": {"mean_U": .7}}}})
        return DoneChild()

    monkeypatch.setattr(overnight, "launch", fake_launch)
    monkeypatch.setattr(overnight.time, "sleep", lambda seconds: None)
    folder, result = overnight.supervise(plan, durable)
    assert len(evaluations) == (6 if different_budgets else 3)
    assert all(item["limit"] == 0 for item in evaluations)
    assert result["matched_final_budget"] is not different_budgets
    assert result["matched_comparison"]["status"] == ("completed" if different_budgets else "not_needed")
    if different_budgets:
        matched = [row for row in evaluations if row["kind"] == "matched"]
        assert len(matched) == 3
        assert all(row["policies"] == ["policy"] and row["checkpoint_counters"]["episodes"] == 25 for row in matched)
    assert result["all_final_evaluations_available"]
    assert result["all_training_successful"]
    assert result["all_durable_copies_available"]
    assert all(Path(row["durable_copy"], DB_NAME).is_file() for row in result["experiments"])


def test_failed_child_launch_is_reported_and_does_not_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight, "SPECTRUM_OUTPUT_DIR", tmp_path/"runs")
    plan = overnight.normalize_plan({"campaign_id": "启动失败", "experiments": [{"name": "a", "config": plan_config()}]}, tmp_path)

    def fail(*args, **kwargs):
        raise OSError("fixture launch failure")

    monkeypatch.setattr(overnight, "launch", fail)
    folder, result = overnight.supervise(plan, tmp_path/"durable")
    assert result["finished"]
    assert not result["all_training_successful"]
    assert result["experiments"][0]["status"] == "failed"
    assert "启动失败" in result["experiments"][0]["errors"][0]


def test_launch_real_child_writes_log_and_constrains_cpu_threads(tmp_path):
    import sys
    log = tmp_path/"真实子进程日志.txt"
    child = overnight.launch([sys.executable, "-c",
        "import os,json;print(json.dumps({k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}))"], log, cpu=True)
    assert child.wait(timeout=15) == 0
    values = json.loads(log.read_text(encoding="utf-8"))
    assert values == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def test_common_checkpoint_uses_largest_shared_equal_budget(tmp_path):
    states = {}
    for name, episodes in (("a", [25, 50, 66]), ("b", [25, 50, 70]), ("c", [25, 50, 73])):
        run = tmp_path/name
        states[name] = {"run_dir": run}
        for episode in episodes:
            directory = run/"检查点存档"/str(episode)
            directory.mkdir(parents=True)
            (directory/"检查点.pt").write_bytes(b"fixture")
            atomic_json(directory/"检查点清单.json", {"file": "检查点.pt",
                        "counters": {"episodes": episode, "env_step": episode*100, "update_step": episode*100-1000}})
    selected = overnight.common_checkpoint_selection(states)
    assert all(row[0] == 50 for row in selected.values())
    sidecar = tmp_path/"b"/"检查点存档"/"50"/"检查点清单.json"
    value = overnight.read_json(sidecar)
    value["counters"]["update_step"] += 1
    atomic_json(sidecar, value)
    selected = overnight.common_checkpoint_selection(states)
    assert all(row[0] == 25 for row in selected.values())


def test_failed_matching_cannot_report_campaign_success():
    result = dict(all_final_evaluations_available=True, all_training_successful=True, all_durable_copies_available=True)
    for status in ("failed", "unavailable", "scheduled"):
        assert not overnight.campaign_successful(dict(result, matched_comparison={"status": status}))
    for status in ("disabled", "not_needed", "completed"):
        assert overnight.campaign_successful(dict(result, matched_comparison={"status": status}))
