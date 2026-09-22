"""仅恢复一个未完成的Spectrum seed43 trial；不重启旧CNN或已耗尽预算任务。"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import uuid


MODULE = "implementations.spectrum_tsac_20260921"
DB_NAME = "运行记录.sqlite"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def node_path(path, plan):
    path = Path(path).resolve()
    if not inside(path, plan["node_project"]) or inside(path, plan["durable_project"]) or path.as_posix().startswith("/home/"):
        raise ValueError("活跃训练/评价只能使用指定节点本地目录："+str(path))
    return path


def normalize_plan(raw, base):
    plan = dict(raw)
    for key in ("config_path", "resume_checkpoint", "node_project", "durable_project", "control_dir"):
        path = Path(plan[key])
        plan[key] = str((path if path.is_absolute() else Path(base)/path).resolve())
    count, wall = plan["remaining_episodes"], plan["remaining_wall_seconds"]
    if isinstance(count, bool) or not isinstance(count, int) or count > 150:
        raise ValueError("恢复回合数必须为整数且不超过150")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall > 5940:
        raise ValueError("恢复剩余墙钟须有限且不超过5940秒")
    plan.setdefault("poll_seconds", 30)
    if isinstance(plan["poll_seconds"], bool) or not isinstance(plan["poll_seconds"], (int, float)) or not 0 < plan["poll_seconds"] <= 60:
        raise ValueError("poll_seconds必须在(0,60]内")
    node_path(plan["node_project"], plan)
    evaluations = []
    for index, item in enumerate(plan.get("cpu_evaluations", [])):
        request = dict(item)
        if request.get("split", "validation") != "validation" or type(request.get("limit", 0)) is not int or request.get("limit", 0) != 0 or request.get("policies", ["policy"]) != ["policy"]:
            raise ValueError("恢复评价仅允许validation、limit=0、policy-only")
        request.setdefault("name", "参考评价-"+str(index+1))
        if not isinstance(request["name"], str) or not request["name"] or any(c in request["name"] for c in "/\\\0:"):
            raise ValueError("评价名称必须是安全单文件名")
        for key in ("run_dir", "checkpoint_path", "durable_eval_dir"):
            path = Path(request[key])
            request[key] = str((path if path.is_absolute() else Path(base)/path).resolve())
        node_path(request["run_dir"], plan)
        request.update(split="validation", limit=0, policies=["policy"], kind="reference")
        evaluations.append(request)
    if len({r["name"] for r in evaluations}) != len(evaluations):
        raise ValueError("参考评价名称重复")
    plan["cpu_evaluations"] = evaluations
    return plan


def load_helpers(plan):
    import importlib
    sys.path.insert(0, plan["node_project"])
    paths = importlib.import_module("project_paths")
    if Path(paths.PROJECT_ROOT).resolve() != Path(plan["node_project"]):
        raise ValueError("导入的算法包不是指定节点部署")
    overnight = importlib.import_module(MODULE+".overnight")
    artifacts = importlib.import_module(MODULE+".artifacts")
    config = importlib.import_module(MODULE+".config")
    train = importlib.import_module(MODULE+".train")
    return SimpleNamespace(output_root=paths.SPECTRUM_OUTPUT_DIR, launch=overnight.launch,
        stop_child=overnight.stop_child, read_progress=overnight.read_progress,
        sync_published=overnight.sync_published, checkpoints=overnight.checkpoints,
        copy_closed_tree=overnight.copy_closed_tree, load_checkpoint=artifacts.load_checkpoint,
        config_hash=artifacts.config_hash, load_config=config.load_config, prepare_dataset=train.prepare_dataset)


def assert_closed(root, require_database=True):
    databases = list(Path(root).rglob(DB_NAME))
    if require_database and not databases:
        raise ValueError("待复制子树没有关闭数据库证据")
    for path in databases:
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as connection:
            statuses = [row[0] for row in connection.execute("SELECT status FROM runs")]
        if not statuses or any(value not in ("completed", "failed", "interrupted") for value in statuses):
            raise ValueError("拒绝复制含活跃数据库的目录："+str(path))


def checkpoint_metadata(path, expected_hash=None):
    path = Path(path)
    metadata = read_json(path.parent/"检查点清单.json")
    if metadata.get("file") != path.name or sha256(path) != metadata.get("sha256"):
        raise ValueError("检查点与sidecar内容SHA不符")
    if expected_hash and metadata["sha256"] != expected_hash:
        raise ValueError("检查点不符合恢复计划冻结SHA")
    if metadata.get("boundary") != "episode":
        raise ValueError("只允许完整回合检查点")
    return metadata


def verify_source(plan, helpers):
    config = helpers.load_config(plan["config_path"])
    if config.train.seed != 43 or config.model.encoder != "cnn_attention_residual" or not config.train.device.startswith("cuda"):
        raise ValueError("此恢复入口只允许Spectrum残差seed43，不能重启CNN或seed42")
    if config.train.episodes != plan["remaining_episodes"] or not math.isclose(config.train.max_wall_seconds, plan["remaining_wall_seconds"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("config中的新增回合/剩余墙钟须与恢复计划完全一致")
    dataset = helpers.prepare_dataset(config)
    sidecar = checkpoint_metadata(plan["resume_checkpoint"], plan["checkpoint_sha256"])
    payload = helpers.load_checkpoint(plan["resume_checkpoint"], config, dataset["manifest_hash"])
    for key, expected in (("run_id", plan["source_run_id"]), ("trial_id", plan["source_trial_id"]),
                          ("checkpoint_id", plan["source_checkpoint_id"])):
        if payload.get(key) != expected or sidecar.get(key) != expected:
            raise ValueError("来源检查点身份不匹配："+key)
    if payload["counters"] != sidecar["counters"] or payload.get("boundary") != "episode":
        raise ValueError("来源计数或恢复边界不一致")
    if payload["counters"]["episodes"]+plan["remaining_episodes"] > 300:
        raise ValueError("本次恢复不能超过原300回合上限")
    if not isinstance(payload.get("initialization"), dict) or not payload["initialization"]:
        raise ValueError("来源缺少必须保留的CNN初始化谱系")
    if config.train.max_env_steps and config.train.max_env_steps <= payload["counters"]["env_step"]:
        raise ValueError("环境步预算已经耗尽")
    return dict(run_id=payload["run_id"], trial_id=payload["trial_id"], checkpoint_id=payload["checkpoint_id"],
                counters=payload["counters"], initialization=payload["initialization"],
                data_manifest_hash=payload["data_manifest_hash"]), helpers.config_hash(config)


def verify_lineage(manifest, source, expected_hash):
    expected = {"trial_id": source["trial_id"], "parent_run_id": source["run_id"],
                "parent_checkpoint_id": source["checkpoint_id"], "initialization": source["initialization"],
                "resume_env_step": source["counters"]["env_step"], "resume_update_step": source["counters"]["update_step"],
                "data_manifest_hash": source["data_manifest_hash"], "config_hash": expected_hash,
                "warmup_collection": "initialized_policy"}
    if manifest.get("run_id") == source["run_id"] or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("恢复run谱系/初始化/计数不符，拒绝继续")


def find_run(helpers, excluded, source, expected_hash):
    matches = []
    for path in Path(helpers.output_root).glob("*/运行清单.json"):
        if path.parent.name in excluded:
            continue
        manifest = read_json(path)
        if manifest.get("mode") == "train" and manifest.get("config_hash") == expected_hash:
            # create_run首次写入尚未含initialization；等待控制器完成第二次原子发布。
            if "initialization" not in manifest:
                continue
            verify_lineage(manifest, source, expected_hash)
            matches.append(path.parent)
    if len(matches) > 1:
        raise ValueError("同配置出现多个新run，拒绝错误配对")
    return matches[0] if matches else None


def prepare_reference(request, plan):
    request = dict(request)
    run_dir = node_path(request["run_dir"], plan)
    manifest = read_json(run_dir/"运行清单.json")
    if manifest["config"]["train"]["seed"] != 42:
        raise ValueError("非训练参考仅允许seed42的250回合评价")
    assert_closed(run_dir, require_database=False)
    metadata = checkpoint_metadata(request["checkpoint_path"], request.get("checkpoint_sha256"))
    if metadata["counters"]["episodes"] != 250:
        raise ValueError("seed42参考必须使用250回合检查点")
    for key, actual in (("checkpoint_counters", metadata["counters"]), ("checkpoint_id", metadata["checkpoint_id"])):
        if key in request and request[key] != actual:
            raise ValueError("参考请求检查点预算/身份不符")
        request[key] = actual
    destination = Path(request["durable_eval_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("参考评价持久目录已非空，拒绝覆盖既有证据")
    return request


def validate_evaluation(result, request):
    report = result.get("report") or {}
    rows = report.get("results", {}).get("policy", [])
    if result.get("status") != "completed" or report.get("split") != "validation" or len(rows) != 15:
        raise ValueError("评价没有完整15景validation结果")
    if report.get("checkpoint_id") != request["checkpoint_id"] or report.get("evaluated_training_counters") != request["checkpoint_counters"]:
        raise ValueError("评价实际检查点/预算不匹配")
    if len({row.get("scenario_id") for row in rows}) != 15:
        raise ValueError("评价场景ID缺失或重复")
    for row in rows:
        value = row.get("mean_satisfaction")
        if not isinstance(row.get("scenario_id"), str) or not row["scenario_id"] or row.get("phase") != "validation" or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("评价场景/phase/U不合法")
        if type(row.get("constraint_violation_count")) is not int or row["constraint_violation_count"] != 0 or row.get("constraint_violations") != []:
            raise ValueError("评价约束记录缺失或存在违约")
    mean = sum(row["mean_satisfaction"] for row in rows)/15
    summary = report.get("summary", {}).get("policy", {})
    if summary.get("scenarios") != 15 or not isinstance(summary.get("mean_U"), (int, float)) or not math.isclose(mean, summary["mean_U"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("评价汇总与逐景不一致")
    output = Path(report["output_dir"]).resolve()
    if (Path(request["run_dir"])/"独立评估").resolve() not in output.parents:
        raise ValueError("拒绝把参考根或越界目录当作评价子树复制")
    assert_closed(output)
    return report, output


def run(plan, *, helpers=None, sleeper=time.sleep, clock=time.monotonic):
    control = Path(plan["control_dir"])
    control.mkdir(parents=True, exist_ok=True)
    progress_file = control/"恢复进度.json"
    if progress_file.exists():
        raise ValueError("已有恢复进度，拒绝重复启动；请核验原PID与记录")
    with (control/"恢复启动锁.json").open("x", encoding="utf-8") as stream:
        json.dump({"supervisor_pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
    state = {"status": "preflight", "finished": False, "training_started": False, "errors": [], "warnings": [],
             "supervisor_pid": os.getpid(), "remaining_episodes": plan["remaining_episodes"],
             "remaining_wall_seconds": plan["remaining_wall_seconds"], "node_project": plan["node_project"],
             "test_set_used": False, "evaluations": []}
    started = clock()

    def publish():
        state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=clock()-started)
        atomic_json(progress_file, state)

    publish()
    if plan["remaining_episodes"] <= 0 or plan["remaining_wall_seconds"] <= 0:
        state.update(status="skipped_no_budget", finished=True)
        publish()
        return state
    training = evaluation = None
    try:
        helpers = helpers or load_helpers(plan)
        source, expected_hash = verify_source(plan, helpers)
        state["source"] = source
        queue = [prepare_reference(request, plan) for request in plan["cpu_evaluations"]]
        Path(helpers.output_root).mkdir(parents=True, exist_ok=True)
        excluded = {p.name for p in Path(helpers.output_root).iterdir() if p.is_dir()}
        command = [sys.executable, "-m", MODULE+".train", "--mode", "train", "--config", plan["config_path"], "--resume", plan["resume_checkpoint"]]
        training = helpers.launch(command, control/"seed43恢复训练日志.txt")
        train_started = clock()
        state.update(status="running", training_started=True, training_pid=training.pid, training_command=command)
        seen, run_dir, final_scheduled = set(), None, False
        active_request = None
        publish()
        while True:
            found = find_run(helpers, excluded, source, expected_hash)
            if found:
                run_dir = node_path(found, plan)
                state["run_dir"] = str(run_dir)
                try:
                    state["progress"] = helpers.read_progress(run_dir)
                    helpers.sync_published(run_dir, Path(plan["durable_project"])/"outputs"/"spectrum_tsac_20260921"/run_dir.name, seen)
                except (OSError, sqlite3.Error) as error:
                    message = "读取/同步暂不可用："+str(error)
                    if message not in state["warnings"]:
                        state["warnings"].append(message)
            code = training.poll()
            if code is None and clock()-train_started > plan["remaining_wall_seconds"]+900:
                helpers.stop_child(training)
                state["errors"].append("训练超过剩余墙钟及900秒收尾宽限，已终止自己的子进程")
                code = training.poll()
            if code is not None:
                state["training_exit_code"] = code
                if code != 0 and "训练非零退出" not in state["errors"]:
                    state["errors"].append("训练非零退出")
                if not final_scheduled:
                    final_scheduled = True
                    saved = helpers.checkpoints(run_dir) if run_dir else []
                    if not saved:
                        state["final_evaluation"] = {"status": "missing", "error": "没有新的完整回合检查点"}
                        state["errors"].append("恢复训练未产生可评价的新检查点")
                    else:
                        _, checkpoint, metadata = saved[-1]
                        checkpoint_metadata(checkpoint, metadata["sha256"])
                        if metadata["trial_id"] != source["trial_id"] or not source["counters"]["episodes"] < metadata["counters"]["episodes"] <= source["counters"]["episodes"]+plan["remaining_episodes"]:
                            raise ValueError("新检查点trial或新增回合超出计划")
                        queue.append({"name": "seed43最终评价", "kind": "final", "run_dir": str(run_dir),
                            "checkpoint_path": str(checkpoint), "checkpoint_id": metadata["checkpoint_id"],
                            "checkpoint_counters": metadata["counters"], "split": "validation", "limit": 0, "policies": ["policy"]})
            if evaluation:
                code = evaluation.poll()
                if code is None and clock()-evaluation_started > 1800:
                    helpers.stop_child(evaluation)
                    active_request["timed_out"] = True
                    code = evaluation.poll()
                if code is not None:
                    try:
                        result = read_json(active_request["result_path"])
                        if code != 0:
                            raise ValueError("评价子进程非零退出："+str(code))
                        report, output = validate_evaluation(result, active_request)
                        destination = active_request.get("durable_eval_dir") or str(Path(plan["durable_project"])/"outputs"/"spectrum_tsac_20260921"/run_dir.name/"独立评估"/output.name)
                        helpers.copy_closed_tree(output, destination)
                        record = {"name": active_request["name"], "kind": active_request["kind"], "status": "completed", "exit_code": code,
                                  "checkpoint_counters": active_request["checkpoint_counters"], "report": report, "durable_eval_dir": destination}
                    except Exception as error:
                        record = {"name": active_request["name"], "kind": active_request["kind"], "status": "failed", "exit_code": code, "error": str(error), "timed_out": active_request.get("timed_out", False)}
                        state["errors"].append("评价失败："+active_request["name"]+"："+str(error))
                    state["evaluations"].append(record)
                    if active_request["kind"] == "final":
                        state["final_evaluation"] = record
                    evaluation, active_request = None, None
            if evaluation is None and queue:
                active_request = dict(queue.pop(0))
                label = f"{len(state['evaluations'])+1:02d}-"+active_request["name"]
                request_path = control/(label+"-评价请求.json")
                active_request["result_path"] = str(control/(label+"-评价结果.json"))
                atomic_json(request_path, active_request)
                evaluation = helpers.launch([sys.executable, "-m", MODULE+".overnight", "--evaluation-worker", str(request_path)], control/(label+"-评价日志.txt"), cpu=True)
                evaluation_started = clock()
                state["active_evaluation"] = {"name": active_request["name"], "pid": evaluation.pid}
            elif evaluation is None:
                state.pop("active_evaluation", None)
            if training.poll() is not None and final_scheduled and "final_evaluation" in state and not queue and evaluation is None:
                if run_dir:
                    try:
                        assert_closed(run_dir)
                        destination = Path(plan["durable_project"])/"outputs"/"spectrum_tsac_20260921"/run_dir.name
                        helpers.copy_closed_tree(run_dir, destination)
                        state["durable_run_dir"] = str(destination)
                    except Exception as error:
                        state["errors"].append("完整关闭run复制失败："+str(error))
                state.update(status="completed" if not state["errors"] else "failed", finished=True)
                publish()
                return state
            publish()
            sleeper(plan["poll_seconds"])
    except BaseException as error:
        if helpers:
            for child in (training, evaluation):
                if child is not None and child.poll() is None:
                    helpers.stop_child(child)
        state["errors"].append(str(error))
        state.update(status="failed", finished=True)
        publish()
        return state
    finally:
        if helpers:
            for child in (training, evaluation):
                if child is not None and child.poll() is None:
                    helpers.stop_child(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    path = Path(args.plan).resolve()
    result = run(normalize_plan(read_json(path), path.parent))
    print(json.dumps({"status": result["status"], "finished": result["finished"], "errors": result["errors"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] in ("completed", "skipped_no_budget") else 1)


if __name__ == "__main__":
    main()
