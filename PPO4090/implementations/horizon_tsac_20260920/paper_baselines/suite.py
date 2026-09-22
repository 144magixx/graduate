"""等待既有实验完整结束，再运行论文基线与冻结预算的验证/测试比较。

本模块在父任务成功门打开之前只使用标准库，不导入torch或启动训练。
所有SQLite写入都发生于节点本地；持久目录只接收已关闭评价的副本。
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid


LEARNED_IDS = ("mlp_sac", "cnn_sac", "mlp_dqn", "mlp_ppo", "tsac_205")
PAPER_IDS = ("fixed", "greedy", "random", "mlp_sac", "cnn_sac", "mlp_dqn", "mlp_ppo")
NAMES = {"fixed": "Fixed", "greedy": "Greedy", "random": "Random", "mlp_sac": "MLP-SAC",
         "cnn_sac": "CNN-SAC", "mlp_dqn": "MLP-DQN", "mlp_ppo": "MLP-PPO", "tsac_205": "T-SAC局部205控制组",
         "horizon_primary": "Horizon T-SAC主参考η=0.5", "horizon_supplemental": "Horizon T-SAC验证选择补充参考"}
DB_NAME = "运行记录.sqlite"


class SuiteBlocked(RuntimeError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    """门控阶段也只用stdlib原子保存状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe(value):
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in "/\\\0:"):
        raise ValueError("campaign_id必须为单个安全文件名")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_parent(path, wait_seconds=28800, poll_seconds=30, *, sleeper=time.sleep, clock=time.monotonic):
    """未finished不启动；finished但失败直接blocked，读取异常有限等待。"""
    for name, value in (("wait_seconds", wait_seconds), ("poll_seconds", poll_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+"必须为有限正数")
    started, last = clock(), "父进度尚不可读"
    required = ("all_training_successful", "all_final_evaluations_available", "all_durable_copies_available")
    while True:
        try:
            parent = _read(path)
            if not isinstance(parent, dict):
                raise ValueError("父进度必须为对象")
            if parent.get("finished") is True:
                failed = [key for key in required if parent.get(key) is not True]
                if failed:
                    raise SuiteBlocked("父实验已结束但未成功满足："+", ".join(failed))
                matching = parent.get("matched_comparison") or {}
                if matching and matching.get("status") not in ("disabled", "not_needed", "completed"):
                    raise SuiteBlocked("父共同预算补评未成功完成："+str(matching.get("status")))
                return parent
            last = "父实验仍未finished"
        except (OSError, ValueError) as error:
            last = str(error)
        remaining = wait_seconds-(clock()-started)
        if remaining <= 0:
            raise SuiteBlocked("等待父实验超时："+last)
        sleeper(min(poll_seconds, 60, remaining))


def normalize_suite_plan(raw, base):
    """成功门之后解析依赖，冻结五学习方法、100EP和完整评价策略。"""
    from ..config import Config, load_config
    from ..overnight import normalize_plan
    from .specs import adapt_config
    value = copy.deepcopy(raw)
    if not isinstance(value.get("final_test", False), bool):
        raise ValueError("final_test必须为冻结的bool")
    value.setdefault("final_test", False)
    value.setdefault("gpu_preflight", True)
    if not isinstance(value["gpu_preflight"], bool):
        raise ValueError("gpu_preflight必须为bool")
    value.setdefault("gpu_preflight_timeout_seconds", 600)
    timeout = value["gpu_preflight_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("GPU短验timeout须为0–600秒范围内的有限正数")
    value.setdefault("wait_seconds", 28800)
    value.setdefault("expected_eval_scenarios", 15)
    if value["expected_eval_scenarios"] != 15:
        raise ValueError("本轮必须使用全部15个validation/test场景")
    if value.get("quick_evaluation", False) or value.get("matched_budget_fallback", False):
        raise ValueError("Suite禁止quick评价与按optimizer次数匹配的旧fallback")
    value["quick_evaluation"] = False
    value["matched_budget_fallback"] = False
    algorithms = [row.get("algorithm") for row in value.get("experiments", [])]
    if sorted(algorithms) != sorted(LEARNED_IDS):
        raise ValueError("Suite必须且仅包含四个论文学习基线与tsac_205控制组，各一次")
    for row in value["experiments"]:
        source = row["config"]
        config = Config.from_dict(source) if isinstance(source, dict) else load_config(Path(base)/source)
        config = adapt_config(config, row["algorithm"])
        config.train.episodes = 100
        row["config"] = config.to_dict()
        row["settings"] = dict(row.get("settings", {}))
        row["final_policies"] = ["policy"]
    value.setdefault("heuristic_settings", {})
    if value["heuristic_settings"].get("random_include_skip", False):
        raise ValueError("论文Random不允许与含主动SKIP的诊断变体混名")
    value["interpretation"] = "五个学习方法同环境交互比较；PPO真实优化次数单列；静态三方法只评价一次。"
    value["comparison_description"] = "四个论文学习基线、T-SAC局部控制组及预登记Horizon主参考；不以测试集调参。"
    return normalize_plan(value, base)


def _full_validation(row, count=15, field="final_evaluation"):
    evaluation = row.get(field) or {}
    report = evaluation.get("report") or {}
    values = report.get("results", {}).get("policy", [])
    if evaluation.get("status") != "completed" or report.get("split") != "validation" or len(values) != count:
        raise SuiteBlocked("父参考缺少全15场景validation："+str(row.get("name")))
    scores = [value.get("mean_satisfaction") for value in values]
    if len({v.get("scenario_id") for v in values}) != count or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
        raise SuiteBlocked("父validation场景/指标不完整")
    return sum(scores)/count


def _evaluation_budget(row, field):
    evaluation = row.get(field) or {}
    budget = evaluation.get("checkpoint_counters")
    if not isinstance(budget, dict) or any(isinstance(budget.get(key), bool) or not isinstance(budget.get(key), int) or budget[key] < 0
                                            for key in ("episodes", "env_step", "update_step")):
        raise SuiteBlocked("父评价缺少合法检查点预算："+str(row.get("name")))
    reported = (evaluation.get("report") or {}).get("evaluated_training_counters")
    if reported is not None and reported != budget:
        raise SuiteBlocked("父评价内部检查点预算不一致")
    return budget


def select_parent_references(parent):
    rows = parent.get("experiments", [])
    primary = [row for row in rows if isinstance(row.get("target_entropy_ratio"), (int, float)) and math.isclose(row["target_entropy_ratio"], .5)]
    if len(primary) != 1:
        raise SuiteBlocked("父实验必须唯一包含预登记η=0.5主参考")
    budgets = [_evaluation_budget(row, "final_evaluation") for row in rows]
    field = "final_evaluation" if all(budget == budgets[0] for budget in budgets) else "matched_evaluation"
    if field == "matched_evaluation":
        if (parent.get("matched_comparison") or {}).get("status") != "completed":
            raise SuiteBlocked("父最终预算不同，必须完成共同预算全validation后才能选择补充η")
        budgets = [_evaluation_budget(row, field) for row in rows]
        if not all(budget == budgets[0] for budget in budgets):
            raise SuiteBlocked("父共同预算补评的实际预算仍不相同")
    scores = [(row, _full_validation(row, field=field)) for row in rows]
    signatures = [sorted((item["scenario_id"], item.get("n_demand"), item.get("total_demand_bps"))
                         for item in row[field]["report"]["results"]["policy"]) for row in rows]
    if not all(signature == signatures[0] for signature in signatures):
        raise SuiteBlocked("父共同预算validation场景/需求不一致")
    selected, score = min(scores, key=lambda pair: (-pair[1], pair[0]["target_entropy_ratio"], pair[0]["name"]))
    return {"primary": primary[0], "supplemental": selected,
            "supplemental_validation_mean_U": score,
            "comparison_basis": field, "comparison_budget": budgets[0],
            "selection_rule": "父同预算全15validation场景等权U最大；最终预算不同则只用matched_evaluation；同分取较小η、再按名称；不读取test指标",
            "same_reference": primary[0]["run_dir"] == selected["run_dir"]}


def _archive(run_dir):
    """只消费归档侧车和真实文件hash，不加载模型，也不用可变最新入口。"""
    run_dir = Path(run_dir).resolve()
    result = {}
    for path in run_dir.glob("检查点存档/*/检查点清单.json"):
        metadata = _read(path)
        checkpoint = (path.parent/metadata["file"]).resolve()
        if run_dir not in checkpoint.parents or not checkpoint.is_file() or _sha(checkpoint) != metadata.get("sha256"):
            raise SuiteBlocked("不可变检查点缺失或hash不一致："+str(path))
        counters = metadata.get("counters", {})
        if any(isinstance(counters.get(key), bool) or not isinstance(counters.get(key), int) or counters[key] < 0 for key in ("episodes", "env_step", "update_step")):
            raise SuiteBlocked("检查点预算字段非法")
        key = (counters["episodes"], counters["env_step"])
        row = {"path": str(checkpoint), "checkpoint_id": metadata["checkpoint_id"], "sha256": metadata["sha256"], "counters": counters}
        previous = result.get(key)
        if previous is None or (counters["update_step"], metadata["checkpoint_id"]) > (previous["counters"]["update_step"], previous["checkpoint_id"]):
            result[key] = row
    return result


def select_common_environment_budget(run_dirs):
    histories = {name: _archive(path) for name, path in run_dirs.items()}
    shared = set.intersection(*(set(rows) for rows in histories.values())) if histories else set()
    if not shared:
        return {"status": "missing", "reason": "没有共同episodes且env_step相同的不可变检查点",
                "available_budgets": {name: [list(pair) for pair in sorted(rows)] for name, rows in histories.items()}}
    episodes, env_step = max(shared)
    return {"status": "matched", "episodes": episodes, "env_step": env_step,
            "optimizer_counts_required_equal": False,
            "checkpoints": {name: rows[(episodes, env_step)] for name, rows in histories.items()}}


def _attach_durable_checkpoints(checkpoints, run_dirs, source_rows):
    for name, checkpoint in checkpoints.items():
        relative = Path(checkpoint["path"]).resolve().relative_to(Path(run_dirs[name]).resolve())
        durable = Path(source_rows[name]["durable_copy"]).resolve()/relative
        if not durable.is_file() or _sha(durable) != checkpoint["sha256"]:
            raise SuiteBlocked("所选检查点的持久副本缺失或hash不符："+name)
        checkpoint["durable_path"] = str(durable)


def _node_output_root():
    from project_paths import HORIZON_OUTPUT_DIR
    return HORIZON_OUTPUT_DIR


def _copy_closed(source, destination):
    from ..overnight import copy_closed_tree
    _assert_closed(source, require_root=False)
    return copy_closed_tree(source, destination)


def _assert_closed(run_dir, require_root=True):
    root = Path(run_dir)
    if require_root and not (root/DB_NAME).is_file():
        raise SuiteBlocked("缺少关闭运行的数据库："+str(run_dir))
    databases = list(root.rglob(DB_NAME))
    if not databases:
        raise SuiteBlocked("没有可验证的关闭运行数据库")
    for database in databases:
        connection = sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)
        try:
            statuses = [row[0] for row in connection.execute("SELECT status FROM runs")]
        finally:
            connection.close()
        if not statuses or any(status not in ("completed", "failed", "interrupted") for status in statuses):
            raise SuiteBlocked("拒绝复制/评价仍active的运行："+str(database))


def ensure_node_local(row, durable_root, campaign_id, method_id):
    """父临时目录还在则复用；否则从已关闭持久副本复制到当前本地输出目录。"""
    durable_root = Path(durable_root).resolve()
    original = Path(row["run_dir"]).resolve() if row.get("run_dir") else None
    temporary = Path(tempfile.gettempdir()).resolve()
    node_root = Path(_node_output_root()).resolve()
    if original and original.is_dir() and durable_root not in original.parents and original != durable_root and (
            temporary in original.parents or node_root in original.parents):
        _assert_closed(original)
        return original
    source = Path(row.get("durable_copy") or "").resolve()
    if not row.get("durable_copy") or row.get("durable_copy") == "failed" or not source.is_dir():
        raise SuiteBlocked("父节点目录已失效且无完整持久副本："+method_id)
    _assert_closed(source)
    target = node_root/("论文基线参考-"+_safe(campaign_id)+"-"+_safe(method_id)+"-"+uuid.uuid4().hex[:8])
    _copy_closed(source, target)
    _assert_closed(target)
    return target


def _supervise(plan, durable_root):
    from ..overnight import supervise
    return supervise(plan, durable_root)


def _preflight(plan, directory):
    """父门之后依次启动一回合GPU验收；任何失败都不进入正式supervise。"""
    from ..overnight import child_environment, stop_child
    from project_paths import PROJECT_ROOT
    folder = Path(directory)/"GPU短验"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for row in plan["experiments"]:
        algorithm = row["algorithm"]
        config = copy.deepcopy(row["config"])
        config["train"].update(episodes=1, max_env_steps=0, warmup_steps=0, update_schedule="episode",
                               updates_per_episode=1, batch_size=2, microbatch_size=2, checkpoint_every=1,
                               max_wall_seconds=300.)
        if not str(config["train"]["device"]).startswith("cuda"):
            config["train"]["device"] = "cuda"
        settings = dict(row["settings"])
        if algorithm == "mlp_ppo":
            settings.update(epochs=1, minibatch_size=64)
        config_path = folder/(algorithm+"-短验配置.json")
        settings_path = folder/(algorithm+"-算法参数.json")
        log_path = folder/(algorithm+"-短验日志.txt")
        _write(config_path, config)
        _write(settings_path, settings)
        _write(Path(directory)/"队列状态.json", {"status": "gpu_preflight", "method": algorithm})
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", "implementations.horizon_tsac_20260920.paper_baselines.train",
                    "--algorithm", algorithm, "--config", str(config_path), "--settings", str(settings_path), "--preflight"],
                    cwd=PROJECT_ROOT, env=child_environment(cpu=False), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt")
            try:
                process.wait(timeout=plan["gpu_preflight_timeout_seconds"])
            except subprocess.TimeoutExpired:
                stop_child(process)
                process.wait(timeout=30)
                raise SuiteBlocked("GPU短验超过有限timeout")
            if process.returncode != 0:
                raise SuiteBlocked("GPU短验进程失败，exit_code="+str(process.returncode))
            output = None
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and candidate.get("run_dir"):
                        output = candidate
                except ValueError:
                    continue
            if output is None:
                raise SuiteBlocked("GPU短验没有返回run_dir")
            run_dir = Path(output["run_dir"]).resolve()
            if Path(_node_output_root()).resolve() not in run_dir.parents:
                raise SuiteBlocked("GPU短验产物不在当前节点本地输出根")
            manifest = _read(run_dir/"运行清单.json")
            report = _read(run_dir/"基线训练结果.json")
            counters = report.get("counters", {})
            if manifest.get("mode") != "paper_baseline_preflight" or manifest.get("paper_baseline", {}).get("algorithm_id") != algorithm:
                raise SuiteBlocked("GPU短验未使用独立preflight模式，拒绝混入研究run")
            if counters.get("episodes") != 1 or counters.get("env_step", 0) <= 0 or counters.get("update_step", 0) <= 0:
                raise SuiteBlocked("GPU短验未完成真实回合和更新")
            if len(report.get("episodes", [])) != 1 or any(ep.get("constraint_violation_count") != 0 for ep in report["episodes"]):
                raise SuiteBlocked("GPU短验存在约束违约或缺失记录")
            _copy_closed(run_dir, folder/"运行记录"/algorithm)
            results.append({"algorithm": algorithm, "status": "completed", "run_dir": str(run_dir), "counters": counters,
                            "research_data": False, "mode": "paper_baseline_preflight"})
        except Exception as error:
            results.append({"algorithm": algorithm, "status": "failed", "reason": str(error)})
            return {"status": "failed", "experiments": results}
    return {"status": "completed", "experiments": results}


def _evaluate(request, result_dir, timeout):
    """单独CPU子进程有限等待；进程结束后调用者才能复制其SQLite。"""
    from ..overnight import child_environment, stop_child
    from project_paths import PROJECT_ROOT
    token = uuid.uuid4().hex[:10]
    job_dir = Path(result_dir)/"评价任务"
    job_dir.mkdir(parents=True, exist_ok=True)
    request_path = job_dir/(request["method_id"]+"-"+request["split"]+"-"+token+"-请求.json")
    output_path = request_path.with_name(request_path.stem.replace("请求", "结果")+".json")
    request = dict(request, result_path=str(output_path))
    _write(request_path, request)
    with request_path.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "implementations.horizon_tsac_20260920.paper_baselines.suite",
                                   "--evaluation-worker", str(request_path)], cwd=PROJECT_ROOT, env=child_environment(cpu=True),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_child(process)
        process.wait(timeout=30)
        return {"status": "failed", "reason": "评价超时，未复制可能中断的数据库"}
    if process.returncode != 0 or not output_path.is_file():
        return {"status": "failed", "reason": "评价子进程失败", "exit_code": process.returncode}
    return _read(output_path)


def evaluation_worker(path):
    request = _read(path)
    try:
        from ..evaluate import evaluate_run
        result = evaluate_run(Path(request["run_dir"])/"运行清单.json", split=request["split"], limit=0,
                              policies=tuple(request["policies"]), checkpoint_path=request["checkpoint"]["path"],
                              heuristic_settings=request.get("heuristic_settings"))
        _write(request["result_path"], {"status": "completed", "report": result})
        return 0
    except BaseException as error:
        import traceback
        _write(request["result_path"], {"status": "failed", "reason": str(error), "traceback": traceback.format_exc()})
        return 1


def _report_rows(result, policy, split, expected, checkpoint):
    report = result.get("report") or {}
    rows = report.get("results", {}).get(policy, [])
    if result.get("status") != "completed" or report.get("split") != split or len(rows) != expected:
        raise SuiteBlocked("评价缺失或不是完整冻结划分")
    if report.get("checkpoint_id") != checkpoint["checkpoint_id"] or report.get("evaluated_training_counters") != checkpoint["counters"]:
        raise SuiteBlocked("评价实际检查点/预算与预选择不一致")
    if len({row.get("scenario_id") for row in rows}) != expected:
        raise SuiteBlocked("评价场景重复或缺失")
    for row in rows:
        value = row.get("mean_satisfaction")
        if row.get("phase") != split or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SuiteBlocked("评价phase或U非法")
        if not isinstance(row.get("constraint_violation_count"), int) or row["constraint_violation_count"] < 0 or not isinstance(row.get("constraint_violations"), list):
            raise SuiteBlocked("缺少真实约束违约记录")
        if row["constraint_violation_count"] != len(row["constraint_violations"]):
            raise SuiteBlocked("约束违约计数与列表不一致")
        if not isinstance(row.get("n_demand"), int) or row["n_demand"] < 1 or any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in ("total_demand_bps", "power_budget_w")):
            raise SuiteBlocked("缺少共同场景/需求/资源口径")
    return rows


def _evaluate_split(split, selections, run_dirs, plan, directory):
    records, missing = {}, []
    tasks = [(name, ["policy"]) for name in selections]
    tasks.append(("heuristics", ["paper_fixed", "paper_greedy", "paper_random"]))
    for name, policies in tasks:
        anchor = "horizon_primary" if name == "heuristics" else name
        checkpoint = selections[anchor]
        request = {"method_id": name, "run_dir": str(run_dirs[anchor]), "split": split,
                   "policies": policies, "checkpoint": checkpoint, "heuristic_settings": plan["heuristic_settings"]}
        _write(Path(directory)/"队列状态.json", {"status": "evaluating", "split": split, "method": name})
        try:
            result = _evaluate(request, directory, plan["evaluation_timeout_seconds"])
            policy_rows = {policy: _report_rows(result, policy, split, plan["expected_eval_scenarios"], checkpoint) for policy in policies}
            output = Path(result["report"]["output_dir"]).resolve()
            if Path(run_dirs[anchor]).resolve() not in output.parents:
                raise SuiteBlocked("评价输出不在所选节点本地run目录")
            _copy_closed(output, Path(directory)/"评价记录"/split/name)
            for policy, rows in policy_rows.items():
                key = policy.removeprefix("paper_") if name == "heuristics" else name
                records[key] = {"rows": rows, "checkpoint": checkpoint, "training_required": name != "heuristics",
                                "reference_only_checkpoint": name == "heuristics", "status": "completed",
                                "durable_evaluation_manifest": str(Path(directory)/"评价记录"/split/name/policy/"运行清单.json")}
        except Exception as error:
            keys = [p.removeprefix("paper_") for p in policies] if name == "heuristics" else [name]
            missing.extend({"method": key, "reason": str(error)} for key in keys)
    return {"records": records, "missing": missing}


def summarize_split(data):
    records = data["records"]
    signatures = {}
    summary = {}
    for name, record in records.items():
        rows = record["rows"]
        signatures[name] = sorted((r["scenario_id"], r.get("n_demand"), r.get("total_demand_bps"), r.get("power_budget_w")) for r in rows)
        fields = ("sgm_mean_positive_demand", "mean_slot_sinr_db", "p05_slot_sinr_db", "delivered_bps",
                  "raw_throughput_bps", "unmet_bps", "power_used_w", "skip_fraction", "served_fraction",
                  "policy_decision_seconds", "inference_seconds", "rollout_seconds")
        means, availability = {}, {}
        for field in fields:
            values = [row.get(field) for row in rows]
            finite = [value for value in values if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)]
            availability[field] = len(finite)
            means[field] = sum(finite)/len(rows) if len(finite) == len(rows) else None
        counts = [row.get("sinr_sample_count") for row in rows]
        sinr_count = sum(counts) if all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in counts) else None
        summary[name] = {"display_name": NAMES[name], "mean_U": sum(r["mean_satisfaction"] for r in rows)/len(rows),
                         "scenarios": len(rows), "constraint_violations": sum(r.get("constraint_violation_count", 0) for r in rows),
                         "training_required": record["training_required"],
                         "scene_equal_weight_means": means, "available_scenarios_by_metric": availability,
                         "sinr_sample_count_total": sinr_count,
                         "sinr_aggregation_note": "各场景槽SINR均值及各场景P05再等权平均；不是合并槽样本后的P05",
                         "durable_evaluation_manifest": record.get("durable_evaluation_manifest"),
                         "checkpoint_counters": record["checkpoint"]["counters"] if record["training_required"] else None}
    matched = bool(signatures) and all(signature == next(iter(signatures.values())) for signature in signatures.values())
    required = set(PAPER_IDS)|{"tsac_205", "horizon_primary"}
    complete = required <= records.keys() and not data["missing"] and matched
    feasible = all(row["constraint_violations"] == 0 for row in summary.values())
    return {"methods": summary, "missing": data["missing"], "same_scenarios_and_demands": matched,
            "complete": complete and feasible, "ranking_available": False,
            "descriptive_order_available": complete and feasible,
            "ranking_scope": "formal_ranking_unavailable_single_seed_unknown_root",
            "ordered_methods": sorted(summary, key=lambda key: -summary[key]["mean_U"]) if complete and feasible else [],
            "formal_multiseed_complete": False}


def _write_summary(directory, result):
    _write(Path(directory)/"论文基线对比结果.json", result)
    lines = ["# 论文基线对比阶段结果", "", "状态："+result["status"]+"。", "",
             "主参考预登记为η=0.5；补充参考只由父全validation选择。按共同环境步比较，PPO优化次数单列。", ""]
    for split in ("validation", "test"):
        value = result.get(split)
        if not value:
            continue
        lines.append(split+"：完整性="+str(value["complete"])+"；同场景="+str(value["same_scenarios_and_demands"])+"。")
        for name, row in value["methods"].items():
            lines.append(f"- {row['display_name']}：{row['scenarios']}景，U={row['mean_U']:.8f}，违约={row['constraint_violations']}；训练计数={row['checkpoint_counters']}。")
            metrics = row["scene_equal_weight_means"]
            display = lambda key: "未记录" if metrics[key] is None else f"{metrics[key]:.6g}"
            lines.append("  场景等权：SGM="+display("sgm_mean_positive_demand")+"；SINR均值(dB)="+display("mean_slot_sinr_db")
                         +"；各景P05均值(dB)="+display("p05_slot_sinr_db")+"；槽样本总数="+str(row["sinr_sample_count_total"])
                         +"；交付/原始吞吐(bps)="+display("delivered_bps")+"/"+display("raw_throughput_bps")
                         +"；功率(W)="+display("power_used_w")+"；SKIP比例="+display("skip_fraction")
                         +"；决策时间(s)="+display("policy_decision_seconds")+"。")
        for missing in value["missing"]:
            lines.append("- 缺失 "+missing["method"]+"："+missing["reason"])
        lines.append("")
    lines.extend(["单种子比较不构成正式多seed结论；正式ranking_available始终为false，描述性数值排序不代表统计显著。",
                  "测试集不用于选择算法、η、检查点或规则参数；参数量与未匹配成本保留在各评价运行清单，不虚构预算对齐。",
                  "预算/场景不齐或任一方法缺失时不输出最终排名；父补充未达公共预算时仅保留validation诊断。"])
    if result.get("reason"):
        lines.append("原因："+result["reason"])
    (Path(directory)/"论文基线对比报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _finish(directory, result):
    _write(Path(directory)/"队列状态.json", {"status": result["status"], "test_set_used": result["test_set_used"]})
    _write_summary(directory, result)
    return directory, result


def run_suite(raw_plan, durable_root, parent_progress, plan_base=None):
    raw_plan = copy.deepcopy(raw_plan)
    campaign = _safe(raw_plan.get("campaign_id", "论文基线-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    raw_plan["campaign_id"] = campaign
    directory = Path(durable_root).resolve()/"论文基线对比"/campaign
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory/"提交计划.json", raw_plan)
    _write(directory/"队列状态.json", {"status": "waiting_parent", "parent_progress": str(parent_progress)})
    result = {"campaign_id": campaign, "status": "blocked", "test_set_used": False, "formal_multiseed_complete": False}
    try:
        # 在此之前没有导入numpy/torch、解析模型配置或启动任何子进程。
        parent = wait_for_parent(parent_progress, raw_plan.get("wait_seconds", 28800), raw_plan.get("poll_seconds", 30))
        _write(directory/"父实验完成快照.json", parent)
        references = select_parent_references(parent)
        plan = normalize_suite_plan(raw_plan, plan_base or Path.cwd())
        _write(directory/"冻结实验计划.json", plan)
        _write(directory/"父参考预选择.json", references)
        if plan["gpu_preflight"]:
            preflight = _preflight(plan, directory)
            _write(directory/"GPU短验结果.json", preflight)
            result["gpu_preflight"] = preflight
            checks = preflight.get("experiments", [])
            if preflight.get("status") != "completed" or len(checks) != 5 or {row.get("algorithm") for row in checks} != set(LEARNED_IDS) or any(row.get("status") != "completed" for row in checks):
                raise SuiteBlocked("GPU短验未全部成功，未启动正式学习基线")
        _write(directory/"队列状态.json", {"status": "training_baselines", "parent_gate_passed": True})
        campaign_dir, children = _supervise(plan, durable_root)
        result["training_campaign_dir"] = str(campaign_dir)
        _write(directory/"学习基线训练结果.json", children)
        wanted = {row["name"]: row["algorithm"] for row in plan["experiments"]}
        rows = {wanted[row["name"]]: row for row in children.get("experiments", []) if row.get("name") in wanted}
        failures = [name for name in LEARNED_IDS if name not in rows or rows[name].get("exit_code") != 0
                    or (rows[name].get("final_evaluation") or {}).get("status") != "completed"
                    or rows[name].get("durable_copy") in (None, "failed")]
        if failures:
            result.update(status="missing", reason="学习方法训练/评价/持久化失败", missing_methods=failures)
            return _finish(directory, result)
        run_dirs = {name: ensure_node_local(row, durable_root, campaign, name) for name, row in rows.items()}
        run_dirs["horizon_primary"] = ensure_node_local(references["primary"], durable_root, campaign, "horizon_primary")
        from ..research_stats import compatibility
        compatible = compatibility([{"manifest": _read(path/"运行清单.json")} for path in run_dirs.values()])
        if not compatible["compatible"]:
            result.update(status="missing", reason="物理/数据/评价口径不兼容", compatibility=compatible)
            return _finish(directory, result)
        selection = select_common_environment_budget(run_dirs)
        result["matched_environment_budget"] = selection
        if selection["status"] != "matched":
            result.update(status="missing", reason=selection["reason"])
            return _finish(directory, result)
        checkpoints = selection["checkpoints"]
        checkpoint_sources = dict(rows, horizon_primary=references["primary"])
        supplemental = {"status": "same_as_primary" if references["same_reference"] else "validation_diagnostic_only",
                        "parent_validation_basis": references["comparison_basis"],
                        "parent_selection_validation": references["supplemental"][references["comparison_basis"]],
                        "parent_final_validation": references["supplemental"]["final_evaluation"]}
        if not references["same_reference"]:
            try:
                path = ensure_node_local(references["supplemental"], durable_root, campaign, "horizon_supplemental")
                candidate = _archive(path).get((selection["episodes"], selection["env_step"]))
                if candidate:
                    check = compatibility([{"manifest": _read(run_dirs["horizon_primary"]/"运行清单.json")}, {"manifest": _read(path/"运行清单.json")}])
                    if check["compatible"]:
                        checkpoints["horizon_supplemental"] = candidate
                        run_dirs["horizon_supplemental"] = path
                        checkpoint_sources["horizon_supplemental"] = references["supplemental"]
                        supplemental["status"] = "matched_supplemental"
            except (SuiteBlocked, OSError, ValueError) as error:
                supplemental["reason"] = str(error)
        _attach_durable_checkpoints(checkpoints, run_dirs, checkpoint_sources)
        frozen_choice = {"selection_rule": references["selection_rule"], "primary_eta": .5,
                         "supplemental_eta": references["supplemental"]["target_entropy_ratio"],
                         "supplemental": supplemental, "matched_budget": selection,
                         "methods": list(checkpoints)+list(PAPER_IDS[:3]), "heuristic_settings": plan["heuristic_settings"],
                         "final_test": plan["final_test"], "test_set_consulted": False,
                         "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write(directory/"测试前选择清单.json", frozen_choice)
        result["supplemental"] = supplemental
        validation = _evaluate_split("validation", checkpoints, run_dirs, plan, directory)
        _write(directory/"验证集逐场景结果.json", validation)
        result["validation"] = summarize_split(validation)
        if plan["final_test"] and result["validation"]["complete"]:
            # 不根据刚得到的validation分数改方法、参数、checkpoint或补充参考。
            result["test_set_used"] = True
            test = _evaluate_split("test", checkpoints, run_dirs, plan, directory)
            _write(directory/"测试集逐场景结果.json", test)
            result["test"] = summarize_split(test)
        result["status"] = "completed" if result["validation"]["complete"] and (not plan["final_test"] or result.get("test", {}).get("complete")) else "missing"
        if plan["final_test"] and not result["validation"]["complete"]:
            result["test_skipped_reason"] = "共同预算validation不完整，未打开test"
    except Exception as error:
        result.update(status="blocked", reason=str(error))
    return _finish(directory, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--durable-root")
    parser.add_argument("--parent-progress")
    parser.add_argument("--evaluation-worker")
    args = parser.parse_args()
    if args.evaluation_worker:
        raise SystemExit(evaluation_worker(args.evaluation_worker))
    if not all((args.plan, args.durable_root, args.parent_progress)):
        parser.error("需要--plan --durable-root --parent-progress")
    plan_path = Path(args.plan).resolve()
    directory, result = run_suite(_read(plan_path), args.durable_root, args.parent_progress, plan_path.parent)
    print(json.dumps({"directory": str(directory), "status": result["status"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
