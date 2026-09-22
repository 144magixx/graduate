"""有界夜间并行训练、只读进度、独立验证与持久副本。"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from project_paths import PROJECT_ROOT, SPECTRUM_OUTPUT_DIR
from .artifacts import atomic_json, config_hash
from .audit import sha256
from .config import Config, load_config
from .telemetry.storage import DB_NAME, readonly


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_name(value):
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in '/\\\x00:'):
        raise ValueError("实验与批次名称必须是单个安全文件名")
    return value


def normalize_plan(value, base):
    if isinstance(value, list):
        value = {"experiments": value}
    result = dict(value)
    result.setdefault("campaign_id", "夜间超参-"+datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    safe_name(result["campaign_id"])
    result.setdefault("max_workers", 3)
    result.setdefault("poll_seconds", 30)
    result.setdefault("evaluation_timeout_seconds", 3600)
    result.setdefault("copy_timeout_seconds", 900)
    result.setdefault("training_grace_seconds", 1800)
    result.setdefault("matched_budget_fallback", False)
    result.setdefault('quick_evaluation',True)
    result.setdefault('quick_evaluation_limit', 3)
    result.setdefault('quick_evaluation_policies', ['policy', 'random'])
    result.setdefault('quick_backup', True)
    if isinstance(result['quick_evaluation_limit'], bool) or not isinstance(result['quick_evaluation_limit'], int) or result['quick_evaluation_limit'] < 0:
        raise ValueError('quick_evaluation_limit必须为非负整数；0表示完整validation')
    if not isinstance(result['quick_backup'], bool):
        raise ValueError('quick_backup必须为布尔值')
    if not isinstance(result['quick_evaluation_policies'], list) or not result['quick_evaluation_policies'] or any(p not in ('policy','random','greedy','equal_power') for p in result['quick_evaluation_policies']):
        raise ValueError('quick评价策略不合法')
    for key in ("max_workers", "poll_seconds", "evaluation_timeout_seconds", "copy_timeout_seconds", "training_grace_seconds"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key}必须为正数")
    if not isinstance(result["max_workers"], int):
        raise ValueError("max_workers必须为整数")
    experiments = result.get("experiments", [])
    if not experiments:
        raise ValueError("夜间计划缺少experiments")
    normalized, names, hashes = [], set(), set()
    for row in experiments:
        name = safe_name(row["name"])
        if name in names:
            raise ValueError("实验名称重复")
        raw = row["config"]
        config = Config.from_dict(raw) if isinstance(raw, dict) else load_config(Path(base)/raw)
        if config.train.max_wall_seconds <= 0:
            raise ValueError("夜间实验必须配置有限正max_wall_seconds")
        if not config.telemetry.enabled:
            raise ValueError("夜间监督需要启用telemetry，以保存可审计进度")
        digest = config_hash(config)
        if digest in hashes:
            raise ValueError("重复配置hash无法唯一配对子进程，请使用不同实验配置")
        names.add(name)
        hashes.add(digest)
        algorithm=row.get('algorithm','horizon')
        if algorithm!='horizon':
            from .paper_baselines.specs import LEARNED
            if algorithm not in LEARNED:raise ValueError('未知论文学习基线')
        policies=row.get('final_policies',["policy", "random", "greedy", "equal_power"])
        if not isinstance(policies,list) or not policies or any(p not in ('policy','random','greedy','equal_power','paper_fixed','paper_greedy','paper_random') for p in policies):raise ValueError('不支持的评价策略')
        initialization = row.get('initialize_from')
        if initialization is not None:
            if algorithm != 'horizon' or not isinstance(initialization, str) or not initialization:
                raise ValueError('仅主训练入口支持非空initialize_from路径')
            source = Path(initialization)
            initialization = str(source if source.is_absolute() else (Path(base)/source).resolve())
        normalized.append({"name": name, "config": config.to_dict(), "config_hash": digest,
                           'algorithm':algorithm,'settings':dict(row.get('settings',{})),'final_policies':policies,
                           'initialize_from': initialization})
    result["experiments"] = normalized
    if result.get('paired_transfer', False):
        if not all(row['algorithm'] == 'horizon' and row['initialize_from'] for row in normalized):
            raise ValueError('配对迁移实验必须全部使用主入口及相同父检查点')
        if len({row['initialize_from'] for row in normalized}) != 1:
            raise ValueError('配对迁移实验必须来自同一父检查点')
        grouped = {}
        for row in normalized:
            grouped.setdefault(row['config']['train']['seed'], []).append(row)
        for seed, pair in grouped.items():
            if len(pair) != 2 or {row['config']['model']['encoder'] for row in pair} != {'cnn_local', 'cnn_attention_residual'}:
                raise ValueError(f'种子{seed}需要恰好一组CNN对照和一组残差模型')
            comparable = []
            for row in pair:
                item = json.loads(json.dumps(row['config']))
                item['model'].pop('encoder')
                item['model'].pop('model_version')
                comparable.append(item)
            if comparable[0] != comparable[1]:
                raise ValueError(f'种子{seed}的配对配置除结构及模型版本外不得不同')
        if result['quick_evaluation_limit'] != 0 or result['quick_evaluation_policies'] != ['policy'] or any(row['final_policies'] != ['policy'] for row in normalized):
            raise ValueError('配对修复必须使用全validation且只评价同一policy协议')
    return result


def match_runs(output_root, experiments, excluded=()):
    """只匹配本次启动后出现且配置hash唯一的训练run。"""
    excluded = set(excluded)
    wanted = {row["config_hash"]: row["name"] for row in experiments}
    candidates = {}
    for path in Path(output_root).glob("*/运行清单.json"):
        if path.parent.name in excluded:
            continue
        try:
            manifest = read_json(path)
            if manifest.get("mode") not in ("train","paper_baseline") or manifest.get("config_hash") not in wanted:
                continue
            name = wanted[manifest["config_hash"]]
            candidates.setdefault(name, []).append(path.parent)
        except (OSError, ValueError):
            continue
    ambiguous = [name for name, paths in candidates.items() if len(paths) != 1]
    if ambiguous:
        raise ValueError("同一配置出现多个新run，拒绝错误配对："+", ".join(ambiguous))
    return {name: paths[0] for name, paths in candidates.items()}


def read_progress(run_dir):
    result = {"run_id": Path(run_dir).name, "status": "starting", "episodes": 0,
              "env_step": 0, "update_step": 0, "last_U": None, "errors": []}
    database = Path(run_dir)/DB_NAME
    if not database.is_file():
        return result
    with closing(readonly(database)) as conn:
        row = conn.execute("SELECT status FROM runs LIMIT 1").fetchone()
        if row:
            result["status"] = row[0]
        row = conn.execute("SELECT COUNT(*) FROM episodes WHERE phase='train' AND status='completed'").fetchone()
        result["episodes"] = row[0]
        row = conn.execute("SELECT MAX(event_seq),MAX(env_step),MAX(update_step) FROM events WHERE phase='train' OR phase IS NULL").fetchone()
        result.update(event_seq=row[0] or 0, env_step=row[1] or 0, update_step=row[2] or 0)
        row = conn.execute("SELECT summary FROM episodes WHERE phase='train' AND summary IS NOT NULL ORDER BY final_event_seq DESC LIMIT 1").fetchone()
        if row:
            metrics = json.loads(row[0])
            result.update(last_U=metrics.get("mean_satisfaction"), last_episode=metrics,
                          env_step=max(result["env_step"], metrics.get("env_step", 0)),
                          update_step=max(result["update_step"], metrics.get("update_step", 0)))
        result["errors"] = [dict(row) for row in conn.execute(
            "SELECT event_seq,event_type,payload FROM events WHERE severity='ERROR' ORDER BY event_seq DESC LIMIT 3")]
    return result


def copy_verified(source, destination, digest=None):
    source, destination = Path(source), Path(destination)
    digest = digest or sha256(source)
    if destination.exists() and destination.stat().st_size == source.stat().st_size and sha256(destination) == digest:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+"."+uuid.uuid4().hex+".tmp")
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != digest:
            raise ValueError("持久副本hash校验失败："+str(source))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_published(run_dir, destination, seen):
    """活跃训练期间只复制数据库已发布的不可变文件；不复制活跃SQLite/WAL。"""
    run_dir, destination = Path(run_dir).resolve(), Path(destination)
    database = run_dir/DB_NAME
    if not database.is_file():
        return
    with closing(readonly(database)) as conn:
        rows = list(conn.execute("SELECT path,sha256 FROM artifacts WHERE status='available'"))
    for row in rows:
        relative, digest = row["path"], row["sha256"]
        if (relative, digest) in seen:
            continue
        source = (run_dir/relative).resolve()
        if run_dir not in source.parents or source.suffix == ".sqlite" or source.name.endswith(("-wal", "-shm")):
            raise ValueError("拒绝复制不安全或活跃数据库工件")
        if sha256(source) != digest:
            raise ValueError("已发布工件hash漂移："+relative)
        copy_verified(source, destination/relative, digest)
        seen.add((relative, digest))


def checkpoints(run_dir):
    rows = []
    for sidecar in Path(run_dir).glob("检查点存档/*/检查点清单.json"):
        try:
            metadata = read_json(sidecar)
            checkpoint = sidecar.parent/metadata["file"]
            if checkpoint.is_file():
                rows.append((metadata["counters"]["episodes"], checkpoint, metadata))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(rows, key=lambda row: row[0])


def common_checkpoint_selection(states):
    """选择各组共同保存的最大完整回合，并核验环境步/更新步相同。"""
    histories = {name: {row[0]: row for row in checkpoints(state["run_dir"])}
                 for name, state in states.items() if state.get("run_dir")}
    if len(histories) != len(states) or any(not rows for rows in histories.values()):
        return None
    shared = set.intersection(*(set(rows) for rows in histories.values()))
    for episode in sorted(shared, reverse=True):
        selection = {name: rows[episode] for name, rows in histories.items()}
        budgets = [row[2]["counters"] for row in selection.values()]
        if all(budget == budgets[0] for budget in budgets):
            return selection
    return None


def copy_closed_tree(source, destination):
    """调用者保证相关训练/评价进程已结束；数据库做闭合备份后复制。"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("持久副本不得位于源目录内")
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(("-wal", "-shm", ".tmp")):
            continue
        out = destination/path.relative_to(source)
        if path.suffix == ".sqlite":
            local = path.with_name("闭合备份-"+uuid.uuid4().hex+".sqlite.tmp")
            try:
                with closing(readonly(path)) as db, closing(sqlite3.connect(str(local))) as backup:
                    db.backup(backup)
                copy_verified(local, out)
            finally:
                if local.exists():
                    local.unlink()
        else:
            copy_verified(path, out)


def evaluation_worker(request_path):
    request = read_json(request_path)
    output = Path(request["result_path"])
    try:
        if request.get("backup_path"):
            from .telemetry.recorder import export_run
            backup = export_run(request["run_dir"], request["backup_path"])
            copy_closed_tree(backup, request["durable_backup_path"])
        from .evaluate import evaluate_run
        result = evaluate_run(Path(request["run_dir"])/"运行清单.json", split="validation",
                              limit=request["limit"], policies=tuple(request["policies"]),
                              checkpoint_path=request["checkpoint_path"])
        atomic_json(output, {"status": "completed", "kind": request["kind"],
                            "checkpoint_counters": request["checkpoint_counters"], "report": result})
        return 0
    except BaseException as error:
        import traceback
        atomic_json(output, {"status": "failed", "kind": request["kind"], "error": str(error),
                            "traceback": traceback.format_exc()})
        return 1


def copy_worker(request_path):
    request = read_json(request_path)
    try:
        copy_closed_tree(request["source"], request["destination"])
        atomic_json(request["result_path"], {"status": "completed", "destination": request["destination"]})
        return 0
    except BaseException as error:
        atomic_json(request["result_path"], {"status": "failed", "error": str(error)})
        return 1


def paired_transfer_comparisons(plan, rows, kind):
    """仅在同续训随机流、同新增交互预算和全部相同validation场景上比较。"""
    if not plan.get('paired_transfer', False):
        return {}
    from .evaluate import paired_summary
    lookup = {row['name']: row for row in rows}
    grouped = {}
    for experiment in plan['experiments']:
        grouped.setdefault(str(experiment['config']['train']['seed']), {})[experiment['config']['model']['encoder']] = lookup[experiment['name']]
    result = {}
    for seed, pair in grouped.items():
        left, right = pair['cnn_local'].get(kind), pair['cnn_attention_residual'].get(kind)
        if any(not value or value.get('status') != 'completed' for value in (left, right)):
            result[seed] = {'status': 'pending'}
            continue
        if left['checkpoint_counters'] != right['checkpoint_counters']:
            result[seed] = {'status': 'unmatched_budget', 'cnn_budget': left['checkpoint_counters'], 'residual_budget': right['checkpoint_counters']}
            continue
        reports = [left['report'], right['report']]
        values = [report.get('results', {}).get('policy', []) for report in reports]
        try:
            signatures = [{(item['scenario_id'], item['n_demand'], item['total_demand_bps']) for item in items} for items in values]
        except (KeyError, TypeError):
            result[seed] = {'status': 'incompatible_evaluation'}
            continue
        if any(report.get('split') != 'validation' for report in reports) or any(len(items) != 15 for items in values) or any(len({item['scenario_id'] for item in items}) != 15 for items in values) or len(signatures[0]) != 15 or signatures[0] != signatures[1]:
            result[seed] = {'status': 'incompatible_evaluation'}
            continue
        if any(item.get('phase') != 'validation' or isinstance(item.get('mean_satisfaction'), bool)
               or not isinstance(item.get('mean_satisfaction'), (int, float))
               or not math.isfinite(item['mean_satisfaction']) or not 0 <= item['mean_satisfaction'] <= 1
               for items in values for item in items):
            result[seed] = {'status': 'incompatible_evaluation'}
            continue
        if any(type(item.get('constraint_violation_count')) is not int or item['constraint_violation_count'] != 0
               or item.get('constraint_violations') != [] for items in values for item in items):
            result[seed] = {'status': 'constraint_violation'}
            continue
        paired = paired_summary(values[1], values[0])
        if paired.get('n_pairs') != 15:
            result[seed] = {'status': 'incompatible_evaluation'}
            continue
        result[seed] = {'status': 'comparable', 'additional_training_budget': left['checkpoint_counters'],
                        'cnn_mean_U': reports[0]['summary']['policy']['mean_U'],
                        'residual_mean_U': reports[1]['summary']['policy']['mean_U'],
                        'residual_minus_cnn': paired,
                        'shared_pretrained_seed': True, 'formal_independent_multiseed': False,
                        'checkpoint_ids': [report['checkpoint_id'] for report in reports]}
    return result


def write_progress(campaign_dir, plan, states, started, finished=False, matching=None):
    rows = []
    for name, state in states.items():
        rows.append({"name": name, "config_hash": state["config_hash"],
                     "target_entropy_ratio": state["config"]["train"]["target_entropy_ratio"],
                     "seed": state["config"]["train"]["seed"],
                     "status": state["status"], "exit_code": state.get("exit_code"),
                     "run_dir": str(state["run_dir"]) if state.get("run_dir") else None,
                     "progress": state.get("progress"), "quick_evaluation": state.get("quick_result"),
                     "final_evaluation": state.get("final_result"),
                     "matched_evaluation": state.get("matched_result"),
                     "durable_copy": state.get("durable_copy"), "errors": state["errors"]})
    result = {"campaign_id": plan["campaign_id"], "updated_at_utc": utc(),
              "elapsed_seconds": time.monotonic()-started, "finished": finished,
              "experiments": rows, "test_set_used": False, "formal_multiseed_complete": False,
              "interpretation": plan.get('interpretation',"同一种子超参数探索；验证集用于选参，不能据此声称五种子论文结论。")}
    result["matched_comparison"] = matching
    complete = [row for row in rows if (row.get("final_evaluation") or {}).get("status") == "completed"]
    budgets = [row["final_evaluation"]["checkpoint_counters"] for row in complete]
    result["all_final_evaluations_available"] = len(complete) == len(rows)
    result["all_training_successful"] = all(row["exit_code"] == 0 for row in rows)
    result["all_durable_copies_available"] = all(row["durable_copy"] not in (None, "failed") for row in rows)
    result["matched_final_budget"] = bool(budgets) and len(complete) == len(rows) and all(b == budgets[0] for b in budgets)
    result['matched_environment_budget']=bool(budgets) and len(complete)==len(rows) and all('env_step' in b and 'episodes' in b for b in budgets) and all(b['env_step']==budgets[0]['env_step'] and b['episodes']==budgets[0]['episodes'] for b in budgets)
    result['paired_transfer_comparisons'] = {phase: paired_transfer_comparisons(plan, rows, phase) for phase in ('quick_evaluation', 'final_evaluation')} if plan.get('paired_transfer', False) else None
    lines = ["# 夜间超参数实验进度", "", f"批次：{plan['campaign_id']}；更新时间：{result['updated_at_utc']}。",
             "", plan.get('comparison_description',"三组只改变目标熵比例；每组使用同一训练种子。测试集未使用。"), ""]
    for row in rows:
        p = row["progress"] or {}
        lines.append(f"- {row['name']}：{row['status']}；回合 {p.get('episodes', 0)}；环境步 {p.get('env_step', 0)}；更新 {p.get('update_step', 0)}；最近训练 U={p.get('last_U')}；退出码={row['exit_code']}。")
        evaluation = row["final_evaluation"] or row["quick_evaluation"]
        if evaluation and evaluation.get("status") == "completed":
            lines.append("  评价："+json.dumps(evaluation["report"]["summary"], ensure_ascii=False)+"。")
        if row["errors"]:
            lines.append("  异常："+"；".join(row["errors"])+"。")
        matched = row["matched_evaluation"]
        if matched and matched.get("status") == "completed":
            lines.append("  共同预算全验证集评价："+json.dumps(matched["report"]["summary"], ensure_ascii=False)
                         +"；预算="+json.dumps(matched["checkpoint_counters"], ensure_ascii=False)+"。")
    lines.extend(["", f"完整验证结果齐备：{result['all_final_evaluations_available']}；最终训练预算一致：{result['matched_final_budget']}。",
                  "若预算不一致，不按最后分数作公平排名；保留实际环境步、更新步和检查点身份。",
                  "这是受控探索性结果；正式独立多seed研究另行安排；缺失或失败评价明确保留，不补造分数。"])
    if result['paired_transfer_comparisons']:
        lines.extend(['', '## 同源继续训练配对比较', '', '来源预训练预算另计；不同续训种子可能采到不同实体总数，只在每个种子内核对新增预算。'])
        for phase, pairs in result['paired_transfer_comparisons'].items():
            for seed, comparison in pairs.items():
                lines.append(f'- {phase}／seed={seed}：'+json.dumps(comparison, ensure_ascii=False)+'。')
    if matching:
        lines.extend(["", "共同预算补充评价状态："+json.dumps(matching, ensure_ascii=False)+"。该补充评价保留各组最后状态；只重新评价policy，既有相同验证场景/种子的随机基线可作共同参考。"])
    atomic_json(Path(campaign_dir)/"夜间进度.json", result)
    Path(campaign_dir, "夜间进度.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if finished:
        atomic_json(Path(campaign_dir)/"夜间最终比较.json", result)
        Path(campaign_dir, "夜间最终比较.md").write_text("\n".join(lines).replace("实验进度", "实验结果", 1)+"\n", encoding="utf-8")
    return result


def child_environment(cpu=False):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def launch(command, log_path, cpu=False):
    log = Path(log_path).open("w", encoding="utf-8")
    try:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_environment(cpu), stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
    finally:
        log.close()
    return child


def stop_child(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        child.kill()
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def available_training_slots(states, max_workers):
    running = sum(state["status"] == "running" for state in states.values())
    return max(0, max_workers-running)


def campaign_successful(result):
    matching_status = (result.get("matched_comparison") or {}).get("status", "disabled")
    return (result["all_final_evaluations_available"] and result["all_training_successful"]
            and result["all_durable_copies_available"]
            and matching_status in ("disabled", "not_needed", "completed"))


def supervise(plan, durable_root):
    durable_root = Path(durable_root).resolve()
    if (durable_root/"outputs"/"spectrum_tsac_20260921").resolve() == SPECTRUM_OUTPUT_DIR.resolve():
        raise ValueError("持久根必须与节点本地运行根分离")
    campaign_dir = durable_root/"夜间实验"/plan["campaign_id"]
    campaign_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(campaign_dir/"夜间计划.json", plan)
    SPECTRUM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = {path.name for path in SPECTRUM_OUTPUT_DIR.iterdir() if path.is_dir()}
    states = {row["name"]: dict(row, status="queued", errors=[], seen=set(),
                               quick_scheduled=False, final_scheduled=False) for row in plan["experiments"]}
    started = time.monotonic()
    eval_queue, active_evaluation, active_copy = [], None, None
    matching = None if plan["matched_budget_fallback"] else {"status": "disabled"}
    while True:
        for name, state in states.items():
            if state["status"] == "queued" and available_training_slots(states, plan["max_workers"]):
                config_path = campaign_dir/(name+"-训练配置.json")
                atomic_json(config_path, state["config"])
                try:
                    if state.get('algorithm','horizon')=='horizon':
                        command=[sys.executable,'-m','implementations.spectrum_tsac_20260921.train','--mode','train','--config',str(config_path)]
                        if state.get('initialize_from'):
                            command.extend(['--initialize-from', state['initialize_from']])
                    else:
                        settings_path=campaign_dir/(name+'-算法参数.json');atomic_json(settings_path,state.get('settings',{}))
                        command=[sys.executable,'-m','implementations.spectrum_tsac_20260921.paper_baselines.train','--algorithm',state['algorithm'],'--config',str(config_path),'--settings',str(settings_path)]
                    state["process"] = launch(command,
                                               campaign_dir/(name+"-训练日志.txt"))
                    state.update(status="running", started=time.monotonic())
                except OSError as error:
                    state.update(status="failed", exit_code=None)
                    state["errors"].append("训练进程启动失败："+str(error))
        try:
            matches = match_runs(SPECTRUM_OUTPUT_DIR, plan["experiments"], excluded)
        except ValueError as error:
            matches = {}
            for state in states.values():
                if str(error) not in state["errors"]:
                    state["errors"].append(str(error))
        for name, state in states.items():
            if name in matches:
                state["run_dir"] = matches[name]
            process = state.get("process")
            if state["status"] == "running":
                limit = state["config"]["train"]["max_wall_seconds"]+plan["training_grace_seconds"]
                if process.poll() is None and time.monotonic()-state["started"] > limit:
                    stop_child(process)
                    state["errors"].append("超出训练墙钟预算及回合收尾宽限，监督器终止；保留最近完整检查点")
                code = process.poll()
                if code is not None:
                    state.update(status="trained" if code == 0 else "failed", exit_code=code)
            run_dir = state.get("run_dir")
            if run_dir:
                try:
                    state["progress"] = read_progress(run_dir)
                    sync_published(run_dir, durable_root/"outputs"/"spectrum_tsac_20260921"/run_dir.name, state["seen"])
                except (OSError, ValueError, sqlite3.Error) as error:
                    message = "进度或不可变工件同步异常："+str(error)
                    if message not in state["errors"]:
                        state["errors"].append(message)
                saved = checkpoints(run_dir)
                if saved and plan['quick_evaluation'] and not state["quick_scheduled"] and state["status"] == "running":
                    eval_queue.append((name, "quick", saved[0]))
                    state["quick_scheduled"] = True
                if state["status"] in ("trained", "failed") and not state["final_scheduled"]:
                    state["final_scheduled"] = True
                    if saved:
                        eval_queue.append((name, "final", saved[-1]))
                    else:
                        state["final_result"] = {"status": "missing", "error": "未产生完整回合检查点"}
            elif state["status"] in ("trained", "failed") and not state["final_scheduled"]:
                state["final_scheduled"] = True
                state["final_result"] = {"status": "missing", "error": "未找到与配置匹配的运行清单"}
        if active_evaluation:
            job = active_evaluation
            process = job["process"]
            if process.poll() is None and time.monotonic()-job["started"] > plan["evaluation_timeout_seconds"]:
                stop_child(process)
                job["timed_out"] = True
            code = process.poll()
            if code is not None:
                try:
                    result = read_json(job["result_path"])
                except (OSError, ValueError):
                    result = {"status": "failed", "error": "评价超时" if job.get("timed_out") else "评价子进程未写结果"}
                result["exit_code"] = code
                states[job["name"]][job["kind"]+"_result"] = result
                active_evaluation = None
        if matching is None and all("final_result" in state for state in states.values()):
            histories = [checkpoints(state["run_dir"]) if state.get("run_dir") else [] for state in states.values()]
            final_budgets = [rows[-1][2]["counters"] for rows in histories if rows]
            if len(final_budgets) == len(states) and all(budget == final_budgets[0] for budget in final_budgets):
                matching = {"status": "not_needed", "reason": "各组最后完整检查点预算一致"}
            else:
                selection = common_checkpoint_selection(states)
                if selection is None:
                    matching = {"status": "unavailable", "reason": "没有同时满足回合、环境步与更新步一致的已保存检查点"}
                else:
                    first = next(iter(selection.values()))
                    matching = {"status": "scheduled", "checkpoint_counters": first[2]["counters"],
                                "policies": ["policy"], "split": "validation", "limit": 0}
                    for name, selected in selection.items():
                        eval_queue.append((name, "matched", selected))
        if matching and matching["status"] == "scheduled" and all("matched_result" in state for state in states.values()):
            matching["status"] = "completed" if all(state["matched_result"].get("status") == "completed" for state in states.values()) else "failed"
        if active_evaluation is None and eval_queue:
            name, kind, (_, checkpoint, metadata) = eval_queue.pop(0)
            state = states[name]
            label = {"quick": "首检查点", "final": "最终", "matched": "共同预算"}[kind]
            request_path = campaign_dir/(name+"-"+label+"-评价请求.json")
            result_path = campaign_dir/(name+"-"+label+"-评价结果.json")
            request = dict(kind=kind, run_dir=str(state["run_dir"]), checkpoint_path=str(checkpoint),
                           checkpoint_counters=metadata["counters"], result_path=str(result_path),
                           limit=plan.get('quick_evaluation_limit', 3) if kind == "quick" else 0,
                           policies=plan.get('quick_evaluation_policies', ['policy', 'random']) if kind == "quick" else ["policy"] if kind == "matched" else state['final_policies'])
            if kind == "quick" and plan.get('quick_backup', True):
                request.update(backup_path=str(SPECTRUM_OUTPUT_DIR/".夜间备份"/plan["campaign_id"]/(name+"-首检查点")),
                               durable_backup_path=str(campaign_dir/"首检查点运行包"/name))
            atomic_json(request_path, request)
            try:
                process = launch([sys.executable, "-m", "implementations.spectrum_tsac_20260921.overnight",
                                  "--evaluation-worker", str(request_path)], campaign_dir/(name+"-"+label+"-评价日志.txt"), cpu=True)
                active_evaluation = dict(name=name, kind=kind, process=process, started=time.monotonic(), result_path=result_path)
            except OSError as error:
                state[kind+"_result"] = {"status": "failed", "error": "评价进程启动失败："+str(error)}
        if active_copy:
            job = active_copy
            process = job["process"]
            if process.poll() is None and time.monotonic()-job["started"] > plan["copy_timeout_seconds"]:
                stop_child(process)
                job["timed_out"] = True
            code = process.poll()
            if code is not None:
                state = states[job["name"]]
                try:
                    copied = read_json(job["result_path"])
                except (OSError, ValueError):
                    copied = {"status": "failed", "error": "副本复制超时" if job.get("timed_out") else "副本进程未写结果"}
                if code == 0 and copied.get("status") == "completed":
                    state["durable_copy"] = copied["destination"]
                else:
                    state["durable_copy"] = "failed"
                    state["errors"].append(f"结束副本第{state['copy_attempts']}次失败："+copied.get("error", str(code)))
                active_copy = None
        if active_copy is None:
            for name, state in states.items():
                if state["status"] not in ("trained", "failed") or "final_result" not in state or not state.get("run_dir"):
                    continue
                if matching is None or (matching["status"] == "scheduled" and "matched_result" not in state):
                    continue
                if state.get("durable_copy") not in (None, "failed") or state.get("copy_attempts", 0) >= 3:
                    continue
                # final完成后该run的训练和全部评价写入器已退出；复制独立运行，不阻塞监督。
                state["copy_attempts"] = state.get("copy_attempts", 0)+1
                attempt = state["copy_attempts"]
                destination = durable_root/"outputs"/"spectrum_tsac_20260921"/state["run_dir"].name
                request_path = campaign_dir/(name+f"-副本请求-{attempt}.json")
                result_path = campaign_dir/(name+f"-副本结果-{attempt}.json")
                atomic_json(request_path, {"source": str(state["run_dir"]), "destination": str(destination), "result_path": str(result_path)})
                try:
                    process = launch([sys.executable, "-m", "implementations.spectrum_tsac_20260921.overnight",
                                      "--copy-worker", str(request_path)], campaign_dir/(name+f"-副本日志-{attempt}.txt"), cpu=True)
                    active_copy = dict(name=name, process=process, started=time.monotonic(), result_path=result_path)
                except OSError as error:
                    state["durable_copy"] = "failed"
                    state["errors"].append("副本进程启动失败："+str(error))
                break
        copies_finished = all(not s.get("run_dir") or s.get("durable_copy") not in (None, "failed") or s.get("copy_attempts", 0) >= 3 for s in states.values())
        finished = all(s["status"] in ("trained", "failed") and "final_result" in s for s in states.values()) and not eval_queue and active_evaluation is None and active_copy is None and copies_finished and matching is not None and matching["status"] != "scheduled"
        result = write_progress(campaign_dir, plan, states, started, finished, matching)
        print(json.dumps({"campaign": plan["campaign_id"], "finished": finished,
                          "states": {n: s["status"] for n, s in states.items()}}, ensure_ascii=False), flush=True)
        if finished:
            return campaign_dir, result
        time.sleep(plan["poll_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--durable-root")
    parser.add_argument("--evaluation-worker")
    parser.add_argument("--copy-worker")
    args = parser.parse_args()
    if args.evaluation_worker:
        raise SystemExit(evaluation_worker(args.evaluation_worker))
    if args.copy_worker:
        raise SystemExit(copy_worker(args.copy_worker))
    if not args.plan or not args.durable_root:
        parser.error("需要--plan与--durable-root")
    path = Path(args.plan).resolve()
    plan = normalize_plan(read_json(path), path.parent)
    directory, result = supervise(plan, args.durable_root)
    successful = campaign_successful(result)
    print(json.dumps({"campaign_dir": str(directory), "complete": successful}, ensure_ascii=False))
    raise SystemExit(0 if successful else 1)


if __name__ == "__main__":
    main()
