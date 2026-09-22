"""冻结评估结果的场景配对与训练seed/场景两层bootstrap；不启动训练。

输入runs每项为 {manifest: 完整运行清单, episodes: [场景指标], status: 可选运行状态}。
调用者先按算法/模型配置分组。同trial恢复分支选择最新完成记录；相同seed不作为
多个独立训练样本。一次bootstrap内所有seed共用场景索引，左右算法共用配对索引，
保留共同测试场景的相关性。区间仅覆盖提供的训练seed与场景，不创造来源独立性。
"""
import json
import math
import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _first(*values):
    return next((value for value in values if value is not None), None)


def _version(manifest, key):
    config = manifest.get("config", {})
    return _first(manifest.get("versions", {}).get(key), config.get("physics", {}).get(key),
                  config.get("env", {}).get(key), config.get(key), manifest.get(key))


def compatibility_signature(manifest):
    """模型与观察消融允许不同；物理数值、资源、数据语义与评价口径必须一致。"""
    config = manifest.get("config", {})
    physics, env, data = config.get("physics", {}), config.get("env", {}), config.get("data", {})
    action_spec = manifest.get("action_spec_signature")
    if action_spec is None and all(value is not None for value in
        (physics.get("num_slots"), env.get("max_block_length"), env.get("power_levels_w"))):
        action_spec = dict(num_slots=physics["num_slots"], max_block_length=env["max_block_length"],
                           power_levels_w=list(env["power_levels_w"]))
    data_fields = ("source_schema", "rate_unit", "traffic_scale", "group_assignment", "service_order", "beamwidth_kind")
    return dict(schema_version=_version(manifest, "schema_version"),
        physics_version=_version(manifest, "physics_version"), metric_version=_version(manifest, "metric_version"),
        action_version=_version(manifest, "action_version"), physics_config=physics or None,
        metric_parameters={"satisfaction_tolerance": env["satisfaction_tolerance"]} if "satisfaction_tolerance" in env else None,
        action_spec_signature=action_spec, power_budget_w=_first(env.get("power_budget_w"), manifest.get("power_budget_w")),
        data_semantics={key: data[key] for key in data_fields} if all(key in data for key in data_fields) else None,
        dataset_version=_first(manifest.get("dataset_version"), data.get("dataset_version")),
        split_hash=_first(manifest.get("data_manifest_hash"), manifest.get("split_hash")))


def compatibility(runs, metric="mean_satisfaction"):
    signatures = [compatibility_signature(run["manifest"]) for run in runs]
    if not signatures:
        return dict(compatible=False, differing_fields=[], missing_fields=["runs"], return_comparable=False)
    differing, missing = [], []
    for key in signatures[0]:
        values = [value[key] for value in signatures]
        if any(value is None for value in values):
            missing.append(key)
        try:
            if len({_canonical(value) for value in values}) > 1:
                differing.append(key)
        except (ValueError, TypeError):
            missing.append(key + ".invalid")
    rewards = [dict(version=_version(run["manifest"], "reward_version"),
                    scale=run["manifest"].get("config", {}).get("env", {}).get("reward_scale")) for run in runs]
    return_comparable = (not differing and not missing and all(r["version"] is not None and r["scale"] is not None for r in rewards)
                         and len({_canonical(r) for r in rewards}) == 1)
    if metric in ("episode_return", "return") and not return_comparable:
        differing.append("reward_semantics")
    return dict(compatible=not differing and not missing, differing_fields=sorted(set(differing)),
                missing_fields=sorted(set(missing)), return_comparable=return_comparable)


def _metric(row, name):
    if "summary" in row and isinstance(row["summary"], dict):
        row = {**row, **row["summary"]}
    aliases = {"mean_satisfaction": ("mean_satisfaction", "U", "utility"),
               "episode_return": ("episode_return", "return"), "return": ("episode_return", "return"),
               "J_root": ("J_root", "j_root"), "j_root": ("j_root", "J_root")}
    value = _first(*(row.get(key) for key in aliases.get(name, (name,))))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _model_identity(manifest):
    if manifest.get('model_not_applicable'):
        return _canonical(dict(algorithm=manifest.get('algorithm'),heuristic_settings=manifest.get('heuristic_settings',{})))
    train=manifest.get("config", {}).get("train", {})
    learning_keys=("target_entropy_ratio","actor_lr","critic_lr","alpha_lr","gamma","tau",
                   "batch_size","replay_capacity","update_schedule","updates_per_step","updates_per_episode")
    return _canonical(dict(algorithm=manifest.get("algorithm"),
        model=manifest.get("config", {}).get("model", manifest.get("model")),
        learning={key:train.get(key) for key in learning_keys},baseline_settings=manifest.get('baseline_settings')))


def _status(run):
    return run.get("status", run["manifest"].get("status", "unknown"))


def _select(runs, metric, phase):
    """只按恢复谱系和时间选末次完成评估，绝不根据指标挑选checkpoint。"""
    runs = list(runs)
    parent = list(range(len(runs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tokens, ids = {}, {}
    for i, run in enumerate(runs):
        manifest = run["manifest"]
        run_id, trial_id = manifest.get("run_id"), manifest.get("trial_id")
        if run_id:
            ids[run_id] = i
        for token in (("run", run_id), ("trial", trial_id), ("shared_parent", manifest.get("parent_run_id"))):
            if token[1] is not None:
                if token in tokens:
                    parent[find(i)] = find(tokens[token])
                tokens[token] = i
    for i, run in enumerate(runs):
        ancestor = run["manifest"].get("parent_run_id")
        if ancestor in ids:
            parent[find(i)] = find(ids[ancestor])
    groups = {}
    for i in range(len(runs)):
        groups.setdefault(find(i), []).append(i)
    failed = [run["manifest"].get("run_id") for run in runs if _status(run) in ("failed", "interrupted", "cancelled")]
    excluded, selection, units = [], [], []
    def rank(index):
        manifest = runs[index]["manifest"]
        seen, ancestor, depth = set(), manifest.get("parent_run_id"), 0
        while ancestor in ids and ancestor not in seen:
            seen.add(ancestor)
            depth += 1
            ancestor = runs[ids[ancestor]]["manifest"].get("parent_run_id")
        return (depth, str(manifest.get("created_at_utc", "")), str(manifest.get("run_id", "")))
    for indices in groups.values():
        candidates = []
        for index in indices:
            run = runs[index]
            manifest = run["manifest"]
            if _status(run) != "completed":
                excluded.append(dict(run_id=manifest.get("run_id"), reason="run_not_completed", status=_status(run)))
                continue
            seed = _first(manifest.get("seed"), manifest.get("config", {}).get("train", {}).get("seed"))
            if seed is None or not manifest.get("trial_id") or not manifest.get("run_id"):
                excluded.append(dict(run_id=manifest.get("run_id"), reason="missing_seed_trial_or_run_identity"))
                continue
            by_scene = {}
            for row in run.get("episodes", run.get("rows", [])):
                row = {**row, **(row.get("summary") or {})} if isinstance(row.get("summary"), dict) else row
                row_phase = _first(row.get("phase"), manifest.get("phase"))
                if row_phase != phase or row.get("truncated") or row.get("status", "completed") != "completed":
                    continue
                value, scenario_id = _metric(row, metric), row.get("scenario_id")
                if value is not None and scenario_id is not None:
                    by_scene.setdefault(str(scenario_id), []).append(value)
            if not by_scene:
                excluded.append(dict(run_id=manifest.get("run_id"), reason="no_finite_completed_evaluation_rows"))
                continue
            candidates.append(dict(index=index, run_id=manifest["run_id"], trial_id=manifest["trial_id"], seed=seed,
                values={key: float(np.mean(values)) for key, values in by_scene.items()},
                repeats={key: len(values) for key, values in by_scene.items()}))
        if not candidates:
            continue
        if len({str(value["seed"]) for value in candidates}) != 1:
            excluded.extend(dict(run_id=value["run_id"], reason="lineage_seed_changed") for value in candidates)
            continue
        priority = max(rank(value["index"])[:2] for value in candidates)
        latest = [value for value in candidates if rank(value["index"])[:2] == priority]
        if len(latest) > 1 and len({_canonical(value["values"]) for value in latest}) > 1:
            excluded.extend(dict(run_id=value["run_id"], reason="ambiguous_evaluation_order_same_lineage") for value in latest)
            continue
        chosen = max(latest, key=lambda value: rank(value["index"]))
        discarded = [runs[i]["manifest"].get("run_id") for i in indices if i != chosen["index"]]
        selection.append(dict(trial_id=chosen["trial_id"], seed=chosen["seed"], selected_run_id=chosen["run_id"],
                              superseded_or_failed_run_ids=discarded, rule="latest_completed_descendant_then_timestamp"))
        units.append(chosen)
    # 同seed的独立重启也不能当成多个独立随机seed；保留最新，显式报告。
    by_seed = {}
    for unit in units:
        by_seed.setdefault(str(unit["seed"]), []).append(unit)
    chosen_units, duplicates = [], []
    for seed, candidates in by_seed.items():
        chosen = max(candidates, key=lambda value: rank(value["index"]))
        chosen_units.append(chosen)
        if len(candidates) > 1:
            duplicates.append(dict(seed=chosen["seed"], selected_run_id=chosen["run_id"],
                discarded_run_ids=[value["run_id"] for value in candidates if value is not chosen]))
    return sorted(chosen_units, key=lambda value: str(value["seed"])), dict(failed_runs=failed,
        excluded_runs=excluded, lineage_selection=selection, duplicate_seed_trials=duplicates)


def _interval(values, rng, resamples):
    if len(values) < 2:
        return None
    draws = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(resamples)]
    return np.quantile(draws, [.025, .975]).tolist()


def _intervals(matrix, random_seed, resamples):
    if not isinstance(resamples, int) or resamples < 100 or resamples > 100000:
        raise ValueError("bootstrap resamples必须为100–100000的整数")
    if matrix.size == 0:
        return None, None
    rng = np.random.default_rng(random_seed)
    scenario_ci = _interval(matrix.mean(0), rng, resamples)
    if matrix.shape[0] < 2:
        return scenario_ci, None
    values = []
    for _ in range(resamples):
        seeds = rng.integers(matrix.shape[0], size=matrix.shape[0])
        scenes = rng.integers(matrix.shape[1], size=matrix.shape[1])
        values.append(float(matrix[np.ix_(seeds, scenes)].mean()))
    return scenario_ci, np.quantile(values, [.025, .975]).tolist()


def _support(runs):
    limitations = []
    datasets = [run["manifest"].get("data_manifest", {}) for run in runs]
    root_known = bool(datasets) and all(dataset.get("strict_root_independence") is True and dataset.get("records")
        and all(record.get("root_scene_id") and record.get("split_group_id") for record in dataset["records"]) for dataset in datasets)
    if not root_known:
        limitations.append("缺少可信root/split_group独立性；区间仅描述现有候选场景，不能证明业务实例泛化")
    root_scenarios = {}
    for run, dataset in zip(runs, datasets):
        evaluated_scenes = {row.get("scenario_id") for row in run.get("episodes", run.get("rows", []))}
        for record in dataset.get("records", []):
            if record.get("root_scene_id") and record.get("scenario_id") in evaluated_scenes:
                root_scenarios.setdefault(record["root_scene_id"], set()).add(record["scenario_id"])
    if any(len(scenes) > 1 for scenes in root_scenarios.values()):
        root_known = False
        limitations.append("多个候选场景共享root；当前场景bootstrap仅为候选层诊断，未提供root聚类区间，禁止正式排名")
    rows = [row for run in runs for row in run.get("episodes", run.get("rows", []))]
    assignment_known = bool(rows) and all(row.get("end_to_end_metric_available") is True for row in rows)
    if not assignment_known:
        limitations.append("原始业务/assignment未完整提供，不推造J_root或端到端业务结论")
    return limitations, root_known, assignment_known


def _expected_scenes(runs, phase):
    return {str(row["scenario_id"]) for run in runs
            for row in run["manifest"].get("data_manifest", {}).get("records", [])
            if row.get("split") == phase and row.get("scenario_id") is not None}


def _prepare(runs, metric, phase):
    if phase not in ("validation", "test"):
        raise ValueError("研究统计只接受冻结validation/test评估，不能把训练回合当独立评估")
    runs = list(runs)
    if any(not isinstance(run.get("manifest"), dict) for run in runs):
        raise ValueError("每个run必须附完整manifest")
    agreement = compatibility(runs, metric)
    if len({_model_identity(run["manifest"]) for run in runs}) > 1:
        agreement["compatible"] = False
        agreement["differing_fields"].append("mixed_algorithm_or_model_within_cohort")
    units, audit = _select(runs, metric, phase)
    limitations, root_known, assignment_known = _support(runs)
    if metric in ("J_root", "j_root") and not assignment_known:
        agreement["compatible"] = False
        agreement["missing_fields"].append("root_assignment")
    return runs, agreement, units, audit, limitations, root_known


def summarize_runs(runs, metric="mean_satisfaction", phase="test", expected_seeds=None, resamples=2000, random_seed=0):
    runs, agreement, units, audit, limitations, root_known = _prepare(runs, metric, phase)
    available = {str(unit["seed"]) for unit in units}
    missing_seeds = [seed for seed in (expected_seeds or []) if str(seed) not in available]
    scenes = sorted(set.intersection(*(set(unit["values"]) for unit in units))) if units else []
    union = set.union(*(set(unit["values"]) for unit in units)) if units else set()
    expected = _expected_scenes(runs, phase)
    missing_scenes = sorted((union | expected)-set(scenes))
    matrix = np.asarray([[unit["values"][scene] for scene in scenes] for unit in units], dtype=np.float64)
    eligible = agreement["compatible"] and bool(units) and bool(scenes)
    scene_ci, seed_ci = _intervals(matrix, random_seed, resamples) if eligible else (None, None)
    if len(units) < 2:
        limitations.append("不足2个独立训练seed，训练seed区间不可用；场景区间仅作诊断")
    if missing_scenes:
        limitations.append("仅对所有保留seed共同评估的场景求均值；缺失场景已单列，不补0")
    if expected_seeds is None:
        limitations.append("缺少预登记seed清单，无法判断未提交或失败的seed，禁止正式排名")
    ranking = bool(eligible and phase == "test" and len(units) >= 5 and len(scenes) >= 2 and not missing_seeds
                   and not missing_scenes and not audit["failed_runs"] and not audit["duplicate_seed_trials"]
                   and not audit["excluded_runs"] and root_known and expected_seeds is not None)
    return dict(metric=metric, phase=phase, **agreement, mean=float(matrix.mean()) if eligible else None,
        n_seeds=len(units), n_scenarios=len(scenes), selected_run_ids=[unit["run_id"] for unit in units],
        scenario_means=[dict(scenario_id=scene, mean=float(matrix[:, i].mean())) for i, scene in enumerate(scenes)] if eligible else [],
        seed_means=[dict(seed=unit["seed"], trial_id=unit["trial_id"], mean=float(matrix[i].mean())) for i, unit in enumerate(units)] if eligible else [],
        scenario_ci95=scene_ci, seed_ci95=seed_ci, seed_interval_available=seed_ci is not None,
        missing_seeds=missing_seeds, missing_scenarios=missing_scenes, **audit, support_limitations=limitations,
        ranking_available=ranking, method="seed_scene_bootstrap_shared_scenario_indices",
        confidence_level=.95, bootstrap_resamples=resamples, random_seed=random_seed,
        estimator="训练seed等权；共同场景等权；同run重复场景先求均值；恢复段不增加seed数")


def compare_runs(left_runs, right_runs, metric="mean_satisfaction", phase="test", expected_seeds=None, resamples=2000, random_seed=0):
    left_runs, right_runs = list(left_runs), list(right_runs)
    left = summarize_runs(left_runs, metric, phase, expected_seeds, resamples, random_seed)
    right = summarize_runs(right_runs, metric, phase, expected_seeds, resamples, random_seed)
    agreement = compatibility(left_runs+right_runs, metric)
    if not left["compatible"] or not right["compatible"]:
        agreement["compatible"] = False
        agreement["differing_fields"] = sorted(set(agreement["differing_fields"]+left["differing_fields"]+right["differing_fields"]))
        agreement["missing_fields"] = sorted(set(agreement["missing_fields"]+left["missing_fields"]+right["missing_fields"]))
    a, _ = _select(left_runs, metric, phase)
    b, _ = _select(right_runs, metric, phase)
    a, b = {str(unit["seed"]): unit for unit in a}, {str(unit["seed"]): unit for unit in b}
    shared_seeds = sorted(a.keys() & b.keys())
    scenarios = sorted(set.intersection(*(set(a[seed]["values"]) & set(b[seed]["values"]) for seed in shared_seeds))) if shared_seeds else []
    unmatched = dict(left=[a[seed]["seed"] for seed in sorted(a.keys()-b.keys())],
                     right=[b[seed]["seed"] for seed in sorted(b.keys()-a.keys())])
    paired_missing_scenarios = sorted(({row["scenario_id"] for row in left["scenario_means"]}
                                      | {row["scenario_id"] for row in right["scenario_means"]}) - set(scenarios))
    eligible = agreement["compatible"] and bool(shared_seeds) and bool(scenarios)
    pairs = [dict(seed=a[seed]["seed"], scenario_id=scene, left=a[seed]["values"][scene], right=b[seed]["values"][scene],
                  difference=b[seed]["values"][scene]-a[seed]["values"][scene]) for seed in shared_seeds for scene in scenarios] if eligible else []
    matrix = np.array([[b[seed]["values"][scene]-a[seed]["values"][scene] for scene in scenarios] for seed in shared_seeds])
    scene_ci, seed_ci = _intervals(matrix, random_seed, resamples) if eligible else (None, None)
    limitations = sorted(set(left["support_limitations"]+right["support_limitations"]))
    if unmatched["left"] or unmatched["right"]:
        limitations.append("仅同seed配对；不匹配的训练seed已单列，不以不同seed伪造配对")
    if not shared_seeds:
        limitations.append("没有共同训练seed；本接口不将非配对试验强行转换为配对结果")
    ranking = bool(eligible and len(scenarios) >= 2 and left["ranking_available"] and right["ranking_available"]
                   and not unmatched["left"] and not unmatched["right"] and not paired_missing_scenarios)
    return dict(metric=metric, phase=phase, **agreement, comparison_available=eligible, delta_direction="right_minus_left",
        mean_difference=float(matrix.mean()) if eligible else None, n_seed_pairs=len(shared_seeds), n_scenario_pairs=len(scenarios),
        paired_differences=pairs, scenario_ci95=scene_ci, seed_ci95=seed_ci, seed_interval_available=seed_ci is not None,
        missing_seeds=dict(left=left["missing_seeds"], right=right["missing_seeds"]), unmatched_seeds=unmatched,
        paired_missing_scenarios=paired_missing_scenarios,
        ranking_available=ranking, left=left, right=right, support_limitations=limitations,
        method="paired_seed_scene_bootstrap_shared_scenario_indices", confidence_level=.95,
        bootstrap_resamples=resamples, random_seed=random_seed)
