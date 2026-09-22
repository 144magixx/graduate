"""冻结模型、统一环境、场景等权与配对统计；不从测试集选模型。"""
import argparse
import copy
import hashlib
import shutil
import json
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone
import numpy as np
from project_paths import COVER_OUTPUT_DIR
from .artifacts import atomic_json, load_checkpoint
from .config import Config


def choose_action(env, policy, rng=None, greedy_candidates=16, equal_power=None):
    """大空间B1显式使用受限候选；非数学上界。"""
    from .env.state import Allocation
    from .env.physics import evaluate_allocation
    obs = env.observe()
    ids = np.flatnonzero(obs["valid_action_mask"])
    if policy == "random":
        return env.action_spec.decode(int(rng.choice(ids)))
    if policy in ("greedy", "equal_power"):
        if equal_power is not None:
            ids = np.array([i for i in ids if i == 0 or env.action_spec.power_levels_w[env.action_spec.decode(int(i)).power_index] == equal_power])
        if len(ids)>greedy_candidates:
            ids = ids[np.unique(np.linspace(0,len(ids)-1,greedy_candidates,dtype=int))]
        scores=[]
        current = env.current_beam_id
        for i in ids:
            action=env.action_spec.decode(int(i))
            power=float(env.action_spec.power_levels_w[action.power_index]) if i else 0.
            ledger=dict(env.ledger)
            ledger[current]=Allocation(current,"allocated" if i else "skipped",action.start,action.length,power)
            result=evaluate_allocation(env.scenario,ledger,env.config,channel_cache=env._channel)
            scores.append(result.metrics["mean_satisfaction"])
        return env.action_spec.decode(int(ids[int(np.argmax(scores))]))
    return policy.act(obs,deterministic=True)


def allocate_candidate(root, candidate, policy, physics_config=None):
    """第一阶段调用接口：同一root共同分母，缺root不补造端到端评价。"""
    from .env.environment import Environment
    from .data.domain import root_metric
    config=physics_config or Config()
    if root is not None:
        candidate.validate(root,config)
    scenario=candidate.beam_table if hasattr(candidate,"beam_table") else candidate
    env=Environment(config)
    obs,_=env.reset(scenario)
    rng=np.random.default_rng(0)
    started=time.perf_counter()
    while not obs["terminal"]:
        obs,_,_,_,_=env.step(choose_action(env,policy,rng))
    evaluation=env.evaluate()
    return {"allocation":[x.to_dict() for x in env.ledger.values()],"U":evaluation.metrics["mean_satisfaction"],
            **root_metric(root,candidate if hasattr(candidate,"beam_table") else None,evaluation.rate_bps),
            "constraint_violations":evaluation.constraint_violations,"inference_seconds":time.perf_counter()-started,
            "metrics":evaluation.metrics}


def paired_summary(first, second, seed=0, resamples=2000):
    a={x["scenario_id"]:x["mean_satisfaction"] for x in first if x.get("mean_satisfaction") is not None}
    b={x["scenario_id"]:x["mean_satisfaction"] for x in second if x.get("mean_satisfaction") is not None}
    shared=sorted(set(a)&set(b))
    values=np.array([a[key]-b[key] for key in shared])
    if not len(values):
        return {"n_pairs":0,"mean_difference":None,"ci95":None}
    draws=np.random.default_rng(seed).choice(values,(resamples,len(values)),replace=True).mean(1)
    return {"n_pairs":len(values),"mean_difference":float(values.mean()),
            "ci95":np.quantile(draws,[.025,.975]).tolist() if len(values)>1 else None,
            "method":"paired_scene_bootstrap","training_seed_interval_available":False}


def evaluate_run(manifest_path, split="validation", limit=3, policies=("policy","random"), export_results=True, checkpoint_path=None, heuristic_settings=None):
    import torch
    from .data.loader import load_scenario
    from .env.environment import Environment
    from .rl import DiscreteSAC
    from .telemetry import Recorder
    from .export import export_allocation
    manifest_path=Path(manifest_path)
    run_dir=manifest_path.parent
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    config=Config.from_dict(manifest["config"])
    config.train.device="cpu" if not torch.cuda.is_available() else config.train.device
    torch.set_num_threads(config.train.torch_num_threads)
    selected_checkpoint=Path(checkpoint_path) if checkpoint_path else run_dir/"检查点.pt"
    saved=load_checkpoint(selected_checkpoint,config,manifest["data_manifest_hash"])
    from .train import prepare_dataset
    # manifest本身亦需核验完整内容，不能仅信任header中的hash。
    verified_manifest=prepare_dataset(config,run_dir/"数据划分.json")
    if verified_manifest != manifest["data_manifest"] or verified_manifest["manifest_hash"] != manifest["data_manifest_hash"]:
        raise ValueError("运行清单内嵌数据划分与已验证划分不一致")
    records=[r for r in verified_manifest["records"] if r["split"]==split]
    if limit: records=records[:limit]
    if not records: raise ValueError(f"{split}没有可用场景")
    results={}
    evaluation_id=uuid.uuid4().hex[:12]
    destination=run_dir/"独立评估"/evaluation_id
    destination.mkdir(parents=True)
    for name in policies:
        policy_dir=destination/name
        policy_dir.mkdir()
        eval_manifest=copy.deepcopy(manifest)
        eval_manifest.update(run_id=manifest["run_id"]+"_eval_"+evaluation_id+"_"+name,
                             parent_run_id=manifest["run_id"],phase=split,mode="evaluation",algorithm=name,
                             checkpoint_id=saved["checkpoint_id"],trial_id=manifest["trial_id"],evaluation_id=evaluation_id,
                             evaluated_training_counters=saved["counters"],
                             created_at_utc=datetime.now(timezone.utc).isoformat())
        if name!='policy':
            eval_manifest.pop('paper_baseline',None)
            eval_manifest.pop('baseline_settings',None)
            eval_manifest.update(model_not_applicable=True,training_required=False,reference_only_checkpoint=True,reference_algorithm=manifest['algorithm'])
        if name=='policy' and manifest.get('paper_baseline'):
            eval_manifest['algorithm']=manifest['algorithm']
        if name.startswith('paper_'):
            from .paper_baselines.specs import method_metadata,DISPLAY
            heuristic_id=name.removeprefix('paper_')
            if heuristic_id not in ('fixed','greedy','random'):raise ValueError('未知论文启发式基线')
            resolved_heuristic_settings=dict(fixed_length=5,fixed_power_w=30.,greedy_length=5,greedy_demand_thresholds_bps=None,random_include_skip=False)
            resolved_heuristic_settings.update(heuristic_settings or {})
            if heuristic_id=='random' and resolved_heuristic_settings['random_include_skip']:
                raise ValueError('含主动SKIP的随机诊断不是论文Random Scheme，请使用单独诊断策略')
            eval_manifest.update(algorithm=DISPLAY[heuristic_id],paper_baseline=method_metadata(heuristic_id,resolved_heuristic_settings,config),
                                 heuristic_settings=resolved_heuristic_settings,
                                 model_not_applicable=True,training_required=False,reference_only_checkpoint=True)
        coverage_paths=[]
        for item in eval_manifest.get("coverage_snapshot",{}).get("artifacts",{}).values():
            source=(run_dir/item["path"]).resolve()
            from .audit import sha256
            if run_dir.resolve() not in source.parents or not source.is_file() or sha256(source)!=item["sha256"]:
                raise ValueError("父运行冻结覆盖审计缺失或hash不一致")
            target=policy_dir/Path(item["path"]).name
            shutil.copy2(source,target)
            item["path"]=target.name
            item.pop("artifact_id",None)
            coverage_paths.append(target)
        recorder=Recorder(policy_dir,eval_manifest,config.telemetry)
        for path in coverage_paths:
            recorder.publish_artifact(path,kind="coverage_snapshot")
        summaries=[]
        evaluation_env_step=0
        try:
            for record in records:
                order_seed=int(hashlib.sha256((str(config.train.seed)+":"+record["scenario_id"]).encode("utf-8")).hexdigest()[:16],16)
                source_path=Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR/record["file"]
                scenario=load_scenario(source_path,config,rng=np.random.default_rng(order_seed))
                from .audit import sha256
                if sha256(source_path)!=record["source_hash"]: raise ValueError("评估输入文件hash漂移")
                env=Environment(config)
                obs,_=env.reset(scenario)
                if name=="policy":
                    if manifest.get('paper_baseline'):
                        from .paper_baselines.specs import build_agent
                        policy=build_agent(manifest['paper_baseline']['algorithm_id'],config,env.action_spec,obs,manifest.get('baseline_settings'))
                    else:policy=DiscreteSAC(config,env.action_spec,obs)
                    policy.load_state_dict(saved["agent"],restore_rng=False)
                    policy.train(False)
                elif name.startswith('paper_'):
                    from .paper_baselines.heuristics import HeuristicPolicy
                    policy=HeuristicPolicy(name.removeprefix('paper_'),env.action_spec,scenario,config,seed=config.train.seed,**resolved_heuristic_settings)
                else: policy=name
                episode_id=uuid.uuid4().hex
                snapshot=env.snapshot()
                recorder.start_episode(episode_id,scenario.scenario_id,snapshot,phase=split,env_step=evaluation_env_step)
                rng=np.random.default_rng(config.train.seed)
                total_return=0.
                decision_seconds=0.
                allocation_compute_seconds=0.
                started=time.perf_counter()
                while not obs["terminal"]:
                    before=snapshot
                    decision_started=time.perf_counter()
                    action=choose_action(env,policy,rng,equal_power=25. if name=="equal_power" else None)
                    decision_seconds+=time.perf_counter()-decision_started
                    obs,reward,terminated,truncated,info=env.step(action)
                    evaluation_env_step+=1
                    allocation_compute_seconds+=time.perf_counter()-decision_started
                    total_return+=reward
                    snapshot=env.snapshot()
                    recorder.record_step(episode_id,scenario.scenario_id,env.cursor,before,snapshot,
                                         {"action":action.to_dict(),"reward":reward,**info},env_step=evaluation_env_step,phase=split)
                elapsed=time.perf_counter()-started
                metrics=dict(env.evaluate().metrics)
                metrics.update(scenario_id=scenario.scenario_id,episode_return=total_return,inference_seconds=allocation_compute_seconds,
                               policy_decision_seconds=decision_seconds,rollout_seconds=elapsed,
                               n_demand=scenario.n_demand,checkpoint_id=saved["checkpoint_id"],phase=split)
                if name.startswith('paper_'):metrics['heuristic_spec']=policy.metadata()
                summaries.append(metrics)
                recorder.end_episode(episode_id,metrics,scenario_id=scenario.scenario_id,env_step=evaluation_env_step,phase=split)
                if export_results:
                    safe_id=hashlib.sha256(scenario.scenario_id.encode("utf-8")).hexdigest()[:16]
                    allocation_dir=policy_dir/("场景-"+safe_id)
                    export_allocation(env,allocation_dir,eval_manifest["run_id"])
                    for filename in ("第二阶段分配结果.npz","第三阶段交接清单.json","导出复算验收.json"):
                        recorder.publish_artifact(allocation_dir/filename,kind="allocation_export")
            recorder.close()
        except BaseException:
            recorder.close("failed")
            raise
        results[name]=summaries
    summary={name:{"scenarios":len(rows),"mean_U":float(np.mean([r["mean_satisfaction"] for r in rows if r["mean_satisfaction"] is not None])) if any(r["mean_satisfaction"] is not None for r in rows) else None,
                   "micro_U":float(sum(r["mean_satisfaction"]*r["n_demand"] for r in rows if r["mean_satisfaction"] is not None)/sum(r["n_demand"] for r in rows)) if sum(r["n_demand"] for r in rows) else None,
                   "inference_p50_seconds":float(np.quantile([r["inference_seconds"] for r in rows],.5)),
                   "inference_p95_seconds":float(np.quantile([r["inference_seconds"] for r in rows],.95)),
                   "constraint_violations":sum(r["constraint_violation_count"] for r in rows)} for name,rows in results.items()}
    report={"evaluation_id":evaluation_id,"split":split,"checkpoint_id":saved["checkpoint_id"],"evaluated_training_counters":saved["counters"],"results":results,"summary":summary,
            "paired":paired_summary(results["policy"],results["random"]) if "policy" in results and "random" in results else None,
            "greedy_candidate_rule":"包含SKIP的合法候选ID等间距至多16个，最大即时全局U；非上界",
            "status":"short_validation_only","formal_multiseed_complete":False,"output_dir":str(destination)}
    atomic_json(destination/"评估结果.json",report)
    lines=["# 独立评估报告","",f"划分：{split}；checkpoint：{saved['checkpoint_id']}。本次为短验证，不用于挑选测试集最优模型或宣称性能提升。",""]
    for name,row in summary.items(): lines.append(f"- {name}：{row['scenarios']}景，平均U={row['mean_U']}，硬约束违约={row['constraint_violations']}，整景耗时P50={row['inference_p50_seconds']:.3f}s。")
    lines.extend(["","全部方案在相同配置重算，并导出无pickle的第三阶段数组；场景平均与需求数加权微平均分别记录。正式至少5训练seed、足量独立场景和层级区间待后续长训练。"])
    (destination/"评估报告.md").write_text("\n".join(lines),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--split",choices=("validation","test"),default="validation")
    parser.add_argument("--limit",type=int,default=3)
    parser.add_argument("--checkpoint",help="可选不可变检查点路径，避免训练中最新入口替换竞态")
    parser.add_argument("--policies",nargs="+",choices=("policy","random","greedy","equal_power","paper_fixed","paper_greedy","paper_random"),default=["policy","random"])
    args=parser.parse_args()
    result=evaluate_run(args.manifest,args.split,args.limit,args.policies,checkpoint_path=args.checkpoint)
    print(json.dumps({"output_dir":result["output_dir"],"summary":result["summary"]},ensure_ascii=False))


if __name__=="__main__": main()
