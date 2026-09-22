"""论文学习基线训练；共享可信环境，PPO只用当前on-policy轨迹。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import traceback
import uuid
import numpy as np
from project_paths import COVER_OUTPUT_DIR
from ..config import load_config
from ..train import prepare_dataset
from .specs import LEARNED, adapt_config, build_agent, method_metadata, DISPLAY


def train_baseline(algorithm, config, settings=None, dataset=None, resume=None, run_kind='paper_baseline'):
    if run_kind not in ('paper_baseline','paper_baseline_preflight'):raise ValueError('未知基线运行类别')
    config=adapt_config(config,algorithm)
    settings=dict(settings or {})
    if config.train.device.startswith('cuda') and config.train.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from ..artifacts import create_run,atomic_json,save_checkpoint,load_checkpoint,restore_rng
    from ..data.loader import load_scenario
    from ..data.augment import augment_scenario
    from ..data.domain import DomainSampler,freeze_run_coverage
    from ..env.environment import Environment
    from ..rl.replay import ReplayBuffer,Transition
    from ..telemetry import Recorder
    torch.set_num_threads(config.train.torch_num_threads)
    if config.train.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    torch.manual_seed(config.train.seed);random.seed(config.train.seed);np.random.seed(config.train.seed)
    torch.use_deterministic_algorithms(config.train.deterministic)
    manifest_data=prepare_dataset(config,dataset)
    saved=load_checkpoint(resume,config,manifest_data['manifest_hash']) if resume else None
    sampler=DomainSampler(manifest_data['records'],config.train.seed+11)
    rngs={'augmentation':np.random.default_rng(config.train.seed+23),'exploration':np.random.default_rng(config.train.seed+37),'order':np.random.default_rng(config.train.seed+53)}
    source=lambda row:Path(row['scenario_path']) if row.get('scenario_path') else COVER_OUTPUT_DIR/row['file']
    first=next(row for row in manifest_data['records'] if row['split']=='train')
    env=Environment(config)
    example,_=env.reset(load_scenario(source(first),config,rng=np.random.default_rng(config.train.seed+59)))
    agent=build_agent(algorithm,config,env.action_spec,example,settings)
    if algorithm=='mlp_ppo':settings=asdict(agent.hyperparameters)
    if algorithm=='mlp_dqn':settings={key:getattr(agent,key) for key in ('epsilon_start','epsilon_end','epsilon_decay_steps')}
    replay=ReplayBuffer(config.train.replay_capacity,config.semantic_versions(),env.action_spec.signature(),config.train.replay_mode,config.train.seed+41)
    counters={'env_step':0,'update_step':0,'episodes':0}
    if saved:
        agent.load_state_dict(saved['agent']);replay.load_state_dict(saved['replay']);sampler.load_state_dict(saved['sampler'])
        counters.update(saved['counters'])
        for name,state in saved['local_rngs'].items():rngs[name].bit_generator.state=state
        restore_rng(saved['rng'])
    run_dir,manifest=create_run(config,manifest_data,saved,run_kind)
    coverage=freeze_run_coverage(run_dir,manifest_data,config,dataset)
    metadata=method_metadata(algorithm,settings,config)
    if hasattr(agent,'agent_spec'):metadata['agent_spec']=agent.agent_spec()
    if algorithm=='mlp_ppo':
        metadata['agent_spec']={'input_dim':agent.input_dim,'hidden_width':agent.width,'hidden_layers':2,'hyperparameters':settings,
                                'rollout_protocol':'collect_with_frozen_behavior_then_update_no_replay'}
        metadata['not_applicable_common_fields']=['warmup_steps','replay_capacity','replay_mode','update_schedule','updates_per_step','updates_per_episode','alpha_lr','target_entropy_ratio','tau','model.critic','model.spectrum_tokens']
        metadata['agent_spec'].update(actor='gate_and_independent_scoring_joint_mask',critic='state_value')
    if algorithm in ('mlp_sac','cnn_sac','tsac_205'):
        metadata['agent_spec']={'input_dim':5+2*config.physics.num_slots,'hidden_width':config.model.d_model,
                               'encoder':config.model.encoder,'actor':'independent_scoring_joint_mask','critic':'three_additive_branches_plus_skip',
                               'convolution_channels':[2,32,64] if algorithm=='cnn_sac' else None,
                               'convolution_kernels':[5,3] if algorithm=='cnn_sac' else None,'mlp_hidden_layers':2 if algorithm!='tsac_205' else None}
    networks=[getattr(agent,key) for key in ('actor','critic','critic_1','critic_2','online') if hasattr(agent,key)]
    metadata['trainable_parameters']=sum(p.numel() for network in networks for p in network.parameters() if p.requires_grad)
    metadata['trace_detail']='basic_all保留全部动作/资源/逐束速率满足度；训练逐槽SINR和干扰数组不记录，终局交接独立复算提供'
    manifest.update(algorithm=DISPLAY[algorithm],paper_baseline=metadata,baseline_settings=settings,
                    coverage_snapshot=coverage,action_spec_signature=env.action_spec.signature(),
                    resume_env_step=counters['env_step'],resume_update_step=counters['update_step'])
    atomic_json(run_dir/'运行清单.json',manifest)
    recorder=Recorder(run_dir,manifest,config.telemetry) if config.telemetry.enabled else None
    if recorder:
        for filename in ('配置快照.json','数据划分.json'):recorder.publish_artifact(run_dir/filename)
        for item in coverage['artifacts'].values():recorder.publish_artifact(run_dir/item['path'],kind=item['kind'],artifact_id=item['artifact_id'])
    started=time.monotonic();heartbeat=started;last_checkpoint=None;summaries=[]

    def basic_snapshot():
        value=env.snapshot()
        for beam in value['beams']:
            beam['sinr_db']=None;beam['interference_w']=None
        return value

    def log_update(values, samples):
        counters['update_step']=agent.update_step
        values=dict(values)
        values['sample_references']=[dict(episode_id=t.episode_id,step_index=t.step_index,env_step=t.env_step,scenario_id=t.scenario_id) for t in samples]
        if algorithm!='mlp_ppo':values['replay']=replay.diagnostics()
        if recorder:recorder.record_update(values,values.get('update_step',counters['update_step']),env_step=counters['env_step'])

    def offpolicy_update(trigger):
        if algorithm=='mlp_ppo' or trigger!=config.train.update_schedule or counters['env_step']<config.train.warmup_steps or not len(replay):return
        count=config.train.updates_per_step if trigger=='env_step' else config.train.updates_per_episode
        for _ in range(count):
            batch=replay.sample(config.train.batch_size)
            log_update(agent.update(batch),batch)

    try:
        for offset in range(config.train.episodes):
            if config.train.max_env_steps and counters['env_step']>=config.train.max_env_steps:break
            row=sampler.sample();scenario=load_scenario(source(row),config,rng=rngs['order'])
            if config.data.augment and scenario.source_schema=='legacy_lon_in_lat':scenario=augment_scenario(scenario,rngs['augmentation'])
            obs,_=env.reset(scenario,config.train.seed+counters['episodes'])
            episode_id=uuid.uuid4().hex
            scenario_path=run_dir/'场景快照'/('场景-'+episode_id+'.json');atomic_json(scenario_path,scenario.to_dict())
            snapshot=basic_snapshot() if recorder else None
            if recorder:
                recorder.publish_artifact(scenario_path,kind='scenario')
                recorder.start_episode(episode_id,scenario.scenario_id,snapshot,env_step=counters['env_step'])
            transitions=[];old_logp=[];old_values=[];total=0.;truncated=False;episode_start=time.monotonic()
            while not obs['terminal']:
                if algorithm=='mlp_ppo':
                    action,lp,value=agent.collect(obs);old_logp.append(lp);old_values.append(value)
                elif algorithm=='mlp_dqn':action=agent.act(obs)
                elif counters['env_step']<config.train.warmup_steps:
                    action=env.action_spec.decode(int(rngs['exploration'].choice(np.flatnonzero(obs['valid_action_mask']))))
                else:action=agent.act(obs)
                following,reward,terminated,_,info=env.step(action);counters['env_step']+=1
                truncated=bool(config.train.max_env_steps and counters['env_step']>=config.train.max_env_steps and not terminated)
                transitions.append(Transition(obs,env.action_spec.encode(action),reward,following,terminated,truncated,scenario.scenario_id,
                                               config.semantic_versions(),episode_id,env.cursor,counters['env_step']))
                total+=reward
                if recorder:
                    before=snapshot;snapshot=basic_snapshot()
                    recorder.record_step(episode_id,scenario.scenario_id,env.cursor,before,snapshot,
                        dict(action=action.to_dict(),reward=reward,terminated=terminated,truncated=truncated,**info),env_step=counters['env_step'])
                obs=following
                if algorithm!='mlp_ppo':
                    if terminated:replay.add_episode(transitions,scenario_id=scenario.scenario_id,n_demand=scenario.n_demand,domain_cell=row.get('domain_cells'))
                    offpolicy_update('env_step')
                now=time.monotonic()
                if recorder and now-heartbeat>=5:
                    recorder.event('performance',{'steps_per_second':(counters['env_step']-manifest['resume_env_step'])/(now-started),
                        'updates_per_second':(counters['update_step']-manifest['resume_update_step'])/(now-started),
                        'gpu_memory_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None},env_step=counters['env_step'],phase='train')
                    heartbeat=now
                if truncated:break
            if algorithm=='mlp_ppo' and transitions:
                diagnostic=agent.update_rollout(transitions,old_log_probs=old_logp,old_values=old_values)
                for item in diagnostic['mini_updates']:
                    log_update(item,[transitions[i] for i in item['rollout_indices']])
                counters['update_step']=agent.update_step
            elif transitions:offpolicy_update('episode')
            counters['episodes']+=1
            metrics=dict(env.evaluate().metrics)
            if metrics['mean_satisfaction'] is not None and not np.isclose(total,config.env.reward_scale*metrics['mean_satisfaction'],atol=1e-8,rtol=1e-8):raise AssertionError('奖励与U不一致')
            metrics.update(episode_return=total,episode_seconds=time.monotonic()-episode_start,env_step=counters['env_step'],update_step=counters['update_step'],
                           episode_id=episode_id,scenario_id=scenario.scenario_id,truncated=truncated)
            summaries.append(metrics)
            if recorder:recorder.end_episode(episode_id,metrics,scenario_id=scenario.scenario_id,env_step=counters['env_step'],terminated=env.terminal,truncated=truncated)
            print(json.dumps(dict(run_id=manifest['run_id'],algorithm=algorithm,episode=counters['episodes'],env_step=counters['env_step'],updates=counters['update_step'],U=metrics['mean_satisfaction']),ensure_ascii=False),flush=True)
            timed_out=bool(config.train.max_wall_seconds and time.monotonic()-started>=config.train.max_wall_seconds)
            due=counters['episodes']%config.train.checkpoint_every==0 or offset+1==config.train.episodes or timed_out or (config.train.max_env_steps and counters['env_step']>=config.train.max_env_steps)
            if env.terminal and due:
                last_checkpoint=save_checkpoint(run_dir,agent,replay,sampler,config,manifest,counters,recorder.flush() if recorder else 0,rngs)
                if recorder:
                    recorder.publish_artifact(last_checkpoint);recorder.publish_artifact(last_checkpoint.parent/'检查点清单.json')
            if truncated or timed_out:break
        report={'algorithm':algorithm,'counters':counters,'episodes':summaries,'elapsed_seconds':time.monotonic()-started,
                'replay':replay.diagnostics(),'sampler':sampler.state_dict(),'paper_baseline':metadata,
                'update_protocol':'on_policy_minibatch_epochs' if algorithm=='mlp_ppo' else 'completed_episode_replay_'+config.train.update_schedule,
                'checkpoint_available':last_checkpoint is not None}
        atomic_json(run_dir/'基线训练结果.json',report)
        if recorder:recorder.close()
        else:
            manifest.update(status='completed');atomic_json(run_dir/'运行清单.json',manifest)
        return run_dir,report
    except BaseException as error:
        atomic_json(run_dir/'中断记录.json',dict(error=str(error),traceback=traceback.format_exc(),counters=counters))
        if recorder:
            try:recorder.event('training_failed',dict(error=str(error),traceback=traceback.format_exc()),severity='ERROR');recorder.close('failed')
            except Exception:pass
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--algorithm',required=True,choices=LEARNED);parser.add_argument('--config');parser.add_argument('--dataset');parser.add_argument('--resume');parser.add_argument('--settings')
    parser.add_argument('--preflight',action='store_true',help='标记有界设备诊断，不能冒充正式基线训练')
    args=parser.parse_args();settings=json.loads(Path(args.settings).read_text(encoding='utf-8')) if args.settings else None
    directory,report=train_baseline(args.algorithm,load_config(args.config),settings,args.dataset,args.resume,'paper_baseline_preflight' if args.preflight else 'paper_baseline')
    print(json.dumps(dict(run_dir=str(directory),counters=report['counters']),ensure_ascii=False))


if __name__=='__main__':main()
