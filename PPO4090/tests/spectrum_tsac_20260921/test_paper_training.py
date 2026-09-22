import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
import torch
from implementations.spectrum_tsac_20260921 import artifacts
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.data.split import manifest_hash
from implementations.spectrum_tsac_20260921.paper_baselines.train import train_baseline
from implementations.spectrum_tsac_20260921.paper_baselines.specs import adapt_config
from implementations.spectrum_tsac_20260921.evaluate import evaluate_run
from implementations.spectrum_tsac_20260921.rl.replay import ReplayBuffer
from tests.spectrum_tsac_20260921.test_schedule import tiny_config
from tests.spectrum_tsac_20260921.test_data import fixture


def dataset_at(root,c):
    rows=[]
    for index,split in enumerate(('train','validation')):
        scenario=fixture(3,c);value=scenario.to_dict();value['scenario_id']=f'fixture_{index}'
        value['demand_bps']=[x*(1+.1*index) for x in value['demand_bps']]
        path=root/f'{split}.json';path.write_text(json.dumps(value),encoding='utf-8')
        rows.append(dict(scenario_id=value['scenario_id'],scenario_path=str(path),source_hash=sha256(path),split=split))
    data={'dataset_version':c.data.dataset_version,'records':rows};data['manifest_hash']=manifest_hash(data)
    path=root/'dataset.json';path.write_text(json.dumps(data),encoding='utf-8');return path


@pytest.mark.parametrize('algorithm',['mlp_sac','cnn_sac','mlp_dqn','mlp_ppo'])
def test_complete_local_baseline_chain(algorithm,tmp_path):
    c=tiny_config();c.train.checkpoint_every=1;c.train.updates_per_step=1
    settings={'epochs':2,'minibatch_size':2} if algorithm=='mlp_ppo' else None
    data=dataset_at(tmp_path,c)
    with patch.object(artifacts,'SPECTRUM_OUTPUT_DIR',tmp_path/'runs'):
        if algorithm=='mlp_ppo':
            with patch.object(ReplayBuffer,'sample',side_effect=AssertionError('PPO must not sample replay')):
                directory,result=train_baseline(algorithm,c,settings,data)
        else:directory,result=train_baseline(algorithm,c,settings,data)
    assert result['counters']['env_step']==6
    assert result['counters']['update_step']==(8 if algorithm=='mlp_ppo' else 4)
    assert result['replay']['transitions']==(0 if algorithm=='mlp_ppo' else 6)
    manifest=json.loads((directory/'运行清单.json').read_text(encoding='utf-8'))
    assert manifest['paper_baseline']['algorithm_id']==algorithm
    evaluated=evaluate_run(directory/'运行清单.json',limit=1,policies=('policy',),export_results=True)
    assert evaluated['summary']['policy']['constraint_violations']==0
    assert evaluated['results']['policy'][0]['phase']=='validation'


def test_paper_heuristics_via_common_evaluator(tmp_path):
    c=tiny_config();c.physics.num_slots=8;c.env.max_block_length=5;c.env.power_levels_w=(5.,10.,20.,25.,30.);c.env.power_budget_w=60
    data=dataset_at(tmp_path,c)
    with patch.object(artifacts,'SPECTRUM_OUTPUT_DIR',tmp_path/'runs'):
        directory,_=train_baseline('mlp_sac',c,dataset=data)
    report=evaluate_run(directory/'运行清单.json',limit=1,policies=('paper_fixed','paper_greedy','paper_random'))
    for name in ('paper_fixed','paper_greedy','paper_random'):
        assert report['summary'][name]['constraint_violations']==0
        assert report['results'][name][0]['heuristic_spec']


def test_ppo_controller_restore_matches_continuous(tmp_path):
    c=tiny_config();c.train.checkpoint_compression=True;data=dataset_at(tmp_path,c);settings={'epochs':2,'minibatch_size':2}
    with patch.object(artifacts,'SPECTRUM_OUTPUT_DIR',tmp_path/'runs'):
        whole,_=train_baseline('mlp_ppo',c,settings,data)
        checkpoints=list((whole/'检查点存档').glob('*/检查点.pt'))
        first=min(checkpoints,key=lambda p:json.loads((p.parent/'检查点清单.json').read_text(encoding='utf-8'))['counters']['episodes'])
        c2=copy.deepcopy(c);c2.train.episodes=1
        resumed,_=train_baseline('mlp_ppo',c2,settings,data,resume=first)
    left=artifacts.load_checkpoint(whole/'检查点.pt');right=artifacts.load_checkpoint(resumed/'检查点.pt')
    assert (whole/'检查点.pt').read_bytes()[:2]==b'\x1f\x8b'
    assert left['counters']==right['counters']
    for key in ('actor','critic'):
        for name,value in left['agent'][key].items():torch.testing.assert_close(value,right['agent'][key][name],rtol=0,atol=0)


def test_offpolicy_episode_schedule_and_reward_scale_are_honored(tmp_path):
    c=tiny_config();c.train.update_schedule='episode';c.train.updates_per_episode=2;c.env.reward_scale=37.
    data=dataset_at(tmp_path,c)
    with patch.object(artifacts,'SPECTRUM_OUTPUT_DIR',tmp_path/'runs'):
        _,report=train_baseline('mlp_dqn',c,dataset=data)
    assert report['counters']['update_step']==4
    for episode in report['episodes']:
        assert episode['episode_return']==pytest.approx(37*episode['mean_satisfaction'])
