import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from implementations.spectrum_tsac_20260921.config import Config
from implementations.spectrum_tsac_20260921.artifacts import atomic_json,load_checkpoint
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.data.loader import scenario_from_rows
from implementations.spectrum_tsac_20260921.data.split import manifest_hash
from implementations.spectrum_tsac_20260921.train import run_training,prepare_dataset
from implementations.spectrum_tsac_20260921.env.environment import Environment
from implementations.spectrum_tsac_20260921.export import export_allocation,reload_and_evaluate


def tiny_config():
    c=Config()
    c.physics.num_slots=4
    c.physics.num_groups=4
    c.env.max_block_length=2
    c.env.power_levels_w=(5.,10.)
    c.env.power_budget_w=15.
    c.model.d_model=16
    c.model.encoder_layers=1
    c.model.encoder="full_pool"
    c.train.batch_size=3
    c.train.microbatch_size=2
    c.train.warmup_steps=0
    c.train.episodes=2
    c.train.replay_capacity=30
    c.data.augment=False
    return c


class IntegrationTests(unittest.TestCase):
    def test_training_checkpoint_exact_boundary_resume_and_export(self):
        c=tiny_config()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            scenario=scenario_from_rows([dict(latitude_deg=30+i,longitude_deg=100+i,demand_bps=2e8,ground_diameter_deg=1.) for i in range(3)],c,"standard","test_fixture",metadata={"source":"unit_test_only"})
            path=root/"测试场景.json"
            atomic_json(path,scenario.to_dict())
            manifest={"dataset_version":"test_only","records":[{"scenario_id":scenario.scenario_id,"scenario_path":str(path),"source_hash":sha256(path),"split":"train"}]}
            manifest["manifest_hash"]=manifest_hash(manifest)
            split=root/"测试划分.json"
            atomic_json(split,manifest)
            with patch("implementations.spectrum_tsac_20260921.artifacts.SPECTRUM_OUTPUT_DIR",root/"runs"):
                whole,report=run_training(c,"integration_fixture",split)
                checkpoints=sorted((whole/"检查点存档").glob("*/检查点.pt"),key=lambda p:json.loads((p.parent/"检查点清单.json").read_text(encoding="utf-8"))["counters"]["episodes"])
                self.assertEqual(len(checkpoints),2)
                first_hash=sha256(checkpoints[0])
                child_c=copy.deepcopy(c)
                child_c.train.episodes=1
                child,child_report=run_training(child_c,"integration_fixture",split,checkpoints[0])
                self.assertEqual(sha256(checkpoints[0]),first_hash)
                expected=load_checkpoint(whole/"检查点.pt")
                actual=load_checkpoint(child/"检查点.pt")
                self.assertEqual(expected["counters"],actual["counters"])
                for key in ("actor","critic_1","critic_2","target_critic_1","target_critic_2"):
                    for name,value in expected["agent"][key].items():
                        torch.testing.assert_close(value,actual["agent"][key][name],rtol=0,atol=0)
                self.assertEqual(expected["sampler"],actual["sampler"])
                changed=copy.deepcopy(c)
                changed.train.gamma=.99
                with self.assertRaisesRegex(ValueError,"gamma"):
                    load_checkpoint(whole/"检查点.pt",changed)
                from implementations.spectrum_tsac_20260921.telemetry.recorder import export_run
                package=export_run(whole,root/"离线包")
                self.assertTrue((package/"运行记录.sqlite").is_file())
                self.assertEqual(len(list((package/"检查点存档").glob("*/检查点.pt"))),2)
            env=Environment(c)
            obs,_=env.reset(scenario)
            while not obs["terminal"]:
                valid=np.flatnonzero(obs["valid_action_mask"])
                obs,_,_,_,_=env.step(int(valid[-1]))
            _,verified=export_allocation(env,root/"交接")
            self.assertTrue(verified["passed"])
            self.assertTrue(reload_and_evaluate(root/"交接")["passed"])

    def test_manifest_tampering_and_physics_config_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"清单.json"
            manifest={"records":[],"manifest_hash":"forged"}
            atomic_json(path,manifest)
            with self.assertRaisesRegex(ValueError,"清单hash"):
                prepare_dataset(Config(),path)
        for name,value in (("slot_bandwidth_hz",-1),("satellite_radius_km",6000),("sinr_margin_db",float("nan"))):
            c=Config()
            setattr(c.physics,name,value)
            with self.assertRaises(ValueError): c.validate()


if __name__=="__main__":unittest.main()
