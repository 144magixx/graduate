import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from implementations.spectrum_tsac_20260921 import artifacts, train
from implementations.spectrum_tsac_20260921.audit import sha256
from implementations.spectrum_tsac_20260921.data.split import manifest_hash
from tests.spectrum_tsac_20260921.test_data import fixture
from tests.spectrum_tsac_20260921.test_schedule import tiny_config


class LongRunControlsTests(unittest.TestCase):
    def run_case(self, config):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'source.json'
            source.write_text(json.dumps(fixture(3,config).to_dict()),encoding='utf-8')
            manifest={'dataset_version':config.data.dataset_version,'records':[dict(scenario_id='fixture',scenario_path=str(source),source_hash=sha256(source),split='train')]}
            manifest['manifest_hash']=manifest_hash(manifest)
            dataset=root/'dataset.json';dataset.write_text(json.dumps(manifest),encoding='utf-8')
            with patch.object(artifacts,'SPECTRUM_OUTPUT_DIR',root/'runs'),contextlib.redirect_stdout(io.StringIO()):
                run_dir,report=train.run_training(config,'longrun_control_fixture',dataset)
            saved=[json.loads(p.read_text(encoding='utf-8'))['counters']['episodes']
                   for p in (run_dir/'检查点存档').glob('*/检查点清单.json')]
            return report,sorted(saved)

    def test_checkpoint_interval_and_final_boundary(self):
        config=tiny_config();config.train.episodes=3;config.train.checkpoint_every=2
        report,saved=self.run_case(config)
        self.assertEqual(saved,[2,3])
        self.assertEqual(report['counters']['env_step'],9)

    def test_wall_budget_finishes_boundary_and_saves(self):
        config=tiny_config();config.train.episodes=20;config.train.checkpoint_every=100
        config.train.max_wall_seconds=1e-9
        report,saved=self.run_case(config)
        self.assertEqual(saved,[1]);self.assertEqual(report['counters']['env_step'],3)
        self.assertFalse(report['episodes'][0]['truncated'])

    def test_invalid_runtime_controls(self):
        for name,value in [('checkpoint_every',0),('checkpoint_every',True),('max_wall_seconds',float('inf'))]:
            config=tiny_config();setattr(config.train,name,value)
            with self.assertRaises(ValueError):config.validate()
