"""设备迁移的只读推理与精确续训边界。"""
import copy
import unittest

import torch

from implementations.horizon_tsac_20260920.env.action import ActionSpec
from implementations.horizon_tsac_20260920.rl import DiscreteSAC
from tests.horizon_tsac_20260920.test_sac import observation, small_config


class DeviceRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inference_does_not_restore_generator_or_global_rng(self):
        config = small_config()
        obs = observation(config)
        source = DiscreteSAC(config, ActionSpec(config), obs)
        expected = source.probabilities(obs)
        state = source.state_dict()
        # 模拟无法由本机CPU Generator接受的CUDA状态；推理不应读取它。
        state["config"]["train"]["device"] = "cuda:0"
        state["action_rng"] = torch.tensor([1], dtype=torch.uint8)
        target = DiscreteSAC(config, ActionSpec(config), obs)
        local_rng_before = target.action_rng.get_state().clone()
        global_rng_before = torch.get_rng_state().clone()
        target.load_state_dict(state, restore_rng=False)
        torch.testing.assert_close(target.action_rng.get_state(), local_rng_before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_rng_before, rtol=0, atol=0)
        torch.testing.assert_close(torch.from_numpy(target.probabilities(obs)), torch.from_numpy(expected), rtol=0, atol=0)

    def test_cross_device_exact_resume_rejected_before_weight_mutation(self):
        config = small_config()
        obs = observation(config)
        agent = DiscreteSAC(config, ActionSpec(config), obs)
        before = copy.deepcopy(agent.actor.state_dict())
        state = agent.state_dict()
        state["config"]["train"]["device"] = "cuda:0"
        for value in state["actor"].values():
            if value.is_floating_point():
                value.add_(1)
        with self.assertRaisesRegex(ValueError, "精确续训不支持跨设备类型"):
            agent.load_state_dict(state)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA测试真实跨设备checkpoint")
    def test_cuda_checkpoint_can_be_evaluated_on_cpu(self):
        config = small_config()
        config.train.device = "cuda:0"
        obs = observation(config)
        source = DiscreteSAC(config, ActionSpec(config), obs)
        state = source.state_dict()
        config.train.device = "cpu"
        target = DiscreteSAC(config, ActionSpec(config), obs)
        target.load_state_dict(state, restore_rng=False)
        probability = target.probabilities(obs)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=5)
        with self.assertRaisesRegex(ValueError, "跨设备类型"):
            target.load_state_dict(state, restore_rng=True)


if __name__ == "__main__":
    unittest.main()
