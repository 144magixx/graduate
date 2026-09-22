import copy
import unittest
import numpy as np
import torch
from torch import nn
from implementations.horizon_tsac_20260920.config import Config
from implementations.horizon_tsac_20260920.data.loader import scenario_from_rows
from implementations.horizon_tsac_20260920.env.environment import Environment
from implementations.horizon_tsac_20260920.rl.replay import Transition
from implementations.horizon_tsac_20260920.paper_baselines.specs import adapt_config,build_agent


class PaperSACTests(unittest.TestCase):
    def setup_agent(self,kind):
        c=Config();c.physics.num_slots=4;c.physics.num_groups=1;c.env.max_block_length=2;c.env.power_levels_w=(5.,10.)
        c.model.d_model=16;c.model.attention_heads=2;c.model.encoder_layers=1;c.train.microbatch_size=2
        c=adapt_config(c,kind)
        s=scenario_from_rows([dict(latitude_deg=30+i*.1,longitude_deg=110+i*.1,demand_bps=2e8,ground_diameter_deg=1.) for i in range(3)],c,'standard')
        e=Environment(c);o,_=e.reset(s);e.step(1);o=e.observe()
        a=build_agent(kind,c,e.action_spec,o)
        return c,e,o,a

    def test_architecture_and_information_are_actual_local_baselines(self):
        for kind in ('mlp_sac','cnn_sac'):
            c,e,o,a=self.setup_agent(kind)
            self.assertFalse(any(isinstance(m,nn.MultiheadAttention) for m in a.actor.encoder.modules()))
            self.assertEqual(any(isinstance(m,nn.Conv1d) for m in a.actor.encoder.modules()),kind=='cnn_sac')
            altered=copy.deepcopy(o);altered['beam_static']*=7;altered['global_features']*=2
            np.testing.assert_array_equal(a.probabilities(o),a.probabilities(altered))
            p=a.probabilities(o);self.assertAlmostEqual(float(p.sum()),1,places=6)
            self.assertTrue(np.all(p[~o['valid_action_mask']]==0))

    def test_both_encoders_receive_training_gradient_and_restore(self):
        torch.set_num_threads(1)
        for kind in ('mlp_sac','cnn_sac'):
            torch.manual_seed(4)
            c,e,o,a=self.setup_agent(kind)
            action=a.act(o);n,r,t,x,_=e.step(action)
            transition=Transition(o,e.action_spec.encode(action),r,n,t,x,'fixture',c.semantic_versions())
            before=[v.detach().clone() for v in a.actor.encoder.parameters()]
            diagnostics=a.update([transition,transition])
            self.assertTrue(np.isfinite(diagnostics['actor_loss']))
            self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before,a.actor.encoder.parameters())))
            restored=build_agent(kind,c,e.action_spec,o);restored.load_state_dict(a.state_dict())
            np.testing.assert_array_equal(a.probabilities(o),restored.probabilities(o))


if __name__=='__main__':unittest.main()
