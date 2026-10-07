"""Parallel forecasting preserves complete decisions across feedback replans."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from evaluate_feedback_joint import CHECKPOINT
from feedback_joint_control import NominalEnvironment,observe
from parallel_rollout_interception import PredictionPool,ParallelRolloutInterceptionController
from rollout_interception import RolloutInterceptionController
from source_arboids import SourcePolicy


class ParallelRolloutContract(unittest.TestCase):
    def test_parallel_decisions_and_plans_are_bit_identical(self):
        torch.set_num_threads(1)
        policy=SourcePolicy(CHECKPOINT)
        settings=dict(block_steps=10,blend=1.,tail_steps=100)
        with PredictionPool(CHECKPOINT,workers=5) as pool:
            for n in (3,6):
                np.random.seed(654+n)
                env=NominalEnvironment(defender_num=n)
                obs,_=env.reset(agility=2.25,noisy_agility=False)
                serial=RolloutInterceptionController(n,policy,**settings)
                parallel=ParallelRolloutInterceptionController(n,policy,pool,**settings)
                for _ in range(20):
                    measurement=observe(env,obs)
                    before=np.random.get_state()
                    a,ia=serial.control(measurement)
                    b,ib=parallel.control(measurement)
                    np.testing.assert_array_equal(a,b)
                    self.assertEqual(ia['template'],ib['template'])
                    for first,second in zip(before,np.random.get_state()):
                        np.testing.assert_array_equal(first,second)
                    if ia['planned']:
                        for key in ('score','baseline_score','positions','attackers','commands'):
                            np.testing.assert_array_equal(serial.plan[key],parallel.plan[key])
                    obs,_,done,_=env.step(env.thrust_to_action(a),'RL')
                    if done:
                        break


if __name__=='__main__':
    unittest.main()
