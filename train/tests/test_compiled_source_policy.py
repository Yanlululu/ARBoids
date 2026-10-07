"""Exact output contract on real source observations for all five fleet sizes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from compiled_source_policy import CompiledSourcePolicy
from evaluate_feedback_joint import CHECKPOINT
from source_arboids import SourcePolicy, numerical_source


class CompiledPolicyContract(unittest.TestCase):
    def test_source_policy_outputs_are_bit_identical(self):
        torch.set_num_threads(1)
        source=SourcePolicy(CHECKPOINT)
        compiled=CompiledSourcePolicy(source)
        for n in (2,3,4,5,6):
            for i,agility in enumerate((1.5,2.25,3.)):
                np.random.seed(9240+n*100+i)
                env=numerical_source().TADEnv(defender_num=n)
                obs,_=env.reset(agility=agility,noisy_agility=False)
                for _ in range(20):
                    expected=source(obs)
                    actual=compiled(obs)
                    np.testing.assert_array_equal(actual,expected)
                    obs,_,done,_=env.step(expected,'AdaRes')
                    if done:
                        break


if __name__=='__main__':
    unittest.main()
