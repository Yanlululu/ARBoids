"""Deterministic source Actor compiled without numerical rewrites.

Weights, per-vessel invocation, input conversion and output dtype remain the
same. This is an optional prediction acceleration, not a new trained policy.
"""
import numpy as np
import torch


class _DeterministicActor(torch.nn.Module):
    def __init__(self,actor):
        super().__init__()
        self.actor=actor

    def forward(self,observations):
        return self.actor(observations,True,False)[0]


class CompiledSourcePolicy:
    def __init__(self,source):
        module=_DeterministicActor(source.actor).eval()
        with torch.no_grad():
            traced=torch.jit.trace(module,torch.zeros(1,16),check_trace=True)
            self.program=torch.jit.freeze(traced,optimize_numerics=False)

    def __call__(self,observations):
        with torch.no_grad():
            return np.stack([self.program(torch.tensor(row[None],dtype=torch.float)).cpu().numpy().flatten()
                             for row in observations])
