"""One frozen opponent interface used in behavior rollouts, every data branch and evaluation.

The supported learned policy is a deterministic feedforward actor exported by
an attacker trainer: [B,8] source observations -> [B,2] normalized commands.
Stateful or stochastic policies need a new explicit state/RNG contract, and are
rejected rather than silently reset during cloned branches. APF stays inside the
provided environment, preserving its time-indexed random disturbances.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import canonical_hash,file_hash

OBSERVATION='source-target-nearest3-raw8-v1'


def attacker_observation(world):
    observation=np.asarray(world.env._get_obs()[1],dtype=np.float32)
    if observation.shape!=(8,) or not np.isfinite(observation).all():
        raise ValueError('Attacker observation must be finite raw source [8]')
    return observation


class APFOpponent:
    def __init__(self):
        body={'kind':'apf','policy':'source-environment-APF','mode':'environment-feedback',
              'observation':OBSERVATION,'future_information':False}
        self.identity=body|{'identity_sha256':canonical_hash(body)}
    def act(self,world):
        return None


class TorchScriptOpponent:
    def __init__(self,checkpoint,metadata,device='cpu'):
        if metadata.get('observation')!=OBSERVATION:
            raise ValueError('Attacker observation contract differs')
        if metadata.get('stateful',False) or metadata.get('stochastic',False):
            raise ValueError('This entry accepts frozen deterministic feedforward actors only')
        expected=metadata.get('checkpoint_sha256')
        if not expected or expected!=file_hash(checkpoint):
            raise ValueError('Frozen attacker checkpoint SHA256 mismatch')
        self.device=device
        self.model=torch.jit.load(str(checkpoint),map_location=device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        body={'kind':'torchscript','checkpoint_sha256':expected,'observation':OBSERVATION,
              'mode':'deterministic-mean','stateful':False,'action':'normalized-left-right[-1,1]',
              'training_identity':metadata.get('training_identity','not-specified')}
        self.identity=body|{'identity_sha256':canonical_hash(body)}
        # Contract-level smoke check. A successful export must have no hidden mutable state.
        probe=torch.zeros(1,8,device=device)
        with torch.no_grad():
            first=self.model(probe);second=self.model(probe)
        self._validate(first)
        if not torch.equal(first,second):raise ValueError('Export has stochastic or stateful execution')
    @staticmethod
    def _validate(action):
        if not isinstance(action,torch.Tensor) or action.shape!=(1,2):
            raise ValueError('Attacker actor must return [1,2] tensor')
        if not torch.isfinite(action).all() or (action.abs()>1.000001).any():
            raise ValueError('Attacker action outside normalized bounds')
    @torch.no_grad()
    def act(self,world):
        x=torch.as_tensor(attacker_observation(world)[None],device=self.device)
        action=self.model(x);self._validate(action)
        return action[0].cpu().numpy().astype(np.float64)


def load_opponent(spec_path=None,device='cpu'):
    if spec_path is None:return APFOpponent()
    path=Path(spec_path);spec=json.loads(path.read_text())
    if spec.get('kind')=='apf':return APFOpponent()
    if spec.get('kind')!='torchscript':raise ValueError('Unsupported opponent kind')
    ck=Path(spec['checkpoint'])
    if not ck.is_absolute():ck=path.parent/ck
    return TorchScriptOpponent(ck,spec,device)


def attacker_training_reward(previous_radius,next_radius,events):
    """New attacker-training specification, never a CITF prediction label.

    Both physical breach and radial early failure are attacker terminal wins in
    the selected source80 protocol. Defender collision is neutral (+0 terminal).
    Defender capture and defended timeout are attacker terminal losses (-100).
    The terminal bonus is applied once via the primary outcome, not flag sums.
    """
    if not np.isfinite([previous_radius,next_radius]).all():raise ValueError('Nonfinite target distance')
    code=int(events['outcome'])
    if code not in (0,1,2,3,4):raise ValueError('Unknown terminal outcome')
    terminal={0:0.,1:100.,2:0.,3:-100.,4:-100.}[code]
    return float((previous_radius-next_radius)/60.-.01+terminal)
