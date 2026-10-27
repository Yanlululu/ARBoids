"""Public-measurement adapter for the same CIV-RL executor used in training."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import civ_rl as c
from source_arboids import source_mixture


class CIVMeasuredController:
    def __init__(self,checkpoint,source=None,inference_device='cpu',*,controller=None):
        if controller is None:
            record=c.torch.load(checkpoint,map_location='cpu',weights_only=False)
            if 'forests' in record:
                from civ_forest_value import load
                controller=load(checkpoint,inference_device)
            elif record.get('value_architecture')=='relational-counterfactual-v1':
                from civ_generalization import load
                controller=load(checkpoint,inference_device)
            else:controller=c.load_controller(checkpoint,inference_device)
        self.controller=controller
        if source is not None:
            expected=Path(__file__).resolve().parent/'experiments/main-seed42-20260921/adares1.pth'
            if c.hashlib.sha256(Path(source).read_bytes()).digest()!=c.hashlib.sha256(expected.read_bytes()).digest():
                raise ValueError('The transport source must match this pilot source policy.')
        self.executor=c.CIVExecutor(self.controller);self.reset()

    def reset(self):
        self.state=c.ExecutionState();self.last_elapsed=None

    def control(self,defenders,attacker,observations,boids,elapsed,total_time=80.):
        expected_time=80. if self.controller.source.environment_protocol=='source' else 60.
        if abs(float(total_time)-expected_time)>1e-6:
            raise ValueError('Deployment horizon must match the paired numerical task.')
        if self.last_elapsed is not None and elapsed<=self.last_elapsed:
            raise ValueError('Each controller call must supply a new measurement tick.')
        def boat(state):
            return SimpleNamespace(x=float(state[0]),y=float(state[1]),theta=float(state[2]),
                pos=np.asarray(state[:2]),vel=np.asarray(state[3:5]),velocity=np.asarray(state[3:6]),
                velocity_r=np.asarray(state[3:6]),agility=1.)
        environment=SimpleNamespace(defender_list=[boat(state) for state in defenders],attacker=boat(attacker),
            boids_actions=np.asarray(boids),Current_T=float(elapsed),Total_T=float(total_time))
        device=next(self.controller.source.parameters()).device
        observations=np.asarray(observations,dtype=np.float32)
        n=len(defenders)
        if observations.shape==(n,14+2*n):
            if np.any(observations[:,-2:]!=0.):raise ValueError('The ROS peer-padding slot must be empty.')
            observations=observations[:,:-2]
        if observations.shape!=(n,14+2*(n-1)):
            raise ValueError('Deployment needs the same public peer observations as numerical training.')
        packet=c.d.mechanism_tensor(c.d.joint_packet(environment,observations),device)
        actions,decisions=self.executor.actions(packet,[self.state])
        physical=actions[0];self.state.step+=1;self.last_elapsed=float(elapsed)
        return source_mixture(physical,environment.boids_actions),dict(control_steps=self.state.step,
            conditional_interventions=int(self.state.choice>0),physical_action=physical,
            unified_decision=decisions.get(0),execution_identity=self.controller.config.identity())
