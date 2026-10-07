"""Vectorized nominal dynamics for the frozen feedback controller.

The controller, feedback QP, candidate policies, score and update schedule stay
unchanged. Only independent vessel integrations inside prediction are batched.
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from feedback_joint_control import (FeedbackJointController, NominalEnvironment,
    nominal_scene, observe, minimum_separation)
from gate_study_control import NominalModel


@lru_cache(maxsize=1)
def model():
    return NominalModel()


class FastNominalEnvironment(NominalEnvironment):
    def step(self,rl_action,controller='RL',att_action=None):
        if controller!='RL' or att_action is not None:
            raise ValueError('The fast nominal adapter implements physical defender commands only.')
        boats=[self.attacker,*self.defender_list]
        obstacles=[SimpleNamespace(pos=b.pos,r=self.Obs_R) for b in self.defender_list]
        attacker=self._APF_navi_step(self.attacker.pos,np.zeros(2),obstacles,self.attacker.theta)
        forces=np.vstack((attacker,self.action_to_thrust(rl_action)))
        forces=np.clip(forces,np.array([b.min_thrust for b in boats])[:,None],
                       np.array([b.max_thrust for b in boats])[:,None])
        state=np.array([[b.x,b.y,b.theta,*b.velocity_r] for b in boats])
        dynamics=model()
        dt=boats[0].dt
        for _ in range(boats[0].N):
            measured_velocity=state[:,3:].copy()
            state[:,:2]+=measured_velocity[:,:2]*dt
            state[:,2]=(state[:,2]+measured_velocity[:,2]*dt)%(2*np.pi)
            state[:,3:]+=dynamics.acceleration(state,forces)*dt
        for i,boat in enumerate(boats):
            boat.x,boat.y,boat.theta=state[i,:3]
            boat.pos[:]=state[i,:2]
            boat.velocity_r=state[i,3:].copy()
            boat.velocity=measured_velocity[i].copy()
            boat.vel=boat.velocity[:2]
            boat.left_thrust,boat.right_thrust=forces[i]
        positions=state[1:,:2]
        self._Boid_navi_step(positions,measured_velocity[1:,:2],state[1:,2],state[0,:2])
        observations,observation=self._get_obs()
        self.att_action=self.thrust_to_action(attacker,self.attacker.agility)
        self.Current_T+=self.Action_T
        done=self._isTerminate()
        return observations,self._get_rewards(done),done,observation


class FastFeedbackJointController(FeedbackJointController):
    def predict(self,measurement,template):
        env=nominal_scene(measurement,self.previous_thrust,self.previous_attacker_thrust,
                          self.velocity_lag,self.attacker_agility)
        env.__class__=FastNominalEnvironment
        current=measurement
        positions=[measurement.defenders.copy()]
        attackers=[measurement.attacker.copy()]
        commands=[]
        distance=minimum_separation(measurement.defenders)
        deviation=0.
        done=0
        held=None
        for _ in range(self.block_steps):
            command,_,reference=self.feedback(current,template)
            if self.prediction=='held':
                held=command.copy() if held is None else held
                command=held
            commands.append(command.copy())
            deviation+=float(np.mean(((command-reference)/1500.)**2))
            observations,_,done,_=env.step(env.thrust_to_action(command),'RL')
            if observations.shape[1]<measurement.observations.shape[1]:
                observations=np.pad(observations,((0,0),(0,measurement.observations.shape[1]-observations.shape[1])))
            current=observe(env,observations)
            positions.append(current.defenders.copy())
            attackers.append(current.attacker.copy())
            distance=min(distance,minimum_separation(current.defenders))
            if done:
                break
        nearest=float(np.linalg.norm(current.defenders[:,:2]-current.attacker[:2],axis=-1).min())
        guard_margin=float(np.linalg.norm(current.attacker[:2])-np.linalg.norm(current.defenders[:,:2],axis=-1).min())
        cost=(nearest/60. + 2.*(max(0.,5.-guard_margin)/60.)**2
              + .1*(max(0.,7.-distance)/2.)**2 + .02*deviation/len(commands))
        score=(int(done==2),int(done==1),-int(done>2),cost)
        return dict(template=template,score=score,outcome=int(done),minimum_distance=distance,
                    positions=np.asarray(positions),attackers=np.asarray(attackers),
                    commands=np.asarray(commands),guard_margin=guard_margin,intercept_distance=nearest)
