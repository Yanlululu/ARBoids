"""Causal, committed-feedback prediction with continuous joint thrust correction.

The rollout and executor call the same feedback function. A block commits only
policy parameters, never a constant physical thrust. The block has no hidden
recursive planner: outer parameters are reconsidered only after its last tick.
"""
import study_runtime
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from source_arboids import numerical_source, snapshot, source_mixture
from gate_study_control import NominalModel
from cbf_source_baseline import CBFController


TEMPLATES = ('baseline', 'more_learned', 'more_boids', 'interceptor_learned', 'interceptor_boids')
SOURCE = numerical_source()


@lru_cache(maxsize=8)
def shared_safety(defenders):
    # CBFpy safety_filter is a stateless QP; sharing avoids recompiling identical
    # dimensions in every episode. Episode policy state is never shared here.
    return CBFController(defenders)


@dataclass(frozen=True)
class Measurement:
    defenders: np.ndarray
    attacker: np.ndarray
    observations: np.ndarray
    boids: np.ndarray
    time: float = 0.
    capture_radius: float = 5.
    total_time: float = 80.

    def __post_init__(self):
        n = len(self.defenders)
        if (self.defenders.shape != (n, 6) or self.attacker.shape != (6,)
                or self.boids.shape != (n, 2) or len(self.observations) != n):
            raise ValueError('Invalid public measurement dimensions.')
        if n < 2 or not all(np.isfinite(x).all() for x in
                           (self.defenders, self.attacker, self.observations, self.boids)):
            raise ValueError('Non-finite public measurements or too few defenders.')


def observe(env, observations):
    """Read publicly measurable states, not velocity_r, RNG, or attacker agility."""
    attacker = np.array([env.attacker.x, env.attacker.y, env.attacker.theta, *env.attacker.velocity])
    return Measurement(snapshot(env), attacker, observations.copy(), env.boids_actions.copy(),
                       float(env.Current_T), float(env.Defend_R), float(env.Total_T))


class NominalEnvironment(SOURCE.TADEnv):
    def generate_random_current(self):
        return np.zeros(3)

    def _APF_navi_step(self, position, goal, obstacles, phi, robot='att'):
        # The source APF formula with only its additive random force removed.
        attraction = .1*(goal-position)
        repulsion = np.zeros(2)
        radius = 50.
        for obstacle in obstacles:
            distance = max(np.linalg.norm(position-obstacle.pos)-obstacle.r, .1)
            if distance < radius:
                direction = position-obstacle.pos
                direction = direction/np.linalg.norm(direction)
                magnitude = 3000.*(1./distance-1./radius)/(distance**2)
                repulsion += magnitude*direction
                radius = distance
        return self.force_to_thrust(attraction+repulsion, phi, robot)


def nominal_attacker_command(measurement, agility=2.25):
    env = NominalEnvironment(defender_num=len(measurement.defenders))
    env.attacker.agility = agility
    env.attacker.reset(measurement.attacker[:2].copy(), measurement.attacker[2])
    from types import SimpleNamespace
    obstacles = [SimpleNamespace(pos=s[:2], r=env.Obs_R) for s in measurement.defenders]
    return env._APF_navi_step(measurement.attacker[:2], np.zeros(2), obstacles, measurement.attacker[2])


def nominal_scene(measurement, previous_thrust=None, previous_attacker_thrust=None,
                  velocity_lag=.05, attacker_agility=2.25):
    """Rebuild a nominal model solely from measurements and known past commands."""
    env = NominalEnvironment(defender_num=len(measurement.defenders))
    env.attacker.agility = attacker_agility
    model = NominalModel()
    states = np.vstack((measurement.attacker, measurement.defenders))
    previous = [previous_attacker_thrust, previous_thrust]
    for index, (boat, state) in enumerate(zip([env.attacker, *env.defender_list], states)):
        boat.reset(state[:2].copy(), float(state[2]))
        observed_velocity = state[3:].copy()
        force = previous[0] if index == 0 else (None if previous[1] is None else previous[1][index-1])
        endpoint_velocity = observed_velocity.copy()
        if force is not None and velocity_lag:
            endpoint_velocity += velocity_lag*model.acceleration(state[None], np.asarray(force,dtype=float)[None])[0]
        boat.velocity_r = endpoint_velocity
        # Preserve the source's timestamped measurement cache at the boundary.
        boat.velocity = observed_velocity
        boat.vel = boat.velocity[:2]
    env.Current_T, env.Total_T = measurement.time, measurement.total_time
    env.Defend_R = measurement.capture_radius
    env.att_action = np.zeros(2)
    env.def_att_dists = np.zeros(env.defender_num)
    env.def_def_dists = np.zeros((env.defender_num, env.defender_num-1))
    env.Pos_Att = env.attacker.pos.copy()
    env.Phi_Att = env.attacker.theta
    env.Pos_Def = np.asarray([b.pos for b in env.defender_list]).flatten()
    env.Phi_Def = np.asarray([b.theta for b in env.defender_list])
    env._Boid_navi_step(measurement.defenders[:, :2], measurement.defenders[:, 3:5],
                        measurement.defenders[:, 2], measurement.attacker[:2])
    env._get_obs()
    env.Rewards = env._get_rewards(0)
    return env


def minimum_separation(states):
    i,j=np.triu_indices(len(states),1)
    return float(np.linalg.norm(states[i,:2]-states[j,:2],axis=-1).min())


def project_thrust(physical, raw_action, boids):
    from source_arboids import actor_thrust
    delta=actor_thrust(raw_action).astype(float)-boids
    denominator=np.sum(delta*delta,axis=-1)
    theta=np.divide(np.sum((physical-boids)*delta,axis=-1),denominator,
                    out=np.zeros(len(raw_action)),where=denominator>1e-20)
    theta=np.clip(theta,0.,1.)
    updated=raw_action.copy()
    updated[:,2]=theta
    return source_mixture(updated,boids).astype(float)


class FeedbackJointController:
    def __init__(self, defenders, policy, block_steps=5, gate_only=False,
                 prediction='feedback', velocity_lag=.05, attacker_agility=2.25,
                 force_planning=False):
        if block_steps not in (5,10) or prediction not in ('feedback','held'):
            raise ValueError('This study fixes block lengths to 5/10 and explicit feedback/held modes.')
        self.defenders, self.policy, self.block_steps = defenders, policy, block_steps
        self.gate_only, self.prediction = gate_only, prediction
        self.velocity_lag, self.attacker_agility = velocity_lag, attacker_agility
        self.force_planning = force_planning
        self.safety = shared_safety(defenders)
        self.reset()

    def reset(self, previous_thrust=None):
        self.remaining=0
        self.template='baseline'
        self.previous_thrust=None if previous_thrust is None else np.array(previous_thrust,dtype=float,copy=True)
        self.previous_attacker_thrust=None
        self.plan=None
        self.plan_count=0
        self.tick=0

    def feedback(self, measurement, template):
        action=self.policy(measurement.observations)
        raw=action.copy()
        offset=np.zeros(self.defenders)
        if template=='more_learned':
            offset[:]=.25
        elif template=='more_boids':
            offset[:]=-.25
        elif template in ('interceptor_learned','interceptor_boids'):
            interceptor=int(np.argmin(np.linalg.norm(measurement.defenders[:,:2]-measurement.attacker[:2],axis=-1)))
            offset[:]=-.25
            offset[interceptor]=.25
            if template=='interceptor_boids':
                offset=-offset
        elif template!='baseline':
            raise ValueError(template)
        action[:,2]=np.clip(action[:,2]+offset,0.,1.)
        _,physical,info=self.safety.control(measurement.defenders,action,measurement.boids)
        if self.gate_only:
            physical=project_thrust(physical,raw,measurement.boids)
            reference=source_mixture(action,measurement.boids).astype(float).reshape(-1)/1000.
            G,h=self.safety.constraints(measurement.defenders.reshape(-1),reference)
            count=self.defenders*(self.defenders-1)//2
            residual=np.asarray(G)@(physical.reshape(-1)/1000.)-np.asarray(h)
            info['cbf_constraint_violation']=float(max(0.,residual[:count].max()))
        physical=np.clip(physical,-500.,1000.)
        if not np.isfinite(physical).all():
            raise FloatingPointError('Continuous feedback returned non-finite thrust.')
        return physical,info,source_mixture(raw,measurement.boids).astype(float)

    def predict(self, measurement, template):
        env=nominal_scene(measurement,self.previous_thrust,self.previous_attacker_thrust,
                          self.velocity_lag,self.attacker_agility)
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
        # Fixed, dimensionless mission score; the first terms prefer valid task
        # outcomes. The original reference remains available at every choice.
        cost=(nearest/60. + 2.*(max(0.,5.-guard_margin)/60.)**2
              + .1*(max(0.,7.-distance)/2.)**2 + .02*deviation/len(commands))
        score=(int(done==2),int(done==1),-int(done>2),cost)
        return dict(template=template,score=score,outcome=int(done),minimum_distance=distance,
                    positions=np.asarray(positions),attackers=np.asarray(attackers),
                    commands=np.asarray(commands),guard_margin=guard_margin,intercept_distance=nearest)

    def control(self, measurement):
        planned=self.remaining==0
        considered=0
        if planned:
            baseline=self.predict(measurement,'baseline')
            candidates=[baseline]
            if self.force_planning or baseline['minimum_distance']<8. or baseline['outcome']==1:
                candidates += [self.predict(measurement,t) for t in TEMPLATES[1:]]
            self.plan=min(candidates,key=lambda p:p['score'])
            self.plan['baseline_score']=baseline['score']
            self.template=self.plan['template']
            self.remaining=self.block_steps
            self.plan_count+=1
            considered=len(candidates)
        physical,info,_=self.feedback(measurement,self.template)
        # This estimate uses current measurements only; it is never the actual
        # future APF disturbance or a hidden applied attacker command.
        self.previous_attacker_thrust=nominal_attacker_command(measurement,self.attacker_agility)
        self.previous_thrust=physical.copy()
        offset=self.block_steps-self.remaining
        predicted=self.plan['positions'][offset+1] if offset+1<len(self.plan['positions']) else None
        self.remaining-=1
        self.tick+=1
        info.update(planned=planned,template=self.template,remaining=self.remaining,
                    plan_index=self.plan_count,plan_offset=offset,candidate_policies=considered,
                    predicted_next_state=None if predicted is None else predicted.copy(),
                    prediction_mode=self.prediction,gate_only=self.gate_only)
        return physical,info
