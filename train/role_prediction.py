"""Public-state closed-loop prediction of conditional residual compositions."""
import numpy as np
import jax.numpy as jnp

from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from feedback_joint_control import observe, nominal_attacker_command
from jit_nominal_environment import JitNominalEnvironment
from jit_rollout import JitRolloutController
from policy.role_residual import RoleResidualPolicy
from policy.role_value import compositions
from train_interaction import outcome_row


class PaperNominal(JitNominalEnvironment):
    def _isTerminate(self):
        self.protocol = 'paper-parameters-v1'
        return TADEnv._isTerminate(self)


class RolePrediction(JitRolloutController):
    prediction_environment_class = PaperNominal

    def __init__(self, defenders, tail_steps=100, tail_policy='baseline', additive=False):
        self.residual = RoleResidualPolicy()
        self.additive = additive
        masks = compositions(defenders)
        self.options = dict(baseline=masks[0], **{str(i): m for i, m in enumerate(masks[1:], 1)})
        super().__init__(defenders, None, block_steps=10, blend=1., capture_margin=.5,
                         failure_cost='delay', tail_steps=tail_steps, tail_policy=tail_policy)
        self.observer.grid = np.linspace(1., 8., 57)
        self.observer.reset()

    def nominal(self, measurement, template):
        return self.residual.compose(dict(obs=np.asarray(measurement.observations, dtype=np.float32),
                                         motion=measurement.defenders), self.options[template])

    def _forecast_force(self, measurement, template):
        action = self.nominal(measurement, template)
        force = 750.*action[:, :2]+250.
        result = self.safety.filter.safety_filter(jnp.asarray(measurement.defenders.reshape(-1)),
                                                 jnp.asarray(force.reshape(-1)/1000.))
        return (1000.*np.clip(np.asarray(result), -.5, 1.)).reshape(self.defenders, 2)

    def control(self, measurement):
        self.attacker_agility = self.observer.update(measurement)
        planned = self.remaining == 0
        if planned:
            predictions = [self.predict(measurement, template) for template in self.options]
            if self.additive:
                scores = np.array([360.*p['score'][0]+180.*p['score'][1]+p['score'][2] for p in predictions])
                surrogate = scores[0]+compositions(self.defenders)@(scores[1:1+self.defenders]-scores[0])
                self.plan = predictions[int(np.argmin(surrogate))]
            else:
                self.plan = min(predictions, key=lambda row: row['score'])
            self.template = self.plan['template']
            self.remaining = self.block_steps
            self.plan_count += 1
        action = self.nominal(measurement, self.template)
        _, force, info = self.safety.control(measurement.defenders, action, measurement.boids)
        self.previous_attacker_thrust = nominal_attacker_command(measurement, self.attacker_agility)
        self.previous_thrust = force.copy()
        self.remaining -= 1
        self.tick += 1
        info.update(planned=planned, template=self.template, candidate_policies=len(self.options))
        return force, info


def role_prediction_episode(seed, defenders, agility, method):
    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(defenders, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility, noisy_agility=False)
        controller = RolePrediction(defenders, tail_steps=0 if method == 'role_short' else 100,
                                    tail_policy='candidate' if method in ('role_sustained', 'role_independent') else 'baseline',
                                    additive=method == 'role_independent')
        np.random.seed(seed+1000000)
        done = 0
        while not done:
            force, _ = controller.control(observe(env, obs))
            obs, _, done, _ = env.step(np.zeros((defenders, 3)), 'AdaRes', defender_thrust=force)
        return outcome_row(env, done), []
