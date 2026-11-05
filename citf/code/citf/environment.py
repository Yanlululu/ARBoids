"""Shared APF and task rules for simulation, source adapters and deployment."""
from functools import lru_cache
import numpy as np


def apf_force(position, goal, obstacles, radii, *, influence=50., repulsion=3000.,
              attraction=.1, tangential=0., rule='all_obstacles'):
    """Sum obstacle forces at a fixed support radius; retain the named legacy rule."""
    if rule not in ('all_obstacles', 'legacy_ordered'):
        raise ValueError('Unknown APF obstacle rule: ' + rule)
    position, goal = np.asarray(position, dtype=float), np.asarray(goal, dtype=float)
    force = attraction * (goal - position)
    radius = float(influence)
    for obstacle, size in zip(obstacles, radii):
        delta = position - np.asarray(obstacle, dtype=float)
        norm = np.linalg.norm(delta)
        distance = max(norm - float(size), .1)
        if distance < radius:
            direction = delta / max(norm, 1e-12)
            perpendicular = np.array([-direction[1], direction[0]])
            magnitude = repulsion * (1. / distance - 1. / radius) / distance ** 2
            force = force + magnitude * ((1. - tangential) * direction + tangential * perpendicular)
            if rule == 'legacy_ordered':
                radius = distance
    return force


def task_outcome(env):
    """Finite defense task; optional radial stopping reproduces the legacy task."""
    attacker_radius = np.linalg.norm(env.attacker.pos)
    paper = getattr(env, 'protocol', 'source') == 'paper-parameters-v1'
    breach = attacker_radius <= env.Target_R if paper else attacker_radius < env.Target_R
    radial = (env.early_attacker_win and
              all(attacker_radius < np.linalg.norm(boat.pos) for boat in env.defender_list))
    if breach or radial:
        return 1
    collisions = env.def_def_dists <= env.Collision_R if paper else env.def_def_dists < env.Collision_R
    if collisions.any():
        return 2
    if (env.def_att_dists < env.Defend_R).any():
        return 3
    return 4 if env.Current_T > env.Total_T - 1e-5 else 0


class DefenseRules:
    """Apply the effective protocol to an archived environment without editing it."""
    def __init__(self, *args, radial_early_failure=False, apf_obstacle_rule='all_obstacles', **kwargs):
        super().__init__(*args, **kwargs)
        self.early_attacker_win = radial_early_failure
        self.apf_obstacle_rule = apf_obstacle_rule

    def _isTerminate(self):
        return task_outcome(self)

    def force_to_thrust(self, force, phi, robot='def'):
        if not np.any(force):
            return np.zeros(2)
        return super().force_to_thrust(force, phi, robot)

    def _APF_navi_step(self, position, goal, obstacles, phi, robot='att'):
        if self.apf_obstacle_rule == 'legacy_ordered':
            return super()._APF_navi_step(position, goal, obstacles, phi, robot)
        force = apf_force(position, goal, [x.pos for x in obstacles], [x.r for x in obstacles])
        noise = getattr(self, '_apf_force_noise', None)
        force = force + (noise() if noise is not None else np.random.normal(0., .3, 2))
        if not np.any(force):
            return np.zeros(2)
        return self.force_to_thrust(force, phi, robot)


@lru_cache(maxsize=None)
def environment_type(base):
    name = 'Defense_' + base.__module__.replace('.', '_') + '_' + base.__name__
    cls = type(name, (DefenseRules, base), {'__module__': __name__})
    globals()[name] = cls
    return cls
