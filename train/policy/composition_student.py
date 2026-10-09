"""Candidate-conditioned students of the frozen pursuit/guard control operator."""
import numpy as np
import torch
from torch import nn

from distill_role_value import measured_features
from evaluate_candidate_roles import adaptive_guard_controller
from feedback_joint_control import nominal_attacker_command


STATE_DIM = 35  # Six-vessel pilot: public geometry, causal history, time, identity.


def student_inputs(controller, measurement):
    """Called after the causal observer update and before the current command."""
    n = controller.defenders
    if n != 6:
        raise ValueError('The initial composition-learning experiment fixes six defenders.')
    base = measured_features(controller, measurement)
    previous = np.zeros((n, 2)) if controller.previous_thrust is None else controller.previous_thrust/1000.
    previous_attacker = (np.zeros(2) if controller.previous_attacker_thrust is None
                         else controller.previous_attacker_thrust/8000.)
    radial = np.arctan2(measurement.attacker[1], measurement.attacker[0])
    common = np.array([(measurement.total_time-measurement.time)/60.,
                       *previous_attacker, float(controller.previous_thrust is not None),
                       np.cos(radial), np.sin(radial)], dtype=np.float32)
    state = np.column_stack((base, previous, np.broadcast_to(common, (n, 6)), np.eye(n))).astype(np.float32)
    controls = np.stack([controller.nominal(measurement, name)[:, :2] for name in controller.options])
    masks = np.stack(list(controller.options.values()))
    modes = np.array([name.startswith('guard-') for name in controller.options], dtype=np.float32)
    response = np.concatenate((masks[..., None], np.broadcast_to(modes[:, None, None], (len(modes), n, 1)),
                               controls, controls-controls[:1]), axis=-1).astype(np.float32)
    return state, response


class CompositionStudent(nn.Module):
    """Shared candidate scorer with nonlinear pursuit/guard and pair responses."""
    def __init__(self, architecture='structured'):
        super().__init__()
        self.architecture = architecture
        if architecture == 'flat':
            self.classifier = nn.Sequential(nn.Linear(6*STATE_DIM, 160), nn.SiLU(),
                nn.Linear(160, 128), nn.SiLU(), nn.Linear(128, 43))
        elif architecture == 'structured':
            self.node = nn.Sequential(nn.Linear(STATE_DIM+6, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
            self.pair = nn.Sequential(nn.Linear(102, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(224, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
            self.register_buffer('pairs', torch.triu_indices(6, 6, 1), persistent=False)
        else:
            raise ValueError('Unknown student architecture.')

    def forward(self, state, response):
        if self.architecture == 'flat':
            return self.classifier(state.flatten(1))
        x = state[:, None].expand(-1, response.shape[1], -1, -1)
        h = self.node(torch.cat((x, response), -1))
        i, j = self.pairs
        relation = torch.cat((x[:, :, i, :4]-x[:, :, j, :4],
                              response[:, :, i, 2:4]-response[:, :, j, 2:4]), -1)
        pairs = self.pair(torch.cat((h[:, :, i], h[:, :, j], relation), -1)).mean(-2)
        mask = response[..., :1]
        pursuit = (h*mask).sum(-2)/mask.sum(-2).clamp_min(1.)
        guard = (h*(1.-mask)).sum(-2)/(1.-mask).sum(-2).clamp_min(1.)
        return self.head(torch.cat((h.mean(-2), h.amax(-2), pursuit, guard, pairs), -1)).squeeze(-1)


class CompositionStudentController:
    """Two-second composition decisions, 0.2-second feedback, no forecasts."""
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)
        self.operator = adaptive_guard_controller(6)
        self.plan_count = 0
        self.mode_plans = dict(baseline=0, guard=0, point=0)
        self.timings = []

    @torch.inference_mode()
    def control(self, measurement):
        import time
        started = time.perf_counter()
        c = self.operator
        c.attacker_agility = c.observer.update(measurement)
        planned = c.remaining == 0
        if planned:
            state, response = student_inputs(c, measurement)
            logits = self.model(torch.from_numpy(state)[None], torch.from_numpy(response)[None])
            c.template = list(c.options)[int(logits.argmax(-1))]
            c.remaining = c.block_steps
            self.plan_count += 1
            self.mode_plans[c.template.split('-')[0]] += 1
        action = c.nominal(measurement, c.template)
        _, force, info = c.safety.control(measurement.defenders, action, measurement.boids)
        c.previous_attacker_thrust = nominal_attacker_command(measurement, c.attacker_agility)
        c.previous_thrust = force.copy()
        c.remaining -= 1
        c.tick += 1
        self.timings.append((planned, time.perf_counter()-started))
        info.update(planned=planned, template=c.template, candidate_policies=len(c.options))
        return force, info


def load_student(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != 'composition-student-v1':
        raise ValueError('Unknown deployment format.')
    model = CompositionStudent(saved['architecture'])
    model.load_state_dict(saved['model'])
    return model.eval().requires_grad_(False)
