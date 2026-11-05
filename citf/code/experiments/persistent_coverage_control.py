"""Persistent inner coverage and pursuit, with joint physical correction.

The local projection can foresee a radial loss after the fleet has already
lost a useful inner position. One defender therefore keeps a coverage role
from initialization, while the other defenders retain the strong pursuit
controller. Full outcomes include captures, timeouts and physical breaches;
APF defense success is not asserted to generalize to learned opponents.
"""
import time

import numpy as np
import torch

from citf.contracts import canonical_hash, file_hash
from experiments.contact_event_policy import retained_teacher
from experiments import joint_viability_projection as viability
from experiments import continuous_safety_projection_native as native
from experiments import interception_reserve as tracking


class CoverageReference:
    def __init__(self, teacher):
        self.teacher = teacher
        self.guard = None
        self.last_time = None

    @torch.no_grad()
    def act(self, world):
        reference, trace = self.teacher.act(world)
        state = torch.as_tensor(world.state(), dtype=torch.float32)[None]
        position = state[0, :, :2]
        radius = world.protocol.target_radius-.5*world.protocol.capture_radius
        target = position[-1]/position[-1].norm().clamp_min(1e-6)*radius
        now = world.env.Current_T
        if self.guard is None or self.last_time is not None and now <= self.last_time:
            self.guard = int((position[:-1]-target).norm(dim=-1).argmin())
        waypoints = target[None, None].expand(1, len(reference), -1)
        guard_force = tracking.tracking_thrust(state, waypoints)[0, self.guard].cpu().numpy()
        command = reference.copy()
        command[self.guard] = guard_force
        trace.update(unmodified_teacher_physical_thrust=reference.tolist(),
            coverage_guard=self.guard, coverage_target=target.cpu().tolist(), coverage_radius=radius,
            coverage_guard_command=guard_force.tolist(),
            coverage_assignment='initial nearest defender to inner boundary waypoint,held throughout task',
            coverage_reference='guard tracks current attacker bearing inside target boundary; other defenders retain strong pursuit')
        self.last_time = now
        return command, trace


class PersistentCoverageController(viability.ViabilityController):
    revision = 'CITF-persistent-inner-coverage-and-joint-projection-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = CoverageReference(self.teacher)

    def act(self, world):
        started = time.perf_counter()
        force, trace = super().act(world)
        original = np.asarray(trace['unmodified_teacher_physical_thrust'])
        trace.update(revision=self.revision, total_physical_residual=(force-original).tolist(),
            source_domain='one persistent inner guard plus retained pursuit,then joint collision and radial correction in full hardware box',
            compute_seconds=time.perf_counter()-started)
        return force, trace


def main():
    import citf.evaluate as evaluation
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value.update(method='CITF-persistent-inner-coverage-development', revision=PersistentCoverageController.revision)
        value['controller'].update(action='one persistent inner guard and remaining strong pursuit,then joint continuous viability projection',
            coverage_assignment='initial closest defender to current attacker bearing on coverage circle; held for task',
            coverage_radius='target_radius minus half capture_radius;12.5m in current physical protocol',
            guard_control='observable velocity/yaw tracking with known inertia and drag plus local separation',
            projection='same2s10step joint collision/radial native solver,12iterations',
            online_information='observed joint state,current attacker bearing,retained reference; no future or task agility',
            reference_has_learning=True, expanded_control_domain=True,
            attribution='persistent fleet role and physical control change,not additional Transformer learning',
            native_binary=file_hash(native.LIBRARY_PATH))
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], coverage=file_hash(__file__),
            joint=file_hash(viability.__file__), tracking=file_hash(tracking.__file__), native=file_hash(native.__file__)))
        return value

    evaluation.runtime_identity = identity
    evaluation.load_behavior = lambda checkpoint, source, world, device: retained_teacher(checkpoint).to(device)
    evaluation.Controller = PersistentCoverageController
    evaluation.main()


if __name__ == '__main__':
    main()
