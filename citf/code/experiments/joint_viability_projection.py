"""Joint continuous correction for collision and radial coverage together.

The collision-only version removed collisions but had zero net gain on768.
This version optimizes both failure mechanisms in the same hardware action
box. Defender forecasts use known physics; the attacker uses measured current
velocity. Neither predicted feasibility nor zero nominal violation is a safety
guarantee under reactive opponents and disturbances.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

from citf.contracts import canonical_hash, file_hash
from experiments.continuous_safety_projection import ProjectionController
from experiments import continuous_safety_projection_native as native
from experiments.contact_event_policy import retained_teacher


def fleet_geometry(state, normalized, horizon):
    """A stationary origin gives the native pair kernel exact radial distances."""
    n = len(state)
    augmented = torch.cat((state, state.new_zeros((1, 6))), 0)
    command = torch.cat((normalized, normalized.new_full((len(normalized), 2), -1/3)), -1)
    distance = native.native_distances(augmented, command, horizon)
    _, j = torch.triu_indices(n+1, n+1, 1, device=state.device)
    return torch.cat((distance[:, j<n], distance[:, j==n]), 1)


class FleetGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, normalized, horizon):
        ctx.save_for_backward(state, normalized)
        ctx.horizon = horizon
        return fleet_geometry(state, normalized[None], horizon)[0]

    @staticmethod
    def backward(ctx, output_gradient):
        state, normalized = ctx.saved_tensors
        n = len(normalized)
        epsilon = 1e-4
        delta = torch.eye(n, dtype=normalized.dtype, device=normalized.device)*epsilon
        with torch.no_grad():
            geometry = fleet_geometry(state, torch.cat((normalized[None]+delta, normalized[None]-delta)), ctx.horizon)
            derivative = (geometry[:n]-geometry[n:])/(2*epsilon)
            gradient = (derivative*output_gradient[None]).sum((-1, -2))
        return None, gradient, None


class ViabilityController(ProjectionController):
    revision = 'CITF-joint-collision-radial-continuous-projection-v1'

    def act(self, world):
        started = time.perf_counter()
        reference, trace = self.teacher.act(world)
        observed = torch.as_tensor(world.state(), dtype=torch.float64)
        state = observed[:-1]
        n = len(state)
        pairs = n*(n-1)//2
        horizon = self.prediction_horizon(world)
        times = torch.arange(1, horizon+1, dtype=torch.float64)*world.protocol.control_dt
        enemy = observed[-1, :2][None]+times[:, None]*observed[-1, 3:5][None]
        enemy_radius = enemy.norm(dim=-1)
        clearance = world.protocol.risk_clearance
        guard_margin = self.model.cfg.guard_margin
        guard_weight = self.model.cfg.guard_weight
        initial = ((np.asarray(reference, np.float64)-250)/750).reshape(-1)
        reference_tensor = torch.from_numpy(initial)

        def costs(geometry):
            pair_minimum = geometry[:pairs].amin(-1)
            coverage = enemy_radius-geometry[pairs:].amin(0)
            collision = (clearance-pair_minimum).clamp_min(0)
            guard = (guard_margin-coverage).clamp_min(0).amax()
            catastrophe = ((world.protocol.collision_radius-pair_minimum).clamp_min(0).square().amax()
                +(-coverage).clamp_min(0).square().amax())
            penalty = 32*collision.square().mean()+guard_weight*guard.square()+1000*catastrophe
            return penalty, catastrophe, pair_minimum.amin(), coverage.amin()

        with torch.no_grad():
            before = costs(fleet_geometry(state, reference_tensor[None], horizon)[0])
        selected = initial.copy()
        after = before
        calls = 0
        result = None
        solver_error = None
        if float(before[2]) < clearance or float(before[3]) < guard_margin:
            metrics=tuple(float(x) for x in before)
            candidates = [dict(raw=initial.copy(),loss=metrics[0],change=0.,catastrophe=metrics[1],
                pair=metrics[2],coverage=metrics[3],metrics=metrics)]

            def objective(raw):
                nonlocal calls
                with torch.enable_grad():
                    value = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
                    geometry = FleetGeometry.apply(state, value, horizon)
                    penalty, catastrophe, pair_minimum, coverage = costs(geometry)
                    change = (value-reference_tensor).square().mean()
                    loss = change+penalty
                    gradient, = torch.autograd.grad(loss, value)
                calls += 1
                if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite joint projection objective or gradient')
                candidates.append(dict(raw=np.asarray(raw).copy(), loss=float(loss.detach()),
                    change=float(change.detach()), catastrophe=float(catastrophe.detach()),
                    pair=float(pair_minimum.detach()), coverage=float(coverage.detach()),
                    metrics=tuple(float(x.detach()) for x in (penalty, catastrophe, pair_minimum, coverage))))
                return float(loss.detach()), gradient.detach().numpy()

            try:
                result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                    bounds=[(-1., 1.)]*len(initial),
                    options=dict(maxiter=self.iterations, maxfun=2*self.iterations, ftol=1e-6))
            except FloatingPointError as error:
                solver_error = str(error)
            feasible = [x for x in candidates if x['pair']>=clearance-1e-3 and x['coverage']>=guard_margin-1e-3]
            if feasible:
                best = min(feasible, key=lambda x: x['change'])
            else:
                best = min(candidates, key=lambda x: (x['catastrophe']>1e-8, x['catastrophe'], x['loss']))
            selected = best['raw']
            after = best['metrics']
        force = np.clip(250+750*selected.reshape(-1, 2), -500, 1000).astype(np.float32)
        trace.update(revision=self.revision, physical_thrust=force.tolist(),
            reference_physical_thrust=reference.tolist(), physical_residual=(force-reference).tolist(),
            projection_applied=bool(np.max(abs(force-reference))>1e-4),
            predicted_pair_before=float(before[2]), predicted_pair_after=float(after[2]),
            predicted_coverage_before=float(before[3]), predicted_coverage_after=float(after[3]),
            predicted_catastrophe_before=float(before[1]), predicted_catastrophe_after=float(after[1]),
            projection_horizon_steps=horizon, projection_objective_calls=calls,
            projection_solver_success=None if result is None else bool(result.success),
            projection_solver_error=solver_error,
            projection_dynamics_branches=1+(1+4*n)*calls,
            prediction='known defender physics,observed constant-velocity attacker,no future information',
            source_domain='joint full common[-500,1000]N force box,collision and radial coverage optimized together',
            compute_seconds=time.perf_counter()-started)
        trace['forward_batches'] += calls
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--projection-horizon', type=int, default=10)
    parser.add_argument('--projection-iterations', type=int, default=12)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    import citf.evaluate as evaluation
    import experiments.continuous_safety_projection as base
    previous = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        value = previous(*positional, **kwargs)
        value.update(method='CITF-joint-viability-continuous-projection-development', revision=ViabilityController.revision)
        value['controller'].update(action='continuous joint collision and radial coverage correction in full hardware box',
            horizon_steps=args.projection_horizon, optimizer_iterations=args.projection_iterations,
            prediction='known defender physics plus observed current-velocity attacker',
            warnings='existing7m clearance and inherited6m radial margin',
            objective='reference change+32mean collisionwarning squared+16radialwarning squared+1000predicted event violation squared',
            attribution='physical joint control change; no additional learning claim',
            reference_has_learning=True, expanded_control_domain=True,
            native_source=file_hash(Path(native.__file__).with_name('projection_dynamics.c')),
            native_binary=file_hash(native.LIBRARY_PATH))
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], viability=file_hash(__file__),
            base=file_hash(base.__file__), native=file_hash(native.__file__)))
        return value

    evaluation.runtime_identity = identity
    evaluation.load_behavior = lambda checkpoint, source, world, device: retained_teacher(checkpoint).to(device)
    evaluation.Controller = lambda *positional, **kwargs: ViabilityController(*positional,
        horizon=args.projection_horizon, iterations=args.projection_iterations, **kwargs)
    evaluation.main()


if __name__ == '__main__':
    main()
