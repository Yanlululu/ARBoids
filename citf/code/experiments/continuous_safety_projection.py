"""Joint physical clearance optimization in the complete hardware force box."""
import argparse
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

from citf.bridge import Controller
from citf.contracts import canonical_hash, file_hash
from citf.physics import forecast
from experiments.contact_event_policy import retained_teacher
from experiments.synchronized_configuration_policy import joint


class ProjectionController(Controller):
    revision = 'CITF-full-domain-physical-separation-projection-v1'

    def __init__(self, *args, horizon=10, iterations=12, **kwargs):
        super().__init__(*args, **kwargs)
        if horizon < 1 or iterations < 1:
            raise ValueError('Projection horizon and iterations must be positive')
        self.teacher = joint.JointPolicyController(self.source, self.model, self.device,
            self.bank, self.budget, self.rounds, self.risk_mode)
        self.horizon, self.iterations = horizon, iterations

    def distances(self, state, normalized, horizon):
        force = 250+750*normalized.reshape(-1, 2)
        position = forecast(state, force, horizon)
        n = len(state)
        i, j = torch.triu_indices(n, n, 1, device=state.device)
        return (position[i]-position[j]).norm(dim=-1)

    def prediction_horizon(self, world):
        remaining = int(np.ceil((world.protocol.time_limit-world.env.Current_T-1e-8)/world.protocol.control_dt))
        horizon = min(self.horizon, remaining)
        if horizon < 1:
            raise ValueError('Cannot project a terminated task')
        return horizon

    def act(self, world):
        started = time.perf_counter()
        reference, trace = self.teacher.act(world)
        state = torch.as_tensor(world.state()[:-1], dtype=torch.float64)
        initial = ((np.asarray(reference, np.float64)-250)/750).reshape(-1)
        reference_tensor = torch.from_numpy(initial)
        horizon = self.prediction_horizon(world)
        clearance = world.protocol.risk_clearance
        with torch.no_grad():
            distances = self.distances(state, reference_tensor, horizon)
            before = float(distances.min())
        selected = initial.copy()
        best_minimum = before
        calls = 0
        optimized = False
        solver = None
        solver_error = None
        if before < clearance:
            candidates = [(before, float('inf'), initial.copy())]

            def objective(raw):
                nonlocal calls
                with torch.enable_grad():
                    value = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
                    pair_distance = self.distances(state, value, horizon)
                    penetration = (clearance-pair_distance.amin(-1)).clamp_min(0)
                    change = (value-reference_tensor).square().mean()
                    loss = change+32*penetration.square().mean()
                    gradient, = torch.autograd.grad(loss, value)
                calls += 1
                minimum = float(pair_distance.min().detach())
                if not np.isfinite(minimum) or not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite projection objective or gradient')
                candidates.append((minimum, float(loss.detach()), np.asarray(raw).copy()))
                return float(loss.detach()), gradient.detach().numpy()

            try:
                solver = minimize(objective, initial, jac=True, method='L-BFGS-B',
                    bounds=[(-1., 1.)]*len(initial),
                    options=dict(maxiter=self.iterations, maxfun=2*self.iterations, ftol=1e-6))
            except FloatingPointError as error:
                # Retain the best finite iterate, including the original command.
                solver = None
                solver_error = str(error)
            # Retain the least-distorting feasible iterate, or the iterate with
            # the largest clearance improvement if the short model is infeasible.
            feasible = [item for item in candidates if item[0] >= clearance-1e-3]
            if feasible:
                best = min(feasible, key=lambda item: float(np.mean((item[2]-initial)**2)))
            else:
                best = max(candidates, key=lambda item: (item[0], -item[1]))
            if best[0] > before+1e-4:
                best_minimum, _, selected = best
                optimized = True
        force = np.clip(250+750*selected.reshape(-1, 2), -500, 1000).astype(np.float32)
        trace.update(revision=self.revision, physical_thrust=force.tolist(),
            reference_physical_thrust=reference.tolist(), physical_residual=(force-reference).tolist(),
            projection_applied=optimized, projection_model_minimum_before=before,
            projection_model_minimum_after=best_minimum, projection_horizon_steps=horizon,
            projection_objective_calls=calls, projection_solver_success=None if solver is None else bool(solver.success),
            projection_solver_error=solver_error,
            projection_model='known defender dynamics from measured states; zero future disturbance,held command',
            source_domain='full common[-500,1000]N force box; nearby joint correction to retained strong control',
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
    import citf.physics as physics
    original = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original(*positional, **kwargs)
        result.update(method='CITF-full-domain-physical-separation-projection-development',
            revision=ProjectionController.revision)
        result['controller'].update(action='joint12dim continuous force correction in common[-500,1000]N',
            projection_horizon_steps=args.projection_horizon, projection_iterations=args.projection_iterations,
            projection_clearance='existing protocol risk_clearance', projection_weight=32.,
            prediction='known defender physics,held current force,measured states,no future disturbance',
            reference_has_learning=True, expanded_control_domain=True,
            attribution='physical control intervention; no added learned mechanism claim')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            projection=file_hash(__file__), physics=file_hash(physics.__file__)))
        return result

    evaluation.runtime_identity = identity
    evaluation.load_behavior = lambda checkpoint, source, world, device: retained_teacher(checkpoint).to(device)
    evaluation.Controller = lambda *positional, **kwargs: ProjectionController(*positional,
        horizon=args.projection_horizon, iterations=args.projection_iterations, **kwargs)
    evaluation.main()


if __name__ == '__main__':
    main()
