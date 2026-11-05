"""Batched central differences for physical distances and autograd for clearance."""
import torch

from citf.contracts import canonical_hash, file_hash
from citf.physics import forecast
from experiments import continuous_safety_projection as original


def physical_distances(state, normalized, horizon):
    force = 250+750*normalized.reshape(len(normalized), len(state), 2)
    position = forecast(state[None].expand(len(normalized), -1, -1), force, horizon)
    i, j = torch.triu_indices(len(state), len(state), 1, device=state.device)
    return (position[:, i]-position[:, j]).norm(dim=-1)


class BatchedDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, normalized, horizon):
        ctx.save_for_backward(state, normalized)
        ctx.horizon = horizon
        return physical_distances(state, normalized[None], horizon)[0]

    @staticmethod
    def backward(ctx, output_gradient):
        state, normalized = ctx.saved_tensors
        count = len(normalized)
        epsilon = 1e-4
        perturbation = torch.eye(count, dtype=normalized.dtype, device=normalized.device)*epsilon
        points = torch.cat((normalized[None]+perturbation, normalized[None]-perturbation), 0)
        with torch.no_grad():
            distances = physical_distances(state, points, ctx.horizon)
            derivative = (distances[:count]-distances[count:])/(2*epsilon)
            gradient = (derivative*output_gradient[None]).sum((-1, -2))
        return None, gradient, None


class FastProjectionController(original.ProjectionController):
    revision = 'CITF-full-domain-physical-separation-projection-batched-gradient-v2'

    def distances(self, state, normalized, horizon):
        if normalized.requires_grad:
            return BatchedDistance.apply(state, normalized, horizon)
        return physical_distances(state, normalized[None], horizon)[0]

    def act(self, world):
        force, trace = super().act(world)
        trace.update(projection_gradient='batched central difference of physical distances,epsilon0.0001 normalized force',
            projection_dynamics_branches=1+(1+4*len(force))*trace['projection_objective_calls'])
        return force, trace


def main():
    import citf.evaluate as evaluation
    saved_identity = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = saved_identity(*args, **kwargs)
        result['controller'].update(projection_gradient='batched central difference of physical distances',
            projection_gradient_epsilon=1e-4, projection_gradient_probe_count='4*defender_count',
            numerical_identity='new actual evaluation; no assumed behavioral equality to automatic differentiation')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'], gradient=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    original.ProjectionController = FastProjectionController
    original.main()


if __name__ == '__main__':
    main()
