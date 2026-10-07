"""Official, unmodified CBFpy safety filter with an external USV configuration.

Only dynamics, barriers and scaling are task-specific. CBF construction,
automatic Lie derivatives and the QP solver are the upstream implementation.
The original ARBoids action is the nominal command for every comparison.
"""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

from study_runtime import directory as _dependencies
if _dependencies.exists():
    sys.path.insert(0, str(_dependencies))

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBF, CBFConfig

from source_arboids import numerical_source, source_mixture


class USVBarrierConfig(CBFConfig):
    def __init__(self, defenders, safe_distance=7., gain=1.):
        self.defenders, self.radius, self.gain = defenders, safe_distance, gain
        self.boat = numerical_source().WAMV()
        self.mass_inverse = jnp.asarray(np.linalg.inv(self.boat.M_RB+self.boat.M_A).A)
        self.pairs = np.triu_indices(defenders, 1)
        super().__init__(n=6*defenders, m=2*defenders,
                         u_min=[-.5]*(2*defenders), u_max=[1.]*(2*defenders),
                         relax_qp=True, cbf_relaxation_penalty=1e3,
                         control_relaxation_penalty=1e5, solver_tol=1e-6)

    def f(self, z):
        state = z.reshape(self.defenders, 6)
        m = self.boat
        c, s = jnp.cos(state[:, 2]), jnp.sin(state[:, 2])
        u = c*state[:, 3]+s*state[:, 4]
        v = -s*state[:, 3]+c*state[:, 4]
        r = state[:, 5]
        velocity = jnp.stack((u, v, r), axis=-1)
        matrix = jnp.broadcast_to(jnp.asarray(m.D), (self.defenders, 3, 3))
        matrix = matrix.at[:, 0, 1].add(-m.m*r)
        matrix = matrix.at[:, 1, 0].add(m.m*r)
        matrix = matrix.at[:, 0, 2].add(m.yDotV*v+m.yDotR*r)
        matrix = matrix.at[:, 1, 2].add(-m.xDotU*u)
        matrix = matrix.at[:, 2, 0].add(-m.yDotV*v-m.yDotR*r)
        matrix = matrix.at[:, 2, 1].add(m.xDotU*u)
        matrix = matrix.at[:, 0, 0].add(-m.xUU*jnp.abs(u))
        matrix = matrix.at[:, 1, 1].add(-m.yVV*jnp.abs(v)-m.yRV*jnp.abs(r))
        matrix = matrix.at[:, 1, 2].add(-m.yVR*jnp.abs(v)-m.yRR*jnp.abs(r))
        matrix = matrix.at[:, 2, 1].add(-m.nVV*jnp.abs(v)-m.nRV*jnp.abs(r))
        matrix = matrix.at[:, 2, 2].add(-m.nVR*jnp.abs(v)-m.nRR*jnp.abs(r))
        body = -jnp.einsum('bij,bj->bi', matrix, velocity) @ self.mass_inverse.T
        acceleration = jnp.stack((c*body[:, 0]-s*body[:, 1],
                                  s*body[:, 0]+c*body[:, 1], body[:, 2]), axis=-1)
        return jnp.concatenate((state[:, 3:6], acceleration), axis=1).reshape(-1)

    def g(self, z):
        state = z.reshape(self.defenders, 6)
        c, s = jnp.cos(state[:, 2]), jnp.sin(state[:, 2])
        m = self.boat
        # Control is physical thrust / 1000, for a well-scaled upstream QP.
        force = jnp.asarray([[1000., 1000.], [0., 0.],
                             [500.*m.width, -500.*m.width]])
        body = self.mass_inverse @ force
        result = jnp.zeros((6*self.defenders, 2*self.defenders))
        for i in range(self.defenders):
            acceleration = jnp.stack((c[i]*body[0]-s[i]*body[1],
                                      s[i]*body[0]+c[i]*body[1], body[2]))
            result = result.at[6*i+3:6*i+6, 2*i:2*i+2].set(acceleration)
        return result

    def h_2(self, z):
        state = z.reshape(self.defenders, 6)
        p = state[self.pairs[0], :2]-state[self.pairs[1], :2]
        return (jnp.sum(p*p, axis=-1)-self.radius**2)/(2*self.radius**2)

    def alpha(self, h):
        return self.gain*h

    def alpha_2(self, h):
        return self.gain*h


class CBFController:
    def __init__(self, defenders, safe_distance=7., gain=1.):
        self.config = USVBarrierConfig(defenders, safe_distance, gain)
        self.filter = CBF.from_config(self.config)
        # Also inspect the official constraint residual after physical saturation.
        self.constraints = jax.jit(lambda z, u: (self.filter.G_qp(z, u), self.filter.h_qp(z, u)))

    def control(self, states, action, boids):
        nominal = source_mixture(action, boids).astype(float)
        z, reference = jnp.asarray(states.reshape(-1)), jnp.asarray(nominal.reshape(-1)/1000.)
        filtered = np.asarray(self.filter.safety_filter(z, reference), dtype=float)
        if not np.isfinite(filtered).all():
            raise FloatingPointError('The original CBFpy solver returned non-finite controls.')
        saturated = np.clip(filtered, -.5, 1.)
        G, h = self.constraints(z, reference)
        residual = np.asarray(G) @ saturated-np.asarray(h)
        count = len(self.config.pairs[0])
        info = dict(warning=bool(np.max(np.asarray(G)@np.asarray(reference)-np.asarray(h)) > 1e-6),
                    candidate_count=0, cbf_constraint_violation=float(max(0., residual[:count].max())),
                    cbf_control_saturation=float(np.max(np.abs(saturated-filtered))))
        return action.copy(), (1000.*saturated).reshape(-1, 2), info


def dependency_provenance():
    import cbfpy, qpax
    result = {'repository': 'https://github.com/StanfordASL/cbfpy',
              'versions': {name: importlib.metadata.version(name)
                           for name in ('cbfpy', 'qpax', 'jax', 'jaxlib', 'numpy', 'scipy')}, 'files': {}}
    for module in (cbfpy, qpax):
        root = Path(module.__file__).parent
        for path in sorted(root.rglob('*.py')):
            result['files'][f'{module.__name__}/{path.relative_to(root).as_posix()}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
