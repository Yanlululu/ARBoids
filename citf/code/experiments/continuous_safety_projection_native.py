"""Native batched physical distances for the continuous projection study.

The small state tensors make eager framework dispatch dominate the physical
arithmetic. A float64 C kernel performs that arithmetic while the same joint
objective and numerical sensitivities remain in the existing solver. Build the
adjacent C source to projection_dynamics_v1.so (Linux) or .dll (Windows) before
using this evaluator. This is independently evaluated research code.
"""
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch

from citf.contracts import canonical_hash, file_hash
from experiments import continuous_safety_projection_fast as fast


LIBRARY_PATH = Path(__file__).with_name('projection_dynamics_v1.dll' if sys.platform == 'win32'
                                     else 'projection_dynamics_v1.so')
LIBRARY = None


def native_distances(state, normalized, horizon):
    global LIBRARY
    if state.ndim != 2 or state.shape[1] != 6 or len(state) < 2:
        raise ValueError('Native projection requires defender states [N,6], N>=2')
    if normalized.ndim != 2 or normalized.shape[1] != 2*len(state) or len(normalized) < 1:
        raise ValueError('Native projection requires commands [B,2*N], B>=1')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 2**31-1:
        raise ValueError('Native projection horizon must be a positive C integer')
    if max(len(state), len(normalized)) > 2**31-1:
        raise ValueError('Native projection dimensions exceed C integer range')
    if not torch.isfinite(state).all() or not torch.isfinite(normalized).all():
        raise ValueError('Native projection inputs must be finite')
    if LIBRARY is None:
        LIBRARY = ctypes.CDLL(str(LIBRARY_PATH))
        pointer = ctypes.POINTER(ctypes.c_double)
        LIBRARY.citf_projection_distances.argtypes = [pointer, pointer, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, pointer]
        LIBRARY.citf_projection_distances.restype = ctypes.c_int
    initial = np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float64)
    command = np.ascontiguousarray(normalized.detach().cpu().numpy(), dtype=np.float64)
    candidates, defenders = len(command), len(initial)
    distances = np.empty((candidates, defenders*(defenders-1)//2, horizon), np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    result = LIBRARY.citf_projection_distances(initial.ctypes.data_as(pointer),
        command.ctypes.data_as(pointer), candidates, defenders, horizon,
        distances.ctypes.data_as(pointer))
    if result:
        raise RuntimeError(f'Physical projection kernel failed: {result}')
    return torch.from_numpy(distances).to(device=normalized.device, dtype=normalized.dtype)


class NativeProjectionController(fast.FastProjectionController):
    revision = 'CITF-full-domain-physical-separation-native-batched-gradient-v3'

    def act(self, world):
        force, trace = super().act(world)
        trace['projection_physics_backend'] = 'native float64 held-thrust dynamics'
        return force, trace


def main():
    import citf.evaluate as evaluation
    saved_identity = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = saved_identity(*args, **kwargs)
        result['controller'].update(projection_physics_backend='native float64,compiled without fast-math',
            projection_native_source=file_hash(Path(__file__).with_name('projection_dynamics.c')),
            projection_native_binary=file_hash(LIBRARY_PATH))
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'], native=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    fast.physical_distances = native_distances
    fast.FastProjectionController = NativeProjectionController
    fast.main()


if __name__ == '__main__':
    main()
