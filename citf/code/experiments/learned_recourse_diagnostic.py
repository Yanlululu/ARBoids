"""Compare one-tick candidate responses under the retained learned feedback.

The root, candidate bank, first action and future seeds match the physical
recourse experiment. Only the frozen controller after that action changes.
Short trajectories remain prediction labels; complete continuations are
diagnostics and never train a terminal-value or outcome predictor.
"""
import sys

import torch

from experiments import recourse_diagnostic as recourse
from experiments import root_response_diagnostic as diagnostic
from experiments import response_data


_initialize_root = diagnostic.initialize


@torch.no_grad()
def retained_action(world):
    return diagnostic.worker['recourse_controller'].act(world)[0]


def install(project, source, checkpoint, out, deadline):
    _initialize_root(project, source, checkpoint, out, deadline)
    w = diagnostic.worker
    w['recourse_controller'] = diagnostic.synchronized.joint.JointPolicyController(
        w['source'], w['model'])
    w['recourse_checkpoint'] = checkpoint
    diagnostic.physical_action = retained_action


def initialize_short(project, source, checkpoint, out, deadline,
                     horizon_steps=16, futures=None):
    install(project, source, checkpoint, out, deadline)
    diagnostic.worker.update(response_hold=1, response_horizon=horizon_steps,
                             response_futures=futures)
    recourse.sample_response = sample_response


def sample_response(world, plans, future_seeds, metadata, horizon_steps=None):
    tensors, metadata, transitions = response_data.sample_response(
        world, plans, future_seeds, metadata, horizon_steps=horizon_steps)
    metadata.update(
        recourse_policy='frozen synchronized learned7101, recomputed every control tick',
        recourse_checkpoint=diagnostic.worker['recourse_checkpoint'],
        prediction_condition='current candidate for one tick, frozen learned feedback until horizon or first task event',
        recourse_diagnostic_revision='learned-feedback-response-v1',
        complete_task_outcomes_are_training_targets=False)
    return tensors, metadata, transitions


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'collect':
        sys.argv.pop(1)
        recourse.initialize = initialize_short
        recourse.sample_response = sample_response
        recourse.main()
    elif mode == 'continue':
        diagnostic.initialize = install
        diagnostic.main()
    else:
        raise ValueError('Expected collect or continue for the paired recourse diagnostic')


if __name__ == '__main__':
    main()
