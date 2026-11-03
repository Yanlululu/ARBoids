"""Compare physical replanning after an observed current configuration.

The roots, initial candidates, short branches and future seeds match the
learned-continuation diagnostic. Only the subsequent feedback controller is
replaced. These outcome-selected cases locate the temporal origin of harms;
they are neither training labels for a value head nor a performance estimate.
"""
from experiments import diagnose_radial_choices as diagnostic
from experiments.restart_configuration_policy import PhysicalPrefixController


base_initialize = diagnostic.initialize
base_diagnose = diagnostic.diagnose


class PhysicalContinuationController:
    def __init__(self, source, model):
        self.controller = PhysicalPrefixController(source, model.cfg)

    def act(self, world):
        return self.controller.act(world), {}


def initialize(*args):
    base_initialize(*args)
    diagnostic.joint.JointPolicyController = PhysicalContinuationController


def diagnose(job):
    result = base_diagnose(job)
    result['continuation_condition'] = (
        'one current candidate action, then physical-prior greedy '
        'replanning with unchanged source, search and weights until actual stop')
    result['continuation_controller'] = 'same-bank physical-prior controller'
    return result


def main():
    diagnostic.initialize = initialize
    diagnostic.diagnose = diagnose
    diagnostic.main()


if __name__ == '__main__':
    main()
