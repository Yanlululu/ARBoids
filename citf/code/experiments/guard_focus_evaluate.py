"""Evaluate the declared radial-focus checkpoint with its envelope objective."""
from citf.contracts import canonical_hash, file_hash
from experiments.guard_focus import FocusConfig
from experiments import guard_prediction as guard
from experiments import guard_envelope


def main():
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller']['predictor_training_revision'] = FocusConfig().revision
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'guard_focus_evaluate': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    guard.GuardConfig = FocusConfig
    guard_envelope.main()


if __name__ == '__main__':
    main()
