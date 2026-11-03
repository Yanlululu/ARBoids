"""Learn longer radial recourse while retaining the short physical guard.

The learned margin covers 6.4 seconds, but the nominal protection and collision
cost retain their established 3.2-second interval. This avoids treating held
thrust for the entire longer interval as the required reference trajectory.
"""
from dataclasses import dataclass

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import long_geometry
from experiments import long_recourse_guard as long


@dataclass(frozen=True)
class DualIntervalConfig(long.LongRecourseConfig):
    revision: str = 'CITF-long-recourse-with-short-nominal-envelope-v1'
    nominal_guard_steps: int = 16


def short_nominal(batch):
    return team.nominal_team(long_geometry.short_collision_batch(batch, 16))


class DualIntervalPredictor(long.LongRecoursePredictor):
    def forward(self, batch, history_cache=None):
        result = super().forward(batch, history_cache=history_cache)
        result['nominal_guard_mu'] = short_nominal(batch)[1]
        return result


def main():
    long.LongRecourseConfig = DualIntervalConfig
    long.LongRecoursePredictor = DualIntervalPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'dual_interval_guard': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        guard.nominal_gaps = short_nominal
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller'].update(nominal_guard_steps=16,
                learned_guard_steps=32, guard_combination='maximum of short nominal and long recourse learned risk')
            result['runtime_code'] = canonical_hash({
                'core': result['runtime_code'], 'dual_interval_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    long.main()


if __name__ == '__main__':
    main()
