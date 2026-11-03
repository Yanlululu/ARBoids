"""Resolve centimetre-scale first-action effects on team radial geometry.

Matched recourse branches show candidate variation around 0.03--0.07 metres
at a typical root. The centered physical-distance objective therefore uses
a 0.1-metre scale, rather than the held-action model's 0.5-metre scale.
Absolute likelihood and the common label-based selection rule are retained.
"""
from dataclasses import dataclass

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import guard_focus as focus
from experiments import response_delta_guard as delta
from experiments import synchronized_recourse as recourse


@dataclass(frozen=True)
class PreciseTeamConfig(recourse.TeamRecourseConfig):
    revision: str = 'CITF-synchronized-first-action-precise-difference-v1'
    response_delta_weight: float = 8.
    response_delta_scale_m: float = .1


class PreciseTeamPredictor(recourse.TeamRecoursePredictor):
    def forward(self, batch, history_cache=None):
        result = super().forward(batch, history_cache=history_cache)
        result.update(response_delta_weight=self.cfg.response_delta_weight,
                      response_delta_scale_m=self.cfg.response_delta_scale_m)
        return result


def main():
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'synchronized_delta': file_hash(__file__)})
    recourse.TeamRecourseConfig, recourse.TeamRecoursePredictor = PreciseTeamConfig, PreciseTeamPredictor
    focus.focused_objective = delta.difference_objective
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'synchronized_delta': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    recourse.main()


if __name__ == '__main__':
    main()
