"""A bounded departure-penalty study on the frozen feedback controller."""
import math

from feedback_joint_fast import FastFeedbackJointController


def penalize_prediction(prediction, departure_penalty):
    """Preserve hard outcome priorities; penalize only non-reference templates."""
    result = dict(prediction)
    score = tuple(result['score'])
    result['unpenalized_score'] = score
    penalty = departure_penalty if result['template'] != 'baseline' else 0.
    result['score'] = (*score[:-1], score[-1] + penalty)
    return result


class FleetSelectionController(FastFeedbackJointController):
    def __init__(self, defenders, policy, block_steps=10, departure_penalty=0., **kwargs):
        if not math.isfinite(departure_penalty) or departure_penalty < 0.:
            raise ValueError('Departure penalty must be finite and non-negative.')
        self.departure_penalty = float(departure_penalty)
        super().__init__(defenders, policy, block_steps, **kwargs)

    def predict(self, measurement, template):
        return penalize_prediction(super().predict(measurement, template), self.departure_penalty)
