"""Isolate repeated-correction effects with a single actual action change.

Weights, candidate sets, task criteria and event triggers are unchanged.
After the first action differing from the retained teacher, all remaining
actions come from that same teacher under the actual resulting state.
"""
import torch

from experiments import contact_event_policy as contact


class SingleCorrectionController(contact.ContactEventController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_used = False

    @torch.no_grad()
    def act(self, world):
        if self.correction_used:
            self.transition = None
            force, trace = self.teacher_controller.act(world)
            trace.update(contact_event=[], correction_budget=1,
                         correction_used=True,
                         selection='retained teacher after one actual correction')
            return force, trace
        force, trace = super().act(world)
        if self.transition is not None and self.transition['changed']:
            self.correction_used = True
        trace.update(correction_budget=1, correction_used=self.correction_used)
        return force, trace


def main():
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['controller']['actual_correction_budget'] = 1
        value['runtime_code'] = canonical_hash(dict(
            core=value['runtime_code'], single_correction=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    contact.ContactEventController = SingleCorrectionController
    contact.main()


if __name__ == '__main__':
    main()
