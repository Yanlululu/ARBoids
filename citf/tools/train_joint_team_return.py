"""Learn the complete fusion configuration's ordinary team return.

The existing per-receiver value views share one actual team Bellman target.
The most adverse learned pair response is already shared during three-vessel
training. Candidate generation, search, endpoint learning and generator
feedback still optimize the same clipped team score. No auxiliary reward,
new value head, imagined transition or intervention target is introduced.
"""
import torch

from train_shared_team_interaction import main, SharedTeamCITF
from unified.learner import Learner


class JointTeamReturnLearner(Learner):
    value_target_kind = 'common-actual-team-return-per-receiver-v1'
    interaction_aggregation_kind = 'positive-local-plus-shared-adverse-team-v1'

    @torch.no_grad()
    def critic_target(self, batch):
        target = batch['r'].clone()
        continuing = ~batch['done'].bool()
        if bool(continuing.any()):
            next_state = batch['next_x'][continuing]
            next_u = self.target.select(next_state)['action']
            first = self.target.q1(next_state, next_u, individual=True)
            second = self.target.q2(next_state, next_u, individual=True)
            target[continuing] += self.gamma*torch.minimum(first, second).mean(-1)
        return target[:, None].expand_as(batch['ri'])

    def payload(self):
        result = super().payload()
        result.update(value_target_kind=self.value_target_kind,
                      interaction_aggregation_kind=self.interaction_aggregation_kind)
        return result

    def restore(self, payload):
        for name in ('value_target_kind', 'interaction_aggregation_kind'):
            saved = payload.get(name)
            if saved is not None and saved != getattr(self, name):
                raise ValueError('Resume retains the registered team-value learning definition.')
        if (payload.get('continuation', {}).get('variant') == 'joint_team_return'
                and payload.get('value_target_kind') != self.value_target_kind):
            raise ValueError('A joint-team-return continuation must retain its target identity.')
        super().restore(payload)


if __name__ == '__main__':
    main(candidate_type=SharedTeamCITF, candidate_name='joint_team_return',
         learner_type=JointTeamReturnLearner,
         candidate_definition='each receiver value learns the same actual mean-team-reward Bellman target with mean per-vessel clipped continuation; positive local responses plus the shared most adverse pair use the same aggregation at every team size; all source tensors and optimizer states retained; original endpoint, candidates, search and generator feedback retained')
