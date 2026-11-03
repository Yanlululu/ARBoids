"""Score simultaneous current changes after single-coordinate proposals.

The complete original physical bank is preserved. A scorer chooses a common
configuration, ranks each defender's alternative command in that context,
and composes pairs, triples and the complete team. Every proposal is then
rescored as a complete joint configuration. Physical and learned scorers use
the same procedure and total unique-configuration budget. No future action
sequence or terminal value is predicted.
"""
from dataclasses import dataclass
from itertools import combinations
import sys
import time

import numpy as np
import torch

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments import joint_configuration_policy as joint


@dataclass(frozen=True)
class CooperativeSearchConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-composed-current-configuration-search-v1'
    action_bank: str = 'physical bank plus single, pair, triple and team compositions'
    composition_mode: str = 'learned'
    physical_bank_budget: int = 96
    composition_budget: int = 160
    search_condition: str = 'rank single-defender alternatives, compose simultaneous changes, rescore every complete configuration'


class CooperativeSearchController(joint.JointPolicyController):
    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        self.model.capture_inputs = self.stochastic
        f, cfg = factorize(world, self.source), self.model.cfg
        if cfg.composition_mode not in ('learned', 'physical'):
            raise ValueError('Unknown composition scorer')
        if self.budget != cfg.composition_budget:
            raise ValueError('Declared composition budget differs from runtime')
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, center = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                         f['prior'].numpy(), self.bank, cfg.physical_bank_budget)
        choices = [row.nonzero()[:, 0].numpy() for row in f['valid']]

        def physical_score(plans):
            batch, _, _ = inputs(world, f, plans)
            return joint.policy.physical_components(batch, cfg)[0][0].cpu().numpy()

        original_choice, physical_trace = whole_configuration_search(
            initial, physical_score, choices, cfg.physical_bank_budget, self.rounds)
        plans = [tuple(map(int, row)) for row in physical_trace['configurations']]
        known = set(plans)
        history_cache = None
        predictor_scored = 0
        forward_batches = 0

        def score_current(rows):
            nonlocal history_cache, predictor_scored, forward_batches
            batch, _, _ = inputs(world, f, np.asarray(rows, dtype=np.int64))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if cfg.composition_mode == 'physical':
                physical = joint.policy.physical_components(batch, cfg)[0][0]
                return -physical, batch, None
            if history_cache is None:
                history_cache = self.model.encode_history(batch)
            output = self.model(batch, history_cache=history_cache)
            predictor_scored += len(rows)
            forward_batches += 1
            return output['logits'][0], batch, output

        logits, _, _ = score_current(plans)
        best = min(range(len(plans)), key=lambda k: (-float(logits[k]), plans[k]))
        base = plans[best]
        alternatives = []
        for i, values in enumerate(choices):
            indices = []
            for value in values:
                if int(value) == base[i]:
                    continue
                candidate = list(base)
                candidate[i] = int(value)
                key = tuple(candidate)
                if key not in known and len(plans) < self.budget:
                    plans.append(key)
                    known.add(key)
                if key in known:
                    indices.append(plans.index(key))
            alternatives.append(indices)
        logits, _, _ = score_current(plans)
        moves = []
        for i, indices in enumerate(alternatives):
            if indices:
                selected = min(indices, key=lambda k: (-float(logits[k]), plans[k]))
                moves.append((i, plans[selected][i]))
        composed = []
        for order in sorted(set((2, 3, len(moves)))):
            if order < 2 or order > len(moves):
                continue
            for selected in combinations(moves, order):
                candidate = list(base)
                for i, value in selected:
                    candidate[i] = value
                key = tuple(candidate)
                if key not in known and len(plans) < self.budget:
                    plans.append(key)
                    known.add(key)
                    composed.append(list(key))
        logits, batch, output = score_current(plans)
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if self.stochastic else min(
            range(len(plans)), key=lambda k: (-float(logits[k]), plans[k]))
        indices = np.asarray(plans[choice], dtype=np.int64)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        physical = joint.policy.physical_components(batch, cfg)[0][0]
        physical_choice = min(range(len(plans)), key=lambda k: (float(physical[k]), plans[k]))
        if output is not None:
            self.transition = dict(features=output['features'][0].cpu().half(),
                baseline=output['baseline'][0].cpu(), action=choice,
                log_probability=float(distribution.log_prob(torch.as_tensor(choice, device=logits.device))),
                changed=int(choice != physical_choice), entropy=float(distribution.entropy()))
            if self.stochastic:
                self.transition['inputs'] = self.model.rollout_inputs
                self.model.rollout_inputs = None
        elif self.stochastic:
            raise ValueError('The pure physical composition is an evaluation control')
        trace = dict(cost=float(-logits[choice]), evaluated_unique=len(plans),
            forward_batches=forward_batches, predictor_scored_configurations=predictor_scored,
            configurations=[list(row) for row in plans], scores=(-logits).cpu().tolist(),
            gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            physical_selected_indices=list(plans[physical_choice]),
            original_physical_selected_indices=original_choice.tolist(),
            original_physical_bank_size=physical_trace['evaluated_unique'],
            source_center_indices=center.tolist(), source_center_scored=tuple(center) in known,
            initial_configurations=initial.tolist(), composition_center=list(base),
            coordinate_proposals=[list(move) for move in moves], composed_configurations=composed,
            composition_mode=cfg.composition_mode, selection=cfg.search_condition,
            compute_seconds=time.perf_counter() - started)
        return force, trace


def main():
    joint.JointPolicyConfig = CooperativeSearchConfig
    joint.JointPolicyController = CooperativeSearchController
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        raise ValueError('This registered first comparison keeps policy weights fixed')
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs.get('checkpoint')
        cfg = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        result['controller'].update(search_condition=cfg['search_condition'],
            composition_mode=cfg['composition_mode'], physical_bank_budget=cfg['physical_bank_budget'],
            total_unique_budget=cfg['composition_budget'],
            physical_control_uses_neural_scores=cfg['composition_mode'] != 'physical',
            search='current single-coordinate proposals plus pair, triple and full-team compositions')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            cooperative_search_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
