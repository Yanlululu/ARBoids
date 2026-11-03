"""Use learned conditional configuration scores to propose coordinated moves.

This is a current-action search, not a future control plan. Single-vessel
responses propose multi-vessel moves, then every proposal is rescored as a
complete configuration. The matched physical controller uses the same search.
"""
import itertools
import numpy as np

from citf.contracts import canonical_hash, file_hash
from experiments import joint_task_geometry as geometry


def coordinated_search(initial, evaluate, valid_choices, budget=192, rounds=1):
    if budget < 1 or rounds < 0:
        raise ValueError('Invalid current configuration search budget')
    cache, batches, phases = {}, 0, []

    def score(plans, phase, limit=None):
        nonlocal batches
        keys = list(dict.fromkeys(tuple(map(int, p)) for p in plans))
        remaining = budget - len(cache)
        if limit is not None:
            remaining = min(remaining, max(0, limit))
        keys = [p for p in keys if p not in cache][:remaining]
        if not keys:
            return
        values = np.asarray(evaluate(np.asarray(keys)), dtype=float)
        if values.shape != (len(keys),) or not np.isfinite(values).all():
            raise ValueError('Invalid complete configuration scores')
        cache.update(zip(keys, values.tolist()))
        batches += 1
        phases.append(dict(phase=phase, count=len(keys)))

    def best():
        return min(cache, key=lambda p: (cache[p], p))

    def neighborhood(anchor):
        result = []
        for i, choices in enumerate(valid_choices):
            for value in choices:
                p = list(anchor)
                p[i] = int(value)
                result.append(tuple(p))
        return result

    score(initial, 'initial')
    if not cache:
        raise ValueError('No complete configuration was evaluated')
    for iteration in range(rounds):
        anchor = best()
        score(neighborhood(anchor), 'conditional_single')
        local = []
        for i, choices in enumerate(valid_choices):
            alternatives = []
            for value in choices:
                p = list(anchor)
                p[i] = int(value)
                key = tuple(p)
                if key in cache and value != anchor[i]:
                    alternatives.append((cache[key] - cache[anchor], int(value)))
            alternatives.sort()
            local.append([(0., anchor[i]), *alternatives[:2]])

        # A beam limits combinatorics for larger fleets. Its additive surrogate
        # only proposes; the actual nonlinear interaction score selects.
        beam = [(0., ())]
        for alternatives in local:
            beam = sorted((cost + delta, prefix + (value,))
                          for cost, prefix in beam for delta, value in alternatives)
            beam = beam[:max(64, 2 * budget)]
        joint = [p for _, p in beam if sum(a != b for a, b in zip(p, anchor)) > 1]
        paired = []
        for i, j in itertools.combinations(range(len(anchor)), 2):
            for (di, vi), (dj, vj) in itertools.product(local[i][1:], local[j][1:]):
                p = list(anchor)
                p[i], p[j] = vi, vj
                paired.append((di + dj, tuple(p)))
        paired.sort()
        # Preserve space for the final local refinement. Interleave broadly
        # coordinated moves and pair moves instead of spending the whole
        # budget on one predicted additive minimum.
        reserve = min(sum(max(0, len(v) - 1) for v in valid_choices),
                      max(0, (budget - len(cache)) // 2))
        proposals = []
        pairs = [p for _, p in paired]
        for index in range(max(len(joint), len(pairs))):
            if index < len(joint):
                proposals.append(joint[index])
            if index < len(pairs):
                proposals.append(pairs[index])
        score(proposals, 'coordinated', budget - len(cache) - reserve)
        score(neighborhood(best()), 'refinement')
        if len(cache) >= budget:
            break
    choice = best()
    return np.asarray(choice), dict(cost=cache[choice], evaluated_unique=len(cache),
        forward_batches=batches, configurations=[list(k) for k in cache],
        scores=list(cache.values()), search_phases=phases,
        selection='conditional single proposals with complete joint rescoring')


def main():
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = previous(*args, **kwargs)
        value['controller']['selection'] = 'conditional single proposals with complete joint rescoring'
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
                                                    joint_search=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    geometry.whole_configuration_search = coordinated_search
    geometry.main()


if __name__ == '__main__':
    main()
