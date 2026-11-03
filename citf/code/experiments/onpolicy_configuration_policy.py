"""Refine current actions with observed geometry from their own state distribution.

The task-return objective, starting learned policy, frozen reference, training
layouts, and update budget match the declared reference-policy comparison.
Only the stopped-motion replay pool gains the complete newly acquired train
shards. Validation trajectories and development-bank labels never enter it.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from citf.contracts import canonical_hash, file_hash
from experiments import reference_configuration_policy as reference


@dataclass(frozen=True)
class OnPolicyReplayConfig(reference.ReferencePolicyConfig):
    revision: str = 'CITF-current-policy-observed-geometry-replay-v1'
    training_signal: str = 'ordinary task policy gradient, frozen learned-policy KL, and old plus current-policy stopped geometry'


_original_worker = reference.initialize_worker


def bind():
    reference.ReferencePolicyConfig = OnPolicyReplayConfig
    reference.initialize_worker = initialize_worker


def initialize_worker(*args):
    bind()
    _original_worker(*args)


def main():
    bind()
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--replay-shards', nargs='+', required=True)
        args, remaining = parser.parse_known_args()
        sys.argv = [sys.argv[0], *remaining]
        extra = list(args.replay_shards)
        load = reference.load_shards
        all_replay_paths = []

        def load_replay(paths, allowed):
            if allowed != 'train':
                raise ValueError('Current-policy replay accepts only training shards')
            all_replay_paths[:] = [*paths, *extra]
            if len(all_replay_paths) != len(set(all_replay_paths)):
                raise ValueError('A replay file was declared twice')
            return load(all_replay_paths, allowed)

        reference.load_shards = load_replay
        save = reference.atomic_save
        recorded = False

        def save_checkpoint(value, path):
            nonlocal recorded
            protocol = dict(value['protocol'])
            protocol['supervised_replay_paths'] = list(all_replay_paths)
            protocol['arguments'] = dict(protocol['arguments'], replay_shards=extra)
            protocol['additional_observed_policy_replay'] = dict(paths=extra,
                target='unchanged observed stopped motion under held fusion coefficients',
                selection='every declared training shard after the complete acquisition',
                sampling='uniform roots from the concatenated old and current-policy train pools')
            protocol['runtime_code'] = canonical_hash(dict(core=protocol['runtime_code'],
                onpolicy_configuration_policy=file_hash(__file__)))
            if not recorded:
                (Path(path).parent / 'protocol.json').write_text(json.dumps(protocol, indent=2))
                recorded = True
            save(dict(value, protocol=protocol), path)

        reference.atomic_save = save_checkpoint
        reference.main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['geometry_replay'] = OnPolicyReplayConfig().training_signal
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            onpolicy_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    reference.main()


if __name__ == '__main__':
    main()
