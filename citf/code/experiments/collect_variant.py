"""Collect development variants against a preallocated immutable layout bank."""
import argparse
from pathlib import Path

import torch

from citf.contracts import LayoutRegistry, canonical_hash, file_hash, tree_hash


class PreallocatedRegistry(LayoutRegistry):
    def register(self, identity, split, alias):
        key = identity['fingerprint']
        entry = self.data['families'].get(key)
        if entry is None or entry['split'] != split or alias not in entry['aliases']:
            raise ValueError('Allocate the exact geometry, split, and alias before parallel collection')
        if canonical_hash(entry['identity']) != canonical_hash(identity):
            raise ValueError('Preallocated physical geometry identity differs')
        return key


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--behavior-checkpoint')
    args, _ = ap.parse_known_args()
    import citf.collect as collection
    implementation = {'collector': file_hash(__file__)}
    if args.behavior_checkpoint:
        checkpoint = torch.load(args.behavior_checkpoint, map_location='cpu', weights_only=False)
        revision = checkpoint['config']['revision']
        if revision == 'CITF-pair-geometry-v1':
            from experiments.pair_geometry import PairGeometryConfig, PairGeometryPredictor
            collection.Config, collection.Predictor = PairGeometryConfig, PairGeometryPredictor
            implementation['predictor'] = file_hash(Path(__file__).with_name('pair_geometry.py'))
        elif revision == 'CITF-stopping-graph-v1':
            from experiments.stopping_graph import StoppingConfig, StoppingPredictor
            collection.Config, collection.Predictor = StoppingConfig, StoppingPredictor
            implementation['predictor'] = file_hash(Path(__file__).with_name('stopping_graph.py'))
        elif revision != collection.Config().revision:
            raise ValueError('Unsupported development behavior revision')
    collection.LayoutRegistry = PreallocatedRegistry
    collection.tree_hash = lambda path: canonical_hash({'core': tree_hash(path), **implementation})
    collection.RUNTIME_REVISION = 'CITF-events-preallocated-variant-collection-v1'
    collection.main()


if __name__ == '__main__':
    main()
