"""Prepare the bounded Gate A matrix using this checkout's actual paths.

Standard-library only: preparation and resource waiting do not load PyTorch.
This creates development identities; it never queues formal training/tests.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--source'); ap.add_argument('--diagnostic-layouts', type=int, default=4)
    a = ap.parse_args(); out = Path(a.out).resolve()
    root = Path(__file__).resolve().parents[2]; project = root / 'source_archive'
    source = Path(a.source).resolve() if a.source else project / 'train/experiments/author-source-fusion-20261011/two-hour-development-20261026/best-m1-400.pth'
    if out.exists() or a.diagnostic_layouts < 1:
        raise ValueError('Use a new development output identity and positive layout count')
    registry = out / 'layout_registry.json'; jobs = []
    for name, role, seed, layouts, cross in [
            ('train_pool', 'train', 9730000, 64, False),
            ('label_validation', 'validation', 9740000, 24, False),
            ('closed_pilot', 'validation', 9750000, 16, True),
            ('diagnostic_roots', 'validation', 9760000, a.diagnostic_layouts, True)]:
        argv = ['-m', 'citf.tasks', '--project', str(project), '--registry', str(registry),
                '--out', str(out / (name + '.json')), '--role', role, '--seed-start', str(seed),
                '--layouts', str(layouts)]
        if cross:
            argv.append('--cross-agilities')
        jobs.append({'job_id': 'allocate_' + name, 'argv': argv,
                     'outputs': [{'path': str(out / (name + '.json')), 'equals': {'role': role}}],
                     'depends_on': [jobs[-1]['job_id']] if jobs else [], 'cooperative_stop': False})
    diagnosis = out / 'branch_diagnosis'
    jobs.append({'job_id': 'branch_diagnosis',
                 'argv': ['-m', 'experiments.branch_diagnosis', '--project', str(project),
                          '--source', str(source), '--tasks', str(out / 'diagnostic_roots.json'),
                          '--registry', str(registry), '--out', str(diagnosis), '--pool', '96',
                          '--futures', '3', '--root-times', '0', '2', '4', '8', '12', '20'],
                 'outputs': [{'path': str(diagnosis / 'summary.json'),
                              'equals': {'completed': True, 'expected_tasks': 2 * a.diagnostic_layouts}}],
                 'depends_on': [jobs[-1]['job_id']], 'cooperative_stop': True,
                 'resume_marker': str(diagnosis / 'protocol.json')})
    files = sorted((root / 'code').rglob('*.py'))
    files = [p for p in files if 'tests' not in p.parts and '__pycache__' not in p.parts]
    files += [source, project / 'train/envs/TADgame.py', project / 'train/envs/modules.py',
              project / 'train/policy/networks.py']
    manifest = {'scope': 'fixed-source development diagnosis; no formal training or final test',
                'cwd': str(root / 'code'), 'source': str(source), 'source_sha256': sha(source),
                'protocol': 'six defenders, source80, agility 2.0/2.5; unchanged current controller',
                'hypothesis': 'State-feedback interactions create learnable configuration-ranking errors beyond held-thrust geometry.',
                'primary_diagnostic': 'held-out-future empirical-oracle gain on common candidates, with physical-family uncertainty',
                'mechanism_diagnostic': 'one third-vessel candidate change while evaluated pair candidates are fixed',
                'next_decision': 'Inspect feedback/nominal regret, coverage and context effects; choose one justified change, then train and evaluate all predictor seeds 7101/7102/7103 with matched learning ablations.',
                'code_sha256': {str(p): sha(p) for p in files}, 'jobs': jobs}
    out.mkdir(parents=True)
    (out / 'matrix.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out / 'matrix.json')


if __name__ == '__main__':
    main()
