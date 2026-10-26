"""Archive and audit the user-designated r20 node and five-seed controls.

This exports existing evidence; it never selects a replacement r20 model or
converts the fixed r20 node into five independent proposed-method runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('train/experiments/author-source-fusion-20261011')
NODE = BASE / 'local-difference-joint-development/RESULT_NODE_20261020_043146_M2_R20.json'
PACKAGE = ROOT / BASE / 'r20-paper-evidence'
SUITES = ('source-development', 'new-development', 'fresh-200', 'new4310-200')
FAMILIES = ('sac', 'rp', 'arboids', 'mappo', 'maac')
SEEDS = tuple(range(42, 47))
CONFIGS = ('F00', 'F10', 'F01', 'F11', 'ordinary_no_M1', 'ordinary_with_M1')
INDEPENDENT = ROOT / BASE / 'r20-independent-evidence'
CONTRASTS = (
    ('M1_alone', 'F10', 'F00'), ('M2_alone', 'F01', 'F00'),
    ('M1_given_M2', 'F11', 'F01'), ('M2_given_M1', 'F11', 'F10'),
    ('M2_vs_ordinary_without_M1', 'F01', 'ordinary_no_M1'),
    ('M2_vs_ordinary_with_M1', 'F11', 'ordinary_with_M1'),
    ('M1_joint_vs_independent', 'F10', 'M1_independent'))
SOURCE_COMMIT = '9eb6df464808af6b9729f39644b4d658e1695e2d'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, getattr(sys.modules.get('torch'), 'Tensor', ())):
        return {'tensor_shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_pth(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def archive(source, target, expected=None):
    source, target = Path(source), Path(target)
    actual = digest(source)
    if expected and actual != expected:
        raise ValueError(f'Hash mismatch: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or digest(target) != actual:
        shutil.copy2(source, target)
    if digest(target) != actual:
        raise ValueError(f'Archived copy differs: {target}')
    return target


def baseline_directory(family, seed):
    if family in ('sac', 'rp', 'arboids'):
        return BASE / 'independent-training' / f'source-six-{family}-seed{seed}-20261021'
    return BASE / 'r20-six-baseline-comparison' / (family if family == 'boids' else f'{family}-seed{seed}')


def compact_path(relative):
    relative = Path(relative)
    return relative.relative_to(BASE) if relative.is_relative_to(BASE) else relative


def fetch_remote(include_checkpoints=False):
    """Retrieve completed training states and finished endpoint files."""
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('connect.cqa1.seetacloud.com', port=22476, username='root',
                   key_filename=[str(Path.home() / '.ssh/id_ed25519'), str(Path.home() / '.ssh/id_rsa')], timeout=20)
    remote_root = '/root/autodl-tmp/ARBoids'
    directories = [baseline_directory('boids', None).as_posix()]
    directories += [baseline_directory(f, s).as_posix() for f in FAMILIES for s in SEEDS]
    program = f'''from pathlib import Path
import hashlib,json,torch
root=Path({remote_root!r})
directories={directories!r}
rows=[]
for relative in directories:
 directory=root/relative
 if not directory.is_dir(): continue
 if directory.name!='boids':
  record=directory/('source-repeat-result.pth' if directory.name.startswith('source-six-') else 'training-result.pth')
  if not record.exists() or not torch.load(record,map_location='cpu',weights_only=False).get('completed'):continue
 for path in sorted(directory.iterdir()):
  if not path.is_file() or path.name.startswith('.') or path.name.endswith('.partial'): continue
  if path.name.startswith('last.') and not {include_checkpoints!r}: continue
  if path.suffix not in ('.pth','.zst','.csv','.yaml','.json'): continue
  if path.stem in {SUITES!r} and not torch.load(path,map_location='cpu',weights_only=False).get('completed'):continue
  sha=hashlib.sha256()
  with path.open('rb') as source:
   for block in iter(lambda:source.read(8*1024*1024),b''):sha.update(block)
  rows.append(dict(relative=path.relative_to(root).as_posix(),sha256=sha.hexdigest(),bytes=path.stat().st_size,mtime=path.stat().st_mtime))
print(json.dumps(rows))
'''
    try:
        stdin, stdout, stderr = client.exec_command(remote_root + '/.venv/bin/python -B -X utf8 -', timeout=300)
        stdin.write(program)
        stdin.channel.shutdown_write()
        records = json.loads(stdout.read().decode('utf-8'))
        errors = stderr.read().decode('utf-8')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(errors)
        metadata_path = PACKAGE / 'metadata/retrieved_baselines.json'
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))['artifacts'] if metadata_path.exists() else []
        combined = {row['relative']: row for row in previous}
        sftp = paramiko.SFTPClient.from_transport(client.get_transport(),
                                                 window_size=8 * 1024 * 1024,
                                                 max_packet_size=256 * 1024)
        sftp.get_channel().settimeout(45)
        try:
            saved = 0
            for record in records:
                destination = PACKAGE / 'raw/baselines' / compact_path(record['relative'])
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or digest(destination) != record['sha256']:
                    temporary = destination.with_name(destination.name + '.partial')
                    offset = temporary.stat().st_size if temporary.exists() else 0
                    if offset > record['bytes']:
                        raise ValueError('Partial download exceeds source size: ' + record['relative'])
                    started = last_progress = time.monotonic()
                    start_offset = offset
                    if record['bytes'] > 10 * 1024 * 1024:
                        print('DOWNLOADING', record['relative'], offset, '/', record['bytes'], flush=True)
                    with sftp.open(remote_root + '/' + record['relative'], 'rb', bufsize=0) as source, temporary.open('ab') as output:
                        while offset < record['bytes']:
                            end = min(offset + 8 * 1024 * 1024, record['bytes'])
                            requests = [(position, min(32768, end - position)) for position in range(offset, end, 32768)]
                            for (_, size), block in zip(requests, source.readv(requests, max_concurrent_prefetch_requests=64)):
                                if len(block) != size:
                                    raise ValueError('Short remote read: ' + record['relative'])
                                output.write(block)
                                offset += len(block)
                            output.flush()
                            now = time.monotonic()
                            if now - last_progress >= 15:
                                print('DOWNLOAD_PROGRESS', record['relative'], offset, '/', record['bytes'],
                                      round((offset - start_offset) / (now - started) / 1024**2, 2), 'MiB/s', flush=True)
                                last_progress = now
                    if offset != record['bytes']:
                        raise ValueError('Downloaded size differs: ' + record['relative'])
                    if digest(temporary) != record['sha256']:
                        raise ValueError('Downloaded file hash differs: ' + record['relative'])
                    temporary.replace(destination)
                    saved += 1
                record['package_path'] = destination.relative_to(PACKAGE).as_posix()
                combined[record['relative']] = record
                if record['relative'].endswith('/source-repeat-result.pth') or record['relative'].endswith('/training-result.pth'):
                    print('ARCHIVED', Path(record['relative']).parent.name, flush=True)
            save_json(metadata_path, {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                                     'artifacts': list(combined.values())})
            print('FETCH_COMPLETE', len(records), 'files;', saved, 'downloaded', flush=True)
        finally:
            sftp.close()
    finally:
        client.close()


def fetch_independent():
    """Retrieve immutable completed evidence; retain large raw training data in place."""
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('connect.cqa1.seetacloud.com', port=22476, username='root',
        key_filename=[str(Path.home()/'.ssh/id_ed25519'), str(Path.home()/'.ssh/id_rsa')], timeout=20)
    remote_root = '/root/autodl-tmp/ARBoids'
    relative = (BASE/'r20-independent-evidence').as_posix()
    program = f'''from pathlib import Path
import hashlib,json,torch
root=Path({remote_root!r});base=root/{relative!r}
records=[]
for path in sorted(base.rglob('*')):
 if not path.is_file() or path.name.endswith('.partial'):continue
 rel=path.relative_to(base).as_posix()
 if any(part.endswith('-root-work') for part in path.relative_to(base).parts):continue
 if rel.startswith('vrx/') and not rel.startswith('vrx/paired/'):continue
 if path.suffix not in ('.pth','.json','.npz','.png','.sdf'):continue
 if 'fitting' in path.name:continue
 large_data=path.name.endswith(('-paired-data.pth','-factual.pth'))
 if large_data and not (path.parent/'result.pth').exists():continue
 if path.suffix=='.pth' and path.name not in ('manifest.pth','analysis-manifest.pth','baseline-runtime.pth'):
  value=torch.load(path,map_location='cpu',weights_only=False)
  endpoint='rows' in value or 'outcomes' in value or large_data or path.name in ('result.pth','selection.pth')
  if endpoint and not value.get('completed'):continue
  if path.parent.name.startswith('seed') and path.stem in ('F01','F11','ordinary_no_M1','ordinary_with_M1') and not value.get('r20_replication',{{}}).get('completed'):continue
 sha=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
 records.append(dict(relative=rel,remote_path=str(path),sha256=sha.hexdigest(),bytes=path.stat().st_size,stored_locally=not large_data))
analysis=torch.load(base/'analysis-manifest.pth',map_location='cpu',weights_only=False)
opponent=analysis['learned_opponent'];path=Path(opponent['path'])
sha=hashlib.sha256()
with path.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
if sha.hexdigest()!=opponent['sha256']:raise ValueError('The frozen common opponent changed before retrieval.')
records.append(dict(relative='dependencies/common-learned-opponent.pth',remote_path=str(path),
 sha256=sha.hexdigest(),bytes=path.stat().st_size,stored_locally=True))
print(json.dumps(records))
'''
    try:
        stdin, stdout, stderr = client.exec_command(remote_root+'/.venv/bin/python -X utf8 -', timeout=300)
        stdin.write(program); stdin.channel.shutdown_write()
        records = json.loads(stdout.read().decode('utf8'))
        errors = stderr.read().decode('utf8')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(errors)
        root = PACKAGE/'raw/independent'
        sftp = paramiko.SFTPClient.from_transport(client.get_transport(),
            window_size=8*1024*1024, max_packet_size=256*1024)
        sftp.get_channel().settimeout(60)
        downloaded = size = 0
        started = last_progress = time.monotonic()
        try:
            for record in records:
                if not record['stored_locally']:
                    continue
                path = root/record['relative']
                if path.exists() and digest(path) == record['sha256']:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name+'.partial')
                # A partial transfer can have belonged to an earlier snapshot.
                with sftp.open(record['remote_path'], 'rb', bufsize=0) as source, temporary.open('wb') as target:
                    for start in range(0, record['bytes'], 8*1024*1024):
                        end = min(start+8*1024*1024, record['bytes'])
                        requests = [(p, min(32768, end-p)) for p in range(start, end, 32768)]
                        for (_, count), block in zip(requests, source.readv(requests, max_concurrent_prefetch_requests=64)):
                            if len(block) != count:
                                raise ValueError('Incomplete evidence transfer: '+record['relative'])
                            target.write(block)
                if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
                    raise ValueError('Transferred evidence hash differs: '+record['relative'])
                temporary.replace(path)
                downloaded += 1; size += record['bytes']
                if time.monotonic()-last_progress > 20:
                    print('EVIDENCE_FETCH', downloaded, 'files', round(size/2**20, 1), 'MiB', flush=True)
                    last_progress = time.monotonic()
        finally:
            sftp.close()
        save_json(PACKAGE/'metadata/retrieved_independent.json', dict(
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(), artifacts=records,
            large_training_data_location='original remote paths; exact hashes retained without redundant local copies'))
        print('EVIDENCE_FETCH_COMPLETE', len(records), 'indexed;', downloaded, 'downloaded;',
              round(time.monotonic()-started, 1), 'seconds', flush=True)
    finally:
        client.close()


def save_csv(name, rows):
    path = PACKAGE / 'tables' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def checked_evaluation(path, tasks, policy_sha):
    data = read_pth(path)
    if not data.get('completed') or len(data['rows']) != len(tasks) or len(data['tasks']) != len(tasks):
        raise ValueError(f'Incomplete evaluation: {path}')
    if canonical(data['tasks']) != canonical(tasks):
        raise ValueError(f'Full ordered task definitions differ: {path}')
    if data.get('policy_sha256') != policy_sha:
        raise ValueError(f'Evaluation policy hash differs: {path}')
    count = 0
    for index, (task, row) in enumerate(zip(tasks, data['rows'])):
        if row is None or row['success'] not in (0, 1):
            raise ValueError(f'Invalid episode at {path}:{index}')
        if canonical([row['seed'], row['defenders'], row['agility']]) != canonical(task[0]):
            raise ValueError(f'Episode identity differs at {path}:{index}')
        if any(not math.isfinite(row[key]) for key in ('duration', 'discounted_return')):
            raise ValueError(f'Nonfinite endpoint metric at {path}:{index}')
        count += int(row['success'])
    if data.get('success') != count or data.get('evaluated') != len(tasks):
        raise ValueError(f'Saved evaluation total differs: {path}')
    return data


def paired(a, b):
    # a is the candidate; b is the reference. The caller checked full task order.
    rescues = [i for i, (x, y) in enumerate(zip(a, b)) if x['success'] and not y['success']]
    harms = [i for i, (x, y) in enumerate(zip(a, b)) if not x['success'] and y['success']]
    return dict(rescued=len(rescues), harmed=len(harms), rescued_indices=rescues, harmed_indices=harms)


def crossed_interval(values, strata=None, draws=10000):
    """Resample training seeds and common task columns, preserving task strata."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.isfinite(values).all() or not values.size:
        raise ValueError('Bootstrap input must be a finite seed-by-task matrix.')
    strata = np.zeros(values.shape[1], dtype=int) if strata is None else np.asarray(strata)
    if len(strata) != values.shape[1]:
        raise ValueError('Bootstrap strata do not match the task columns.')
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    generator = np.random.default_rng(20261026)
    samples = []
    for start in range(0, draws, 256):
        count = min(256, draws-start)
        seeds = generator.integers(values.shape[0], size=(count, values.shape[0]))
        tasks = np.concatenate([group[generator.integers(len(group), size=(count, len(group)))]
                                for group in groups], axis=1)
        samples.extend(values[seeds[:, :, None], tasks[:, None, :]].mean(axis=(1, 2)))
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def paired_statistics(a, b, strata, repeated=False):
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError('Paired comparisons require the same seed and task axes.')
    if left.ndim == 1:
        left, right = left[None, :], right[None, :]
    difference = left-right
    rescues = ((left == 1) & (right == 0)).sum(axis=1)
    harms = ((left == 0) & (right == 1)).sum(axis=1)
    ci = crossed_interval(difference, strata)
    result = dict(gain_percentage_points=100*float(difference.mean()),
        ci95_low_percentage_points=100*ci[0], ci95_high_percentage_points=100*ci[1],
        training_repetitions=left.shape[0] if repeated else 0, paired_tasks=left.shape[1],
        rescued_by_seed=rescues.tolist(), harmed_by_seed=harms.tolist(),
        seed_gains_percentage_points=(100*difference.mean(axis=1)).tolist(),
        interval_method='crossed bootstrap of seeds and stratified common tasks' if repeated
                        else 'stratified paired-task bootstrap conditional on the frozen models')
    if repeated:
        means = difference.mean(axis=1)
        signs = np.asarray([[1 if mask & (1 << i) else -1 for i in range(len(means))]
                            for mask in range(2**len(means))])
        result['seed_sign_flip_p_two_sided'] = float(
            (np.abs((signs*means).mean(axis=1)) >= abs(means.mean())-1e-14).mean())
        result['hypothesis_test_unit'] = 'independent training seed; paired frozen task population'
    else:
        discordant = int(rescues[0]+harms[0]); smaller = int(min(rescues[0], harms[0]))
        result['mcnemar_exact_p_two_sided'] = min(1., 2*sum(math.comb(discordant, i)
            for i in range(smaller+1))/2**discordant) if discordant else 1.
        result['hypothesis_test_unit'] = 'paired task; conditional on one frozen model per method'
    return result


def holm_adjust(rows, key, family_size=None):
    eligible = sorted((row for row in rows if key in row), key=lambda row: row[key])
    family_size = len(eligible) if family_size is None else family_size
    if family_size < len(eligible):
        raise ValueError('A planned comparison family cannot omit available comparisons.')
    previous = 0.
    for rank, row in enumerate(eligible):
        previous = max(previous, min(1., (family_size-rank)*row[key]))
        row[key+'_holm'] = previous
        row['multiplicity_family_size'] = family_size


def collect_traces():
    """Observe exact saved-task replays; keep both favorable and adverse examples."""
    import r20_independent_evidence as e
    manifest = e.read(e.MANIFEST)
    e.runtime(manifest)
    d = sys.modules['develop_source_fusion']
    evaluations = {}
    for label in (*CONFIGS, 'M1_independent'):
        path = e.OUTPUT/'fixed'/(label+'-heldout.pth')
        evaluations[label] = checked_evaluation(path, manifest['heldout_tasks'],
                                                manifest['models'][label]['sha256'])
    examples = []
    selected = set()
    for contrast, candidate, reference in CONTRASTS:
        if contrast in ('M2_alone', 'M1_given_M2'):
            continue
        effect = paired(evaluations[candidate]['rows'], evaluations[reference]['rows'])
        for direction in ('rescued', 'harmed'):
            indices = effect[direction+'_indices']
            index = indices[0] if indices else None
            examples.append(dict(contrast=contrast, candidate=candidate, reference=reference,
                direction=direction, count=len(indices), task_index=index,
                selection_rule='first discordant task in the frozen order; illustrative only'))
            if index is not None:
                selected.update(((candidate, index), (reference, index)))
    e.save(dict(completed=True, manifest_sha256=e.digest(e.MANIFEST), examples=examples),
           e.OUTPUT/'traces/selection.pth')
    source = Path(__file__).read_text(encoding='utf8')
    for label, index in sorted(selected):
        output = e.OUTPUT/'traces'/f'{label}-task{index:03d}.pth'
        expected = evaluations[label]['rows'][index]
        if output.exists():
            previous = e.read(output)
            if previous['completed'] and previous['row'] == expected:
                continue
            raise ValueError('A saved trajectory has a different endpoint identity.')
        e.resource_wait()
        policy = Path(manifest['models'][label]['path'])
        if e.digest(policy) != manifest['models'][label]['sha256']:
            raise ValueError('A trajectory policy changed.')
        scene, future = manifest['heldout_tasks'][index]
        frames, actions = [], []

        def snapshot(env):
            vessels = [env.attacker, *env.defender_list]
            return dict(time=float(env.Current_T),
                positions=np.asarray([v.pos for v in vessels], dtype=np.float64).copy(),
                headings=np.asarray([v.theta for v in vessels], dtype=np.float64),
                velocities=np.asarray([v.velocity for v in vessels], dtype=np.float64).copy(),
                minimum_defender_distance=float(env.def_def_dists.min()),
                minimum_attacker_distance=float(env.def_att_dists.min()),
                attacker_target_distance=float(np.linalg.norm(env.attacker.pos)))

        def observe(frame, event, result):
            if frame.f_code is not d.joint_task_step.__code__:
                return
            env = frame.f_locals['env']
            if event == 'call':
                if not frames:
                    frames.append(snapshot(env))
                actions.append(np.asarray(frame.f_locals['action']).copy())
            elif event == 'return':
                frames.append(snapshot(env))

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            actual = d.joint_episode((scene, str(policy), policy.stat().st_mtime_ns,
                                     False, 0, 0, 0, 0, future))
        finally:
            sys.setprofile(previous)
        if actual != expected:
            raise ValueError(f'Trajectory replay changed the recorded outcome: {label} task {index}')
        if len(frames) != len(actions)+1 or not actions:
            raise ValueError('A trajectory is missing physical-step observations.')
        trace = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        trace['commands'] = np.asarray(actions)
        e.save(dict(completed=True, method=label, task_index=index, task=(scene, future),
            row=actual, exact_endpoint_replay=True, trajectory=trace,
            policy_sha256=manifest['models'][label]['sha256'],
            manifest_sha256=e.digest(e.MANIFEST), runtime_sha256=manifest['runtime_sha256'],
            diagnostic_source=source, diagnostic_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            evidence_role='illustrative first rescue and first harm; all paired outcomes retained separately'), output)
        print(dict(phase='r20_exact_trajectory', method=label, task=index,
                   physical_steps=len(actions), success=actual['success']), flush=True)


def build_independent():
    root = PACKAGE/'raw/independent'
    if not (root/'manifest.pth').exists():
        return dict(completed=False, pending=['independent evidence not retrieved']), []
    manifest = read_pth(root/'manifest.pth')
    manifest_sha = digest(root/'manifest.pth')
    if any(hashlib.sha256(source.encode()).hexdigest() != manifest['runtime_sha256'][name]
           for name, source in manifest['runtime_sources'].items()):
        raise ValueError('The frozen runtime source and identity differ.')
    tasks = manifest['heldout_tasks']
    strata = np.asarray([task[0][2] for task in tasks])
    groups = [('all-400', np.arange(len(tasks)))]+[
        (f'agility-{a:g}', np.flatnonzero(strata == a)) for a in np.unique(strata)]
    pending, by_seed, episodes, model_records = [], [], [], []
    objective_rows, budget_rows = [], []
    populations = {}

    def add(category, label, seed, path, identity):
        if not path.exists() or not read_pth(path).get('completed'):
            pending.append(str(path.relative_to(root)))
            return None
        data = checked_evaluation(path, tasks, identity)
        if data['manifest_sha256'] != manifest_sha:
            raise ValueError('An endpoint was evaluated against another prospective manifest.')
        if category != 'original-baseline' and data['runtime_sha256'] != manifest['runtime_sha256']:
            raise ValueError('An endpoint used another r20 runtime.')
        values = np.asarray([row['success'] for row in data['rows']], dtype=float)
        populations.setdefault((category, label), {})[seed] = values
        for suite, indices in groups:
            rows = [data['rows'][int(i)] for i in indices]
            by_seed.append(dict(category=category, method=label, training_seed=seed, suite=suite,
                success=int(values[indices].sum()), episodes=len(indices),
                success_rate_percent=100*float(values[indices].mean()),
                collision_rate_percent=100*float(np.mean([r['collision'] for r in rows])),
                physical_breach_rate_percent=100*float(np.mean([r['physical_breach'] for r in rows])),
                capture_rate_percent=100*float(np.mean([r['capture'] for r in rows])),
                timeout_rate_percent=100*float(np.mean([r['timeout'] for r in rows])),
                mean_duration_seconds=float(np.mean([r['duration'] for r in rows])),
                mean_discounted_return=float(np.mean([r['discounted_return'] for r in rows])),
                policy_sha256=identity))
        for index, row in enumerate(data['rows']):
            episodes.append(dict(category=category, method=label, training_seed=seed,
                task_index=index, task_json=canonical(tasks[index]), **plain(row)))
        return data

    for label in (*CONFIGS, 'M1_independent'):
        add('fixed-r20-reference', label, None, root/'fixed'/(label+'-heldout.pth'),
            manifest['models'][label]['sha256'])
    for family, seed in [('boids', None)]+[(f, s) for f in FAMILIES for s in SEEDS]:
        label = family if seed is None else f'{family}-seed{seed}'
        path = root/'baselines'/label/'heldout.pth'
        identity = None
        if family != 'boids' and path.exists():
            data = read_pth(path)
            training = data['source_training']
            native = family in ('arboids', 'sac', 'rp')
            identity = training['selected_checkpoint_sha256'] if native else training['best_sha256']
            weight = PACKAGE/'raw/baselines'/compact_path(baseline_directory(family, seed))/(
                'best-development.pth' if native else 'best.pth')
            if not training['completed'] or not weight.exists() or digest(weight) != identity:
                raise ValueError('An unseen baseline lacks its completed training and exact selected model.')
        add('original-baseline', family, seed, path, identity)
    completed_seeds = []
    for seed in SEEDS:
        directory = root/f'seed{seed}'
        result = read_pth(directory/'result.pth') if (directory/'result.pth').exists() else None
        if result is None:
            pending.append(f'seed{seed}/complete-pipeline')
        elif not result['completed'] or result['manifest_sha256'] != manifest_sha:
            raise ValueError('An independent training pipeline has invalid completion provenance.')
        else:
            completed_seeds.append(seed)
        models = {}
        for label in (*CONFIGS, 'M1_independent'):
            model_path = directory/(label+'.pth')
            identity = digest(model_path) if model_path.exists() else None
            if result and (result['models'][label]['sha256'] != identity):
                raise ValueError('An independently trained model differs from its completion record.')
            if model_path.exists():
                model = read_pth(model_path)
                models[label] = model
                record = model['r20_replication']
                if record['training_seed'] != seed or record['manifest_sha256'] != manifest_sha:
                    raise ValueError('An independent model has the wrong seed or recipe identity.')
                if 'successor_task_value' in model and model['successor_task_value'] is not None:
                    fit = model['successor_task_fit']
                    if not record.get('completed') or record['inherited_value_weights'] or record['heldout_used_for_training']:
                        raise ValueError('The independent estimator was not freshly fitted without heldout outcomes.')
                    dispatch = record.get('collection_dispatch')
                    if dispatch:
                        proof_path = root/'verification/root-dispatch-replay.pth'
                        proof = read_pth(proof_path)
                        if (digest(proof_path) != dispatch['audit_sha256'] or not proof['completed']
                                or proof['manifest_sha256'] != manifest_sha
                                or proof['dispatcher_sha256'] != dispatch['dispatcher_sha256']
                                or any(row['mismatches'] for row in proof['roots'])):
                            raise ValueError('Parallel root collection lacks its exact frozen-sampler replay.')
                        native_groups = dispatch.get('native_future_groups')
                        if native_groups:
                            group_proof_path = root/'verification/future-group-replay.pth'
                            group_proof = read_pth(group_proof_path)
                            if (digest(group_proof_path) != native_groups['audit_sha256']
                                    or not group_proof['completed']
                                    or group_proof['manifest_sha256'] != manifest_sha
                                    or group_proof['runtime_sha256'] != manifest['runtime_sha256']
                                    or group_proof['dispatcher_sha256'] != native_groups['dispatcher_sha256']
                                    or group_proof['legacy_dispatcher_sha256'] != dispatch['dispatcher_sha256']
                                    or native_groups['manifest_sha256'] != manifest_sha
                                    or native_groups['predecessor_sha256'] != fit['predecessor_sha256']
                                    or hashlib.sha256(group_proof['dispatcher_source'].encode()).hexdigest()
                                        != native_groups['dispatcher_sha256']
                                    or {row['workers'] for row in group_proof['roots']} != {2, 4}
                                    or any(row['mismatches'] for row in group_proof['roots'])
                                    or not group_proof.get('scheduler_replay', {}).get('completed')
                                    or group_proof['scheduler_replay']['mismatches']):
                                raise ValueError('Native future groups lack their exact frozen-sampler replay.')
                            for root_execution in native_groups['roots']:
                                for continuation in root_execution['continuations']:
                                    if (continuation['dispatcher_sha256'] != native_groups['dispatcher_sha256']
                                            or continuation['workers'] not in (1, 2, 3, 4)
                                            or continuation['retained_observations'] < 0
                                            or continuation['inference_batch'] != 32
                                            or continuation['inference_device'] != 'cpu'):
                                        raise ValueError('A future-group continuation changed its numerical path.')
                    if fit['updates'] != 24000 or len(model['successor_task_value']['states']) != 3:
                        raise ValueError('The independently fitted estimator has a different training budget.')
                    model_records.append(dict(seed=seed, method=label, policy_sha256=identity,
                        training=plain({key:value for key,value in fit.items()
                            if key not in ('root_diagnostics','local_fits')}), replication=plain(record),
                        detailed_training_diagnostics='retained in the exact policy checkpoint',
                        source_training=plain(model.get('independent_source', {}))))
                    history = record['objective_history']
                    expected_updates = [(member, update) for member in range(3) for update in range(1000, 8001, 1000)]
                    if [(row['member'], row['update']) for row in history] != expected_updates:
                        raise ValueError('An independent estimator lacks its complete objective history.')
                    for row in history:
                        if not math.isfinite(row['loss']):
                            raise ValueError('A completed estimator has a nonfinite training objective.')
                        objective_rows.append(dict(training_seed=seed, method=label,
                            member=row['member'], member_update=row['update'], objective=row['loss'],
                            objective_definition=fit['objective']))
                    budget_rows.append(dict(training_seed=seed, method=label,
                        source_training_steps=1000000, source_training_defenders=3,
                        additional_measured_simulation_steps=fit['simulation_steps'],
                        measurement_shared_with=('ordinary_no_M1' if label == 'F01' else
                            'ordinary_with_M1' if label == 'F11' else 'F01' if label == 'ordinary_no_M1' else 'F11'),
                        measured_terminal_observations=fit['unique_terminal_observations'],
                        roots=fit['roots'], factual_trajectory_replays=fit['factual_trajectory_replays'],
                        total_optimizer_updates=fit['updates'], ensemble_members=3,
                        optimizer_updates_per_member=8000, hidden_width=fit['width'],
                        learned_parameter_count=sum(v.numel() for state in model['successor_task_value']['states'] for v in state.values()),
                        policy_sha256=identity, training_observation_sha256=fit['training_observation_sha256']))
            add('independent-pipeline', label, seed, directory/(label+'-heldout.pth'), identity)
        for proposed, ordinary, predecessor in (('F01', 'ordinary_no_M1', 'F00'),
                                                ('F11', 'ordinary_with_M1', 'F10')):
            if all(k in models for k in (proposed, ordinary, predecessor)):
                a, b = (models[k]['successor_task_fit'] for k in (proposed, ordinary))
                if a['training_observation_sha256'] != b['training_observation_sha256']:
                    raise ValueError('A proposed/ordinary pair used different measured observations.')
                if a['predecessor_sha256'] != digest(directory/(predecessor+'.pth')) or a['predecessor_sha256'] != b['predecessor_sha256']:
                    raise ValueError('A proposed/ordinary pair used different predecessors.')
    summary = []
    for (category, label), members in populations.items():
        seed_order = sorted(members, key=lambda s: -1 if s is None else s)
        matrix = np.asarray([members[seed] for seed in seed_order])
        repeated = category != 'fixed-r20-reference' and label != 'boids'
        expected = 5 if repeated else 1
        for suite, indices in groups:
            values = matrix[:, indices]
            rates = 100*values.mean(axis=1)
            interval = crossed_interval(values, strata[indices])
            summary.append(dict(category=category, method=label, suite=suite,
                available_models=len(members), expected_models=expected, complete=len(members) == expected,
                training_repetitions=len(members) if repeated else 0,
                distinct_tasks=len(indices), mean_success_rate_percent=float(rates.mean()),
                sample_sd_percentage_points=float(rates.std(ddof=1)) if len(rates)>1 else None,
                ci95_low_percent=100*interval[0], ci95_high_percent=100*interval[1],
                interval_method='crossed seed/task bootstrap' if repeated else 'conditional task bootstrap'))
    effects = []
    for category in ('fixed-r20-reference', 'independent-pipeline'):
        comparisons = list(CONTRASTS)
        if category == 'independent-pipeline':
            comparisons += [('complete_vs_'+family, 'F11', 'baseline:'+family) for family in ('boids', *FAMILIES)]
        selected = []
        for contrast, candidate, reference in comparisons:
            a = populations.get((category, candidate), {})
            external = reference.startswith('baseline:')
            family = reference.split(':')[-1]
            b = populations.get(('original-baseline' if external else category, family), {})
            repeated = category == 'independent-pipeline'
            seeds = list(SEEDS) if repeated else [None]
            if not all(s in a and (s in b or (external and family == 'boids' and None in b)) for s in seeds):
                continue
            left = np.asarray([a[s] for s in seeds])
            right = np.asarray([b[None if external and family == 'boids' else s] for s in seeds])
            row = dict(category=category, contrast=contrast, candidate=candidate, reference=reference,
                       **paired_statistics(left, right, strata, repeated))
            effects.append(row); selected.append(row)
        holm_adjust(selected, 'seed_sign_flip_p_two_sided' if category == 'independent-pipeline'
                    else 'mcnemar_exact_p_two_sided', family_size=len(comparisons))
        four = [populations.get((category, label), {}) for label in ('F11', 'F10', 'F01', 'F00')]
        seeds = list(SEEDS) if category == 'independent-pipeline' else [None]
        if all(all(s in values for s in seeds) for values in four):
            matrix = np.asarray([four[0][s]-four[1][s]-four[2][s]+four[3][s] for s in seeds])
            interval = crossed_interval(matrix, strata)
            effects.append(dict(category=category, contrast='factorial_interaction',
                candidate='(F11-F10)', reference='(F01-F00)',
                gain_percentage_points=100*float(matrix.mean()),
                ci95_low_percentage_points=100*interval[0], ci95_high_percentage_points=100*interval[1],
                paired_tasks=len(tasks), training_repetitions=len(seeds) if seeds[0] is not None else 0,
                interval_method='crossed seed/task bootstrap of the paired difference of differences'))
    save_csv('independent_by_seed.csv', by_seed)
    save_csv('independent_summary.csv', summary)
    save_csv('independent_paired_effects.csv', effects)
    save_csv('independent_episodes.csv', episodes)
    save_csv('independent_training_objectives.csv', objective_rows)
    save_csv('independent_training_budget.csv', budget_rows)
    save_json(PACKAGE/'metadata/independent_training.json', dict(models=model_records,
        completed_seeds=completed_seeds, source_training_steps=1000000, source_training_defenders=3,
        original_external_baseline_training_defenders=6,
        budget_comparison='M2 and ordinary replacements match observations, capacity and optimizer updates; external baselines do not receive the extra finite-intervention data',
        seed42_source='the existing completed seed-42 source actor is reused; all repeated M2 estimators use newly measured states and fresh value initialization',
        fixed_reference_excluded_from_independent_mean=True))
    diagnostics, diagnostic_lines = build_diagnostics(root, manifest)
    pending.extend(diagnostics['pending'])
    replay = read_pth(root/'verification/training-replay.pth') if (root/'verification/training-replay.pth').exists() else None
    replay_passed = bool(replay and replay['completed'] and replay['runtime_sha256'] == manifest['runtime_sha256']
                         and all(x['parameters_exact'] and x['max_absolute_difference'] == 0 for x in replay['arms'].values()))
    if not replay_passed:
        pending.append('exact training replay')
    result = dict(completed=not pending, completed_training_seeds=completed_seeds,
        required_training_seeds=list(SEEDS), fixed_models_completed=sum(k[0] == 'fixed-r20-reference' for k in populations),
        original_baseline_models_completed=sum(len(v) for k, v in populations.items() if k[0] == 'original-baseline'),
        independent_model_endpoints_completed=sum(len(v) for k, v in populations.items() if k[0] == 'independent-pipeline'),
        distinct_unseen_endpoint_tasks=len(tasks), manifest_sha256=manifest_sha,
        exact_proposed_training_replay_passed=replay_passed, diagnostics=diagnostics,
        pending=pending, reported_episode_rows=len(episodes))
    save_json(PACKAGE/'metadata/independent_status.json', result)
    independent_figures(root, summary, by_seed, effects, diagnostics)
    lines = ['', '## 独立测试与重复', '',
        '固定 r20 节点与重新训练的五条流水线分别统计。400 个新任务在训练、验证选模和历史开发任务之外冻结；各方法使用同一完整任务顺序。', '',
        '| 证据类型 | 方法 | 成功率（%） | 训练重复数 |', '|---|---|---:|---:|']
    for row in summary:
        if row['suite'] != 'all-400':
            continue
        value = f"{row['mean_success_rate_percent']:.2f}"
        if row['sample_sd_percentage_points'] is not None:
            value += f" ± {row['sample_sd_percentage_points']:.2f}"
        if not row['complete']:
            value += '（重复未齐）'
        lines.append(f"| {row['category']} | {row['method']} | {value} | {row['training_repetitions']} |")
    lines += ['', '表中标准差按训练种子计算。区间同时重采样训练种子和按灵活度分层的共同任务；成对比较保留同一任务列，未把 5×400 次评价当成 2000 个独立训练样本。固定节点的区间与 McNemar 检验仅对该固定模型成立。',
        '固定节点七项比较、五种子十三项比较分别使用 Holm 校正，未完成的计划比较仍计入比较族。五种子比较使用训练种子层面的符号翻转检验，并报告跨种子与任务的自助区间；五个种子下双侧检验的最小非零 p 值为 0.0625。',
        'M2 与普通回报替代项使用相同实测数据、网络规模和 24,000 次更新。相对外部基线，新增有限干预采样的计算成本单独列出，不声称总训练计算量相等。',
        '训练复现：F01 与 F11 的原始三成员参数均逐项完全复现，最大绝对差为 0；这验证了训练计算分支，不将恢复的代码冒称完整历史工程的原始字节快照。', '']
    lines += ['| 比较对象 | 配对比较 | 增益（百分点） | 95% 区间 | 救回 / 损害 | 校正后 p |',
              '|---|---|---:|---:|---|---:|']
    for row in effects:
        if row['contrast'] == 'factorial_interaction':
            continue
        repeated = row['category'] == 'independent-pipeline'
        test = 'seed_sign_flip_p_two_sided_holm' if repeated else 'mcnemar_exact_p_two_sided_holm'
        units = '五种子重复' if repeated else '固定 r20'
        counts = ', '.join(f'{a}/{b}' for a,b in zip(row['rescued_by_seed'],row['harmed_by_seed']))
        lines.append(f"| {units} | {row['candidate']} − {row['reference'].replace('baseline:', '')} | "
            f"{row['gain_percentage_points']:+.2f} | [{row['ci95_low_percentage_points']:+.2f}, "
            f"{row['ci95_high_percentage_points']:+.2f}] | {counts} | {row[test]:.4g} |")
    lines += ['', '五种子救回/损害按 42–46 顺序逐一列出。区间为逐比较区间，p 值按对应比较族作 Holm 校正。', '']
    lines.extend(diagnostic_lines)
    if pending:
        lines += ['', f'独立证据仍有 {len(pending)} 项未齐，完整列表见 `metadata/independent_status.json`。']
    return result, lines


def nested_interval(values, strata):
    """Resample independent states and their own shared-future measurements."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == a) for a in np.unique(strata)]
    rng = np.random.default_rng(20261026)
    means = []
    for start in range(0, 10000, 250):
        states = np.concatenate([g[rng.integers(len(g), size=(250, len(g)))] for g in groups], axis=1)
        futures = rng.integers(values.shape[1], size=(*states.shape, values.shape[1]))
        means.extend(values[states[:, :, None], futures].mean(axis=(1, 2)))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def build_diagnostics(root, manifest):
    if not (root/'analysis-manifest.pth').exists():
        return dict(completed=False, pending=['diagnostic manifest']), []
    analysis = read_pth(root/'analysis-manifest.pth')
    identity = digest(root/'analysis-manifest.pth')
    manifest_identity = digest(root/'manifest.pth')
    baseline_identity = digest(root/'baseline-runtime.pth')
    if analysis['r20_manifest_sha256'] != manifest_identity:
        raise ValueError('The diagnostic protocol does not match the frozen r20 population.')
    opponent = root/'dependencies/common-learned-opponent.pth'
    if not opponent.exists() or digest(opponent) != analysis['learned_opponent']['sha256']:
        raise ValueError('The scope evidence lacks its exact common learned opponent.')
    def remapped_scene(seed, scene):
        return int(np.random.SeedSequence([20261025, 835104, 10, seed, int(scene)]).generate_state(1)[0])
    historical = {int(row['scene'][0]) for design in manifest['designs'].values() for row in design['roots']}
    historical.update(int(scene[0]) for tasks in manifest['factual_tasks'].values() for scene, _ in tasks)
    training_scenes = {seed: {remapped_scene(seed, scene) for scene in historical} for seed in SEEDS}
    prospective = [int(scene[0]) for tasks in (manifest['heldout_tasks'], analysis['mechanism_tasks'],
        *analysis['scope_tasks'].values()) for scene, _ in tasks]+[row['seed'] for row in analysis['vrx_tasks']]
    if (len(prospective) != len(set(prospective)) or set(prospective) & historical
            or any(values & (set(prospective) | historical) for values in training_scenes.values())
            or any(training_scenes[a] & training_scenes[b] for a in SEEDS for b in SEEDS if a < b)):
        raise ValueError('An intervention-training or prospective scene population overlaps.')
    archive(ROOT/'vrx/run_r20_evidence.py', PACKAGE/'runtime/frozen_vrx/vrx/run_r20_evidence.py',
            analysis['vrx_adapter_sha256'])
    pending, scope_rows, scope_episodes, scope_effects = [], [], [], []
    scope_complete = unsupported = 0
    for group, tasks in analysis['scope_tasks'].items():
        matched = {}
        for label in analysis['vrx_methods']:
            path = root/'scope'/group/(label+'.pth')
            if not path.exists() or not read_pth(path).get('completed'):
                pending.append(f'scope/{group}/{label}')
                continue
            data = read_pth(path)
            if data['analysis_sha256'] != identity or canonical(data['tasks']) != canonical(tasks):
                raise ValueError('A scope evaluation changed its frozen task design.')
            if data['policy_sha256'] != analysis['models'][label]['sha256']:
                raise ValueError('A scope evaluation changed its policy.')
            scope_complete += 1
            if not data['supported']:
                if tasks[0][0][1] == 6 or label not in ('F01', 'F11', 'mappo') or data['rows'] or data['evaluated']:
                    raise ValueError('Unexpected unsupported-fleet scope record.')
                unsupported += 1
                scope_rows.append(dict(group=group, method=label, supported=False,
                    episodes=0, planned_tasks=len(tasks), success=None, success_rate_percent=None,
                    reason=data['reason'], training_seed=42))
                continue
            data = checked_evaluation(path, tasks, analysis['models'][label]['sha256'])
            values = np.asarray([r['success'] for r in data['rows']])
            matched[label] = values
            interval = crossed_interval(values, [scene[2] for scene, _ in tasks])
            scope_rows.append(dict(group=group, method=label, supported=True, episodes=len(tasks),
                planned_tasks=len(tasks), success=int(values.sum()), success_rate_percent=100*float(values.mean()),
                ci95_low_percent=100*interval[0], ci95_high_percent=100*interval[1],
                training_seed=42, policy_sha256=data['policy_sha256'],
                collision=sum(r['collision'] for r in data['rows']),
                physical_breach=sum(r['physical_breach'] for r in data['rows']),
                capture=sum(r['capture'] for r in data['rows']), timeout=sum(r['timeout'] for r in data['rows'])))
            for index, row in enumerate(data['rows']):
                scope_episodes.append(dict(group=group, method=label, task_index=index,
                    task_json=canonical(tasks[index]), **plain(row)))
        comparisons = [('M1_alone', 'F10', 'F00'), ('M2_alone', 'F01', 'F00'),
                       ('M2_given_M1', 'F11', 'F10'), ('complete_vs_ARBoids', 'F11', 'arboids')]
        current = []
        for contrast, a, b in comparisons:
            if a in matched and b in matched:
                row = dict(group=group, contrast=contrast, candidate=a, reference=b,
                    **paired_statistics(matched[a], matched[b], [scene[2] for scene, _ in tasks]))
                scope_effects.append(row); current.append(row)
        holm_adjust(current, 'mcnemar_exact_p_two_sided')
    save_csv('scope_summary.csv', scope_rows)
    save_csv('scope_episodes.csv', scope_episodes)
    save_csv('scope_paired_effects.csv', scope_effects)

    mechanism_states, mechanism_plans, mechanism_outcomes, mechanism_summary, m1_rows = [], [], [], [], []
    mechanism_complete = 0
    for arm in ('F01', 'F11'):
        ordinary = 'ordinary_no_M1' if arm == 'F01' else 'ordinary_with_M1'
        records = []
        for index, task in enumerate(analysis['mechanism_tasks']):
            path = root/'mechanism'/arm/f'root-{index:03d}.pth'
            if not path.exists():
                pending.append(f'mechanism/{arm}/{index}')
                continue
            row = read_pth(path)
            if not row['completed'] or row['analysis_sha256'] != identity or canonical((row['scene'], row['future'])) != canonical(task):
                raise ValueError('A mechanism observation changed its root identity.')
            if row['index'] != index or row['arm'] != arm or len(row['futures']) != analysis['mechanism_futures']:
                raise ValueError('A mechanism observation changed its future population.')
            count = len(row['futures'])
            measured = {q: np.asarray([row['outcomes'][q, j]['success'] for j in range(count)], dtype=float)
                        for q in row['plans']}
            if any(not np.isin(v, [0, 1]).all() for v in measured.values()):
                raise ValueError('A mechanism outcome is not binary.')
            factual = measured[0]
            selected_outcomes = {}
            for label in (arm, ordinary):
                prediction = row['predictions'][label]
                choice = prediction['choice'] if prediction['accepted'] else 0
                selected = measured[choice]
                selected_outcomes[label] = selected
                means = {q: float(v.mean()) for q, v in measured.items()}
                scores = prediction['plans']
                agreements = []
                for left in row['plans']:
                    for right in row['plans']:
                        if left >= right or means[left] == means[right]:
                            continue
                        predicted = scores[left]['score']-scores[right]['score']
                        product = predicted*(means[left]-means[right])
                        agreements.append(1. if product > 0 else .5 if predicted == 0 else 0.)
                for q in row['plans']:
                    item = scores[q]
                    estimate = float(np.mean(item['members']))
                    if not math.isfinite(item['score']) or not 0 <= estimate <= 1:
                        raise ValueError('A mechanism prediction is not finite or probability bounded.')
                    mechanism_plans.append(dict(arm=arm, state_index=index, method=label,
                        plan_index=q, option=item['option'], decision_score=item['score'],
                        factual_centered_score=item['score']-scores[0]['score'],
                        predecision_potential_estimate=estimate,
                        empirical_success=means[q], empirical_gain=means[q]-float(factual.mean()),
                        selected=q == choice, futures=count))
                mechanism_states.append(dict(arm=arm, state_index=index, method=label,
                    task_json=canonical(task), root_step=row['root_elapsed'],
                    factual_success=float(factual.mean()), selected_success=float(selected.mean()),
                    selected_gain=float((selected-factual).mean()), accepted=prediction['accepted'],
                    measured_regret=max(means.values())-float(selected.mean()),
                    ranking_agreement=float(np.mean(agreements)) if agreements else None,
                    chosen_plan=choice, evaluated_plans=len(measured), futures=count))
            for q, values in measured.items():
                for j, value in enumerate(values):
                    mechanism_outcomes.append(dict(arm=arm, state_index=index, plan_index=q,
                        future_index=j, future_seed=row['futures'][j], success=int(value),
                        **{k:v for k,v in row['outcomes'][q, j].items() if k != 'success'}))
            if arm == 'F11':
                costs = {r['selection']: r for r in row['m1_candidate_objective']}
                if set(costs) != {'joint', 'independent'}:
                    raise ValueError('M1 lacks its same-state selector comparison.')
                m1_rows.append(dict(state_index=index, task_json=canonical(task),
                    joint_unary=costs['joint']['unary_cost'], independent_unary=costs['independent']['unary_cost'],
                    joint_pair_risk=costs['joint']['pair_risk'], independent_pair_risk=costs['independent']['pair_risk'],
                    joint_total=costs['joint']['total_cost'], independent_total=costs['independent']['total_cost'],
                    cost_reduction=costs['independent']['total_cost']-costs['joint']['total_cost']))
            records.append(dict(index=index, agility=task[0][2], factual=factual, selected=selected_outcomes))
            mechanism_complete += 1
        if not records:
            continue
        root_strata = [r['agility'] for r in records]
        for label in (arm, ordinary):
            difference = np.asarray([r['selected'][label]-r['factual'] for r in records])
            interval = nested_interval(difference, root_strata)
            rows = [r for r in mechanism_states if r['arm'] == arm and r['method'] == label]
            mechanism_summary.append(dict(arm=arm, method=label, states=len(records), complete=len(records) == 40,
                expected_gain_percentage_points=100*float(difference.mean()),
                ci95_low_percentage_points=100*interval[0], ci95_high_percentage_points=100*interval[1],
                acceptance_rate_percent=100*float(np.mean([r['accepted'] for r in rows])),
                mean_measured_regret=float(np.mean([r['measured_regret'] for r in rows])),
                empirical_positive_gain_states=int((difference.mean(axis=1)>0).sum()),
                empirical_negative_gain_states=int((difference.mean(axis=1)<0).sum()),
                mean_ranking_agreement=statistics.mean(r['ranking_agreement'] for r in rows if r['ranking_agreement'] is not None)
                    if any(r['ranking_agreement'] is not None for r in rows) else None,
                interval_method='stratified state bootstrap with within-state resampling of paired future draws'))
        difference = np.asarray([r['selected'][arm]-r['selected'][ordinary] for r in records])
        interval = nested_interval(difference, root_strata)
        mechanism_summary.append(dict(arm=arm, method='proposed-minus-ordinary', states=len(records), complete=len(records) == 40,
            expected_gain_percentage_points=100*float(difference.mean()),
            ci95_low_percentage_points=100*interval[0], ci95_high_percentage_points=100*interval[1],
            interval_method='stratified state bootstrap with within-state paired future resampling'))
    save_csv('mechanism_states.csv', mechanism_states)
    save_csv('mechanism_plans.csv', mechanism_plans)
    save_csv('mechanism_outcomes.csv', mechanism_outcomes)
    save_csv('mechanism_summary.csv', mechanism_summary)
    save_csv('M1_same_state_objective.csv', m1_rows)

    vrx_trials, vrx_summary, vrx_effects, transports = [], [], [], []
    selected_vrx = {}
    transport_identity = None
    initial = {}
    for index, task in enumerate(analysis['vrx_tasks']):
        for label in analysis['vrx_methods']:
            candidates = sorted((root/'vrx/paired').glob(f'case{index:03d}-{label}-attempt*/result.json'))
            valid = []
            for path in candidates:
                row = json.loads(path.read_text(encoding='utf8'))
                if (row['analysis_sha256'] != identity or row['paired_scene_index'] != index
                        or row['method'] != label or row['seed'] != task['seed']
                        or row['policy_sha256'] != analysis['models'][label]['sha256']
                        or row['deployment_adapter_sha256'] != analysis['vrx_adapter_sha256']
                        or row['agility'] != task['agility'] or row['action_period'] != task['action_period']
                        or row['duration_limit'] != task['duration'] or row['num_robots'] != task['defenders']+1):
                    raise ValueError('A VRX trial changed its task, policy or deployment identity.')
                if (row['r20_manifest_sha256'] != manifest_identity
                        or row['frozen_runtime_sha256'] != manifest['runtime_sha256']
                        or row['baseline_runtime_sha256'] != baseline_identity
                        or not row['source_unchanged'] or row['source']['commit'] != SOURCE_COMMIT):
                    raise ValueError('A VRX trial changed the frozen policy runtime or author release.')
                common = {'vrx/run_gate_contribution.py': row['transport_file_sha256']['run_gate_contribution.py'],
                    'vrx/run_gate_contribution_sync_start.py': row['startup_adapter_sha256'],
                    'vrx/run_experiment.py': row['shared_transport_sha256'],
                    'train/source_arboids.py': row['transport_file_sha256']['source_arboids.py']}
                if transport_identity is None:
                    transport_identity = common
                    for relative, expected_hash in common.items():
                        archive(ROOT/relative, PACKAGE/'runtime/frozen_vrx'/relative, expected_hash)
                elif transport_identity != common:
                    raise ValueError('VRX methods did not use the same transport implementation.')
                if row['passed']:
                    valid.append((path, row))
                else:
                    transports.append(dict(case=index, method=label, attempt=row['attempt'], result=str(path.relative_to(root))))
            if len(valid) > 1:
                raise ValueError('A valid VRX task was retried after observing its task outcome.')
            if not valid:
                pending.append(f'vrx/{index}/{label}')
                continue
            path, row = valid[0]
            if index in initial and initial[index] != row['initial_poses']:
                raise ValueError('VRX methods did not share identical initial poses.')
            initial[index] = row['initial_poses']
            trajectory_path = path.parent/'trajectory.npz'
            if not trajectory_path.exists():
                pending.append(f'vrx/{index}/{label}/trajectory')
                continue
            with np.load(trajectory_path) as trajectory:
                policy_time = trajectory['PolicyWallSeconds'].astype(float)
                control_time = trajectory['ControlWallSeconds'].astype(float)
                feedback = trajectory['FeedbackSkewSeconds'].astype(float)
                if len(policy_time) != row['control_steps'] or not np.isfinite(policy_time).all():
                    raise ValueError('A VRX trajectory does not match its measured control steps.')
            selected_vrx[index, label] = dict(row=row, inference=policy_time, control=control_time, feedback=feedback)
            vrx_trials.append(dict(case=index, method=label, agility=task['agility'], seed=task['seed'],
                success=int(row['success']), outcome=row['outcome'], outcome_code=row['outcome_code'],
                duration_seconds=row['simulation_seconds'], wall_seconds=row['wall_seconds'],
                control_steps=row['control_steps'], control_deadline_misses=row['control_deadline_misses'],
                inference_mean_seconds=float(policy_time.mean()), inference_p95_seconds=float(np.quantile(policy_time, .95)),
                control_p95_seconds=float(np.quantile(control_time, .95)),
                maximum_feedback_skew_seconds=float(feedback.max()), attempt=row['attempt'],
                policy_sha256=row['policy_sha256'], result=str(path.relative_to(root))))
    for label in analysis['vrx_methods']:
        trials = [r for r in vrx_trials if r['method'] == label]
        if not trials:
            continue
        flags = [r['success'] for r in trials]
        interval = crossed_interval(flags, [r['agility'] for r in trials])
        times = np.concatenate([x['inference'] for (i, m), x in selected_vrx.items() if m == label])
        control = np.concatenate([x['control'] for (i, m), x in selected_vrx.items() if m == label])
        commands = sum(r['control_steps'] for r in trials)
        misses = sum(r['control_deadline_misses'] for r in trials)
        vrx_summary.append(dict(method=label, trials=len(trials), complete=len(trials) == 100,
            success=sum(flags), success_rate_percent=100*statistics.mean(flags),
            ci95_low_percent=100*interval[0], ci95_high_percent=100*interval[1],
            capture=sum(r['outcome_code'] == 3 for r in trials), timeout=sum(r['outcome_code'] == 4 for r in trials),
            collision=sum(r['outcome_code'] == 2 for r in trials), breach=sum(r['outcome_code'] == 1 for r in trials),
            control_steps=commands, control_deadline_misses=misses,
            control_deadline_miss_percent=100*misses/commands,
            inference_median_seconds=float(np.median(times)), inference_p95_seconds=float(np.quantile(times, .95)),
            inference_maximum_seconds=float(times.max()), control_p95_seconds=float(np.quantile(control, .95)),
            model_role='one frozen development model or fixed seed-42 original baseline; prospective real Gazebo tasks'))
    methods = list(analysis['vrx_methods'])
    for i, candidate in enumerate(methods):
        for reference in methods[:i]:
            if not all((index, label) in selected_vrx for index in range(100) for label in (candidate, reference)):
                continue
            left = [int(selected_vrx[index, candidate]['row']['success']) for index in range(100)]
            right = [int(selected_vrx[index, reference]['row']['success']) for index in range(100)]
            vrx_effects.append(dict(candidate=candidate, reference=reference,
                **paired_statistics(left, right, [t['agility'] for t in analysis['vrx_tasks']])))
    holm_adjust(vrx_effects, 'mcnemar_exact_p_two_sided')
    save_csv('vrx_trials.csv', vrx_trials)
    save_csv('vrx_summary.csv', vrx_summary)
    save_csv('vrx_paired_effects.csv', vrx_effects)
    save_json(PACKAGE/'metadata/vrx_transport_attempts.json', dict(invalid_attempts=transports,
        valid_task_failures_are_retained=True, preflight_trials_excluded=True,
        verified_common_transport_files=transport_identity,
        policy_inference='same local Unix socket adapter and public measurements for every method'))

    traces_complete = False
    selection_path = root/'traces/selection.pth'
    if selection_path.exists():
        selection = read_pth(selection_path)
        if selection['manifest_sha256'] != digest(root/'manifest.pth'):
            raise ValueError('Illustrative cases have another test population.')
        traces_complete = True
        example_metrics = []
        for example in selection['examples']:
            index = example['task_index']
            if index is None:
                example_metrics.append(dict(example))
                continue
            paired_traces = {}
            for label in (example['candidate'], example['reference']):
                path = root/'traces'/f'{label}-task{index:03d}.pth'
                if not path.exists():
                    pending.append(str(path.relative_to(root))); traces_complete = False
                    continue
                trace = read_pth(path)
                expected = read_pth(root/'fixed'/(label+'-heldout.pth'))['rows'][index]
                if (not trace['completed'] or not trace['exact_endpoint_replay'] or trace['row'] != expected
                        or trace['policy_sha256'] != manifest['models'][label]['sha256']):
                    raise ValueError('An illustrative trajectory changed the actual saved endpoint.')
                paired_traces[label] = trace
            metrics = dict(example)
            if len(paired_traces) == 2:
                candidate, reference = (paired_traces[example[key]] for key in ('candidate','reference'))
                n = min(len(candidate['trajectory']['commands']), len(reference['trajectory']['commands']))
                difference = np.max(np.abs(candidate['trajectory']['commands'][:n]-reference['trajectory']['commands'][:n]),axis=(1,2))
                first = np.flatnonzero(difference > 1e-8)
                metrics['first_command_difference_seconds'] = float(candidate['trajectory']['time'][first[0]]) if len(first) else None
                for label, trace in (('candidate',candidate),('reference',reference)):
                    metrics[label+'_minimum_defender_distance_metres'] = float(trace['trajectory']['minimum_defender_distance'].min())
                    metrics[label+'_minimum_attacker_distance_metres'] = float(trace['trajectory']['minimum_attacker_distance'].min())
                    metrics[label+'_terminal_time_seconds'] = trace['row']['duration']
                    metrics[label+'_outcome'] = trace['row']['outcome']
            example_metrics.append(metrics)
        save_csv('trajectory_examples.csv', example_metrics)
    else:
        pending.append('trajectory selection')
    status = dict(completed=not pending, pending=pending,
        scope_configurations_completed=scope_complete, scope_configurations_required=60,
        scope_unsupported_configurations=unsupported, scope_valid_episodes=len(scope_episodes),
        independent_mechanism_states=mechanism_complete, required_mechanism_states=80,
        mechanism_future_measurements=len(mechanism_outcomes), vrx_valid_trials=len(vrx_trials),
        vrx_required_trials=600, vrx_invalid_transport_attempts=len(transports),
        exact_illustrative_trajectories_complete=traces_complete, analysis_sha256=identity)
    save_json(PACKAGE/'metadata/diagnostic_protocol.json', dict(
        analysis_sha256=identity, scope_tasks=analysis['scope_tasks'], mechanism_tasks=analysis['mechanism_tasks'],
        intervention_training_and_prospective_scenes_disjoint=True,
        intervention_training_scenes_per_seed={seed:len(values) for seed, values in training_scenes.items()},
        distinct_prospective_scene_identifiers=len(prospective),
        mechanism_root_rule=analysis['mechanism_root_rule'], mechanism_plan_rule=analysis['mechanism_plan_rule'],
        mechanism_continuation=analysis['mechanism_continuation'], mechanism_futures=analysis['mechanism_futures'],
        vrx_tasks=analysis['vrx_tasks'], models=analysis['models'],
        common_learned_opponent=analysis['learned_opponent'],
        cross_fleet_policy=analysis['cross_fleet_policy'], baseline_seed_selection=analysis['baseline_seed_selection']))
    reductions = [row['cost_reduction'] for row in m1_rows]
    mean_reduction_text = f'平均降低 {statistics.mean(reductions):.4f} 个目标单位。' if reductions else ''
    lines = ['', '## 机制、适用范围与 VRX', '',
        f'独立机制状态：{mechanism_complete}/80；每个状态使用 8 个共同的新未来扰动，测量有限干预后回到同一前驱策略的任务结局。候选是两种估计器前三名与实际选择的并集，后悔值仅相对这一实测子集。',
        'M1 同状态对照记录共同候选与风险矩阵下的单船成本、成对风险和联合目标。M2 记录预测排序、实际选中干预的实测收益及损害；这些诊断不参与拟合。',
        'M1 的冻结目标为单艇偏好成本之和加 32 倍成对风险之和。独立选择器根据其他艇的边缘候选概率逐艇决策；联合选择器搜索完整艇队配置，并按实际组合的成对风险修正选择。两者使用相同候选和预测，因此直接对照检验的是协调选择的作用。',
        f"M1 联合选择在 {sum(v>1e-8 for v in reductions)}/{len(reductions)} 个状态下降低目标成本，"
        f"{sum(abs(v)<=1e-8 for v in reductions)} 个持平，{sum(v < -1e-8 for v in reductions)} 个升高；"
        f'{mean_reduction_text}任务成功收益由独立终点配对对照另行检验。', '',
        '| 前驱状态分布 | 估计器 / 比较 | 选中干预平均收益（百分点） | 95% 区间 | 非平局计划排序一致率（%） |',
        '|---|---|---:|---:|---:|']
    for row in mechanism_summary:
        rank = row.get('mean_ranking_agreement')
        lines.append(f"| {row['arm']} | {row['method']} | {row['expected_gain_percentage_points']:+.2f} | "
            f"[{row['ci95_low_percentage_points']:+.2f}, {row['ci95_high_percentage_points']:+.2f}] | "
            +(f'{100*rank:.2f}' if rank is not None else '—')+' |')
    lines += ['', '单项估计器的收益相对同一事实延续；proposed-minus-ordinary 为两估计器选中干预之差。排序一致率先在各状态的实测候选非平局对上计算，再按状态平均。',
        f'适用范围：{scope_complete}/60 个配置已完成，其中 {unsupported} 个配置明确不受当前固定表示支持。F01、F11 的当前 M2 表示和 MAPPO 均固定为六艇，跨艇数结果不以零成功率代替“不支持”。',
        '下表为固定 r20 模型及预先指定的种子 42 基线；灵活度和艇数变化各使用 100 个共同任务，学习对手使用 200 个共同任务。', '',
        '| 条件 | F00 | F10 | F01 | F11 | ARBoids | MAPPO |', '|---|---:|---:|---:|---:|---:|---:|']
    for group in analysis['scope_tasks']:
        values = {row['method']:row for row in scope_rows if row['group'] == group}
        cells = []
        for method in ('F00','F10','F01','F11','arboids','mappo'):
            row = values.get(method)
            cells.append('待完成' if row is None else '不支持' if not row['supported'] else f"{row['success_rate_percent']:.1f}%")
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines += ['', '学习对手是另一开发防御器上训练并冻结的共同对手，所有方法面对同一对手；这里没有进行针对 r20 的对手适应训练。', '',
        '| VRX 方法 | 有效试验 | 成功数 | 控制超时比例（%） | 推理 P95（ms） |', '|---|---:|---:|---:|---:|']
    for row in vrx_summary:
        lines.append(f"| {row['method']} | {row['trials']} | {row['success']} | {row['control_deadline_miss_percent']:.3f} | {1000*row['inference_p95_seconds']:.2f} |")
    lines += ['', 'VRX 使用 100 个共同场景、6 种方法，共 600 次真实 Gazebo 试验；同场景初始位置逐项核对，预检试验不混入统计。所有有效任务失败保留，时延来自逐控制步记录。数值仿真和 VRX 分表呈现。',
        '轨迹图按冻结任务顺序选择每项比较的第一个救回与第一个损害，只用于机制解释；每条回放的完整终点记录与原始评价严格相同。', '']
    return status, lines


def independent_figures(root, summary, by_seed, effects, diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    output = PACKAGE/'figures'
    output.mkdir(exist_ok=True)
    colors = {'F00':'#7c8790', 'F10':'#227a72', 'F01':'#a97832', 'F11':'#2857a3',
        'ordinary_no_M1':'#bd986a', 'ordinary_with_M1':'#8b68a0', 'M1_independent':'#83aaa5',
        'arboids':'#c35644', 'mappo':'#775d94', 'maac':'#a78ab5', 'rp':'#a9af75',
        'sac':'#c8ae75', 'boids':'#a7b1b6'}
    labels = {'ordinary_no_M1':'Ordinary\nwithout M1', 'ordinary_with_M1':'Ordinary\nwith M1',
        'M1_independent':'Independent\nM1 selection', 'arboids':'ARBoids', 'mappo':'MAPPO',
        'maac':'MAAC', 'sac':'SAC', 'rp':'RP', 'boids':'Boids'}

    def finish(figure, name):
        for extension in ('pdf', 'png'):
            figure.savefig(output/(name+'.'+extension), dpi=250, bbox_inches='tight')
        plt.close(figure)

    def table(name):
        with (PACKAGE/'tables'/name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def bars(axis, rows, order):
        mapping = {r['method']: r for r in rows}
        for i, method in enumerate(order):
            if method not in mapping:
                continue
            row = mapping[method]
            rate = float(row['mean_success_rate_percent'])
            lo, hi = float(row['ci95_low_percent']), float(row['ci95_high_percent'])
            axis.bar(i, rate, width=.65, color=colors[method])
            axis.errorbar(i, rate, yerr=[[max(0., rate-lo)], [max(0., hi-rate)]],
                          fmt='none', color='#263442', capsize=3, linewidth=1)
            axis.text(i, min(103., hi+1.5), f'{rate:.1f}', ha='center', fontsize=8)
        axis.set(xticks=np.arange(len(order)), xticklabels=[labels.get(x, x) for x in order],
                 ylabel='Success (%)', ylim=(0, 108))

    fixed = [r for r in summary if r['category'] == 'fixed-r20-reference' and r['suite'] == 'all-400']
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    bars(ax, fixed, ('F00','F10','F01','F11','ordinary_no_M1','ordinary_with_M1','M1_independent'))
    ax.set_title('Frozen r20 reference on 400 unseen paired tasks; 95% conditional task intervals', fontsize=9)
    finish(fig, 'unseen_fixed_r20_controls')
    proposed = [r for r in summary if r['category'] == 'independent-pipeline' and r['suite'] == 'all-400' and r['complete']]
    if len(proposed) == 7:
        original = [r for r in summary if r['category'] == 'original-baseline' and r['suite'] == 'all-400' and r['complete']]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        bars(axes[0], original+[r for r in proposed if r['method'] == 'F11'], ('boids','sac','rp','arboids','mappo','maac','F11'))
        bars(axes[1], proposed, ('F00','F10','F01','F11','ordinary_no_M1','ordinary_with_M1','M1_independent'))
        for axis in axes:
            axis.tick_params(axis='x', labelsize=8)
        axes[0].set_title('Original baselines and the complete repeated method')
        axes[1].set_title('Matched four configurations and mechanism replacements')
        fig.suptitle('Five independent training pipelines; 400 common unseen tasks; crossed seed/task 95% intervals', fontsize=10)
        finish(fig, 'unseen_five_seed_comparison')
    contrast_labels = {'M1_alone':'F10 − F00', 'M2_alone':'F01 − F00', 'M1_given_M2':'F11 − F01',
        'M2_given_M1':'F11 − F10', 'M2_vs_ordinary_without_M1':'F01 − ordinary without M1',
        'M2_vs_ordinary_with_M1':'F11 − ordinary with M1', 'M1_joint_vs_independent':'Joint − independent M1',
        'factorial_interaction':'Factorial interaction'}
    categories = ['fixed-r20-reference']+(['independent-pipeline'] if proposed and len(proposed) == 7 else [])
    fig, axes = plt.subplots(1, len(categories), figsize=(6.4*len(categories), 4.4), layout='constrained', squeeze=False)
    for axis, category in zip(axes[0], categories):
        rows = [r for r in effects if r['category'] == category and r['contrast'] in contrast_labels]
        for i, row in enumerate(rows):
            x, lo, hi = (row[k] for k in ('gain_percentage_points','ci95_low_percentage_points','ci95_high_percentage_points'))
            color = '#227a72' if x >= 0 else '#bd5845'
            axis.errorbar(x, i, xerr=[[max(0., x-lo)], [max(0., hi-x)]], fmt='o', color=color, capsize=3)
        axis.axvline(0, color='#8d969c', linewidth=.8)
        axis.set(yticks=range(len(rows)), yticklabels=[contrast_labels[r['contrast']] for r in rows],
                 xlabel='Paired success difference (percentage points)', title=category)
        axis.invert_yaxis()
    finish(fig, 'unseen_paired_mechanism_effects')

    mechanisms = table('mechanism_summary.csv')
    m1 = table('M1_same_state_objective.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), layout='constrained')
    for axis, arm in zip(axes[:2], ('F01','F11')):
        rows = [r for r in mechanisms if r['arm'] == arm and r['method'] != 'proposed-minus-ordinary']
        for i, row in enumerate(rows):
            x, lo, hi = (float(row[k]) for k in ('expected_gain_percentage_points','ci95_low_percentage_points','ci95_high_percentage_points'))
            axis.errorbar(i, x, yerr=[[max(0.,x-lo)],[max(0.,hi-x)]], fmt='o', capsize=5, color=colors[row['method']])
        axis.axhline(0, color='#8d969c', linewidth=.8)
        axis.set(xticks=range(len(rows)), xticklabels=['M2', 'Ordinary return'][:len(rows)],
                 ylabel='Measured gain over factual continuation (pp)', title=arm+' independent states')
    if m1:
        x = np.asarray([float(r['independent_total']) for r in m1]); y = np.asarray([float(r['joint_total']) for r in m1])
        maximum = max(float(x.max()), float(y.max()), 1.)*1.05
        axes[2].scatter(x, y, color=colors['F10'], alpha=.7, s=22)
        axes[2].plot([0,maximum],[0,maximum], ':', color='#8d969c')
        axes[2].set(xlabel='Independent selection cost', ylabel='Joint selection cost',
                    title='M1: identical state, candidates and objective')
    fig.suptitle('Mechanism measurements: 40 states per arm; 8 shared futures per state; no fitting', fontsize=10)
    finish(fig, 'independent_mechanism_measurements')

    measured_plans = table('mechanism_plans.csv')
    fig, axes = plt.subplots(1,4,figsize=(13,3.5),layout='constrained')
    for axis, (arm, method) in zip(axes, (('F01','F01'), ('F01','ordinary_no_M1'),
                                         ('F11','F11'), ('F11','ordinary_with_M1'))):
        rows = [row for row in measured_plans if row['arm'] == arm and row['method'] == method]
        axis.scatter([float(row['factual_centered_score']) for row in rows],
            [100*float(row['empirical_gain']) for row in rows], s=12, alpha=.35,
            color=colors[method], edgecolors='none')
        axis.axhline(0,color='#8d969c',linewidth=.7)
        axis.axvline(0,color='#8d969c',linewidth=.7)
        axis.set(title=labels.get(method,method).replace('\n',' '),
            xlabel='Decision score relative to factual', ylabel='Measured success gain (pp)',ylim=(-105,105))
        agreement=next(float(row['mean_ranking_agreement']) for row in mechanisms if row['arm']==arm and row['method']==method)
        axis.text(.03,.03,f'State-mean ranking agreement: {100*agreement:.1f}%',
            transform=axis.transAxes,fontsize=8)
    fig.suptitle('Predictions before measurement; shared selected-plan subset and 8 common futures; points share states',fontsize=10)
    finish(fig,'independent_value_prediction_vs_measurement')

    scopes = table('scope_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for label in ('F00','F01','F10','F11','arboids','mappo'):
        rows = sorted((r for r in scopes if r['method'] == label and r['group'].startswith('agility-')
                       and r['supported'] == 'True'), key=lambda r: float(r['group'].split('-')[-1]))
        if rows:
            axes[0].plot([float(r['group'].split('-')[-1]) for r in rows], [float(r['success_rate_percent']) for r in rows],
                'o-', label=labels.get(label,label), color=colors[label], linewidth=1.5, markersize=4)
        rows = sorted((r for r in scopes if r['method'] == label and r['group'].startswith('fleet-')
                       and r['supported'] == 'True'), key=lambda r: int(r['group'].split('-')[-1]))
        if rows:
            axes[1].plot([int(r['group'].split('-')[-1]) for r in rows], [float(r['success_rate_percent']) for r in rows],
                'o-', label=labels.get(label,label), color=colors[label], linewidth=1.5, markersize=4)
    axes[0].set(xlabel='Attacker agility', ylabel='Success (%)', ylim=(0,105), title='Six-vessel agility stress; 100 tasks per point')
    axes[1].set(xlabel='Number of defenders', ylabel='Success (%)', ylim=(0,105), xticks=[3,4,5,7,8],
                title='Fleet transfer at agility 2.25; 100 tasks per point')
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(.02,.02,'F01, F11 and MAPPO: fixed six-vessel representation', transform=axes[1].transAxes, fontsize=8)
    finish(fig, 'prospective_operating_range')
    learned = [r for r in scopes if r['group'] == 'common-learned-opponent' and r['supported'] == 'True']
    if learned:
        fig, ax = plt.subplots(figsize=(7.5,3.8), layout='constrained')
        bars(ax, [dict(r, mean_success_rate_percent=r['success_rate_percent']) for r in learned], ('F00','F10','F01','F11','arboids','mappo'))
        ax.set_title('One common frozen learned opponent; 200 paired tasks; no r20-specific adaptation', fontsize=9)
        finish(fig, 'common_learned_opponent')
    vrx = table('vrx_summary.csv')
    if vrx:
        fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
        bars(axes[0], [dict(r,mean_success_rate_percent=r['success_rate_percent']) for r in vrx], ('F00','F10','F01','F11','arboids','mappo'))
        axes[0].set_title('100 shared Gazebo scenes per method')
        for label in ('F00','F10','F01','F11','arboids','mappo'):
            values = []
            for path in (root/'vrx/paired').glob(f'case*-{label}-attempt*/trajectory.npz'):
                result = json.loads((path.parent/'result.json').read_text(encoding='utf8'))
                if not result['passed']:
                    continue
                with np.load(path) as trajectory:
                    values.extend(trajectory['PolicyWallSeconds'].tolist())
            if values:
                ordered = np.sort(np.asarray(values))*1000
                axes[1].plot(ordered, np.arange(1,len(ordered)+1)/len(ordered), color=colors[label],label=labels.get(label,label))
        axes[1].axvline(200,color='#bd5845',linestyle=':',linewidth=1)
        axes[1].set(xscale='log',xlabel='Measured policy inference (ms)',ylabel='Empirical cumulative probability',
                    title='All native control-step measurements')
        axes[1].legend(frameon=False,ncol=2,fontsize=8)
        finish(fig,'vrx_paired_success_and_latency')

    if diagnostics['exact_illustrative_trajectories_complete']:
        examples = read_pth(root/'traces/selection.pth')['examples']
        outcome_names = {1:'source failure',2:'collision',3:'capture',4:'timeout success'}
        for contrast in dict.fromkeys(r['contrast'] for r in examples):
            rows = [r for r in examples if r['contrast'] == contrast]
            fig, axes = plt.subplots(2,2,figsize=(9,8.6),layout='constrained')
            for i, example in enumerate(rows):
                index = example['task_index']
                paired = {} if index is None else {label: read_pth(root/'traces'/f'{label}-task{index:03d}.pth')
                    for label in (example['candidate'], example['reference'])}
                points = np.concatenate([np.asarray([[-15.,-15.],[15.,15.]])]+
                    [data['trajectory']['positions'].reshape(-1,2) for data in paired.values()])
                lower, upper = points.min(axis=0), points.max(axis=0)
                center = (lower+upper)/2
                half_span = max(35.,float((upper-lower).max())/2+8.)
                for j, label in enumerate((example['candidate'],example['reference'])):
                    axis = axes[i,j]
                    if index is None:
                        axis.text(.5,.5,'No '+example['direction']+' tasks',transform=axis.transAxes,ha='center')
                        axis.set_axis_off();continue
                    data = paired[label]
                    positions = data['trajectory']['positions']
                    axis.add_patch(Circle((0,0),15,color='#e2e9ed',zorder=0))
                    for vessel in range(1,positions.shape[1]):
                        axis.plot(positions[:,vessel,0],positions[:,vessel,1],color=plt.cm.tab10(vessel-1),linewidth=1.25)
                        axis.scatter(*positions[0,vessel],s=18,facecolors='none',edgecolors=plt.cm.tab10(vessel-1))
                        axis.scatter(*positions[-1,vessel],s=18,color=plt.cm.tab10(vessel-1))
                    axis.plot(positions[:,0,0],positions[:,0,1],color='#263442',linewidth=2,label='Attacker')
                    axis.scatter(*positions[0,0],s=30,marker='x',color='#263442')
                    axis.scatter(*positions[-1,0],s=25,color='#263442')
                    axis.set(xlim=(center[0]-half_span,center[0]+half_span),ylim=(center[1]-half_span,center[1]+half_span),
                        aspect='equal',xlabel='x (m)',ylabel='y (m)',
                        title=f"{label}: {outcome_names[data['row']['outcome']]} at {data['row']['duration']:.1f} s\n{example['direction']}; paired task {index}")
                    axis.grid(alpha=.15)
            fig.suptitle(contrast_labels.get(contrast,contrast)+'; first rescue and first harm in frozen task order\n'
                'Dark line: attacker; colors: defenders; open / cross: start; filled: final; shaded: target region',fontsize=9)
            finish(fig,'trajectory_'+contrast)

        examples = [row for row in examples if row['contrast']=='M2_given_M1' and row['task_index'] is not None]
        if len(examples) == 2:
            fig, axes = plt.subplots(2,2,figsize=(9.4,6.2),layout='constrained')
            for column, example in enumerate(examples):
                for method in (example['candidate'],example['reference']):
                    trace = read_pth(root/'traces'/f"{method}-task{example['task_index']:03d}.pth")['trajectory']
                    for axis, measure in zip(axes[:,column],('minimum_defender_distance','minimum_attacker_distance')):
                        axis.plot(trace['time'],trace[measure],color=colors[method],label=method,linewidth=1.5)
                        axis.scatter(trace['time'][-1],trace[measure][-1],color=colors[method],s=18)
                        axis.axhline(5.,color='#ac5449',linewidth=.6,linestyle=':')
                        axis.set(xlabel='Time (s)',ylabel='Minimum pair distance (m)' if measure=='minimum_defender_distance' else 'Nearest attacker distance (m)')
                axes[0,column].set_title(f"{example['direction'].capitalize()}: paired task {example['task_index']}")
                axes[0,column].legend(frameon=False)
            fig.suptitle('Exact F11 / F10 replay; dotted lines: original numerical collision / capture thresholds',fontsize=10)
            finish(fig,'M2_rescue_and_harm_clearance')

    objectives = table('independent_training_objectives.csv')
    fitted = {(int(row['training_seed']), row['method']) for row in objectives}
    fit_methods = ('F01','ordinary_no_M1','F11','ordinary_with_M1')
    if all((seed, method) in fitted for seed in SEEDS for method in fit_methods):
        fig, axes = plt.subplots(1,4,figsize=(13,3.4),layout='constrained')
        for axis, method in zip(axes,fit_methods):
            per_seed = []
            for seed in SEEDS:
                rows = [row for row in objectives if row['method'] == method and int(row['training_seed']) == seed]
                x = np.asarray(sorted({int(row['member_update']) for row in rows}))
                y = np.asarray([np.mean([float(row['objective']) for row in rows if int(row['member_update']) == update]) for update in x])
                per_seed.append(y)
                axis.plot(x/1000,y,color=colors[method],alpha=.25,linewidth=.8)
            axis.plot(x/1000,np.mean(per_seed,axis=0),color=colors[method],linewidth=2)
            axis.set(title=labels.get(method,method).replace('\n',' '),xlabel='Updates per member (thousand)',
                ylabel='Within-method training objective')
        fig.suptitle('Fresh value fitting: five seeds, three members, 24,000 total updates; objective scales differ',fontsize=10)
        finish(fig,'independent_value_training_objectives')

    fig, axes = plt.subplots(1, 5, figsize=(14.5,3.3),layout='constrained')
    for axis, family in zip(axes,FAMILIES):
        curves = []
        for seed in SEEDS:
            directory = PACKAGE/'raw/baselines'/compact_path(baseline_directory(family,seed))
            if family in ('sac','rp','arboids'):
                with (directory/'metrics1.csv').open(encoding='utf8',newline='') as stream:
                    rows = list(csv.DictReader(stream))
                x=np.asarray([int(r['step']) for r in rows]);y=np.asarray([float(r['def_sr']) for r in rows])
            else:
                rows=read_pth(directory/'training-result.pth')['validation_history']
                x=np.asarray([r['steps'] for r in rows]);y=np.asarray([r['success_rate'] for r in rows])
            curves.append((x,y))
            axis.plot(x/1e6,100*y,color=colors[family],alpha=.22,linewidth=.7)
        start=max(x[0] for x,y in curves);end=min(x[-1] for x,y in curves)
        grid=np.linspace(start,end,100)
        mean=np.mean([np.interp(grid,x,y) for x,y in curves],axis=0)
        axis.plot(grid/1e6,100*mean,color=colors[family],linewidth=1.8)
        axis.set(title=labels[family],xlabel='Environment steps (million)',ylim=(0,105))
    axes[0].set_ylabel('Training / development success (%)')
    fig.suptitle('Completed original-baseline training; faint lines = five seeds; no heldout task feedback',fontsize=10)
    finish(fig,'completed_baseline_training_curves')


def figures(summary, by_seed, ablation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    out = PACKAGE / 'figures'
    out.mkdir(exist_ok=True)
    labels = {'sac': 'SAC', 'rp': 'RP', 'arboids': 'ARBoids', 'mappo': 'MAPPO', 'maac': 'MAAC', 'boids': 'Boids', 'r20': 'r20 (fixed)'}
    order = ('boids', 'sac', 'rp', 'arboids', 'mappo', 'maac', 'r20')
    fig, ax = plt.subplots(figsize=(8.2, 3.9), layout='constrained')
    totals = {r['method']: r for r in summary if r['suite'] == 'all-856'}
    for i, method in enumerate(order):
        row = totals.get(method)
        if not row:
            continue
        values = [r['success_rate_percent'] for r in by_seed if r['method'] == method and r['suite'] == 'all-856']
        ax.bar(i, row['mean_success_rate_percent'], color='#1d6b70' if method == 'r20' else '#b7cbd1', width=.65)
        if row['training_repetitions'] > 1:
            ax.errorbar(i, row['mean_success_rate_percent'], yerr=row['sample_sd_percentage_points'], color='#183c48', capsize=4, linewidth=1.1)
            ax.scatter(np.linspace(i-.16, i+.16, len(values)), values, s=15, color='#274b5b', zorder=3)
    ax.set(xticks=range(len(order)), xticklabels=[labels[x] for x in order], ylabel='Success (%)', ylim=(0, 103))
    ax.set_title('856 matched development tasks; points = training seeds, error bars = sample SD', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('baseline_five_seeds.' + extension), dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.9), layout='constrained')
    for offset, suite in enumerate(SUITES):
        vals = [next(r['success_rate_percent'] for r in ablation if r['configuration'] == c and r['suite'] == suite) for c in CONFIGS]
        ax.bar(np.arange(6) + (offset-1.5)*.19, vals, width=.19, label=suite)
    ax.set(xticks=range(6), xticklabels=['F00', 'F10', 'F01', 'F11 / r20', 'Ordinary\nwithout M1', 'Ordinary\nwith M1'], ylabel='Success (%)', ylim=(0, 105))
    ax.legend(ncol=2, frameon=False, fontsize=8, loc='upper left')
    ax.set_title('Original indexed r20 ablations and matched ordinary-return controls', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('r20_ablation.' + extension), dpi=250)
    plt.close(fig)


def build(require_complete=False):
    PACKAGE.mkdir(parents=True, exist_ok=True)
    node = json.loads((ROOT / NODE).read_text(encoding='utf-8'))
    archive(ROOT / NODE, PACKAGE / 'metadata' / NODE.name)
    unavailable, artifact_records = [], []
    for name, info in node['artifacts'].items():
        source = ROOT / info['relative_path']
        target = PACKAGE / 'raw/r20' / compact_path(info['relative_path'])
        if name.startswith('code:'):
            if not source.exists() or digest(source) != info['sha256']:
                unavailable.append({'artifact': name, 'expected_sha256': info['sha256'],
                                    'reason': 'marking-time source has changed; no exact r20 training source was indexed'})
                continue
        if not source.exists() and target.exists():
            source = target
        archive(source, target, info['sha256'])
        artifact_records.append({'artifact': name, 'path': target.relative_to(PACKAGE).as_posix(), 'sha256': info['sha256']})
    raw_r20 = PACKAGE / 'raw/r20'
    tasks = {suite: read_pth(raw_r20 / compact_path(node['artifacts']['eval:F11:' + suite]['relative_path']))['tasks'] for suite in SUITES}
    if sum(len(v) for v in tasks.values()) != 856:
        raise ValueError('Unexpected task count')
    save_json(PACKAGE / 'metadata/tasks.json', {'identity': 'suite plus zero-based row index plus full saved task specification', 'suites': tasks})
    evaluations, episodes, ablation, effects = {}, [], [], []

    def add_episodes(category, method, seed, suite, data):
        for i, row in enumerate(data['rows']):
            episodes.append(dict(category=category, method=method, training_seed=seed, suite=suite,
                                 row_index=i, task_json=canonical(tasks[suite][i]), **plain(row)))

    for config in CONFIGS:
        sha = node['artifacts']['model:' + config]['sha256']
        counts = []
        for suite in SUITES:
            path = raw_r20 / compact_path(node['artifacts'][f'eval:{config}:{suite}']['relative_path'])
            d = checked_evaluation(path, tasks[suite], sha)
            evaluations[config, suite] = d
            counts.append(d['success'])
            ablation.append(dict(configuration=config, suite=suite, success=d['success'], episodes=len(tasks[suite]),
                                 success_rate_percent=100*d['success']/len(tasks[suite]), policy_sha256=sha))
            add_episodes('indexed-r20-ablation', config, None, suite, d)
        if counts != node['summary']['success_counts'][config]:
            raise ValueError('Indexed success counts differ: ' + config)
        ablation.append(dict(configuration=config, suite='all-856', success=sum(counts), episodes=856,
                             success_rate_percent=100*sum(counts)/856, policy_sha256=sha))
    contrasts = [('M1_alone', 'F10', 'F00'), ('M2_alone', 'F01', 'F00'), ('M1_given_M2', 'F11', 'F01'),
                 ('M2_given_M1', 'F11', 'F10'), ('M2_vs_ordinary_without_M1', 'F01', 'ordinary_no_M1'),
                 ('M2_vs_ordinary_with_M1', 'F11', 'ordinary_with_M1')]
    contributions = []
    for label, candidate, reference in contrasts:
        for suite in SUITES:
            a, b = evaluations[candidate, suite], evaluations[reference, suite]
            delta = a['success'] - b['success']
            contributions.append(dict(contrast=label, suite=suite, candidate=candidate, reference=reference,
                                      success_difference=delta, gain_percentage_points=100*delta/len(tasks[suite])))
            effects.append(dict(comparison=label, suite=suite, candidate=candidate, reference=reference, **paired(a['rows'], b['rows'])))
        total_delta = sum(evaluations[candidate, suite]['success'] - evaluations[reference, suite]['success'] for suite in SUITES)
        contributions.append(dict(contrast=label, suite='all-856', candidate=candidate, reference=reference,
                                  success_difference=total_delta, gain_percentage_points=100*total_delta/856))
    rank = []
    for old in node['rank_correction_4310']:
        suite = old['suite']
        artifact = node['artifacts'][old['evaluation']]
        model_key = 'comparison:model:rank_before' if old['stage'] == 'before' else 'model:F11'
        d = checked_evaluation(raw_r20 / compact_path(artifact['relative_path']), tasks[suite], node['artifacts'][model_key]['sha256'])
        p = paired(d['rows'], evaluations['F10', suite]['rows'])
        if p['rescued_indices'] != old['rescued_row_indices_zero_based'] or p['harmed_indices'] != old['harmed_row_indices_zero_based']:
            raise ValueError('Rank-correction paired identities differ')
        rank.append(dict(stage=old['stage'], suite=suite, success=d['success'], baseline_success=old['baseline_success'], **p))
        if old['stage'] == 'before':
            add_episodes('rank-correction-before', 'F11-r18', None, suite, d)
            before_rank_rows = d['rows']
    direct_rank = paired(evaluations['F11', 'new4310-200']['rows'], before_rank_rows)
    effects.append(dict(comparison='rank_correction_direct_before_after', suite='new4310-200',
                        candidate='F11-r20', reference='F11-r18', **direct_rank))

    by_seed, training, pending = [], [], []
    methods = [('boids', None)] + [(f, s) for f in FAMILIES for s in SEEDS]
    for family, seed in methods:
        relative = baseline_directory(family, seed)
        directory = PACKAGE / 'raw/baselines' / compact_path(relative)
        local = ROOT / relative
        if not all((directory / (suite + '.pth')).exists() for suite in SUITES) and local.exists():
            for path in local.iterdir():
                if path.is_file() and path.suffix in ('.pth', '.yaml', '.csv', '.json') and not path.name.startswith('last.'):
                    archive(path, directory / path.name)
        if not all((directory / (suite + '.pth')).exists() for suite in SUITES):
            pending.append(dict(method=family, seed=seed))
            continue
        train_record, policy_sha = None, None
        if family != 'boids':
            native = family in ('sac', 'rp', 'arboids')
            record_path = directory / ('source-repeat-result.pth' if native else 'training-result.pth')
            model_path = directory / ('best-development.pth' if native else 'best.pth')
            if not record_path.exists() or not model_path.exists():
                pending.append(dict(method=family, seed=seed, reason='training provenance not yet archived'))
                continue
            train_record = read_pth(record_path)
            if not train_record['completed'] or not train_record['fresh_initialization'] or train_record['seed'] != seed:
                raise ValueError('Training completion or seed identity differs: ' + str(record_path))
            if train_record.get('training_steps', train_record.get('steps')) != 1000000:
                raise ValueError('Training budget differs: ' + str(record_path))
            policy_sha = digest(model_path)
            expected_sha = train_record['selected_checkpoint_sha256'] if native else train_record['best_sha256']
            if policy_sha != expected_sha:
                raise ValueError('Selected model hash differs: ' + str(model_path))
            if native:
                import yaml
                if train_record['family'] != family or train_record['source_commit'] != SOURCE_COMMIT:
                    raise ValueError('Native family/source differs')
                for filename, key in [('adares1.pth', 'checkpoint_sha256'), ('config.yaml', 'configuration_sha256')]:
                    if digest(directory / filename) != train_record[key]:
                        raise ValueError('Native final checkpoint/config differs')
                config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
                expected = yaml.safe_load((ROOT / 'third_party/arboids_release/train/configs/train.yaml').read_text(encoding='utf-8'))
                expected['agent'].update(defender_num=6, adaptive=family == 'arboids', residual=family != 'sac')
                if 'seed' in config['training']:
                    expected['training']['seed'] = seed
                if config != expected:
                    raise ValueError('Native training recipe differs from the declared source protocol')
                with (directory / 'metrics1.csv').open(newline='', encoding='utf-8') as stream:
                    metrics = list(csv.DictReader(stream))
                if int(metrics[-1]['step']) != 1000000:
                    raise ValueError('Native training stream did not finish its budget')
                selected = max(metrics, key=lambda r: (float(r['def_sr']), float(r['reward'])))
                saved = train_record['development_selection']
                if int(selected['step']) != saved['step'] or any(not math.isclose(float(selected[k]), saved[k], abs_tol=1e-12, rel_tol=1e-12) for k in ('def_sr', 'reward')):
                    raise ValueError('Native selected checkpoint differs from the training-development maximum')
            else:
                best = read_pth(model_path)
                history = train_record['validation_history']
                selected = max(history, key=lambda h: (h['success'], h['mean_return']))
                if best['training_steps'] != selected['steps'] or canonical(best['selection']) != canonical(selected):
                    raise ValueError('Best checkpoint does not match the predeclared training-only selection')
                if best['seed'] != seed or best['algorithm'] != family or train_record['spec']['source_commit'] != SOURCE_COMMIT:
                    raise ValueError('Baseline family/source differs')
            training.append(dict(method=family, seed=seed, policy_sha256=policy_sha, record=plain(train_record)))
        total, total_rescues, total_harms = 0, 0, 0
        for suite in SUITES:
            data = checked_evaluation(directory / (suite + '.pth'), tasks[suite], policy_sha)
            total += data['success']
            p = paired(evaluations['F11', suite]['rows'], data['rows'])
            total_rescues += p['rescued']; total_harms += p['harmed']
            by_seed.append(dict(method=family, training_seed=seed, suite=suite, success=data['success'], episodes=len(tasks[suite]),
                                success_rate_percent=100*data['success']/len(tasks[suite]), r20_rescued=p['rescued'], r20_harmed=p['harmed'], policy_sha256=policy_sha))
            effects.append(dict(comparison='r20_vs_baseline', suite=suite, candidate='r20', reference=family, training_seed=seed, **p))
            add_episodes('external-baseline', family, seed, suite, data)
        by_seed.append(dict(method=family, training_seed=seed, suite='all-856', success=total, episodes=856,
                            success_rate_percent=100*total/856, r20_rescued=total_rescues, r20_harmed=total_harms, policy_sha256=policy_sha))
    for suite in (*SUITES, 'all-856'):
        row = next(r for r in ablation if r['configuration'] == 'F11' and r['suite'] == suite)
        by_seed.append(dict(method='r20', training_seed=None, suite=suite, success=row['success'], episodes=row['episodes'],
                            success_rate_percent=row['success_rate_percent'], policy_sha256=row['policy_sha256']))
    summary = []
    for method in ('boids', *FAMILIES, 'r20'):
        for suite in (*SUITES, 'all-856'):
            rows = [r for r in by_seed if r['method'] == method and r['suite'] == suite]
            if not rows:
                continue
            rates = [r['success_rate_percent'] for r in rows]
            learned = method in FAMILIES
            summary.append(dict(method=method, suite=suite, training_repetitions=len(rows) if learned else 0,
                                evaluation_repetitions=len(rows), required_training_repetitions=5 if learned else 0,
                                completed=len(rows) == 5 if learned else True,
                                mean_success_rate_percent=statistics.mean(rates),
                                sample_sd_percentage_points=statistics.stdev(rates) if len(rates) > 1 else None,
                                total_success=sum(r['success'] for r in rows), total_evaluations=sum(r['episodes'] for r in rows),
                                inference_scope='training-seed variation on fixed development tasks' if learned else ('fixed selected development node' if method == 'r20' else 'deterministic untrained controller')))
    upstream = json.loads((PACKAGE / 'metadata/retrieved_upstream.json').read_text(encoding='utf-8'))
    for row in upstream['artifacts']:
        if digest(PACKAGE / row['package_path']) != row['sha256']:
            raise ValueError('Retrieved upstream source changed')
    save_json(PACKAGE / 'metadata/training_records.json', training)
    final_checkpoints = []
    checkpoint_storage = json.loads((PACKAGE / 'metadata/checkpoint-storage.json').read_text(encoding='utf-8'))
    for run in training:
        family, seed, record = run['method'], run['seed'], run['record']
        directory = PACKAGE / 'raw/baselines' / compact_path(baseline_directory(family, seed))
        if family in ('sac', 'rp', 'arboids'):
            path, expected, kind = directory / 'adares1.pth', record['checkpoint_sha256'], 'final saved actor'
        elif family == 'maac' and seed == 42:
            path = PACKAGE / 'checkpoints/maac-seed42/last.pth.zst'
            storage = next(x for x in checkpoint_storage['records'] if 'maac-seed42/' in x['original_path'])
            if storage['original_sha256'] != record['last_sha256'] or not checkpoint_storage['verified_lossless']:
                raise ValueError('The compressed MAAC42 checkpoint has lost its original byte-identity mapping')
            expected, kind = storage['compressed_sha256'], 'losslessly compressed complete training state'
        else:
            path = directory / ('last.pth.zst' if family == 'maac' else 'last.pth')
            expected, kind = record['last_sha256'], 'complete training state'
        exists = path.is_file()
        if exists and digest(path) != expected:
            raise ValueError('Final checkpoint hash differs: ' + str(path))
        final_checkpoints.append(dict(method=family, seed=seed, path=path.relative_to(PACKAGE).as_posix(),
                                      kind=kind, exists=exists, stored_sha256=expected,
                                      original_recorded_sha256=record.get('last_sha256', record.get('checkpoint_sha256'))))
    save_json(PACKAGE / 'metadata/final_checkpoints.json', final_checkpoints)
    save_json(PACKAGE / 'metadata/indexed_artifacts.json', artifact_records)
    save_json(PACKAGE / 'metadata/provenance_boundaries.json', {
        'result_role': 'development', 'all_suites_development': True, 'layout4310_used_for_development_selection': True,
        'r20_is_fixed_selected_node_not_five_independent_pipeline_repetitions': True,
        'exact_r20_training_source_recovered': False, 'unavailable_indexed_code': unavailable,
        'training_data_container_flags': node['training_data_containers'],
        'upstream_files_retrieved': len(upstream['artifacts']),
        'upstream_without_historical_hash': sum(not x.get('historically_indexed_sha256') for x in upstream['artifacts']),
        'native_baseline_selection': 'original training-development success, then reward, first exact tie',
        'mappo_maac_selection': 'training-only 50-episode success, then return, first exact tie',
        'standard_deviation_unit': 'independent training seed; same fixed 856 endpoint tasks for every seed',
        'no_pooled_binomial_independence_assumption': True,
        'incomplete_or_interrupted_attempts_excluded_from_results': True})
    source_root = ROOT / 'third_party/arboids_release'
    source_manifest = json.loads((source_root / 'provenance.json').read_text(encoding='utf-8'))
    if source_manifest['commit'] != SOURCE_COMMIT:
        raise ValueError('Pinned source commit differs')
    for name, sha in source_manifest['files'].items():
        archive(source_root / name, PACKAGE / 'runtime/author_release' / name, sha)
    archive(source_root / 'provenance.json', PACKAGE / 'runtime/author_release/provenance.json')
    for filename in ('r20_baseline_comparison.py', 'source_arboids.py', 'mechanism_controls.py',
                     'develop_source_fusion.py', 'conditional_fusion_value.py', 'local_intervention_value.py',
                     'plain_return.py', 'run_remote_evidence.py', 'export_r20_paper_evidence.py',
                     'r20_independent_evidence.py', 'r20_diagnostic_evidence.py', 'study_runtime.py',
                     'train.py', 'utils/manager.py', 'utils/config.py', 'utils/protocol.py'):
        source = ROOT / 'train' / filename
        archive(source, PACKAGE / 'runtime/current_snapshot/train' / filename)
    runtime_versions = {digest(p): p.relative_to(PACKAGE).as_posix() for p in (PACKAGE / 'runtime').glob('r20_baseline_comparison-*.py')}
    referenced_versions = set()
    for run in training:
        record = run['record']
        if 'spec' in record:
            referenced_versions.add(record['spec']['runtime_sha256'])
        for event in record.get('resume_events', []):
            referenced_versions.update(event[k] for k in ('previous_runtime_sha256', 'runtime_sha256') if k in event)
    if referenced_versions - runtime_versions.keys():
        raise ValueError('A MAPPO/MAAC training runtime has not been archived with its exact hash')
    save_json(PACKAGE / 'metadata/baseline_runtime_versions.json', {
        'exact_archived_versions': runtime_versions, 'referenced_by_completed_training': sorted(referenced_versions),
        'current_snapshot_is_not_a_claim_of_exact_historical_r20_or_native_endpoint_source': True})
    save_csv('ablation.csv', ablation)
    save_csv('mechanism_contributions.csv', contributions)
    save_csv('paired_effects.csv', effects)
    save_csv('rank_correction_4310.csv', rank)
    save_csv('baselines_by_seed.csv', by_seed)
    save_csv('baselines_summary.csv', summary)
    save_csv('episodes.csv', episodes)
    figures(summary, by_seed, ablation)
    prospective, prospective_lines = build_independent()
    complete = not pending
    checkpoints_complete = all(x['exists'] for x in final_checkpoints) and len(final_checkpoints) == 25
    status = dict(node_id=node['node_id'], built_at_utc=datetime.now(timezone.utc).isoformat(),
                  indexed_node_verified=True, completed_baseline_training_runs=len(training), required_baseline_training_runs=25,
                  five_seed_baselines_complete=complete, pending=pending, episode_rows=len(episodes),
                  all_final_checkpoints_archived=checkpoints_complete,
                  development_archive_complete=complete and checkpoints_complete,
                  evidence_archive_complete=complete and checkpoints_complete and prospective['completed'],
                  independent_evidence=prospective,
                  distinct_development_tasks=856, distinct_unseen_endpoint_tasks=prospective.get('distinct_unseen_endpoint_tasks',0),
                  development_only=not (PACKAGE/'raw/independent/manifest.pth').exists(),
                  exact_r20_training_source_recovered=False,
                  exact_proposed_training_computation_replayed=prospective.get('exact_proposed_training_replay_passed',False))
    save_json(PACKAGE / 'status.json', status)
    lines = ['# r20 论文证据', '', f"固定节点：`{node['node_id']}`。", '',
             '完整性状态：'+('所有计划证据已齐。' if status['evidence_archive_complete'] else '仍有计划实验未完成；未完成部分不计入完整统计。')]
    lines += prospective_lines
    lines += ['', '## 历史开发证据', '',
             f"历史 r20 已核验：**805/856（{100*805/856:.6f}%）**；四套开发任务分别为 245/256、194/200、186/200、180/200。",
             '同观测、同网络规模、同更新次数下，F11 比含 M1 的普通回报对照多成功 **71/856**，提高 **8.2944 个百分点**。',
             f"4310 排名修正：相对 F10 的救回/损害由 **12/12** 变为 **12/3**；修正前后直接配对为救回 **{direct_rank['rescued']}**、损害 **{direct_rank['harmed']}**，净增 9 次成功。",
             f"五种子外部基线：**{len(training)}/25 次训练已完整归档**，种子 42–46。Boids 无训练种子。", '',
             '| 方法 | 856 任务成功率（%） | 完整训练次数 |', '|---|---:|---:|']
    for row in summary:
        if row['suite'] != 'all-856':
            continue
        sd = row['sample_sd_percentage_points']
        value = f"{row['mean_success_rate_percent']:.4f}" + (f" ± {sd:.4f}" if sd is not None else '')
        lines.append(f"| {row['method']} | {value} | {row['training_repetitions']} |")
    lines += ['', '均值和样本标准差按独立训练种子计算；每个种子使用同一套固定 856 任务。r20 是指定的单个固定节点，不能将这一列称为完整方法的五种子均值。', '',
              '## 文件', '',
              '- `raw/r20/`：原索引中的模型、24 份评价、训练容器、排名修正前模型与评价，以及找回的上游训练文件。',
              '- `raw/baselines/`：完整外部基线的选中权重、训练记录、最终权重（已取回时）及逐任务评价。',
              '- `raw/interrupted_attempts/`：写盘中断前留下的记录，作为恢复来源保存，不计入完整结果表。',
              '- `tables/`：基线逐种子与汇总、六配置消融、机制贡献、成对救回/损害、排名修正和逐任务记录。',
              '- `figures/`：可用于排版的 PDF 和 PNG 图。历史基线图误差线为种子间标准差，新测试区间按图注中的任务、种子或状态层级重采样。',
              '- `raw/independent/`：冻结设计、独立模型与逐任务结果、机制原始测量、配对轨迹和真实 VRX 记录。大型干预采样保留远端原路径与字节哈希，见 `metadata/retrieved_independent.json`。',
              '- `tables/independent_training_budget.csv` 与 `independent_training_objectives.csv`：实际采样步数、共享数据的对照关系、拟合预算和各成员训练目标。',
              '- `metadata/`：完整任务定义、原节点索引、训练配置、执行记录、来源和校验信息。',
              '- `runtime/author_release/`：按原提交校验的作者环境、奖励、观测和网络代码。`current_snapshot` 明确是归档时快照。',
              '- `runtime/frozen_vrx/`：与 600 次试验逐项核对哈希的控制适配与共同 ROS 传输代码；实际策略代码来自 `raw/independent/manifest.pth` 的冻结源码。',
              '- `verify.py` 与 `SHA256SUMS`：归档完整性校验。执行 `python -X utf8 verify.py`。', '',
              '## 证据范围', '',
              '四套任务均为开发证据，4310 未用于拟合，但参与开发选择。此归档不将它们重新命名为独立正式测试。',
              'F11 合并训练容器原有 `completed=false`，保持原值；24 份评价的 `completed=true` 单独核验。',
              '原先缺少的 10 份上游中间文件现已从远端找回并保存当前字节哈希；原索引未给出其历史哈希，不能仅凭路径证明历史字节身份。',
              '原节点未保存完整工程的精确训练源码快照；现已恢复并逐参数复现两项 M2 和两项普通回报对照的计算分支。当前工程快照仍不冒充完整历史工程的原始字节。',
              'F00 是内部消融起点，与重新训练的独立 ARBoids 基线不同。原始作者基线与 MAPPO/MAAC 的选模规则分别保存在训练元数据中。',
              'MAPPO 使用分散高斯策略与集中式价值网络；MAAC 是适用于双推进器连续动作的注意力 SAC 适配，不能称为原仓库离散动作代码的逐字复现。',
              '中断与恢复记录保留在执行元数据和原始训练记录中，未把中断结果补入完整结果表。', '']
    if pending:
        lines += ['尚待完成：' + ', '.join(f"{x['method']}-{x['seed']}" for x in pending), '']
    (PACKAGE / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (PACKAGE / 'verify.py').write_text('''from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
expected={entry['path'] for entry in manifest['files']}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in ('manifest.json','SHA256SUMS')}
if actual!=expected:raise SystemExit('FILE SET DIFFERS: '+str(sorted(actual^expected)))
checksums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines()}
if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()!=checksums['manifest.json']:raise SystemExit('MANIFEST HASH MISMATCH')
for entry in manifest['files']:
 path=root/entry['path']
 if not path.is_file():raise SystemExit('MISSING '+entry['path'])
 sha=hashlib.sha256()
 with path.open('rb') as source:
  for block in iter(lambda:source.read(8*1024*1024),b''):sha.update(block)
 if path.stat().st_size!=entry['bytes'] or sha.hexdigest()!=entry['sha256'] or entry['sha256']!=checksums[entry['path']]:raise SystemExit('MISMATCH '+entry['path'])
print('VERIFIED',len(manifest['files']),'files')
print('Five-seed baselines complete:',manifest['status']['five_seed_baselines_complete'])
print('All planned evidence complete:',manifest['status']['evidence_archive_complete'])
print('Independent model endpoints:',manifest['status']['independent_evidence']['independent_model_endpoints_completed'])
''', encoding='utf-8')
    files = []
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_file() and path.name not in ('manifest.json', 'SHA256SUMS') and not path.name.endswith('.partial'):
            files.append(dict(path=path.relative_to(PACKAGE).as_posix(), bytes=path.stat().st_size, sha256=digest(path)))
    save_json(PACKAGE / 'manifest.json', {'status': status, 'files': files})
    (PACKAGE / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in files) + digest(PACKAGE/'manifest.json') + '  manifest.json\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if require_complete and not status['evidence_archive_complete']:
        raise SystemExit('Planned r20 evidence is still incomplete; the exact pending scope is recorded in status.json.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-remote', action='store_true')
    parser.add_argument('--include-checkpoints', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--collect-traces', action='store_true')
    parser.add_argument('--fetch-independent', action='store_true')
    args = parser.parse_args()
    if args.collect_traces:
        collect_traces()
        return
    if args.fetch_remote:
        fetch_remote(args.include_checkpoints)
    if args.fetch_independent:
        fetch_independent()
    build(args.require_complete)


if __name__ == '__main__':
    main()
