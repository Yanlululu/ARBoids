"""Stage-one composition learning; teacher preferences are not student Q labels."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from distill_role_value import initialize, rollout
from evaluate_candidate_roles import adaptive_guard_controller, guard_composition_controller
from interaction_evaluation import OriginalPolicy
from interaction_rollout import CandidateRolePolicy
from policy.composition_student import (CompositionStudent, CompositionStudentController,
                                        load_student, student_inputs)
from train_interaction import atomic_json, atomic_torch, episode


def recording_teacher():
    controller = adaptive_guard_controller(6)
    original_predict, original_control = controller.predict, controller.control
    controller.records = []

    def predict(measurement, template):
        if template == 'baseline':
            controller.input_state, controller.input_response = student_inputs(controller, measurement)
            controller.scores = []
        prediction = original_predict(measurement, template)
        controller.scores.append(prediction['score'])
        return prediction

    def control(measurement):
        force, info = original_control(measurement)
        if info['planned']:
            controller.records.append(dict(state=controller.input_state, response=controller.input_response,
                scores=np.asarray(controller.scores, dtype=np.float64), time=measurement.time,
                choice=list(controller.options).index(controller.template)))
        return force, info

    controller.predict, controller.control = predict, control
    return controller


def collect(task):
    seed, agility, split = task
    row, controller = rollout(seed, agility, recording_teacher)
    row.update(scene_seed=seed, agility=agility, split=split)
    return row, [dict(r, scene_seed=seed, agility=agility, split=split) for r in controller.records]


def preference(scores):
    """Preserve lexicographic collision/breach priority; one-second soft BC."""
    failure = scores[..., 0]*2+scores[..., 1]
    eligible = failure == failure.min(-1, keepdim=True).values
    logits = -scores[..., 2].float()
    return torch.softmax(logits.masked_fill(~eligible, -torch.inf), -1)


def fit(data, architecture, seed, updates, output):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompositionStudent(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.tensor(data['state'], device=device)
    z = torch.tensor(data['response'], device=device)
    target = preference(torch.tensor(data['scores'], device=device))
    train = np.where(data['split'] == 'train')[0]
    valid = np.where(data['split'] == 'validation')[0]
    rng = np.random.default_rng(seed+1703)
    best, history = float('inf'), []
    start = time.perf_counter()
    for step in range(1, updates+1):
        model.train()
        index = rng.choice(train, min(64, len(train)), replace=False)
        logits = model(x[index], z[index])
        loss = -(target[index]*logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 200 == 0 or step == updates:
            model.eval()
            with torch.inference_mode():
                pred = torch.cat([model(x[part], z[part]) for part in np.array_split(valid, max(1, len(valid)//128))])
                val_loss = float(-(target[valid]*pred.log_softmax(-1)).sum(-1).mean())
                accuracy = float((pred.argmax(-1).cpu() == torch.tensor(data['choice'][valid])).float().mean())
                regret = (target[valid].max(-1).values-target[valid].gather(1, pred.argmax(-1)[:, None]).squeeze(1)).mean()
            history.append(dict(step=step, train_loss=float(loss.detach()), validation_loss=val_loss,
                                validation_top1=accuracy, preference_regret=float(regret)))
            if val_loss < best:
                best = val_loss
                atomic_torch(output, dict(format='composition-student-v1', architecture=architecture,
                    training_seed=seed, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    selected_step=step, validation_loss=val_loss, training_objective='teacher-preference-BC',
                    parameter_count=sum(p.numel() for p in model.parameters())))
    return dict(architecture=architecture, seed=seed, history=history,
                parameter_count=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-start)


def evaluate(task):
    method, checkpoint, scene, agility = task
    if method in ('source', 'prior'):
        policy = OriginalPolicy(checkpoint) if method == 'source' else CandidateRolePolicy()
        row, _ = episode(policy, scene, defenders=6, agility=agility, safety=True)
        extra = {}
    else:
        if method == 'teacher43':
            factory = lambda: adaptive_guard_controller(6)
        elif method == 'teacher22':
            factory = lambda: guard_composition_controller(6)
        else:
            model = load_student(checkpoint)
            factory = lambda: CompositionStudentController(model)
        row, controller = rollout(scene, agility, factory)
        extra = dict(decisions=controller.plan_count)
        if hasattr(controller, 'mode_plans'):
            extra['mode_plans'] = controller.mode_plans
        if hasattr(controller, 'timings'):
            extra['timings'] = controller.timings
    return dict(row, method=method, scene_seed=scene, agility=agility, **extra)


def summarize(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        result[method] = dict(episodes=len(group), mean_capped_time=float(np.mean([r['capture_time'] for r in group])),
            **{key:sum(r[key] for r in group) for key in ('capture', 'success', 'breach', 'collision', 'timeout')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--validation-count', type=int, default=32)
    parser.add_argument('--test-count', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[42, 101])
    parser.add_argument('--scene-base', type=int, default=1300000000)
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in ('count', 'validation_count', 'test_count', 'workers', 'updates')):
        parser.error('Scene counts, workers and updates must be positive.')
    if (len(set(args.training_seeds)) != len(args.training_seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.training_seeds)):
        parser.error('Training seeds must be distinct integers in [0, 2**32).')
    args.output, args.source = args.output.resolve(), args.source.resolve()
    if args.output.exists():
        parser.error('Existing experiment evidence must be retained.')
    if not args.source.is_file():
        parser.error('A source checkpoint file is required.')
    train_root = Path(__file__).resolve().parent
    files = [train_root/name for name in ('train_composition_student.py', 'policy/composition_student.py',
        'distill_role_value.py', 'evaluate_candidate_roles.py', 'role_prediction.py',
        'policy/role_residual.py', 'policy/role_value.py', 'jit_rollout.py', 'jit_nominal_environment.py',
        'feedback_joint_control.py', 'attacker_motion_observer.py', 'envs/TADgame.py')]
    banks = {split: [(args.scene_base+cell*10000000+offset+i, agility, split)
                     for cell, agility in enumerate((4., 6.)) for i in range(count)]
             for split, offset, count in [('train', 0, args.count),
                 ('validation', 20000000, args.validation_count), ('test', 40000000, args.test_count)]}
    initial = [seed for bank in banks.values() for seed, _, _ in bank]
    streams = initial+[seed+1000000 for seed in initial]
    if len(streams) != len(set(streams)) or min(streams)<0 or max(streams)>=2**32:
        parser.error('All initial and physical-future random streams must be distinct.')
    initialize()
    protocol = dict(stage='structured-student-stage-one-development', arguments={k:str(v) if isinstance(v, Path) else v
        for k,v in vars(args).items()}, banks=banks, physical_future_offset=1000000,
        candidate_count=43, composition_period_seconds=2., feedback_period_seconds=.2, theta=0.,
        teacher_target='lexicographic teacher preference, not student-policy value',
        objective='soft behavior cloning, temperature one second, best collision/breach class only',
        selection='minimum whole-episode-held-out preference cross-entropy every 200 updates',
        task_gate=dict(capture_loss_max_percentage_points=3., breach_increase_max=0, collision_increase_max=0,
                       capped_time_ratio_max=1.10, reference='teacher43', confidence='development descriptive only'),
        comparisons=['structured-vs-flat same examples/objective/updates', 'teacher22', 'teacher43', 'prior', 'source'],
        execution='team coordinator, only public measurements and causal history, no prediction in student',
        source_checkpoint_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        files={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text(encoding='utf-8')) for p in files})
    atomic_json(args.output/'protocol.json', protocol)
    start, records, teacher_rows = time.perf_counter(), [], []
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(args.workers, initializer=initialize, mp_context=context) as pool:
        tasks = banks['train']+banks['validation']
        for future in as_completed([pool.submit(collect, t) for t in tasks]):
            row, states = future.result()
            teacher_rows.append(row)
            records.extend(states)
            if len(teacher_rows)%16 == 0:
                print(dict(stage='labels', episodes=len(teacher_rows), total=len(tasks), states=len(records)), flush=True)
    records.sort(key=lambda r:(r['scene_seed'], r['time']))
    data = {key:np.array([r[key] for r in records]) for key in records[0]}
    np.savez_compressed(args.output/'teacher-labels.npz', **data)
    atomic_json(args.output/'teacher-episodes.json', teacher_rows)
    atomic_json(args.output/'collection.json', dict(seconds=time.perf_counter()-start,
        episodes=len(teacher_rows), states=len(records), candidate_forecasts=len(records)*43))
    print(dict(stage='collected', states=len(records), seconds=time.perf_counter()-start), flush=True)
    paths, fits = {}, []
    for seed in args.training_seeds:
        for architecture in ('flat', 'structured'):
            method = architecture+'-'+str(seed)
            paths[method] = args.output/(method+'.pth')
            fits.append(fit(data, architecture, seed, args.updates, paths[method]))
            atomic_json(args.output/'training.json', fits)
            print(dict(stage='trained', method=method, seconds=fits[-1]['seconds'],
                       best_validation=min(h['validation_loss'] for h in fits[-1]['history'])), flush=True)
    methods = dict(paths, teacher43=None, teacher22=None, prior=None, source=args.source)
    tasks = [(method, path, scene, agility) for scene, agility, _ in banks['test'] for method, path in methods.items()]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=initialize, mp_context=context) as pool:
        for future in as_completed([pool.submit(evaluate, t) for t in tasks]):
            rows.append(future.result())
            if len(rows)%32 == 0:
                atomic_json(args.output/'results.json', dict(complete=False, episodes=rows, summary=summarize(rows)))
                print(dict(stage='evaluation', episodes=len(rows), total=len(tasks)), flush=True)
    for name, saved in protocol['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != saved['sha256']:
            raise RuntimeError('Frozen source changed: '+name)
    result = dict(complete=True, episodes=sorted(rows, key=lambda r:(r['scene_seed'], r['method'])),
                  summary=summarize(rows), wall_seconds=time.perf_counter()-start, formal_evidence=False)
    atomic_json(args.output/'results.json', result)
    print(dict(stage='complete', summary=result['summary'], seconds=result['wall_seconds']), flush=True)


if __name__ == '__main__':
    main()
