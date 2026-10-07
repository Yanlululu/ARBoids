"""Independent paired task, intervention-calibration and latency evaluations."""
import study_runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from envs.TADgame import TADEnv
from envs.snapshot import seed_random, preserved_random_state
from policy.networks import ActorAdap
from policy.interaction_sac import make_agent, tensor_packet
from interaction_rollout import (DeploymentPolicy, public_packet, execute, compact_snapshot,
                                FrozenPolicy, frozen_payload, branch_return)
from train_interaction import episode, outcome_row, atomic_json


SEEDS = (202, 303, 404, 505, 606)
CELLS = [(3, a) for a in (1.5, 2., 2.5, 3.)] + [(n, 2.25) for n in range(2, 8)]
METRICS = ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')
CORE_CELLS = [(3, 2.25), (6, 2.25)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def formal_evidence_standard():
    return dict(version='AD-capped-capture-v1', primary_metric='capture_time',
        primary_definition='Episode capture time if captured; otherwise Tmax=60 s. Average over every episode.',
        primary_reference='arboids_cbf', primary_cell='n3-a2.25', strong_interaction_cell='n6-a2.25',
        strong_interaction_definition='Six defenders in the same task arena at agility 2.25; fixed before formal outcomes.',
        secondary_metrics=['capture','success','collision'], formal_seeds=list(SEEDS),
        episodes_per_cell_per_seed=200, calibration_states_per_seed=64, calibration_repetitions=4,
        interval='10000 paired hierarchical bootstrap draws: training seeds, then scenarios/states within seed.',
        effect='Full minus prespecified reference; seconds for capped time, native units for E_env.',
        robustness=['all five seed effects','10% trimmed paired-state sensitivity','leave-one-seed-out effects'],
        percentage_improvement_threshold=None, require_every_seed_to_win=False,
        primary_direction='lower', calibration_definition='E_env = mean absolute error of Delta Q against complete-environment Delta G.',
        practical_meaning='Interpret the absolute time saved and joint CR/SR/collision outcomes in the 60-s task; record the rationale. No fixed percentage cutoff.',
        seed_rule='Keep all five independent training seeds; do not require every seed to win or select a subset.',
        A='Full improves capped capture time against ARBoids+CBF with credible uncertainty and task relevance under common safety and comparable budgets.',
        B='Full improves independent E_env against Same-info; robust across states/seeds and linked to learning efficiency, control selection or task performance.',
        C='Full improves against no_peer in the prespecified six-defender condition; a small three-defender contrast is permitted.',
        D='Permutation tests configuration matching; four-branch contrasts test conditional interaction; executed controls and fixed trajectories connect coefficients to behavior.',
        attribution=dict(same_info='Only remove conditional-consequence supervision and its label generation.',
            no_peer='Only mask teammate learned/Boids candidates in the gate; preserve teammate state, joint critic, supervision, CBF and budget.'),
        formal_entry=['implementation invariants','replicated conditional-value trend','interpretable no_peer contrast',
                      'short and model-return development controls','fresh development confirmation','frozen method and test protocol'])


class OriginalPolicy:
    def __init__(self, path):
        self.actor = ActorAdap(6, 8, 3, 512).eval().requires_grad_(False)
        self.actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(torch.as_tensor(obs, dtype=torch.float32), True, False)[0].numpy()

    def choose_action(self, packet, deterministic=True):
        return self(packet['obs']), 0.


class BoidsPolicy:
    def choose_action(self, packet, deterministic=True):
        return np.zeros((len(packet['obs']), 3), dtype=np.float32), 0.


def long_reference_controller(policy, defenders):
    from adaptive_interval_rollout import AdaptiveIntervalController
    from jit_nominal_environment import JitNominalEnvironment

    class PaperNominal(JitNominalEnvironment):
        def _isTerminate(self):
            self.protocol = 'paper-parameters-v1'
            return TADEnv._isTerminate(self)

    controller = AdaptiveIntervalController(defenders, policy, blend=1., capture_margin=.5,
        failure_cost='delay', tail_steps=100, tail_policy='candidate', agility_threshold=1.75)
    controller.prediction_environment_class = PaperNominal
    return controller


def long_reference_episode(policy, seed, defenders, agility, trajectory=False):
    from feedback_joint_control import observe

    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(defenders, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility)
        controller = long_reference_controller(policy, defenders)
        np.random.seed(seed + 1_000_000)
        done, history = 0, []
        while not done:
            thrust, info = controller.control(observe(env, obs))
            action = policy(obs)
            obs, _, done, _ = env.step(action, 'AdaRes', defender_thrust=thrust)
            if trajectory:
                history.append(dict(time=float(env.Current_T), motion=public_packet(env, obs)['motion'],
                    attacker=env.attacker.pos.copy(), gates=action[:, 2], action=action, thrust=thrust))
        return outcome_row(env, done), history


def save_trajectory(path, history):
    if history:
        keys = ('time', 'motion', 'attacker', 'gates', 'action', 'thrust')
        np.savez_compressed(path, **{k: np.stack([r[k] for r in history]) for k in keys})


def task_evaluation(checkpoint, output, training_seed, arm, episodes=200, cells=None, scene_base=510000000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs=dict(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest(checkpoint),
        training_seed=training_seed,arm=arm,episodes=episodes,
        cells=[list(c) for c in cells] if cells is not None else None,scene_base=scene_base)
    completion=output/'completed.json'
    if completion.exists():
        previous=json.loads(completion.read_text())
        if previous.get('input')!=inputs: raise ValueError('Task evaluation inputs changed; do not mix checkpoints.')
        if previous.get('complete'):
            if digest(output/'episodes.csv')!=previous['episodes_sha256']: raise ValueError('Task episodes changed.')
            return previous
    elif (output/'episodes.csv').exists():
        raise ValueError('Existing task episodes have no checkpoint provenance.')
    atomic_json(completion,dict(complete=False,input=inputs))
    if arm == 'boids':
        policy = BoidsPolicy()
    elif arm in ('original', 'long_reference'):
        policy = OriginalPolicy(checkpoint)
    else:
        policy = DeploymentPolicy(checkpoint)
    csv_path = output / 'episodes.csv'
    fields = ['arm', 'training_seed', 'cell', 'scene_seed', 'defenders', 'agility', *METRICS, 'outcome', 'duration']
    completed = set()
    if csv_path.exists():
        with csv_path.open(encoding='utf-8') as f:
            completed = {(r['cell'], int(r['scene_seed'])) for r in csv.DictReader(f)}
    selected = list(enumerate(CELLS)) if cells is None else [(CELLS.index(c), c) for c in cells]
    for cell_id, (n, agility) in selected:
        cell = f'n{n}-a{agility:g}'
        for i in range(episodes):
            seed = scene_base + SEEDS.index(training_seed) * 1000000 + cell_id * 10000 + i
            if (cell, seed) in completed:
                continue
            if arm == 'long_reference':
                row, history = long_reference_episode(policy, seed, n, agility, trajectory=i == 0)
            else:
                row, history = episode(policy, seed, n, agility, safety=arm not in ('boids', 'original'), trajectory=i == 0)
            row = dict(arm=arm, training_seed=training_seed, cell=cell, scene_seed=seed, defenders=n, agility=agility, **row)
            exists = csv_path.exists()
            with csv_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            if history:
                save_trajectory(output / f'{cell}.npz', history)
        print(f'[IA-TEST] {arm} seed={training_seed} {cell} complete', flush=True)
    if digest(checkpoint)!=inputs['checkpoint_sha256']: raise ValueError('Task checkpoint changed during evaluation.')
    atomic_json(completion, dict(complete=True, input=inputs, arm=arm, training_seed=training_seed,
                episodes_sha256=digest(csv_path),episodes_per_cell=episodes, cells=[CELLS[j] for j, _ in selected]))


def calibration(resume, common_pretrain, output, training_seed, states=64, repetitions=4):
    inputs=dict(checkpoint=str(Path(resume).resolve()),checkpoint_sha256=digest(resume),
        source=str(Path(common_pretrain).resolve()),source_sha256=digest(common_pretrain),
        training_seed=training_seed,states=states,repetitions=repetitions)
    saved = torch.load(resume, map_location='cpu', weights_only=False)
    agent = make_agent(saved['config'])
    if agent.legacy:
        raise ValueError('The legacy individual critic has no joint conditional-value estimate.')
    agent.load_state_dict(saved['agent'])
    policy = FrozenPolicy(frozen_payload(agent, online_critic=True))
    del saved
    source = OriginalPolicy(common_pretrain)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completion=output/'completed.json'
    if completion.exists():
        prior=json.loads(completion.read_text())
        if prior.get('input')!=inputs: raise ValueError('Calibration inputs changed.')
        if prior.get('complete'):
            if digest(output/'calibration.csv')!=prior['calibration_sha256']: raise ValueError('Calibration states changed.')
            return prior
    rows = []
    with preserved_random_state():
        for index in range(states):
            seed = 520000000 + SEEDS.index(training_seed) * 1000000 + index
            seed_random(seed)
            defenders = 3 if index < states//2 else 6
            env = TADEnv(defenders, protocol='paper-parameters-v1')
            obs, _ = env.reset(2.25)
            packet = public_packet(env, obs)
            snapshot = compact_snapshot(env)
            for _ in range(10 + index % 40):
                snapshot = compact_snapshot(env)
                action, _ = source.choose_action(packet)
                packet, _, done, _, _, _ = execute(env, packet, action)
                if done:
                    break
            packet = public_packet(snapshot.environment)
            action, _ = policy.action(packet, torch.Generator().manual_seed(seed + 100))
            alternative = action.copy()
            boat, reference = index % defenders, (0., .5, 1.)[(index // 3) % 3]
            alternative[boat, 2] = reference
            with torch.no_grad():
                p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
                a = policy.critic(p, torch.as_tensor(action).unsqueeze(0))
                b = policy.critic(p, torch.as_tensor(alternative).unsqueeze(0))
                predicted = float((torch.minimum(*a) - torch.minimum(*b)).item())
            differences = []
            count = 0
            for repeat in range(repetitions):
                future, noise = seed + repeat * 10000, seed + repeat * 10000 + 1
                ga, na, _ = branch_return(policy, snapshot, action, future, noise, 300)
                gb, nb, _ = branch_return(policy, snapshot, alternative, future, noise, 300)
                differences.append(ga - gb)
                count += na + nb
            actual = float(np.mean(differences))
            rows.append(dict(state=index, scene_seed=seed, defenders=defenders, training_seed=training_seed,
                boat=boat, reference=reference, zero_prediction_error=abs(actual),
                predicted_difference=predicted, environment_difference=actual,
                absolute_error=abs(predicted-actual), monte_carlo_std=float(np.std(differences)), simulated_steps=count))
    with (output / 'calibration.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if digest(resume)!=inputs['checkpoint_sha256'] or digest(common_pretrain)!=inputs['source_sha256']:
        raise ValueError('Calibration inputs changed while running.')
    atomic_json(completion, dict(complete=True,input=inputs,calibration_sha256=digest(output/'calibration.csv'),
                states=states, repetitions=repetitions,
                E_env=float(np.mean([r['absolute_error'] for r in rows])),
                E_delta=float(np.mean([r['absolute_error'] for r in rows])),
                definition='Complete-environment soft-return difference; no truncated bootstrap tail.'))


def hierarchical_interval(values, rng, repeats=10000):
    """Resample training seeds, then paired scenes within each selected seed."""
    values = np.asarray(values, dtype=float)
    groups, scenes = values.shape
    estimates = []
    for _ in range((repeats + 499) // 500):
        seed_index = rng.integers(groups, size=(500, groups, 1))
        scene_index = rng.integers(scenes, size=(500, groups, scenes))
        estimates.extend(values[seed_index, scene_index].mean(axis=(1, 2)).tolist())
    return [float(x) for x in np.quantile(estimates[:repeats], [.025, .975])]


def paired_effect(values, rng, *, seeds=SEEDS, repeats=10000):
    """Keep training seeds as the independent units and expose state-outlier sensitivity."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or len(values)!=len(seeds) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('A finite paired matrix for every prespecified training seed is required.')
    means=values.mean(axis=1)
    trim=int(values.shape[1]*.1)
    trimmed=np.sort(values,axis=1)[:,trim:-trim] if trim else values
    omitted=[float(np.delete(means,i).mean()) for i in range(len(means))] if len(means)>1 else []
    return dict(difference=float(means.mean()),ci95=hierarchical_interval(values,rng,repeats),
        seed_effects={str(s):float(v) for s,v in zip(seeds,means)},
        seed_standard_deviation=float(means.std(ddof=1)) if len(means)>1 else None,
        negative_seed_count=int((means<0).sum()),positive_seed_count=int((means>0).sum()),
        median_paired_difference=float(np.median(values)),trimmed_10pct_sensitivity=float(trimmed.mean()),
        leave_one_seed_out=omitted,paired_units_per_seed=values.shape[1])


def formal_evidence(study):
    """A--D estimands from fixed formal banks; significance alone never marks a claim supported."""
    study=Path(study);rng=np.random.default_rng(78102026)
    standard=formal_evidence_standard()
    manifest=json.loads((study/'manifest.json').read_text())
    if manifest['candidate_protocol'].get('formal_evidence')!=standard:
        raise ValueError('The A--D evidence standard must be registered before formal evaluation.')
    frozen=study/'reviews/review-protocol-freeze'
    freeze=json.loads((frozen/'evidence.json').read_text())
    decision=json.loads((frozen/'decision.json').read_text())
    if (decision.get('decision')!='continue' or decision.get('assessment')!='frozen' or
            decision.get('evidence_sha256')!=digest(frozen/'evidence.json') or
            freeze.get('frozen_protocol')!=manifest['candidate_protocol'] or freeze.get('frozen_source')!=manifest['code']):
        raise ValueError('A current frozen protocol and source are required for formal evidence.')
    artifacts={}
    def bind(path):
        path=Path(path)
        artifacts[str(path.relative_to(study)) if path.is_relative_to(study) else str(path)]=digest(path)
    bind(frozen/'evidence.json');bind(frozen/'decision.json')
    def completed(directory,filename,checksum,**expected):
        path=directory/'completed.json';record=json.loads(path.read_text())
        if not record.get('complete') or any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError('Incomplete or incompatible formal evaluation: '+str(directory))
        if record.get(checksum)!=digest(directory/filename): raise ValueError('Formal evaluation rows changed.')
        for field,hash_field in (('checkpoint','checkpoint_sha256'),('source','source_sha256')):
            if field in record.get('input',{}):
                source=Path(record['input'][field])
                if digest(source)!=record['input'][hash_field]: raise ValueError('Evaluated source checkpoint changed.')
                bind(source)
        bind(path);bind(directory/filename)
        return record
    def read_rows(path):
        with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))
    task={}
    arms=('full','arboids_cbf','same_info','no_peer','short','model_value')
    for seed in SEEDS:
        for arm in arms:
            progress_path=study/f'training/seed-{seed}/{arm}/progress.json'
            progress=json.loads(progress_path.read_text())
            if progress.get('step')!=1000000 or progress.get('complete') is not True:
                raise ValueError('All five seeds and six arms must reach the fixed one-million-step endpoint.')
            bind(progress_path)
            directory=study/f'primary/seed-{seed}/{arm}'
            record=completed(directory,'episodes.csv','episodes_sha256',episodes_per_cell=200,cells=[list(c) for c in CORE_CELLS])
            if not record.get('input',{}).get('checkpoint_sha256'): raise ValueError('Missing task checkpoint provenance.')
            rows=read_rows(directory/'episodes.csv')
            keys={(r['cell'],int(r['scene_seed'])) for r in rows}
            expected={(f'n{n}-a{a:g}',560000000+SEEDS.index(seed)*1000000+CELLS.index((n,a))*10000+i)
                      for n,a in CORE_CELLS for i in range(200)}
            if len(rows)!=len(keys) or keys!=expected or any(int(r['training_seed'])!=seed or r['arm']!=arm for r in rows):
                raise ValueError('The fixed five-seed core task bank is incomplete or mismatched.')
            for r in rows:
                capture,duration=float(r['capture']),float(r['duration'])
                capped=duration if capture==1 else 60.
                if (any(float(r[k]) not in (0.,1.) for k in METRICS if k!='capture_time') or
                        not 0<=duration<=60.000001 or not np.isclose(float(r['capture_time']),capped,rtol=0,atol=1e-6)):
                    raise ValueError('Primary capped capture time must include every noncapture at Tmax=60 s.')
                task[arm,seed,r['cell'],int(r['scene_seed'])]=r
    arm_summaries=[]
    for arm in arms:
        for cell in ('n3-a2.25','n6-a2.25'):
            summary=dict(arm=arm,cell=cell,episodes=1000,seed_means={},mean={},ci95={},seed_standard_deviation={})
            for metric in METRICS:
                values=np.asarray([[float(task[k][metric]) for k in sorted(task) if k[:3]==(arm,seed,cell)] for seed in SEEDS])
                means=values.mean(axis=1)
                summary['mean'][metric]=float(means.mean())
                summary['ci95'][metric]=hierarchical_interval(values,rng)
                summary['seed_means'][metric]={str(s):float(v) for s,v in zip(SEEDS,means)}
                summary['seed_standard_deviation'][metric]=float(means.std(ddof=1))
            arm_summaries.append(summary)
    def task_contrast(reference,cell):
        effects={}
        for metric in ('capture_time','capture','success','collision'):
            matrix=[]
            for seed in SEEDS:
                scenes=sorted(k[3] for k in task if k[:3]==('full',seed,cell))
                matrix.append([float(task['full',seed,cell,s][metric])-float(task[reference,seed,cell,s][metric]) for s in scenes])
            effects[metric]=paired_effect(matrix,rng)
        return effects
    task_effects={arm:{cell:task_contrast(arm,cell) for cell in ('n3-a2.25','n6-a2.25')}
                  for arm in arms if arm!='full'}
    calibration={}
    for seed in SEEDS:
        for arm in arms:
            if arm=='arboids_cbf': continue
            directory=study/f'calibration/seed-{seed}/{arm}'
            record=completed(directory,'calibration.csv','calibration_sha256',states=64,repetitions=4)
            if not record.get('input',{}).get('checkpoint_sha256'): raise ValueError('Missing calibration checkpoint provenance.')
            rows=read_rows(directory/'calibration.csv')
            rows=sorted(rows,key=lambda r:int(r['state']))
            if ([int(r['state']) for r in rows]!=list(range(64)) or
                    any(int(r['training_seed'])!=seed or int(r['scene_seed'])!=520000000+SEEDS.index(seed)*1000000+i or
                        int(r['defenders'])!=(3 if i<32 else 6) for i,r in enumerate(rows))):
                raise ValueError('Independent environment calibration must contain the fixed balanced state bank.')
            if any(not np.isclose(float(r['absolute_error']),abs(float(r['predicted_difference'])-float(r['environment_difference'])),
                                  rtol=1e-7,atol=1e-8) for r in rows):
                raise ValueError('E_env must be the absolute prediction error against the complete environment return.')
            calibration[arm,seed]=rows
    value_effects={}
    for arm in ('same_info','no_peer','short','model_value'):
        matrix=np.asarray([[float(a['absolute_error'])-float(b['absolute_error'])
            for a,b in zip(calibration['full',seed],calibration[arm,seed])] for seed in SEEDS])
        value_effects[arm]={label:paired_effect(matrix[:,indices],rng)
            for label,indices in (('all',slice(None)),('three_defenders',slice(0,32)),('six_defenders',slice(32,64)))}
    matching,interaction,second_moment,thrust,examples=[],[],[],[],[]
    for seed in SEEDS:
        directory=study/f'mechanism/seed-{seed}'
        completed(directory,'states.json','states_sha256')
        data=json.loads((directory/'states.json').read_text())
        if data.get('input',{}).get('repetitions')!=4: raise ValueError('Four mechanism continuations are required.')
        rows=data['states']
        expected=530000000+SEEDS.index(seed)*1000000
        if len(rows)!=32 or [r['scene_seed'] for r in rows]!=list(range(expected,expected+32)):
            raise ValueError('The prespecified mechanism states are incomplete or mismatched.')
        seed_matching,seed_interaction,seed_second,seed_thrust=[],[],[],[]
        for index,row in enumerate(rows):
            returns={k:np.asarray(v,dtype=float) for k,v in row['returns'].items()}
            if any(v.shape!=(4,) or not np.isfinite(v).all() for v in returns.values()):
                raise ValueError('Mechanism inference requires all four finite paired continuations.')
            x=returns['00']-returns['10']-returns['01']+returns['11']
            seed_matching.append(float((returns['00']-returns['permuted']).mean()))
            seed_interaction.append(float(x.mean()))
            # Cross-repeat products estimate the squared conditional mean without positive Monte Carlo variance bias.
            seed_second.append(float((x.sum()**2-(x*x).sum())/12))
            forces=row['executed_first_thrust']
            seed_thrust.append(float(np.linalg.norm(np.asarray(forces['00'])-np.asarray(forces['permuted']))))
            if index<2:
                if not row.get('trajectory_examples'): raise ValueError('A prespecified trajectory example is missing.')
                examples.append(dict(seed=seed,state=index,scene_seed=row['scene_seed']))
        matching.append(seed_matching);interaction.append(seed_interaction);second_moment.append(seed_second);thrust.append(seed_thrust)
    result=dict(standard=standard,complete=True,artifact_inputs=artifacts,arm_summaries=arm_summaries,
        A=dict(reference='arboids_cbf',cell='n3-a2.25',effects=task_effects['arboids_cbf']['n3-a2.25']),
        B=dict(reference='same_info',E_env=value_effects['same_info'],task_effects=task_effects['same_info']['n3-a2.25'],
            interpretation='Value-error improvement must be linked to learning, control selection or task benefit; accuracy alone is insufficient.'),
        C=dict(reference='no_peer',strong=task_effects['no_peer']['n6-a2.25'],ordinary=task_effects['no_peer']['n3-a2.25']),
        D=dict(configuration_matching=paired_effect(matching,rng),conditional_interaction=paired_effect(interaction,rng),
            noise_corrected_interaction_second_moment=paired_effect(second_moment,rng),
            executed_permutation_thrust_change_N=paired_effect(thrust,rng),trajectory_examples=examples,
            interpretation='Permutation measures matching. Signed four-branch interaction and cross-repeat squared interaction are distinct; positive absolute noisy interaction alone is not evidence.'),
        auxiliary_controls=dict(task=task_effects,conditional_value=value_effects),
        claim_status='Evidence computed; practical relevance and A--D support require the recorded evidence review. No percentage or unanimous-seed rule.')
    a=result['A']['effects']['capture_time'];b=result['B']['E_env']['all'];c=result['C']['strong']['capture_time'];d=result['D']
    task_link=any(result['B']['task_effects'][k]['difference']*direction<0
                  for k,direction in (('capture_time',1),('capture',-1),('success',-1)))
    result['statistical_checks']=dict(
        A=a['ci95'][1]<0,
        B=b['ci95'][1]<0 and b['trimmed_10pct_sensitivity']<0 and all(v<0 for v in b['leave_one_seed_out']) and task_link,
        C=c['ci95'][1]<0,
        D=(d['configuration_matching']['ci95'][0]>0 or
           d['conditional_interaction']['ci95'][0]>0 or d['conditional_interaction']['ci95'][1]<0 or
           d['noise_corrected_interaction_second_moment']['ci95'][0]>0) and
          d['executed_permutation_thrust_change_N']['ci95'][0]>0)
    result['B']['task_direction_consistent']=task_link
    result['practical_relevance_assessed']=False
    return result


def summarize(study):
    study = Path(study)
    rows = []
    for path in sorted((study / 'evaluation').glob('seed-*/*/episodes.csv')):
        if int(path.parents[1].name.split('-')[1]) not in SEEDS:
            continue
        with path.open(encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return dict(complete=False, reason='No independent task evaluations completed.')
    lookup = {(r['arm'], int(r['training_seed']), r['cell'], int(r['scene_seed'])): r for r in rows}
    summaries, pairs = [], []
    rng = np.random.default_rng(7102026)
    for arm in sorted({r['arm'] for r in rows}):
        for cell in sorted({r['cell'] for r in rows}):
            selected = [r for r in rows if r['arm'] == arm and r['cell'] == cell]
            if not selected:
                continue
            summary = dict(arm=arm, cell=cell, episodes=len(selected),
                trained_seeds=len({r['training_seed'] for r in selected}),
                **{m: float(np.mean([float(r[m]) for r in selected])) for m in METRICS})
            groups = [[r for r in selected if int(r['training_seed']) == seed] for seed in SEEDS]
            if all(len(g) == 200 for g in groups):
                summary['ci95'] = {m: hierarchical_interval([[float(r[m]) for r in g] for g in groups],rng) for m in METRICS}
            summaries.append(summary)
            if arm == 'full':
                continue
            by_seed = []
            for seed in SEEDS:
                matched = [(r, lookup.get(('full', seed, cell, int(r['scene_seed'])))) for r in selected
                           if int(r['training_seed']) == seed]
                matched = [(a, b) for a, b in matched if b is not None]
                by_seed.append(matched)
            if any(len(group) != 200 for group in by_seed):
                continue
            for metric in METRICS:
                difference = np.asarray([[float(b[metric])-float(a[metric]) for a, b in group] for group in by_seed])
                pairs.append(dict(reference=arm, cell=cell, metric=metric, **paired_effect(difference,rng)))
    expected = {(arm, seed, f'n{n}-a{agility:g}', 510000000+SEEDS.index(seed)*1000000+ci*10000+i)
        for arm in ('full','same_info','arboids_cbf','model_value','short','no_peer','boids','original','long_reference')
        for seed in SEEDS for ci,(n,agility) in enumerate(CELLS) for i in range(200)}
    result = dict(complete=len(rows)==len(lookup) and set(lookup)==expected, protocol='paper-parameters-v1',
                  summaries=summaries, paired_comparisons=pairs)
    if result['complete']: result['formal_evidence']=formal_evidence(study)
    atomic_json(study / 'analysis.json', result)
    return result


def runtime(checkpoint, output, episodes=16):
    policy = DeploymentPolicy(checkpoint)
    rows = []
    for n in (3, 6):
        # Warm both network and the CBF specialization outside the timing sample.
        env = TADEnv(n, protocol='paper-parameters-v1')
        env.reset(2.25)
        p = public_packet(env)
        policy.control(p['obs'], p['motion'], env.boids_actions)
        for i in range(episodes):
            seed_random(400000000 + n*10000 + i)
            obs, _ = env.reset(2.25)
            done = 0
            while not done:
                started = time.perf_counter()
                p = public_packet(env, obs)
                force = policy.control(p['obs'], p['motion'], env.boids_actions, env.Current_T)
                duration = time.perf_counter()-started
                rows.append(dict(defenders=n, scene=i, seconds=duration))
                obs, _, done, _ = env.step(policy.last_action, 'AdaRes', defender_thrust=force)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'calls.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summaries=[]
    for n in (3,6):
        values=np.asarray([r['seconds'] for r in rows if r['defenders']==n])
        summaries.append(dict(defenders=n, calls=len(values), median=float(np.median(values)),
            p95=float(np.quantile(values,.95)), p99=float(np.quantile(values,.99)),
            maximum=float(values.max()), deadline_misses=int((values>.2).sum())))
    atomic_json(output/'completed.json', dict(complete=True, summaries=summaries,
        timing_scope='public observation adaptation, candidate exchange emulation, actor and common CBF; excludes external communication'))


def runtime_matrix(study, output, episodes=16):
    from feedback_joint_control import observe
    study, output = Path(study), Path(output)
    manifest = json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    seed = SEEDS[0]
    inherited = manifest['inherited_pretraining'].get(f'pretrain-{seed}')
    pretrain = inherited['path'] if inherited else study/f'pretrain/seed-{seed}/actor.pth'
    rows = []
    for arm in ('full','same_info','arboids_cbf','short','model_value','no_peer','boids','original','long_reference'):
        if arm=='boids': policy=BoidsPolicy()
        elif arm in ('original','long_reference'): policy=OriginalPolicy(pretrain)
        else: policy=DeploymentPolicy(study/f'training/seed-{seed}/{arm}/policy.pth')
        for n in (3,6):
            for i in range(-1,episodes):
                seed_random(400000000+n*10000+i)
                env=TADEnv(n,protocol='paper-parameters-v1')
                obs,_=env.reset(2.25)
                planner=long_reference_controller(policy,n) if arm=='long_reference' else None
                done=0
                while not done:
                    start=time.perf_counter()
                    packet=public_packet(env,obs)
                    if planner:
                        force,_=planner.control(observe(env,obs))
                        action=policy(obs)
                    elif arm in ('boids','original'):
                        action,_=policy.choose_action(packet)
                        force=action[:,2:3]*(750.*action[:,:2]+250.)+(1-action[:,2:3])*env.boids_actions
                    else:
                        force=policy.control(obs,packet['motion'],env.boids_actions)
                        action=policy.last_action
                    elapsed=time.perf_counter()-start
                    if i>=0: rows.append(dict(arm=arm,defenders=n,scene=i,seconds=elapsed))
                    obs,_,done,_=env.step(action,'AdaRes',defender_thrust=force)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'calls.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summaries=[]
    for arm in sorted({r['arm'] for r in rows}):
        for n in (3,6):
            x=np.array([r['seconds'] for r in rows if r['arm']==arm and r['defenders']==n])
            summaries.append(dict(arm=arm,defenders=n,calls=len(x),median=float(np.median(x)),
                p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max()),
                deadline_misses=int((x>.2).sum())))
    atomic_json(output/'completed.json',dict(complete=True,summaries=summaries,
        timing_scope='CPU observation adapter and controller, including shared CBF when applicable; excludes external communication'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['task', 'calibration', 'summarize', 'runtime', 'evidence'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pretrain', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--seed', type=int, default=SEEDS[0])
    parser.add_argument('--arm', default='full')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--states', type=int, default=64)
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--primary-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'task':
        task_evaluation(args.checkpoint, args.output, args.seed, args.arm, args.episodes,
                        CORE_CELLS if args.primary_only else None,
                        560000000 if args.primary_only else 510000000)
    elif args.mode == 'calibration':
        calibration(args.checkpoint, args.pretrain, args.output, args.seed, args.states, args.repetitions)
    elif args.mode == 'runtime':
        if args.study: runtime_matrix(args.study,args.output)
        else: runtime(args.checkpoint,args.output)
    elif args.mode == 'evidence':
        atomic_json(args.output/'reviews/formal-evidence/completed.json',formal_evidence(args.output))
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
