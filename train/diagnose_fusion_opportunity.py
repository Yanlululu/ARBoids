"""Frozen original-ARBoids fusion opportunities, with new-future validation."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from distill_role_value import initialize as initialize_runtime
from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from interaction_evaluation import OriginalPolicy
from interaction_rollout import compact_snapshot, execute, public_packet, safety_controller
from source_arboids import source_mixture
from train_interaction import atomic_json, atomic_torch, outcome_row


POLICY = None
FROZEN = None


def initialize(checkpoint):
    global POLICY
    initialize_runtime()
    POLICY = OriginalPolicy(checkpoint)


def single_boat_actions(baseline):
    actions, interventions = [baseline.copy()], [dict(boat=None, gate=None)]
    for boat in range(len(baseline)):
        for gate in (0., .5, 1.):
            changed = baseline.copy()
            changed[boat, 2] = gate
            actions.append(changed)
            interventions.append(dict(boat=boat, gate=gate))
    return np.stack(actions), interventions


def complete_branch(snapshot, first, future_seed):
    """One prescribed 0.2-s action, then the same frozen own-state policy."""
    env = snapshot.restore(future_seed=future_seed)
    packet, done, step, discounted_return, undiscounted_return = public_packet(env), 0, 0, 0., 0.
    while not done:
        action = first if step == 0 else POLICY.choose_action(packet, deterministic=True)[0]
        packet, reward, done, _, _, _ = execute(env, packet, action, safety=True)
        mean_reward = float(np.mean(reward))
        discounted_return += .99**step*mean_reward
        undiscounted_return += mean_reward
        step += 1
    return dict(outcome_row(env, done), discounted_return=discounted_return,
                undiscounted_return=undiscounted_return, simulated_steps=step)


def pick(search, interventions, objective):
    baseline = search[0]
    if objective == 'task':
        eligible = [i for i, row in enumerate(search) if row['collision'] <= baseline['collision']
                    and row['breach'] <= baseline['breach'] and row['capture'] >= baseline['capture']]
        return min(eligible, key=lambda i:(search[i]['capture_time'], i != 0, i))
    return min(range(len(search)), key=lambda i:(-search[i]['discounted_return'], i != 0, i))


def mean_rows(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'outcome'}


def diagnose(task):
    root, scene, defenders, agility, index, output, search_repeats, validation_repeats = task
    with preserved_random_state():
        seed_random(scene)
        env = TADEnv(defenders, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility, noisy_agility=False)
        np.random.seed(scene+1000000)
        packet = public_packet(env, obs)
        prefix_steps = 10+(index*13)%51
        for _ in range(prefix_steps):
            snapshot = compact_snapshot(env)
            action = POLICY.choose_action(packet, deterministic=True)[0]
            packet, _, done, _, _, _ = execute(env, packet, action, safety=True)
            if done:
                break
        env = snapshot.restore()
        packet = public_packet(env)
        baseline = POLICY.choose_action(packet, deterministic=True)[0]
        actions, interventions = single_boat_actions(baseline)
        nominal = np.stack([source_mixture(a, env.boids_actions) for a in actions])
        filtered = np.stack([safety_controller(defenders).control(packet['motion'], a, env.boids_actions)[1]
                             for a in actions])
        atomic_torch(output/'states'/f'{root}.pth', dict(snapshot=snapshot, packet=packet, actions=actions,
                     nominal=nominal, filtered=filtered, scene_seed=scene))
        raw, search = [], []
        for choice, action in enumerate(actions):
            outcomes = []
            for repeat in range(search_repeats):
                future = 1600000000+root*1000+repeat
                row = complete_branch(snapshot, action, future)
                outcomes.append(row)
                raw.append(dict(row, phase='search', choice=choice, future_seed=future))
            search.append(mean_rows(outcomes))
        choices = {objective:pick(search, interventions, objective) for objective in ('task', 'reward')}
        validation = {}
        for choice in sorted({0, *choices.values()}):
            outcomes = []
            for repeat in range(validation_repeats):
                future = 1700000000+root*1000+repeat
                row = complete_branch(snapshot, actions[choice], future)
                outcomes.append(row)
                raw.append(dict(row, phase='validation', choice=choice, future_seed=future))
            validation[choice] = mean_rows(outcomes)
        metrics = ('capture_time', 'capture', 'success', 'collision', 'breach', 'discounted_return')
        result = dict(root=root, scene_seed=scene, defenders=defenders, agility=agility,
            time=float(env.Current_T), interventions=interventions, selected=choices,
            search=search, validation=validation, branches=raw,
            maximum_nominal_change=float(np.max(np.linalg.norm(nominal-nominal[:1], axis=(1, 2)))),
            maximum_executed_change=float(np.max(np.linalg.norm(filtered-filtered[:1], axis=(1, 2)))),
            selected_executed_change={key:float(np.linalg.norm(filtered[value]-filtered[0])) for key,value in choices.items()},
            validation_delta={key:{m:validation[value][m]-validation[0][m] for m in metrics}
                              for key,value in choices.items()},
            search_delta={key:{m:search[value][m]-search[0][m] for m in metrics} for key,value in choices.items()})
        atomic_json(output/'roots'/f'{root}.json', result)
        return result


def summary(roots):
    rng = np.random.default_rng(20261009)
    cells = sorted({(r['defenders'], r['agility']) for r in roots})
    output = {}
    for cell in cells:
        rows = [r for r in roots if (r['defenders'],r['agility']) == cell]
        take = rng.integers(0, len(rows), size=(10000, len(rows)))
        methods = {}
        for objective in ('task', 'reward'):
            delta = np.array([r['validation_delta'][objective]['capture_time'] for r in rows])
            methods[objective] = dict(mean_capped_time_delta=float(delta.mean()),
                exploratory_state_paired_95ci=np.quantile(delta[take].mean(-1), [.025,.975]).tolist(),
                improved_states=int((delta < -1e-9).sum()), worsened_states=int((delta > 1e-9).sum()),
                nonbaseline_selections=sum(r['selected'][objective] != 0 for r in rows),
                search_time_delta=float(np.mean([r['search_delta'][objective]['capture_time'] for r in rows])),
                validation_mean_deltas={metric:float(np.mean([r['validation_delta'][objective][metric] for r in rows]))
                    for metric in ('capture','success','breach','collision','discounted_return')})
        output[str(cell)] = dict(states=len(rows), executed_change_states=sum(r['maximum_executed_change']>1e-6 for r in rows),
            median_maximum_executed_change=float(np.median([r['maximum_executed_change'] for r in rows])), objectives=methods)
    return output


def initialize_gradient(checkpoint):
    global FROZEN
    from interaction_rollout import FrozenPolicy
    initialize_runtime()
    FROZEN = FrozenPolicy(torch.load(checkpoint, map_location='cpu', weights_only=True))


def own_policy_state(root, output):
    """One uniform pre-action state from a new complete on-policy episode."""
    scene = 1520000000 + root
    with preserved_random_state():
        seed_random(scene)
        env = TADEnv(3, protocol='paper-parameters-v1')
        obs, _ = env.reset(2.25, noisy_agility=True)
        np.random.seed(1530000000 + root)
        generator = torch.Generator().manual_seed(1540000000 + root)
        reservoir = np.random.default_rng(1550000000 + root)
        packet, done, steps, snapshot = public_packet(env, obs), 0, 0, None
        while not done:
            steps += 1
            if reservoir.integers(steps) == 0:
                snapshot = compact_snapshot(env)
            action, _ = FROZEN.action(packet, generator)
            packet, _, done, _, _, _ = execute(env, packet, action, safety=FROZEN.safety,
                attacker_action=FROZEN.attacker_action(env), reward_objective=FROZEN.reward_objective)
        saved = dict(snapshot=snapshot, packet=public_packet(snapshot.environment),
            scene_seed=scene, condition_agility=2.25, collection_steps=steps,
            collection_outcome=outcome_row(env, done))
        atomic_torch(output/'states'/f'{root}.pth', saved)
        return saved


def gradient_root(task):
    """Audit the actor's actual min-Q direction under that critic's own policy."""
    from policy.interaction_sac import tensor_packet
    root, states_path, output, repetitions = task
    saved = (own_policy_state(root, output) if states_path is None else
             torch.load(states_path/f'{root}.pth', map_location='cpu', weights_only=False))
    snapshot, packet = saved['snapshot'], saved['packet']
    action, _ = FROZEN.action(packet, torch.Generator().manual_seed(1800000000+root))
    tensor = torch.tensor(action[None], requires_grad=True)
    tp = {k:v[None] for k,v in tensor_packet(packet).items()}
    q1,q2 = FROZEN.critic(tp,tensor)
    derivative = torch.autograd.grad(torch.minimum(q1,q2).sum(), tensor)[0][0,:,2].detach().numpy()
    boat = int(np.argmax(np.abs(derivative)))
    direction = float(np.sign(derivative[boat]))
    actions = [action.copy() for _ in range(3)]
    actions[1][boat,2] = np.clip(action[boat,2]+.1*direction,0.,1.)
    actions[2][boat,2] = np.clip(action[boat,2]-.1*direction,0.,1.)
    with torch.inference_mode():
        prediction = []
        for candidate in actions:
            a,b = FROZEN.critic(tp,torch.tensor(candidate[None]))
            prediction.append(float(torch.minimum(a,b)))
    rows = []
    for choice, first in enumerate(actions):
        for repeat in range(repetitions):
            future, policy_seed = 1810000000+root*1000+repeat,1820000000+root*1000+repeat
            with preserved_random_state():
                env = snapshot.restore(future_seed=future)
                generator = torch.Generator().manual_seed(policy_seed)
                torch.randn((1,env.defender_num,3),generator=generator)
                current, done, steps, soft_return = public_packet(env),0,0,0.
                while not done:
                    candidate,logp = (first.copy(),0.) if steps==0 else FROZEN.action(current,generator)
                    current,reward,done,_,_,_ = execute(env,current,candidate,safety=FROZEN.safety,
                        attacker_action=FROZEN.attacker_action(env),reward_objective=FROZEN.reward_objective)
                    soft_return += FROZEN.gamma**steps*(float(np.mean(reward))-FROZEN.alpha*logp)
                    steps += 1
                rows.append(dict(outcome_row(env,done),choice=choice,repeat=repeat,future_seed=future,
                    policy_seed=policy_seed,soft_return=soft_return,simulated_steps=steps))
    baseline = [r for r in rows if r['choice']==0]
    deltas = {}
    for choice in (1,2):
        alternatives=[r for r in rows if r['choice']==choice]
        difference=np.array([r['soft_return']-b['soft_return'] for r,b in zip(alternatives,baseline)])
        deltas[str(choice)]=dict(predicted=prediction[choice]-prediction[0],measured=float(difference.mean()),
            standard_error=float(difference.std(ddof=1)/np.sqrt(repetitions)),
            time_difference=float(np.mean([r['capture_time']-b['capture_time'] for r,b in zip(alternatives,baseline)])),
            differences=difference.tolist())
    result=dict(root=root,defenders=len(action),agility=saved.get('condition_agility',float(snapshot.environment.attacker.agility)),
        realized_agility=float(snapshot.environment.attacker.agility),time=float(snapshot.environment.Current_T),
        boat=boat,derivative=derivative.tolist(),actions=np.asarray(actions).tolist(),
        q=prediction,deltas=deltas,branches=rows)
    atomic_json(output/'roots'/f'{root}.json',result)
    return result


def run_gradient(args):
    import json
    if args.on_policy_roots:
        checkpoint = torch.load(args.critic_checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['config']['agent']['defender_num'] != 3 or checkpoint.get('stage') != 'gate':
            raise ValueError('This on-policy condition is the three-vessel gate-stage training distribution.')
        selected = list(range(args.count))
    else:
        source=json.loads((args.states_from/'results.json').read_text())
        if not source['complete']:
            raise ValueError('The source-state experiment must be complete.')
        selected=[r['root'] for r in source['roots'] if (r['agility']==2.25 and r['root']%16<8)
                  or (r['agility']>2.25 and r['root']%8<4)]
    protocol=dict(stage='frozen-critic-direction-diagnosis',roots=selected,repetitions=args.validation_repeats,
        checkpoint=str(args.critic_checkpoint),checkpoint_sha256=hashlib.sha256(args.critic_checkpoint.read_bytes()).hexdigest(),
        states_from=str(args.states_from),states_protocol_sha256=(None if args.on_policy_roots else
            hashlib.sha256((args.states_from/'protocol.json').read_bytes()).hexdigest()),
        on_policy_roots=args.on_policy_roots,
        state_distribution=('one uniform pre-action state per new complete checkpoint-policy episode; three vessels; '
            'agility 2.25 with original training noise +/-0.5' if args.on_policy_roots else
            'original-policy states; six-vessel conditions are outside this three-vessel critic training distribution'),
        source=Path(__file__).read_text(encoding='utf-8'),
        intervention='fixed sampled candidates; largest absolute min-Q gate derivative, plus/minus 0.1 on one vessel',
        continuation='same frozen checkpoint stochastic policy; matching gamma, reward and entropy convention; no learned tail',
        scope='diagnostic of this trained critic and actor update direction, not a matched supervision ablation')
    atomic_json(args.output/'protocol.json',protocol)
    (args.output/'roots').mkdir()
    if args.on_policy_roots:
        (args.output/'states').mkdir()
    started, results=time.perf_counter(),[]
    with ProcessPoolExecutor(args.workers,initializer=initialize_gradient,initargs=(args.critic_checkpoint,),
                             mp_context=mp.get_context('spawn')) as pool:
        tasks=[(r,None if args.on_policy_roots else args.states_from/'states',args.output,args.validation_repeats) for r in selected]
        for f in as_completed([pool.submit(gradient_root,t) for t in tasks]):
            results.append(f.result())
            print(dict(stage='critic-direction',states=len(results),total=len(tasks)),flush=True)
    summary={}
    rng=np.random.default_rng(20261009)
    for n,a in sorted({(r['defenders'],r['agility']) for r in results}):
        group=[r for r in results if (r['defenders'],r['agility'])==(n,a)]
        values=np.array([r['deltas']['1']['measured'] for r in group])
        pred=np.array([r['deltas']['1']['predicted'] for r in group])
        take=rng.integers(0,len(group),(10000,len(group)))
        summary[str((n,a))]=dict(states=len(group),predicted_improvement=float(pred.mean()),
            measured_improvement=float(values.mean()),measured_95ci=np.quantile(values[take].mean(-1),[.025,.975]).tolist(),
            positive_measured_states=int((values>0).sum()),delta_q_mae=float(np.abs(pred-values).mean()),
            zero_predictor_mae=float(np.abs(values).mean()),
            capped_time_difference=float(np.mean([r['deltas']['1']['time_difference'] for r in group])))
    atomic_json(args.output/'results.json',dict(complete=True,summary=summary,roots=results,
        wall_seconds=time.perf_counter()-started,simulated_steps=sum(b['simulated_steps'] for r in results for b in r['branches'])))
    print(dict(complete=True,summary=summary),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--stress-count', type=int, default=8)
    parser.add_argument('--search-repeats', type=int, default=4)
    parser.add_argument('--validation-repeats', type=int, default=8)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--critic-checkpoint', type=Path)
    parser.add_argument('--states-from', type=Path)
    parser.add_argument('--on-policy-roots', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing experiment evidence.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    if args.critic_checkpoint:
        if args.on_policy_roots == (args.states_from is not None):
            parser.error('--critic-checkpoint requires exactly one of --states-from and --on-policy-roots.')
        run_gradient(args)
        return
    cells = [(3,2.25,args.count),(6,2.25,args.count),(6,4.,args.stress_count),(6,6.,args.stress_count)]
    tasks = []
    for cell, (n, agility, count) in enumerate(cells):
        for i in range(count):
            tasks.append((len(tasks),1450000000+cell*10000000+i,n,agility,i,args.output,
                          args.search_repeats,args.validation_repeats))
    files = [Path(__file__), Path('train/interaction_rollout.py'), Path('train/interaction_evaluation.py'),
             Path('train/policy/networks.py'), Path('train/envs/TADgame.py'), Path('train/envs/modules.py'),
             Path('train/envs/snapshot.py'), Path('train/cbf_source_baseline.py')]
    protocol = dict(stage='original-fusion-opportunity-diagnosis', cells=cells,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        roots=[list(t[:5]) for t in tasks], prefix='one state per episode, before step 10+(index*13)%51; last valid preterminal if earlier',
        candidates='baseline plus one-vessel theta in {0,.5,1}; other current gates and all proposals fixed',
        continuation='same frozen deterministic original policy, own-state feedback, real complete environment, common CBF',
        task_selection='on search futures preserve observed collision/breach/capture, then minimize capped time',
        reward_selection='on search futures maximize gamma=.99 original paper reward; no entropy for this deterministic policy',
        validation='new independent futures; common randomness within each intervention pair; state is sampling unit',
        inference_limit='offline opportunities at sampled states; no deployable oracle or action-space impossibility claim',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        files={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text(encoding='utf-8')) for p in files})
    atomic_json(args.output/'protocol.json',protocol)
    (args.output/'states').mkdir()
    (args.output/'roots').mkdir()
    started, roots = time.perf_counter(), []
    with ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(args.source,),mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(diagnose,t) for t in tasks]):
            roots.append(future.result())
            print(dict(stage='fusion-opportunities',states=len(roots),total=len(tasks)),flush=True)
    for p,saved in protocol['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=saved['sha256']:
            raise RuntimeError('Frozen source changed: '+p)
    result=dict(complete=True,summary=summary(roots),roots=sorted(roots,key=lambda r:r['root']),
        wall_seconds=time.perf_counter()-started,simulated_steps=sum(b['simulated_steps'] for r in roots for b in r['branches']))
    atomic_json(args.output/'results.json',result)
    print(dict(complete=True,summary=result['summary'],seconds=result['wall_seconds']),flush=True)


if __name__=='__main__':
    main()
