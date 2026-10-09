"""Matched fixed-policy TD/absolute/difference critics for original fusion actions."""
import study_runtime
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import diagnose_fusion_opportunity as opportunity
from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from interaction_rollout import public_packet, execute, compact_snapshot
from policy.interaction_sac import TwinTeamCritic
from train_interaction import atomic_json, atomic_torch


KEYS = ('obs', 'motion', 'central')
ARMS = ('td', 'absolute', 'difference')


def collect(task):
    split, root, repeats, output = task
    bank = 0 if split == 'train' else 1
    scene = 2150000000 + bank*10000000 + root
    with preserved_random_state():
        seed_random(scene)
        env = TADEnv(3, protocol='paper-parameters-v1')
        obs, _ = env.reset(2.25, noisy_agility=True)
        np.random.seed(2180000000 + bank*10000000 + root)
        reservoir = np.random.default_rng(2210000000 + bank*10000000 + root)
        packet, done, transitions, snapshot = public_packet(env, obs), 0, [], None
        while not done:
            if reservoir.integers(len(transitions)+1) == 0:
                snapshot = compact_snapshot(env)
            action, _ = opportunity.POLICY.choose_action(packet, deterministic=True)
            following, reward, done, _, _, _ = execute(env, packet, action, safety=True)
            transitions.append(dict(**packet, **{'next_'+k:v for k,v in following.items()},
                action=action, reward=np.array([np.mean(reward)],dtype=np.float32),
                done=np.array([bool(done)],dtype=np.float32)))
            packet = following
        total = 0.
        for index in range(len(transitions)-1, -1, -1):
            row = transitions[index]
            total = float(row['reward'][0]) + .99*total
            row['return'] = np.array([total],dtype=np.float32)
            row['next_action'] = (transitions[index+1]['action'] if index+1<len(transitions)
                                  else np.zeros_like(row['action']))
        packet = public_packet(snapshot.environment)
        baseline, _ = opportunity.POLICY.choose_action(packet, deterministic=True)
        actions = np.stack([baseline.copy() for _ in range(4)])
        for choice, gate in enumerate((0., .5, 1.), 1):
            actions[choice,root%3,2] = gate
        branches = []
        for choice, first in enumerate(actions):
            for repeat in range(repeats):
                future = 2250000000 + bank*10000000 + root*100 + repeat
                branches.append(dict(opportunity.complete_branch(snapshot, first, future),
                    choice=choice, repeat=repeat, future_seed=future))
        atomic_torch(output/split/f'{root}.pth',dict(root=root,scene_seed=scene,snapshot=snapshot,
            packet=packet,actions=actions,boat=root%3,transitions=transitions,branches=branches))
        return dict(split=split,root=root,real_steps=len(transitions),
                    simulated_steps=sum(b['simulated_steps'] for b in branches))


def load_data(output, split, count, device):
    roots = [torch.load(output/split/f'{i}.pth', map_location='cpu', weights_only=False) for i in range(count)]
    packet = {k:torch.tensor(np.stack([r['packet'][k] for r in roots]),dtype=torch.float32,device=device) for k in KEYS}
    actions = torch.tensor(np.stack([r['actions'] for r in roots]),dtype=torch.float32,device=device)
    returns = np.array([[np.mean([b['discounted_return'] for b in r['branches'] if b['choice']==c])
                         for c in range(4)] for r in roots],dtype=np.float32)
    transitions = [t for r in roots for t in r['transitions']]
    replay = ({k:torch.tensor(np.stack([t[k] for t in transitions]),dtype=torch.float32,device=device)
               for k in transitions[0]} if split == 'train' else None)
    return roots,packet,actions,torch.tensor(returns,device=device),replay


def subset(packet, index):
    return {k:v[index] for k,v in packet.items()}


def critic_scores(critic, packet, actions):
    heads = [critic(packet, actions[:,i]) for i in range(actions.shape[1])]
    return [torch.cat([pair[h] for pair in heads],dim=1) for h in (0,1)]


def train(args, data, seed):
    _,packet,actions,returns,replay = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    value = TwinTeamCritic(512,128,'nominal-thrust-v2',args.critic_state_coordinates).to(args.device)
    for head in (value.q1,value.q2):
        with torch.no_grad(): head.node[0].weight[:,7:10].zero_()
    optimizer = torch.optim.Adam(value.parameters(),lr=1e-4,eps=1e-5)
    real = {k:replay[k] for k in KEYS}
    following = {k:replay['next_'+k] for k in KEYS}
    for step in range(args.warmup_updates):
        index = rng.integers(len(replay['action']),size=args.batch_size)
        q = value(subset(real,index),replay['action'][index])
        loss = sum(F.mse_loss(h,replay['return'][index]) for h in q)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        for head in (value.q1,value.q2): head.node[0].weight.grad[:,7:10].zero_()
        optimizer.step()
    models = {arm:copy.deepcopy(value) for arm in ARMS}
    optimizers = {arm:torch.optim.Adam(models[arm].parameters(),lr=1e-4,eps=1e-5) for arm in ARMS}
    target = copy.deepcopy(value).requires_grad_(False)
    history = []
    started = time.perf_counter()
    for step in range(1,args.updates+1):
        index = rng.integers(len(replay['action']),size=args.batch_size)
        root_index = rng.integers(len(actions),size=64)
        with torch.no_grad():
            a,b = target(subset(following,index),replay['next_action'][index])
            y = replay['reward'][index]+.99*(1-replay['done'][index])*.5*(a+b)
        losses = {}
        for arm,model in models.items():
            q = model(subset(real,index),replay['action'][index])
            td_loss = sum(F.mse_loss(h,y) for h in q)
            aux = torch.zeros((),device=args.device)
            if arm != 'td':
                qs = critic_scores(model,subset(packet,root_index),actions[root_index])
                g = returns[root_index]
                if arm == 'absolute':
                    # Same three paired interventions as the difference arm.
                    aux = sum(.5*(F.mse_loss(h[:,1:],g[:,1:])+F.mse_loss(h[:,:1],g[:,:1])) for h in qs)
                else:
                    aux = sum(F.mse_loss(h[:,1:]-h[:,:1],g[:,1:]-g[:,:1]) for h in qs)
            loss = td_loss+.1*aux
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite fixed-policy loss.')
            optimizers[arm].zero_grad(set_to_none=True); loss.backward(); optimizers[arm].step()
            losses[arm]=dict(td=float(td_loss.detach()),auxiliary=float(aux.detach()))
        with torch.no_grad():
            for lag,current in zip(target.parameters(),models['td'].parameters()): lag.lerp_(current,.005)
        if step%500==0 or step==args.updates:
            history.append(dict(step=step,losses=losses))
            print(dict(stage='fixed-critic',seed=seed,step=step,losses=losses),flush=True)
    checkpoint = dict(seed=seed,models={k:v.state_dict() for k,v in models.items()},
                      target=target.state_dict(),optimizers={k:v.state_dict() for k,v in optimizers.items()},
                      updates=args.updates,warmup_updates=args.warmup_updates,history=history)
    atomic_torch(args.output/f'critics-{seed}.pth',checkpoint)
    return models,dict(seed=seed,seconds=time.perf_counter()-started,history=history)


@torch.no_grad()
def evaluate(models, data, seed):
    roots,packet,actions,returns,_ = data
    truth = returns.cpu().numpy()
    results = []
    for arm,model in models.items():
        qa,qb = model if isinstance(model,tuple) else critic_scores(model,packet,actions)
        score = torch.minimum(qa,qb).cpu().numpy()
        choices = score.argmax(axis=1)
        predictions = score[:,1:]-score[:,:1]
        observed = truth[:,1:]-truth[:,:1]
        rows=[]
        for i,r in enumerate(roots):
            baseline = [b for b in r['branches'] if b['choice']==0]
            chosen = [b for b in r['branches'] if b['choice']==int(choices[i])]
            metrics = {k:float(np.mean([a[k]-b[k] for a,b in zip(chosen,baseline)]))
                for k in ('discounted_return','capture_time','capture','success','breach','collision')}
            difference = np.array([[b['discounted_return']-base['discounted_return']
                for b,base in zip([x for x in r['branches'] if x['choice']==c],baseline)] for c in range(1,4)])
            rows.append(dict(root=i,choice=int(choices[i]),q=score[i].tolist(),observed_returns=truth[i].tolist(),
                metrics=metrics,delta_variance=(difference.var(axis=1,ddof=1)/len(baseline)).tolist()))
        variance = np.array([r['delta_variance'] for r in rows])
        rng = np.random.default_rng(20261009)
        sample = rng.integers(len(rows),size=(10000,len(rows)))
        time_delta=np.array([r['metrics']['capture_time'] for r in rows])
        return_delta=np.array([r['metrics']['discounted_return'] for r in rows])
        results.append(dict(seed=seed,arm=arm,rows=rows,
            delta_mae=float(np.abs(predictions-observed).mean()),zero_mae=float(np.abs(observed).mean()),
            delta_mse=float(((predictions-observed)**2).mean()),
            noise_corrected_delta_mse=float(((predictions-observed)**2-variance).mean()),
            zero_noise_corrected_mse=float((observed**2-variance).mean()),
            mean_time_delta=float(time_delta.mean()),time_95ci=np.quantile(time_delta[sample].mean(1),[.025,.975]).tolist(),
            mean_return_delta=float(return_delta.mean()),return_95ci=np.quantile(return_delta[sample].mean(1),[.025,.975]).tolist(),
            nonbaseline_choices=int((choices!=0).sum()),
            task_deltas={k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('capture','success','breach','collision')}))
    return results


def centered_scores(model, baseline_value, packet, actions):
    scores = critic_scores(model,packet,actions)
    return tuple(v+h-h[:,:1] for v,h in zip(baseline_value,scores))


def run_centered(args):
    """Separate on-policy value anchoring from the intervention response fit."""
    parent = json.loads((args.center_from/'protocol.json').read_text())
    parent_coordinates = parent['arguments'].get('critic_state_coordinates', 'dynamics-v1')
    if args.critic_state_coordinates is not None and args.critic_state_coordinates != parent_coordinates:
        raise ValueError('Centered continuation must retain the parent critic state coordinates.')
    args.critic_state_coordinates = parent_coordinates
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=parent['source_sha256']:
        raise ValueError('The frozen continuation policy must match the parent experiment.')
    protocol = dict(stage='centered-conditional-value-development',parent=str(args.center_from),
        parent_protocol_sha256=hashlib.sha256((args.center_from/'protocol.json').read_bytes()).hexdigest(),
        parent_results_sha256=hashlib.sha256((args.center_from/'results.json').read_bytes()).hexdigest(),
        source=Path(__file__).read_text(encoding='utf-8'),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        formula='Q(s,a)=B_TD(s,a_pi)+A(s,a)-A(s,a_pi); B_TD frozen; paired signals train A only',
        initialization='same parent TD model in both arms, action input weights zeroed; other network architecture unchanged',
        comparison=f'same data, initialization, minibatches, .1 loss scale and {args.updates} Adam updates; absolute versus difference labels',
        scope='parent test scenes are now reused development data; improvement would require fresh confirmation before a learning claim',
        model_seeds=[42,101],additional_simulator_steps=0)
    atomic_json(args.output/'protocol.json',protocol)
    training=load_data(args.center_from,'train',parent['arguments']['train_states'],args.device)
    testing=load_data(args.center_from,'test',parent['arguments']['test_states'],args.device)
    _,packet,actions,returns,_=training
    _,test_packet,test_actions,_,_=testing
    results=[];started=time.perf_counter()
    for seed in (42,101):
        saved=torch.load(args.center_from/f'critics-{seed}.pth',map_location=args.device,weights_only=True)
        base=TwinTeamCritic(512,128,'nominal-thrust-v2',parent_coordinates).to(args.device)
        base.load_state_dict(saved['models']['td']);base.requires_grad_(False)
        with torch.no_grad():
            baseline_value=base(packet,actions[:,0])
            test_baseline=base(test_packet,test_actions[:,0])
        models={name:copy.deepcopy(base).requires_grad_(True) for name in ('centered-absolute','centered-difference')}
        for model in models.values():
            with torch.no_grad():
                for head in (model.q1,model.q2): head.node[0].weight[:,7:10].zero_()
        optimizers={name:torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-5) for name,model in models.items()}
        rng=np.random.default_rng(seed);history=[]
        for step in range(1,args.updates+1):
            index=rng.integers(len(actions),size=64);g=returns[index];losses={}
            for name,model in models.items():
                qs=centered_scores(model,tuple(v[index] for v in baseline_value),subset(packet,index),actions[index])
                if name=='centered-difference':
                    loss=sum(F.mse_loss(h[:,1:]-h[:,:1],g[:,1:]-g[:,:1]) for h in qs)
                else:
                    loss=sum(.5*(F.mse_loss(h[:,1:],g[:,1:])+F.mse_loss(h[:,:1],g[:,:1])) for h in qs)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite conditional response loss.')
                optimizers[name].zero_grad(set_to_none=True);(.1*loss).backward();optimizers[name].step()
                losses[name]=float(loss.detach())
            if step%500==0 or step==args.updates:
                history.append(dict(step=step,losses=losses))
                print(dict(stage='centered-critic',seed=seed,step=step,losses=losses),flush=True)
        atomic_torch(args.output/f'critics-{seed}.pth',dict(seed=seed,base=base.state_dict(),
            models={k:v.state_dict() for k,v in models.items()},optimizers={k:v.state_dict() for k,v in optimizers.items()},
            history=history,updates=args.updates))
        with torch.no_grad():
            scores={k:centered_scores(v,test_baseline,test_packet,test_actions) for k,v in models.items()}
        results.extend(evaluate(scores,testing,seed))
    atomic_json(args.output/'results.json',dict(complete=True,results=results,seconds=time.perf_counter()-started))
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-states',type=int,default=256)
    p.add_argument('--test-states',type=int,default=128)
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--warmup-updates',type=int,default=1500)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',default='cuda')
    p.add_argument('--critic-state-coordinates',choices=['dynamics-v1','execution-v2'])
    p.add_argument('--center-from',type=Path)
    args=p.parse_args()
    if args.output.exists(): p.error('Preserve completed and interrupted experiment evidence.')
    if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,set(range(24)))
    torch.set_num_threads(1)
    if args.center_from:
        run_centered(args)
        return
    if args.critic_state_coordinates is None:
        args.critic_state_coordinates = 'execution-v2'
    files=[Path(__file__),Path('train/diagnose_fusion_opportunity.py'),Path('train/policy/interaction_sac.py'),
        Path('train/interaction_rollout.py'),Path('train/interaction_evaluation.py'),Path('train/policy/networks.py'),
        Path('train/envs/TADgame.py'),Path('train/envs/modules.py'),Path('train/envs/snapshot.py'),Path('train/cbf_source_baseline.py')]
    protocol=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        files={str(f):dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),text=f.read_text(encoding='utf-8')) for f in files},
        policy='fixed deterministic original pretrained ARBoids; original paper reward, gamma .99; common CBF',
        roots='one uniform pre-action state per independent episode; three defenders; agility 2.25 +/- .5',
        intervention='baseline plus theta 0/.5/1 on root-index modulo 3; fixed other candidates and gates; one step only',
        futures=f'{args.repeats} complete common-random-number futures for all four actions; disjoint train/test episodes and random streams',
        fit=f'same initialization, replay minibatches, shared TD-only lagged target, {args.warmup_updates} state-only MC warmup and {args.updates} critic updates; no actor update',
        auxiliary='same full-return samples; weight .1; absolute average branch MSE versus paired difference MSE',
        model_seeds=[42,101],selection='fixed final update; no test-based checkpoint or hyperparameter selection',
        inference='development diagnostic of fixed-policy critics; first-action choices are privileged critic probes, not a deployed learned actor')
    atomic_json(args.output/'protocol.json',protocol)
    for split in ('train','test'): (args.output/split).mkdir()
    tasks=[(split,i,args.repeats,args.output) for split,n in [('train',args.train_states),('test',args.test_states)] for i in range(n)]
    collected=[];started=time.perf_counter()
    with ProcessPoolExecutor(args.workers,initializer=opportunity.initialize,initargs=(args.source,),mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(collect,t) for t in tasks]):
            collected.append(f.result())
            if len(collected)%32==0 or len(collected)==len(tasks):
                print(dict(stage='fixed-policy-labels',states=len(collected),total=len(tasks)),flush=True)
    atomic_json(args.output/'collection.json',dict(complete=True,roots=collected,seconds=time.perf_counter()-started))
    training=load_data(args.output,'train',args.train_states,args.device)
    results=[];training_runs=[]
    # Test outcomes are loaded only after the first set of critics is fixed.
    testing=None
    for seed in (42,101):
        models,run=train(args,training,seed);training_runs.append(run)
        if testing is None: testing=load_data(args.output,'test',args.test_states,args.device)
        results.extend(evaluate(models,testing,seed))
    for filename,saved in protocol['files'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=saved['sha256']:
            raise RuntimeError('Frozen source changed: '+filename)
    atomic_json(args.output/'results.json',dict(complete=True,results=results,training=training_runs,
        real_steps=sum(r['real_steps'] for r in collected if r['split']=='train'),
        simulated_steps=sum(r['simulated_steps'] for r in collected),seconds=time.perf_counter()-started))
    print(json.dumps([ {k:v for k,v in r.items() if k!='rows'} for r in results]),flush=True)


if __name__=='__main__': main()
