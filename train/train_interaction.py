"""Train IA-CRRL study arms and the matched peer-candidate control with resumable state."""
import study_runtime
import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random, preserved_random_state
from policy.interaction_sac import JointReplay, make_agent
from interaction_rollout import public_packet, execute, compact_snapshot, SnapshotPool, InterventionSampler


ARMS = ('arboids_cbf', 'same_info', 'model_value', 'short', 'full', 'no_peer')


def arm_config(config, arm):
    if arm not in ARMS:
        raise ValueError('Unknown study arm: ' + arm)
    config = copy.deepcopy(config)
    config['interaction'].update(arm=arm, auxiliary=('none' if arm in ('arboids_cbf', 'same_info')
                                                    else 'value' if arm == 'model_value' else 'difference'))
    if arm == 'short':
        config['interaction']['horizon_steps'] = 10
    # Preserve the original five configuration dictionaries and their checkpoints.
    config['interaction'].pop('peer_candidates', None)
    if arm == 'no_peer':
        config['interaction']['peer_candidates'] = False
    return config


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def outcome_row(env, outcome):
    events = env.physical_events()
    captured = outcome == 3
    return dict(success=int(outcome in (3, 4)), capture=int(captured), collision=int(events['collision']),
                breach=int(events['breach']), timeout=int(outcome == 4), outcome=outcome,
                capture_time=float(env.Current_T if captured else env.Total_T),
                duration=float(env.Current_T))


def episode(policy, seed, defenders=3, agility=2.25, *, safety=True, trajectory=False, attacker=None):
    with preserved_random_state():
        seed_random(int(seed))
        env = TADEnv(defenders, protocol='paper-parameters-v1', LearningSide='Att' if attacker else 'Def')
        obs, _ = env.reset(agility, noisy_agility=False)
        np.random.seed(int(seed) + 1_000_000)
        packet, done, history = public_packet(env, obs), 0, []
        while not done:
            action, _ = policy.choose_action(packet, deterministic=True)
            attacker_action = None
            if attacker is not None:
                _, att_obs = env._get_obs()
                attacker_action = attacker(att_obs)
            before = packet
            packet, _, done, thrust, _, _ = execute(env, packet, action, safety=safety, attacker_action=attacker_action)
            if trajectory:
                history.append(dict(time=float(env.Current_T), motion=packet['motion'].copy(),
                                    attacker=np.asarray(env.attacker.pos).copy(), gates=action[:, 2].copy(),
                                    action=action.copy(), thrust=thrust.copy(), before=before['motion'].copy()))
        return outcome_row(env, done), history


def evaluate(policy, episodes, seed_base, defenders=3, agility=2.25, safety=True, attacker=None):
    rows = [episode(policy, seed_base + i, defenders, agility, safety=safety, attacker=attacker)[0] for i in range(episodes)]
    return {key: float(np.mean([r[key] for r in rows]))
            for key in ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')}


def export_policy(agent, output, step, *, endpoint=False):
    artifact = dict(format='interaction-sac-deployment-v1',
        config=agent.config, actor={k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
        step=int(step), stage=agent.stage)
    if endpoint and not agent.legacy:
        artifact.update(critic={k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
                        alpha=float(agent.alpha), gamma=agent.gamma)
    name = agent.stage + '-endpoint.pth' if endpoint else 'policy.pth'
    atomic_torch(Path(output) / name, artifact)


def export_probe(agent, output, step, checks):
    """Fixed development checkpoint, including the lagged critic used for model labels."""
    from interaction_rollout import frozen_payload
    artifact = frozen_payload(agent, online_critic=True)
    artifact.update(format='interaction-sac-deployment-v1', step=int(step), stage=agent.stage,
                    target={k: v.detach().cpu() for k, v in agent.target.state_dict().items()},
                    learning_checks=checks)
    atomic_torch(Path(output) / f'probe-{step}.pth', artifact)


def validate(config):
    if config['environment'] != dict(protocol='paper-parameters-v1', total_time=60., agility_noise_half_width=.5):
        raise ValueError('This study requires the fixed 60-second paper protocol.')
    t, i = config['training'], config['interaction']
    if i.get('peer_candidates', True) != (i['arm'] != 'no_peer'):
        raise ValueError('Only the matched no-peer arm may mask peer candidates.')
    positive = ('joint_steps', 'warm_steps', 'replay_capacity', 'eval_interval',
                'eval_episodes', 'log_interval', 'checkpoint_interval')
    if t['gate_steps'] < 0 or any(t[k] <= 0 for k in positive) or t['warm_steps'] >= t['gate_steps'] + t['joint_steps']:
        raise ValueError('Invalid training intervals or step budget.')
    if any(i[k] <= 0 for k in ('intervention_interval', 'pairs_per_batch', 'snapshot_capacity', 'snapshot_interval', 'horizon_steps')):
        raise ValueError('Invalid intervention settings.')
    if i['auxiliary'] not in ('none', 'value', 'difference') or i['auxiliary_weight'] < 0:
        raise ValueError('Invalid auxiliary objective.')
    if not i['safety']:
        raise ValueError('Training study arms must share the CBF filter.')
    if i.get('bootstrap_source', 'coupled') not in ('coupled', 'real_td'):
        raise ValueError('Invalid bootstrap source.')
    if i.get('critic_control_coordinates','raw-v1') not in ('raw-v1','nominal-thrust-v2'):
        raise ValueError('Invalid critic control coordinates.')
    if not isinstance(i.get('label_repetitions',1),int) or i.get('label_repetitions',1)<1:
        raise ValueError('A positive integer label continuation count is required.')
    if not isinstance(t.get('critic_warmup_updates',0),int) or t.get('critic_warmup_updates',0)<0:
        raise ValueError('Critic warmup updates must be a nonnegative integer.')
    if t.get('critic_warmup_updates',0) and (i['arm']=='arboids_cbf' or i.get('bootstrap_source')!='real_td'):
        raise ValueError('Critic warmup requires the matched real-TD evaluator.')
    if t.get('critic_warmup_updates',0) and t['warm_steps']<2:
        raise ValueError('Collect real replay before critic warmup.')
    if i.get('reward_objective', 'paper-reward-v1') not in ('paper-reward-v1', 'capped-time-v1'):
        raise ValueError('Unknown training reward objective.')
    if i.get('reward_objective') == 'capped-time-v1' and (
            config['rl']['GAMMA'] != 1. or i.get('entropy_objective') != 'task-return-v4'):
        raise ValueError('Capped-time training requires undiscounted task returns without entropy rewards.')
    if i.get('gate_objective', 'critic-v1') not in ('critic-v1', 'paired-improvement-v1'):
        raise ValueError('Unknown gate objective.')
    if i.get('intervention_scope', 'single') not in ('single', 'team'):
        raise ValueError('Unknown gate intervention scope.')
    if i.get('intervention_scope') == 'team' and i.get('gate_objective') != 'paired-improvement-v1':
        raise ValueError('Team intervention is an explicit direct-gate exploration variant.')
    if i.get('gate_objective') == 'paired-improvement-v1':
        if (i['auxiliary'] != 'difference' or i['horizon_steps'] < 300 or
                not i.get('deterministic_rollouts') or i.get('reward_objective') != 'capped-time-v1'):
            raise ValueError('Direct gate improvement requires complete paired deployment-policy task returns.')
        if not 1 <= i.get('gate_updates_per_batch', 64) <= i['intervention_interval']:
            raise ValueError('Invalid number of gate updates per intervention batch.')
    if config['agent']['defender_num'] < 2:
        raise ValueError('Candidate interaction requires at least two defenders.')


def recondition_bootstrap(output, device='cpu', updates=10000, workers=4, *, isolate=False):
    """Repair a development checkpoint only after a fixed-policy heldout audit.

    The actor, replay, simulator, environment-step counter and training RNG are
    preserved. A fresh teacher is fitted to real replay, never to model labels.
    This is explicitly charged development recovery, not a fresh-run comparison.
    """
    from interaction_review import bootstrap_calibration, source_for_seed, digest
    from interaction_rollout import frozen_payload
    output = Path(output)
    study = output.parents[2]
    checkpoint = output/'resume.pth'
    fingerprint = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = copy.deepcopy(saved['config'])
    if config['training']['seed'] not in (42,101) or config['interaction']['arm'] == 'arboids_cbf':
        raise ValueError('Only development joint-critic checkpoints may be reconditioned.')
    if isolate and (config['interaction']['arm'] not in ('no_peer','model_value') or
                    saved['step']>250000 or saved['agent']['stage']!='gate'):
        raise ValueError('Exact evaluator isolation is limited to the existing development controls.')
    if config['interaction'].get('bootstrap_source','coupled') != 'coupled':
        raise ValueError('This checkpoint already uses the independent bootstrap protocol.')
    if updates <= 0:
        raise ValueError('A fixed positive recovery budget is required.')
    before = dict(config=config, actor=saved['agent']['actor'], critic=saved['agent']['critic'],
        target=saved['agent']['target'], stage=saved['agent']['stage'],
        alpha=float(saved['agent']['log_alpha'].exp()), gamma=config['rl']['GAMMA'])
    config = copy.deepcopy(config)
    config['interaction']['bootstrap_source'] = 'real_td'
    validate(config)
    started = time.perf_counter()
    same_info = config['interaction']['auxiliary'] == 'none'
    with preserved_random_state():
        seed_random(401000000+config['training']['seed'])
        agent = make_agent(config, device)
        agent.actor.load_state_dict(saved['agent']['actor'])
        agent.actor_optimizer.load_state_dict(saved['agent']['actor_optimizer'])
        agent.alpha_optimizer.load_state_dict(saved['agent']['alpha_optimizer'])
        agent.log_alpha.data.copy_(saved['agent']['log_alpha'].to(agent.device))
        agent.set_stage(saved['agent']['stage'])
        # CPU/CUDA exp can differ by one ulp. Both audit branches use the
        # unchanged temperature evaluated on the learner's actual device.
        before['alpha'] = float(agent.alpha)
        if same_info or isolate:
            # Preserve the complete current value function while separating future
            # real-TD updates from auxiliary gradients. This is no claim of repair.
            for name in ('critic','bootstrap_critic'):
                getattr(agent,name).load_state_dict(saved['agent']['critic'])
            for name in ('critic_optimizer','bootstrap_optimizer'):
                getattr(agent,name).load_state_dict(copy.deepcopy(saved['agent']['critic_optimizer']))
            agent.target.load_state_dict(saved['agent']['target'])
            actual_updates, checks = 0, saved.get('learning_checks',{})
        else:
            replay = JointReplay(1)
            replay.load_state_dict(saved['replay'])
            for count in range(1,updates+1):
                checks = agent.learn_bootstrap(replay)
                if count % 1000 == 0 or count == updates:
                    print('[IA-BOOTSTRAP] '+json.dumps(dict(update=count, **checks)),flush=True)
            agent.initialize_critic_from_bootstrap()
            del replay
            actual_updates = updates
        learning_seconds = time.perf_counter()-started
        after = frozen_payload(agent, online_critic=True)
        after['target'] = {k:v.detach().cpu().clone() for k,v in agent.target.state_dict().items()}
        calibration = bootstrap_calibration(dict(before=before, repaired=after), source_for_seed(study,config['training']['seed']),
            scene_base=400000000+(config['training']['seed']-42)*10000, states=64, repetitions=2, workers=workers)
        a,b = (calibration['summary'][k]['all'] for k in ('before','repaired'))
        identity_split = all(torch.equal(before[name][k],after[name][k])
                             for name in ('actor','critic','target') for k in before[name])
        finite = all(torch.isfinite(v).all().item() for name in ('actor','critic','target') for v in after[name].values())
        passed = (identity_split and finite and calibration['summary']['before']==calibration['summary']['repaired']) if (same_info or isolate) else (b['horizon_100_label_mae'] <= .5*a['horizon_100_label_mae'] and
            b['tail_value_mae'] is not None and b['tail_value_mae'] <= .5*a['tail_value_mae'] and
            b['environment_mae'] <= max(a['environment_mae'],1.05*b['zero_prediction_mae']))
        record = dict(protocol='real-transition-only-bootstrap-v1',complete=False,
            input=dict(seed=config['training']['seed'],arm=config['interaction']['arm']),
            previous_checkpoint_sha256=fingerprint,
            step=saved['step'],actor_unchanged=all(torch.equal(v,after['actor'][k]) for k,v in before['actor'].items()),
            additional_environment_steps=0,additional_model_training_steps=0,bootstrap_updates=actual_updates,
            learning_wall_seconds=learning_seconds,calibration=calibration,passed=bool(passed),
            acceptance='At the fixed recovery endpoint, halve H100 label and tail-value errors; root delta error must not worsen. '
                       'Same-info is a numerically identical teacher split, without reconditioning updates.')
        record_path=study/f'reviews/bootstrap-repair/{config["training"]["seed"]}-{config["interaction"]["arm"]}/completed.json'
        if isolate:
            record.update(protocol='exact-development-bootstrap-isolation-v1',
                acceptance='Actor, critic, target and fixed-policy calibration must be identical and finite; only future bootstrap updates are isolated. This does not establish improved accuracy or task performance.')
        if record_path.exists():
            prior=json.loads(record_path.read_text())
            if prior.get('complete'):
                raise RuntimeError('A completed recovery cannot be silently replaced.')
            record['prior_attempts']=prior.pop('prior_attempts',[])+[prior]
        atomic_json(record_path,record)
        if not passed or not record['actor_unchanged']:
            raise RuntimeError('Bootstrap reconditioning failed heldout calibration; the checkpoint is unchanged.')
        if digest(checkpoint) != fingerprint:
            raise RuntimeError('The checkpoint changed during reconditioning; refusing to overwrite it.')
        saved.update(config=config,agent=agent.state_dict(),auxiliary=None,learning_checks=checks,
            bootstrap_updates=actual_updates,bootstrap_repair=record,
            elapsed=saved['elapsed']+learning_seconds)
        atomic_torch(checkpoint,saved)
        export_policy(agent,output,saved['step'])
        export_probe(agent,output,saved['step'],checks)
        (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        atomic_json(output/'progress.json',dict(step=saved['step'],total=config['training']['gate_steps']+config['training']['joint_steps'],
            stage=agent.stage,simulated_steps=saved['simulated_steps'],elapsed=saved['elapsed'],complete=False,
            updates=max(0,saved['step']-config['training']['warm_steps']+1),bootstrap_updates=actual_updates,learning_checks=checks))
        record.update(complete=True,repaired_probe_sha256=digest(output/f'probe-{saved["step"]}.pth'))
        atomic_json(record_path,record)
    return dict(output=str(output),step=saved['step'],bootstrap_updates=actual_updates,calibration=calibration['summary'],passed=True)


def run_training(config, exp=None, device='cpu', *, output=None, max_steps=None, attacker=None,
                 stop_after_stage=None, stop_at_step=None):
    config = copy.deepcopy(config)
    validate(config)
    torch.set_num_threads(1)
    t, interaction = config['training'], config['interaction']
    warmup_kind = t.get('critic_warmup_target', 'real_td')
    if warmup_kind not in ('real_td', 'complete_real_return', 'complete_real_state_return'):
        raise ValueError('Unknown critic warmup target.')
    warmup_from_returns = warmup_kind in ('complete_real_return', 'complete_real_state_return')
    if warmup_from_returns and (not t.get('critic_warmup_updates') or
            t['warm_steps'] > t['replay_capacity'] or interaction['arm'] == 'arboids_cbf'):
        raise ValueError('Complete-return warmup requires a fixed pre-actor budget and intact joint replay.')
    output = Path(output if output is not None else exp.exp_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(t['resume']) if t.get('resume') else output / 'resume.pth'
    if resume.exists():
        # Full resumes contain our own simulator and RNG objects; deployment uses weights_only=True.
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved.get('format') != 'interaction-sac-resume-v1':
            raise ValueError('Unsupported resume format.')
        old, new = copy.deepcopy(saved['config']), copy.deepcopy(config)
        for c in (old, new):
            for key in ('resume', 'pretrain_checkpoint'):
                c['training'].pop(key, None)
        if old != new:
            raise ValueError('Resume configuration differs from the saved scientific settings.')
    else:
        saved = None
        if (output / 'metrics.csv').exists():
            raise RuntimeError('Existing metrics have no recoverable checkpoint.')
    seed_random(int(t['seed']))
    agent = make_agent(config, device)
    replay, pool = JointReplay(t['replay_capacity']), SnapshotPool(interaction['snapshot_capacity'])
    step, simulated, episodes, prior_elapsed, auxiliary, losses = 0, 0, 0, 0., None, {}
    bootstrap_updates, bootstrap_repair, actor_updates, critic_updates = 0, None, 0, 0
    critic_warmup_updates = 0
    critic_warmup_complete = not t.get('critic_warmup_updates',0)
    env, packet, done = None, None, True
    if saved is not None:
        agent.load_state_dict(saved['agent'])
        replay.load_state_dict(saved['replay'])
        pool = saved['pool']
        step, simulated, episodes = saved['step'], saved['simulated_steps'], saved['episodes']
        prior_elapsed, auxiliary, done = saved['elapsed'], saved['auxiliary'], saved['done']
        losses = saved.get('learning_checks', {})
        bootstrap_updates, bootstrap_repair = saved.get('bootstrap_updates',0), saved.get('bootstrap_repair')
        actor_updates = saved.get('actor_updates', max(0, step - t['warm_steps'] + 1))
        critic_updates = saved.get('critic_updates', max(0, step - t['warm_steps'] + 1))
        critic_warmup_updates = saved.get('critic_warmup_updates',0)
        critic_warmup_complete = saved.get('critic_warmup_complete',not t.get('critic_warmup_updates',0))
        if saved['snapshot'] is not None:
            env = copy.deepcopy(saved['snapshot'].environment)
            packet = public_packet(env)
        saved['random'].restore()
        del saved
        metrics_path = output / 'metrics.csv'
        if metrics_path.exists():
            with metrics_path.open(encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [row for row in reader if int(row['step']) <= step]
            with metrics_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    elif t.get('initial_state'):
        initial = torch.load(t['initial_state'], map_location='cpu', weights_only=False)
        if initial['config']['interaction']['arm'] != interaction['arm']:
            raise ValueError('Cannot change study arms between adversarial phases.')
        agent.load_state_dict(initial['agent'])
        agent.set_stage('joint' if t['gate_steps'] == 0 else 'gate')
        del initial
    elif t.get('pretrain_checkpoint'):
        weights = torch.load(t['pretrain_checkpoint'], map_location='cpu', weights_only=True)
        if agent.legacy:
            agent.actor.load_state_dict(weights, strict=True)
        else:
            agent.actor.initialize_source(weights)
    elif not t.get('allow_random_initialization', False):
        raise ValueError('A verified common pretraining checkpoint is required.')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    started = time.perf_counter()
    total = t['gate_steps'] + t['joint_steps']
    stop_at = total if max_steps is None else min(total, step + int(max_steps))
    if stop_at_step is not None:
        if not 0 < stop_at_step <= total or step > stop_at_step:
            raise ValueError('The absolute development checkpoint is outside the remaining training budget.')
        stop_at = min(stop_at, int(stop_at_step))
    if stop_after_stage not in (None, 'gate', 'joint'):
        raise ValueError('Unknown absolute training-stage boundary.')
    if stop_after_stage == 'gate':
        if not t['gate_steps'] or step > t['gate_steps']:
            raise ValueError('The gate endpoint cannot be reconstructed after joint learning.')
        stop_at = min(stop_at, t['gate_steps'])
    if step in (t['gate_steps'], total) and step > 0 and not (output / (agent.stage + '-endpoint.pth')).exists():
        export_policy(agent, output, step, endpoint=True)
    sampler = InterventionSampler(interaction['workers'])
    metrics_fields = ['step', 'stage', 'elapsed', 'simulated_steps', 'success', 'capture', 'collision',
                      'breach', 'timeout', 'capture_time']

    def checkpoint():
        elapsed = prior_elapsed + time.perf_counter() - started
        atomic_torch(output / 'resume.pth', dict(format='interaction-sac-resume-v1', config=config,
            agent=agent.state_dict(), replay=replay.state_dict(), pool=pool, step=step, simulated_steps=simulated,
            episodes=episodes, elapsed=elapsed, auxiliary=auxiliary, done=bool(done),
            snapshot=None if env is None else compact_snapshot(env), random=RandomState.capture(),
            learning_checks=losses,bootstrap_updates=bootstrap_updates,bootstrap_repair=bootstrap_repair,
            actor_updates=actor_updates, critic_updates=critic_updates,
            critic_warmup_updates=critic_warmup_updates,critic_warmup_complete=critic_warmup_complete))
        export_policy(agent, output, step)
        if step in (t['gate_steps'], total):
            export_policy(agent, output, step, endpoint=True)
        if (stop_at_step is not None and step == stop_at_step and not agent.legacy and
                not (output/f'probe-{step}.pth').exists()):
            export_probe(agent, output, step, losses)
        atomic_json(output / 'progress.json', dict(step=step, total=total, stage=agent.stage,
                    simulated_steps=simulated, elapsed=elapsed, complete=step == total,
                    updates=max(actor_updates, critic_updates), bootstrap_updates=bootstrap_updates,
                    actor_updates=actor_updates, critic_updates=critic_updates,
                    critic_warmup_updates=critic_warmup_updates, learning_checks=losses))

    try:
        while step < stop_at:
            stage = 'gate' if step < t['gate_steps'] else 'joint'
            if agent.stage != stage:
                agent.set_stage(stage)
                auxiliary = None
            if done:
                env = TADEnv(config['agent']['defender_num'], LearningSide='Att' if attacker else 'Def',
                             **config['environment'])
                agility = t.get('fixed_agility', 2.25 if stage == 'gate' else
                    2. + .25 * min(3, (step - t['gate_steps']) // t.get('curriculum_interval', 250000)))
                with preserved_random_state():
                    np.random.seed(100000000 + int(t['seed']) * 10000 + episodes)
                    obs, _ = env.reset(agility, noisy_agility=True)
                packet, done = public_packet(env, obs), False
                episodes += 1
            warmup_target=t.get('critic_warmup_updates',0)
            if step+1>=t['warm_steps'] and not critic_warmup_complete:
                # Finish before collecting the next transition, so interruption
                # and resume retain the same replay and policy-update boundary.
                real_returns = replay.complete_returns(agent.gamma) if warmup_from_returns else None
                while critic_warmup_updates<warmup_target:
                    with preserved_random_state():
                        seed_random(709000000+int(t['seed'])*10000+critic_warmup_updates)
                        losses=(agent.learn_bootstrap_returns(replay,real_returns) if real_returns is not None
                                else agent.learn_bootstrap(replay))
                    critic_warmup_updates+=1;bootstrap_updates+=1
                    if critic_warmup_updates%1000==0 or critic_warmup_updates==warmup_target:
                        print('[IA-CRITIC-WARMUP] '+json.dumps(dict(updates=critic_warmup_updates,**losses)),flush=True)
                agent.initialize_critic_from_bootstrap()
                if real_returns is not None:
                    agent.target.load_state_dict(agent.bootstrap_critic.state_dict())
                auxiliary=None;critic_warmup_complete=True
            if step % interaction['snapshot_interval'] == 0 and interaction['auxiliary'] != 'none':
                pool.add(env)
            action, logp = agent.choose_action(packet)
            attacker_action = None
            if attacker is not None:
                _, att_obs = env._get_obs()
                attacker_action = attacker(att_obs)
            following, reward, outcome, thrust, _, _ = execute(env, packet, action, attacker_action=attacker_action,
                reward_objective=interaction.get('reward_objective', 'paper-reward-v1'))
            replay.store(packet, action, thrust, reward, following, bool(outcome),
                policy_cost=float(agent.alpha)*logp if warmup_from_returns else None)
            packet, done, step = following, bool(outcome), step + 1
            if (interaction['auxiliary'] != 'none' and step % interaction['intervention_interval'] == 0
                    and (not warmup_target or critic_warmup_updates==warmup_target)):
                auxiliary, used = sampler.generate(agent, pool, step=step,
                    attacker=None if attacker is None else attacker.state_dict())
                simulated += used
            if step >= t['warm_steps']:
                if agent.legacy:
                    losses = agent.learn(replay, auxiliary)
                else:
                    losses = agent.learn(replay, auxiliary,
                        diagnostics=step % t['log_interval'] == 0 or step == stop_at,
                        update_actor=(stage != 'gate' or interaction.get('gate_objective') != 'paired-improvement-v1'
                            or step % interaction['intervention_interval'] < interaction.get('gate_updates_per_batch', 64)))
                    bootstrap_updates += int(losses.get('bootstrap_updated', agent.bootstrap_critic is not None))
                actor_updates += int(losses.get('actor_updated', True))
                critic_updates += int(losses.get('critic_updated', True))
            if step % t['eval_interval'] == 0 or step in (t['gate_steps'], total):
                scores = evaluate(agent, t['eval_episodes'], 310000000 + int(t['seed']) * 10000,
                                  config['agent']['defender_num'], attacker=attacker)
                row = dict(step=step, stage=stage, elapsed=prior_elapsed + time.perf_counter()-started,
                           simulated_steps=simulated, **scores)
                exists = (output / 'metrics.csv').exists()
                with (output / 'metrics.csv').open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=metrics_fields)
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                print('[IA-EVAL] ' + json.dumps(row), flush=True)
            if step % t['log_interval'] == 0:
                status = dict(step=step, total=total, stage=stage, simulated_steps=simulated,
                              elapsed=prior_elapsed + time.perf_counter()-started, complete=False,
                              bootstrap_updates=bootstrap_updates, actor_updates=actor_updates,
                              critic_updates=critic_updates, **losses)
                atomic_json(output / 'progress.json', status)
                print('[IA-TRAIN] ' + json.dumps(status), flush=True)
            if step % t['checkpoint_interval'] == 0 or step in (stop_at, t['gate_steps']):
                checkpoint()
        if (stop_at_step is not None and step == stop_at_step and not agent.legacy and
                not (output/f'probe-{step}.pth').exists()):
            export_probe(agent, output, step, losses)
        return dict(output=str(output), step=step, complete=step == total)
    except KeyboardInterrupt:
        checkpoint()
        raise
    finally:
        sampler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/interaction-aware-sac.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrain', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--stop-after-stage', choices=['gate', 'joint'])
    parser.add_argument('--stop-at-step', type=int)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--repair-bootstrap', action='store_true')
    parser.add_argument('--recondition-bootstrap-if-needed', action='store_true')
    parser.add_argument('--isolate-bootstrap-if-needed', action='store_true')
    parser.add_argument('--critic-control-coordinates', choices=['raw-v1','nominal-thrust-v2'])
    parser.add_argument('--entropy-objective', choices=['joint-sum-v1','stage-mean-v2','proposal-mean-v3','task-return-v4'])
    parser.add_argument('--reward-objective', choices=['paper-reward-v1','capped-time-v1'])
    parser.add_argument('--gate-objective', choices=['critic-v1','paired-improvement-v1'])
    parser.add_argument('--intervention-scope', choices=['single','team'])
    parser.add_argument('--defenders', type=int)
    parser.add_argument('--bootstrap-estimator', choices=['min','mean'])
    parser.add_argument('--critic-warmup-target', choices=['real_td','complete_real_return','complete_real_state_return'])
    parser.add_argument('--policy-learning-rate', type=float)
    parser.add_argument('--long-horizon-steps', type=int)
    parser.add_argument('--label-repetitions', type=int)
    parser.add_argument('--critic-warmup-updates', type=int)
    parser.add_argument('--warm-steps', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.repair_bootstrap:
        # The scheduler and all owned training jobs must already be stopped.
        import fcntl
        study=args.output.resolve().parents[2]
        with (study/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status=json.loads((study/'status.json').read_text())
            if status.get('active'):
                raise RuntimeError('Stop all training jobs before bootstrap reconditioning.')
            print(json.dumps(recondition_bootstrap(args.output,args.device)),flush=True)
        return
    config = arm_config(yaml.safe_load(args.config.read_text(encoding='utf-8')), args.arm)
    config['training'].update(seed=args.seed, pretrain_checkpoint=None if args.pretrain is None else str(args.pretrain))
    if args.critic_control_coordinates is not None:
        config['interaction']['critic_control_coordinates']=args.critic_control_coordinates
    if args.entropy_objective is not None:
        config['interaction']['entropy_objective']=args.entropy_objective
    if args.reward_objective is not None:
        config['interaction']['reward_objective'] = args.reward_objective
        if args.reward_objective == 'capped-time-v1':
            config['rl']['GAMMA'] = 1.
            config['interaction']['entropy_objective'] = 'task-return-v4'
    if args.gate_objective is not None:
        config['interaction']['gate_objective'] = args.gate_objective
        if args.gate_objective == 'paired-improvement-v1':
            config['interaction'].update(deterministic_rollouts=True, gate_updates_per_batch=64,
                                          gate_anchor_weight=.05)
    if args.defenders is not None:
        config['agent']['defender_num'] = args.defenders
    if args.intervention_scope is not None:
        config['interaction']['intervention_scope'] = args.intervention_scope
    if args.bootstrap_estimator is not None:
        config['interaction']['bootstrap_estimator']=args.bootstrap_estimator
    if args.critic_warmup_target is not None:
        config['training']['critic_warmup_target']=args.critic_warmup_target
    if args.policy_learning_rate is not None:
        if args.policy_learning_rate <= 0: parser.error('Policy learning rate must be positive.')
        config['rl'].update(actor_learning_rate=args.policy_learning_rate,temperature_learning_rate=args.policy_learning_rate)
    if args.long_horizon_steps is not None:
        if args.long_horizon_steps <= 0: parser.error('Long horizon must be positive.')
        if args.arm != 'short': config['interaction']['horizon_steps']=args.long_horizon_steps
    if args.label_repetitions is not None:
        config['interaction']['label_repetitions']=args.label_repetitions
    if args.critic_warmup_updates is not None:
        config['training']['critic_warmup_updates']=args.critic_warmup_updates
    if args.warm_steps is not None:
        config['training']['warm_steps'] = args.warm_steps
    if args.formal:
        # The new confirmatory budget is one million additional real transitions in total.
        config['training'].update(joint_steps=750000, curriculum_interval=187500)
    if args.smoke:
        config['training'].update(gate_steps=20, joint_steps=20, warm_steps=8, replay_capacity=128,
            eval_interval=20, eval_episodes=2, checkpoint_interval=20, log_interval=10,
            allow_random_initialization=args.pretrain is None)
        config['rl']['batch_size'] = 16
        config['interaction'].update(workers=0, pairs_per_batch=2, intervention_interval=10,
                                     horizon_steps=2, snapshot_interval=2)
    if args.recondition_bootstrap_if_needed and args.isolate_bootstrap_if_needed:
        parser.error('Choose either evaluator rebuilding or exact isolation.')
    if (args.recondition_bootstrap_if_needed or args.isolate_bootstrap_if_needed) and (args.output/'resume.pth').exists():
        previous=yaml.safe_load((args.output/'config.yaml').read_text(encoding='utf-8'))
        if previous['interaction'].get('bootstrap_source','coupled') == 'coupled':
            recondition_bootstrap(args.output,args.device,isolate=args.isolate_bootstrap_if_needed)
    print(json.dumps(run_training(config, device=args.device, output=args.output, max_steps=args.max_steps,
                                  stop_after_stage=args.stop_after_stage, stop_at_step=args.stop_at_step)), flush=True)


if __name__ == '__main__':
    main()
