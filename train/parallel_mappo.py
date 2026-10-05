"""Persistent CPU rollout workers; a single learner owns all PPO updates."""
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp
import random

import numpy as np
import torch

from policy.mappo import PredictiveMAPPO
from policy.rollout_buffer import RolloutBuffer
from train_mappo import make_env, play_episode, continue_episode, play_episodes_batched


_agent = _environment = _config = None


def _initialize(config):
    global _agent, _environment, _config
    torch.set_num_threads(1)
    _agent = PredictiveMAPPO(config, 'cpu')
    _environment = make_env(config)
    _config = copy.deepcopy(config)


def _load_policy(config, actor, critic):
    # State dicts do not include Python policy settings, e.g. the inherited
    # logit bound. A reference policy must run with its own full configuration.
    if config != _config:
        _initialize(config)
    _agent.actor.load_state_dict({k: torch.as_tensor(v) for k, v in actor.items()})
    _agent.critic.load_state_dict({k: torch.as_tensor(v) for k, v in critic.items()})


def _episodes(arguments):
    config, actor, critic, seeds, agility, noisy, deterministic, retain, batch_size = arguments
    _load_policy(config, actor, critic)
    if batch_size > 1 and not deterministic:
        output = play_episodes_batched(_agent, seeds, agility, noisy, batch_size)
        return [(transitions if retain else None, summary) for transitions, summary in output]
    output = []
    for seed in seeds:
        np.random.seed(int(seed))
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        transitions, summary = play_episode(_agent, _environment, agility, noisy, deterministic)
        summary['seed'] = int(seed)
        output.append((transitions if retain else None, summary))
    return output


def collision_branch_cases(rollout, budget, lookback, seed, max_cases=None):
    """Select fresh training failures, never validation/audit scenarios."""
    if budget < 0 or lookback < 1:
        raise ValueError('Branch count must be nonnegative and lookback positive.')
    if max_cases is not None and max_cases < 1:
        raise ValueError('The maximum number of independent start states must be positive.')
    failures = [(episode, summary) for episode, summary in zip(rollout.episodes, rollout.summaries)
                if summary['collision'] and not summary.get('curriculum', False)]
    if not failures or not budget:
        return []
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(failures))[:min(budget, max_cases or budget)]
    cases = []
    for index in chosen:
        episode, summary = failures[index]
        if summary['seed'] < 100000:
            raise ValueError('Curriculum sources must be training seeds, outside reserved evaluation blocks.')
        start = max(0, len(episode) - lookback)
        cases.append(dict(source_seed=summary['seed'], source_step=start,
                          prefix=[transition['executed_action'].copy() for transition in episode[:start]],
                          state=episode[start]['state'].copy(), obs=episode[start]['obs'].copy(), seeds=[]))
    for index in range(budget):
        cases[index % len(cases)]['seeds'].append(int(rng.integers(100000, 2**31-1)))
    return cases


def restore_branch_start(env, case, agility, noisy):
    # The APF attacker is deterministic; all environmental randomness is NumPy
    # current/reset noise. Stored executed actions reproduce the prefix exactly.
    np.random.seed(case['source_seed'])
    obs = env.reset(agility, noisy)
    for action in case['prefix']:
        transition = env.step(action)
        if transition.terminated:
            raise ValueError('A branch prefix reached a terminal event.')
        obs = transition.observation
    if (not np.array_equal(env.env.centralized_state(), case['state']) or
            not np.array_equal(np.asarray(obs, dtype=np.float32), case['obs'])):
        raise ValueError('Prefix replay did not reproduce the collected training state.')
    return obs


def breach_branch_cases(agent, rollout, budget, seed, max_cases=4, lead_steps=5, intervention_threshold=.05):
    """Rewind breaches before the first effective predictive intervention.

    Late pre-breach states can already be unrecoverable. Use actual executed
    prefixes and current-policy raw messages; labels still come only from new
    independent continuations. Without a predictive residual, start at reset.
    """
    if budget < 0 or max_cases < 1 or lead_steps < 0 or intervention_threshold <= 0:
        raise ValueError('Invalid early breach recovery sampling settings.')
    failures = [(e, s) for e, s in zip(rollout.episodes, rollout.summaries)
                if s['breach'] and not s.get('curriculum', False)]
    if not failures or not budget:
        return []
    rng = np.random.default_rng(seed)
    cases = []
    for index in rng.permutation(len(failures))[:min(budget, max_cases)]:
        episode, summary = failures[index]
        if summary['seed'] < 100000:
            raise ValueError('Breach recovery must use training seeds, never evaluation scenarios.')
        start = 0
        if agent.settings.compatibility_prior:
            data = {key: agent.tensor(np.asarray([t[key] for t in episode]), key=='mask')
                    for key in ('obs', 'proposals', 'boids', 'edges', 'mask', 'gate_raw')}
            with torch.no_grad():
                distribution, _, anchor = agent.actor.gate_distribution(
                    data['obs'], data['proposals'], data['boids'], data['edges'], data['mask'], return_anchor=True)
                shift = distribution.mean - anchor
                change = (data['gate_raw'].clamp(0., 1.) - (data['gate_raw']-shift).clamp(0., 1.)).abs()
                changed = (change.flatten(1).max(-1).values > intervention_threshold).nonzero().flatten()
            if len(changed):
                start = max(0, int(changed[0])-lead_steps)
        cases.append(dict(source_seed=summary['seed'], source_step=start,
            prefix=[t['executed_action'].copy() for t in episode[:start]], state=episode[start]['state'].copy(),
            obs=episode[start]['obs'].copy(), seeds=[], recovery_kind='breach'))
    for i in range(budget):
        case = cases[i % len(cases)]
        future_seed = int(rng.integers(100000, 2**31-1))
        while future_seed in case['seeds']:
            future_seed = int(rng.integers(100000, 2**31-1))
        case['seeds'].append(future_seed)
    return cases


def _branches(arguments):
    config, actor, critic, cases, agility, noisy = arguments
    _load_policy(config, actor, critic)
    output, restore_steps = [], 0
    for case in cases:
        observation = restore_branch_start(_environment, case, agility, noisy)
        snapshot = copy.deepcopy(_environment)
        restore_steps += len(case['prefix'])
        for seed in case['seeds']:
            reference = None
            if config.get('training', {}).get('branch_paired_reference', False):
                # A separate deterministic continuation with the same exogenous
                # currents. Never feed its future outcome to the acting policy.
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                _, reference = continue_episode(_agent, copy.deepcopy(snapshot), observation.copy(),
                                                deterministic=True)
            # Independent futures from the same present-time state. No branch
            # is labelled with the source episode's later collision outcome.
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            transitions, summary = continue_episode(_agent, copy.deepcopy(snapshot), observation.copy(),
                gate_scale=config.get('training', {}).get('branch_gate_scale', 1.),
                gate_steps=config.get('training', {}).get('branch_gate_steps', 0))
            summary.update(seed=seed, curriculum=True, source_seed=case['source_seed'],
                           source_step=case['source_step'], gate_steps=config.get('training', {}).get('branch_gate_steps', 0),
                           recovery_kind=case.get('recovery_kind', 'collision'))
            if reference is not None:
                summary.update({f'paired_reference_{name}': reference[name] for name in
                    ('success', 'capture', 'timeout', 'breach', 'collision', 'task_return', 'steps')})
            output.append((transitions, summary))
    return output, restore_steps


def summarize(rows):
    metrics = {key + '_rate': float(np.mean([row[key] for row in rows]))
               for key in ('success', 'capture', 'timeout', 'breach', 'collision')}
    metrics['mean_task_return'] = float(np.mean([row['task_return'] for row in rows]))
    metrics['gate_mean'] = float(np.mean([row['gate_mean'] for row in rows]))
    return metrics


class ParallelRollouts:
    def __init__(self, config, workers=4, environments_per_worker=1):
        if min(workers, environments_per_worker) < 1:
            raise ValueError('workers and environments per worker must be positive')
        self.workers = workers
        self.environments_per_worker = environments_per_worker
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                        initializer=_initialize, initargs=(config,))

    def run(self, agent, seeds, agility=2., noisy=False, deterministic=False, retain=True):
        actor = {k: v.detach().cpu().numpy().copy() for k, v in agent.actor.state_dict().items()}
        critic = {k: v.detach().cpu().numpy().copy() for k, v in agent.critic.state_dict().items()}
        groups = [s.tolist() for s in np.array_split(np.asarray(seeds), self.workers) if len(s)]
        # Immutable copies: every episode in an update uses the same behavior policy.
        arguments = [(copy.deepcopy(agent.config), actor, critic, group, agility, noisy, deterministic, retain,
                      self.environments_per_worker)
                     for group in groups]
        episodes = [episode for group in self.pool.map(_episodes, arguments) for episode in group]
        if retain:
            rollout = RolloutBuffer()
            for transitions, summary in episodes:
                rollout.add_episode(transitions, summary)
            return rollout
        return [summary for _, summary in episodes]

    def resume_cases(self, agent, cases, agility=2., noisy=False):
        """Collect current-policy complete continuations; never replay old losses."""
        if not cases:
            return RolloutBuffer(), 0
        actor = {k: v.detach().cpu().numpy().copy() for k, v in agent.actor.state_dict().items()}
        critic = {k: v.detach().cpu().numpy().copy() for k, v in agent.critic.state_dict().items()}
        groups = [cases[index::self.workers] for index in range(self.workers)]
        arguments = [(copy.deepcopy(agent.config), actor, critic, group, agility, noisy)
                     for group in groups if group]
        rollout, restored_steps = RolloutBuffer(), 0
        for episodes, restored in self.pool.map(_branches, arguments):
            restored_steps += restored
            for transitions, summary in episodes:
                rollout.add_episode(transitions, summary)
        return rollout, restored_steps

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
