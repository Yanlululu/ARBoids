"""Frozen-policy candidate-message stress tests and measured CPU inference cost."""
import argparse
from collections import deque
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from evidence_eval import ENVIRONMENT, PhysicalPolicy, sha256
from evidence_stats import paired_fixed, holm
from RL.control import fuse_thrust, to_thrust
from RL.guidance import preserve_rng
from RL.observations import tensor_frame
from train_mappo import make_env, minimum_spacing, summarize


CONDITIONS = (('ideal', 0., 0), ('drop-0.1', .1, 0), ('drop-0.3', .3, 0),
              ('drop-0.5', .5, 0), ('message-delay-1', 0., 1), ('message-delay-2', 0., 2))


class CandidateLink:
    """Independent directed packet loss, last-packet hold, explicit packet age."""
    def __init__(self, drop, delay, seed, period=.2):
        self.drop, self.delay, self.period = drop, delay, period
        self.rng = np.random.default_rng(int(seed) + 512191)
        self.queue = deque()
        self.features, self.age = None, None

    def transmit(self, fresh):
        if self.features is None:
            self.features = torch.zeros_like(fresh[..., :4])
            self.age = torch.zeros_like(fresh[..., 4:5])
        self.age = self.age + self.period
        self.queue.append(fresh[..., :4].clone())
        if len(self.queue) > self.delay:
            arriving = self.queue.popleft()
            received = torch.as_tensor(self.rng.random(fresh.shape[:-1]) >= self.drop,
                                       device=fresh.device).unsqueeze(-1)
            self.features = torch.where(received, arriving, self.features)
            self.age = torch.where(received, torch.full_like(self.age, self.delay * self.period), self.age)
        return torch.cat((self.features, self.age), -1)


@torch.no_grad()
def evaluate_messages(policy, seeds, drop=0., delay=0, duration=60.):
    if policy.kind != 'channel' or not 0 <= drop <= 1 or delay < 0:
        raise ValueError('A channel actor and valid link parameters are required')
    rows = []
    with preserve_rng():
        for seed in seeds:
            np.random.seed(int(seed))
            env = make_env(dict(environment=dict(ENVIRONMENT, total_time=duration)))
            obs, _ = env.reset(2., noisy_agility=False)
            initial = np.r_[env.attacker.pos, env.attacker.theta,
                            np.asarray([d.pos for d in env.defender_list]).ravel(),
                            [d.theta for d in env.defender_list]]
            link = CandidateLink(drop, delay, seed, env.Action_T)
            reward, energy, steps, spacing = 0., 0., 0, minimum_spacing(env)
            while True:
                frame = tensor_frame(env.structured_frame(policy.motion_features), policy.device)
                _, candidate, _ = policy.actor.propose(frame, True)
                messages = link.transmit(policy.actor.exchange(frame, candidate))
                dist, _ = policy.actor.gate_distribution(frame, candidate, messages)
                command = fuse_thrust(frame['boids'], to_thrust(candidate), torch.sigmoid(dist.mean),
                                      policy.actor.fusion)[0].cpu().numpy()
                obs, r, outcome, info = env.step_thrust(command)
                if not np.isfinite(command).all() or not np.isfinite(r).all():
                    raise FloatingPointError('Non-finite communication-stress evaluation')
                steps += 1
                reward += float(np.mean(r))
                energy += float(np.mean(command ** 2)) * env.Action_T
                spacing = min(spacing, minimum_spacing(env))
                if steps > int(np.ceil(duration / env.Action_T)) + 1:
                    raise RuntimeError('Message evaluation exceeded its horizon')
                if outcome:
                    break
            rows.append(dict(seed=int(seed), outcome_code=int(outcome), success=int(outcome in (3, 4)),
                             reward=reward, steps=steps, time_seconds=env.Current_T, min_spacing=spacing,
                             thrust_squared_integral=energy, initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
    return dict(passed=True, policy=policy.metadata(), drop_probability=drop, message_delay_cycles=delay,
                rows=rows, summary=summarize(rows), environment=dict(ENVIRONMENT, total_time=duration))


@torch.no_grad()
def benchmark(policy, defenders, repeats=1000):
    with preserve_rng():
        np.random.seed(1981000)
        env = make_env(dict(environment=ENVIRONMENT), defenders=defenders)
        obs, _ = env.reset(2., noisy_agility=False)
        frame = env.structured_frame(policy.motion_features)
        for _ in range(100):
            policy.act([frame], [obs])
        times = np.empty(repeats)
        for i in range(repeats):
            tick = time.perf_counter_ns()
            policy.act([frame], [obs])
            times[i] = (time.perf_counter_ns()-tick) / 1e6
    return dict(defenders=defenders, repeats=repeats, warmup_calls=100,
                mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, .95)), p99_ms=float(np.quantile(times, .99)),
                scope='One team actor call including frame tensor conversion; excludes sensing, simulator and network latency')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(1)
    policies = dict(channel=PhysicalPolicy('channel', root / 'inputs/channel-frozen.pth'),
                    reference=PhysicalPolicy('arboids', root / 'inputs/arboids-reference.pth'))
    out = root / 'results/deployment'
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, drop, delay in CONDITIONS:
        path = out / f'{name}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if (not result['passed'] or result['policy'] != policies['channel'].metadata()
                    or result['evaluator_sha256'] != sha256(__file__)):
                raise RuntimeError(f'Cached deployment evaluation changed: {path}')
        else:
            result = evaluate_messages(policies['channel'], range(9060000, 9060256), drop, delay)
            result['evaluator_sha256'] = sha256(__file__)
            path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        data[name] = result
        print(json.dumps(dict(condition=name, **result['summary'])), flush=True)
    comparisons, values = {}, []
    for name, _, _ in CONDITIONS[1:]:
        comparisons[name] = {}
        for metric, outcome in (('success', None), ('collision', 2), ('breach', 1)):
            a, b = data[name]['rows'], data['ideal']['rows']
            if outcome is None:
                a, b = [[dict(r, outcome_code=r['success']) for r in rows] for rows in (a, b)]
            result = paired_fixed(a, b, 1 if outcome is None else outcome)
            comparisons[name][metric] = result
            values.append(result)
    for value, p in zip(values, holm([r['exact_mcnemar_two_sided_p'] for r in values])):
        value['holm_15_exploratory_p'] = p
    report = dict(passed=True, source_sha256=sha256(__file__), conditions={k:v['summary'] for k,v in data.items()},
                  comparisons=comparisons, policies={k:p.metadata() for k,p in policies.items()},
                  latency={k:[benchmark(p, n) for n in (3, 8)] for k,p in policies.items()},
                  hardware=dict(platform=platform.platform(), processor=platform.processor(), torch=str(torch.__version__),
                                torch_threads=torch.get_num_threads(), shared_host=True),
                  communication=dict(candidate_floats_per_sender=4, bytes_per_float=4, action_hz=5,
                                     broadcast_payload_bytes_per_second='16 * N * 5',
                                     unicast_payload_bytes_per_second='16 * N * (N - 1) * 5',
                                     age='Computed locally in seconds since packet generation',
                                     scope='Candidate payload only; excludes headers, retransmission and observation/state sharing'))
    (root / 'artifacts/deployment-evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Frozen-policy deployment evidence', '',
             '| Candidate-message condition | Success / 256 | Collision / 256 | Target breach / 256 |',
             '|---|---:|---:|---:|']
    for name, s in report['conditions'].items():
        lines.append(f'| {name} | {s["successes"]} | {s["collisions"]} | {s["breaches"]} |')
    lines += ['', 'Loss holds the last received packet; unavailable initial packets use zero features. '
              'Age is measured in seconds. Geometry observations remain current. The policy was not retrained on corruption.', '',
              '| Policy | Team size | Median actor call (ms) | p95 (ms) | p99 (ms) |', '|---|---:|---:|---:|---:|']
    for arm, benchmarks in report['latency'].items():
        for r in benchmarks:
            lines.append(f'| {arm} | {r["defenders"]} | {r["median_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["p99_ms"]:.3f} |')
    lines += ['', 'CPU measurements were made on a shared host and exclude sensing, networking and simulator steps. '
              'They do not measure an end-to-end real-time deadline.',
              'A 3-vessel team sends 240 bytes/s of ideal candidate broadcast payload, or 480 bytes/s as directed unicasts; '
              'packet headers and the existing state-observation channel are additional costs.', '']
    (root / 'artifacts/deployment-evidence.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
