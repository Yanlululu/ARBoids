"""Analyze a completed, paired VRX cohort without treating timeouts as captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from evidence_eval import sha256
from evidence_stats import paired_fixed, holm


def analyze(path, output):
    data = json.loads(path.read_text(encoding='utf-8'))
    expected = data['protocol']['episodes']
    if not data['passed'] or expected != 100 or len(data['pairs']) != expected:
        raise ValueError('The fixed 100-pair VRX cohort has not completed')
    rows = {arm: [] for arm in ('channel', 'reference')}
    for pair in data['pairs']:
        if pair['channel']['initial_poses'] != pair['reference']['initial_poses']:
            raise ValueError('VRX initial conditions differ within a pair')
        initial = hashlib.sha256(json.dumps(pair['channel']['initial_poses'], sort_keys=True).encode()).hexdigest()
        for arm in rows:
            trial = pair[arm]
            if (not trial['passed'] or trial['seed'] != pair['seed'] or trial['outcome_code'] not in (1, 2, 3, 4)
                    or trial['checkpoint_sha256'] != data['protocol'][arm]['sha256']):
                raise ValueError(f'Invalid or changed VRX trial: {pair["seed"]}, {arm}')
            rows[arm].append(dict(seed=pair['seed'], outcome_code=trial['outcome_code'],
                                  success=int(trial['success']), time_seconds=trial['simulation_seconds'],
                                  initial_sha256=initial))
    comparisons = {}
    for metric, outcome in (('success', None), ('collision', 2), ('breach', 1)):
        a, b = rows['channel'], rows['reference']
        if outcome is None:
            a, b = [[dict(r, outcome_code=r['success']) for r in x] for x in (a, b)]
        result = paired_fixed(a, b, 1 if outcome is None else outcome)
        plus = binomtest(result['candidate_only'], expected).proportion_ci(confidence_level=.975)
        minus = binomtest(result['reference_only'], expected).proportion_ci(confidence_level=.975)
        result['conservative_paired_95_interval'] = [plus.low-minus.high, plus.high-minus.low]
        comparisons[metric] = result
    adjusted = holm([r['exact_mcnemar_two_sided_p'] for r in comparisons.values()])
    for result, p in zip(comparisons.values(), adjusted):
        result['holm_3_outcomes_p'] = p
    summary = {}
    for arm, values in rows.items():
        summary[arm] = dict(episodes=expected, successes=sum(r['success'] for r in values),
                            captures=sum(r['outcome_code'] == 3 for r in values),
                            timeout_denials=sum(r['outcome_code'] == 4 for r in values),
                            collisions=sum(r['outcome_code'] == 2 for r in values),
                            breaches=sum(r['outcome_code'] == 1 for r in values),
                            mean_episode_seconds=float(np.mean([r['time_seconds'] for r in values])))
    both_captured = [(a['time_seconds'], b['time_seconds']) for a, b in zip(rows['channel'], rows['reference'])
                     if a['outcome_code'] == b['outcome_code'] == 3]
    conditional_time = dict(pairs=len(both_captured), scope='Descriptive subset in which both policies captured; not an unconditional efficiency estimate')
    if both_captured:
        times = np.asarray(both_captured)
        conditional_time['mean_difference_seconds'] = float((times[:, 0] - times[:, 1]).mean())
    report = dict(passed=True, scope='Two frozen checkpoints, 100 paired VRX scenarios, three defenders, agility 2.25',
                  summary=summary, comparisons=comparisons, conditional_capture_time=conditional_time,
                  protocol=data['protocol'], input_sha256=sha256(path), analysis_sha256=sha256(__file__))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'vrx-statistics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    a, b = summary['channel'], summary['reference']
    lines = ['# VRX 配对实验结果', '',
             f'100 个配对场景中，新模型成功 {a["successes"]} 次，原 ARBoids 成功 {b["successes"]} 次；'
             f'碰撞分别为 {a["collisions"]} 和 {b["collisions"]} 次，突破目标分别为 {a["breaches"]} 和 {b["breaches"]} 次。', '',
             '| 方法 | 捕获 | 超时防守 | 碰撞 | 突破目标 |', '|---|---:|---:|---:|---:|']
    for name, values in summary.items():
        lines.append(f'| {name} | {values["captures"]} | {values["timeout_denials"]} | {values["collisions"]} | {values["breaches"]} |')
    lines += ['', '| 指标 | 新减原，百分点 | 保守配对 95% 区间 | Holm 校正 p |', '|---|---:|---:|---:|']
    for metric, result in comparisons.items():
        lo, hi = result['conservative_paired_95_interval']
        lines.append(f'| {metric} | {100*result["difference"]:.2f} | [{100*lo:.2f}, {100*hi:.2f}] | {result["holm_3_outcomes_p"]:.6g} |')
    if a['collisions'] == b['collisions'] == 0:
        lines += ['', '两种策略均未观察到碰撞，因此这组 VRX 结果不能证明新模型的碰撞率更低。']
    if comparisons['success']['holm_3_outcomes_p'] >= .05:
        lines += ['', '成功率差异未通过预设的校正检验；应报告点估计和区间，不能据此宣称 VRX 上已显著优于原模型。']
    lines += ['', '范围为两个冻结模型。初始状态逐对相同，运行顺序随机化；Gazebo 的后续异步物理演化不要求逐位相同。'
              '基础设施失败不计作任务失败。成功包含捕获与超时防守，二者已分别列出。',
              '这些同条件结果不与原论文表中来自其他实验协议的百分比直接相减，也不代表真实艇试验。', '']
    (output / 'vrx-evidence.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output)['summary']))
