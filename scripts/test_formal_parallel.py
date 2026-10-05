import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_formal_evidence_parallel as adapter


class ServerAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((adapter.STUDY/'manifest.json').read_text(encoding='utf-8'))
        cls.manifest=adapter.build_manifest(cls.original,{'local_completed_jobs':[]})
        cls.jobs={j['name']:j for j in cls.manifest['jobs']}

    def test_training_dependencies_require_completed_predecessors(self):
        for seed in (101,202,303,404,505):
            self.assertEqual(self.jobs[f'train-{seed}-pretrain']['dependencies'],[])
            self.assertEqual(self.jobs[f'train-{seed}-full']['dependencies'],[f'train-{seed}-pretrain'])
            self.assertEqual(self.jobs[f'train-{seed}-arboids']['dependencies'],[f'train-{seed}-pretrain'])
            self.assertEqual(self.jobs[f'train-{seed}-fixed_rule']['dependencies'],[f'train-{seed}-arboids'])
            self.assertEqual(self.jobs[f'primary-{seed}-full']['dependencies'],[f'train-{seed}-full'])
            self.assertEqual(self.jobs[f'generalization-{seed}-team-4-full']['dependencies'],[f'train-{seed}-full'])

    def test_budgets_seeds_episodes_and_sampling_are_unchanged(self):
        for old,new in zip(self.original['jobs'],self.manifest['jobs']):
            self.assertEqual(old['name'],new['name'])
            for flag in ('--steps','--seed','--scene-seed','--first-seed','--episodes','--workers','--agility',
                         '--defenders','--attacker','--message-delay-steps','--message-drop-probability'):
                if flag in old['command']:
                    self.assertEqual(adapter.option(old['command'],flag),adapter.option(new['command'],flag))
        for key in ('training_seeds','pretrain_steps','continuation_steps','evaluation_selection'):
            self.assertEqual(self.original[key],self.manifest[key])

    def test_vrx_slots_use_separate_domains_and_disjoint_cpu_sets(self):
        with patch.object(adapter.os,'sched_getaffinity',return_value=set(range(32)),create=True):
            slots=adapter.make_slots()
        assigned=[c for slot in slots for c in slot['cpus']]
        self.assertEqual(len(assigned),20)
        self.assertEqual(len(set(assigned)),20)
        vrx=[s for s in slots if s['group']=='vrx']
        self.assertEqual(len({s['domain'] for s in vrx}),2)
        command,result=adapter.command_for(self.jobs['vrx-000-candidate'],vrx[0],2)
        self.assertIn('export ROS_DOMAIN_ID=100',command[command.index('bash')+2])
        self.assertIn('infrastructure-attempt-02',str(result))
        self.assertEqual(adapter.option(command,'--seed'),'87000000')

    def test_retry_cannot_replace_a_real_task_failure(self):
        result={'passed':False,'control_steps':0,'error':'TimeoutError: Waiting for all vessel poses and thrust bridge subscribers'}
        self.assertTrue(adapter.startup_retry_allowed(result,'Network is unreachable'))
        for altered in (dict(result,control_steps=1),dict(result,outcome_code=1),dict(result,passed=True)):
            self.assertFalse(adapter.startup_retry_allowed(altered,'Network is unreachable'))
        self.assertFalse(adapter.startup_retry_allowed(result,'unrelated error'))

    def test_randomized_vrx_order_is_preserved(self):
        ordered=[j['name'] for j in sorted([j for j in self.manifest['jobs'] if j['category']=='vrx'],key=adapter.priority)]
        self.assertEqual(ordered,[j['name'] for j in self.original['jobs'] if j['category']=='vrx'])


if __name__=='__main__':
    unittest.main(verbosity=2)
