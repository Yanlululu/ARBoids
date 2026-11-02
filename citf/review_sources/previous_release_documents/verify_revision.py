"""Run bounded real-environment interface checks, never formal research claims.

Run from anywhere. Creates a new directory and refuses overwrite. Requires the
source files and source weight supplied in this delivery, not a remote service.
"""
from pathlib import Path
import subprocess,json,sys,os

ROOT=Path(__file__).resolve().parents[1]

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a new verification directory')
    out.mkdir(parents=True)
    project=ROOT/'source_archive'
    source=next((project/'train/experiments').rglob('best-m1-400.pth'))
    registry=out/'layout_registry.json'
    def call(name,args,expect_success=True):
        print('CHECK',name,flush=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT/'code',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,env=env)
        (out/(name+'.log')).write_text(result.stdout,encoding='utf8')
        if (result.returncode==0)!=expect_success:raise RuntimeError(name+' unexpected exit; see log')
    call('unit_tests',['-m','unittest','discover','-s','tests','-v'])
    for role,seed in [('train',9621000),('validation',9622000),('test',9623000)]:
        call('tasks_'+role,['-m','citf.tasks','--project',project,'--registry',registry,'--out',out/(role+'_tasks.json'),
             '--role',role,'--seed-start',seed,'--layouts',2 if role!='test' else 1])
    for role in ['train','validation']:
        call('collect_'+role,['-m','citf.collect','--project',project,'--source',source,'--registry',registry,
             '--tasks',out/(role+'_tasks.json'),'--out',out/(role+'.pt'),'--root-times',0,2,'--max-per-layout',2,
             '--candidates',4,'--futures',2,'--max-roots',2,'--device','cpu'])
    common=['-m','citf.train','--train',out/'train.pt','--validation',out/'validation.pt',
            '--seed',809,'--updates',4,'--batch-roots',2,'--validate-every',2,'--save-every',1,'--device','cpu']
    call('continuous',common+['--out',out/'continuous'])
    call('interrupted',common+['--out',out/'resumed','--max-updates-this-call',2])
    call('resumed',common+['--out',out/'resumed','--resume'])
    call('behavior_tasks',['-m','citf.tasks','--project',project,'--registry',registry,'--out',out/'behavior_tasks.json',
         '--role','train','--seed-start',9624000,'--layouts',1])
    call('learned_behavior_collect',['-m','citf.collect','--project',project,'--source',source,'--registry',registry,
         '--tasks',out/'behavior_tasks.json','--out',out/'behavior.pt','--behavior-checkpoint',out/'continuous/best.pt',
         '--behavior-bank',2,'--behavior-budget',4,'--behavior-rounds',0,'--root-times',0.2,'--max-per-layout',1,
         '--candidates',4,'--futures',2,'--max-roots',1])
    eval_args=['-m','citf.evaluate','--project',project,'--source',source,'--registry',registry,'--tasks',out/'test_tasks.json',
         '--checkpoint',out/'continuous/best.pt','--out',out/'controller_check','--bank',2,'--budget',4,'--rounds',0,'--device','cpu']
    call('export_runtime',eval_args+['--prepare-runtime',out/'runtime.json'])
    call('closed_loop_interface',eval_args)
    call('reject_unsigned_final',eval_args+['--out',out/'unsigned_final','--phase','final'],expect_success=False)
    import torch
    x=torch.load(out/'continuous/last.pt',map_location='cpu',weights_only=False)
    y=torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=False)
    equality={'updates':x['update'],'weights_bitwise_equal':all(torch.equal(x['state'][k],y['state'][k]) for k in x['state']),
              'scheduler_equal':x['scheduler']==y['scheduler'],'best_selection_equal':x['best']==y['best'],
              'sampler_rng_equal':x['sampler_rng']==y['sampler_rng']}
    if not all(v for k,v in equality.items() if k!='updates'):raise AssertionError(equality)
    (out/'resume_equivalence.json').write_text(json.dumps(equality,indent=2))
    counts={r:json.loads((out/(r+'.json')).read_text()) for r in ['train','validation','behavior']}
    learned=torch.load(out/'behavior.pt',map_location='cpu',weights_only=False)
    assert learned['roots'][0]['behavior_identity']['method']=='CITF'
    status={'evidence_role':'bounded interface verification, not formal performance',
       'unit_tests':48,'resume':equality,'train_roots':counts['train']['roots'],'validation_roots':counts['validation']['roots'],
       'train_branch_steps':counts['train']['branch_transitions'],'validation_branch_steps':counts['validation']['branch_transitions'],
       'learned_behavior_roots':counts['behavior']['roots'],'learned_behavior_branch_steps':counts['behavior']['branch_transitions'],
       'full_training_runs':0,'vrx_trials':0,'independent_closed_loop_interface_tasks':1,
       'raw_task_result':'controller_check/summary.json','formal_advantage_claim':False}
    (out/'verification_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
