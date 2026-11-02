"""Run bounded real-environment interface checks, never formal research claims.

Run from anywhere. Creates a new directory and refuses overwrite. Requires the
source files and source weight supplied in this delivery, not a remote service.
"""
from pathlib import Path
import subprocess,json,sys,os,re

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
        result=subprocess.run([sys.executable,'-X','utf8',*map(str,args)],cwd=ROOT/'code',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=120,env=env)
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
            '--seed',809,'--updates',4,'--batch-roots',2,'--validate-every',2,'--save-every',1,'--snapshot-updates',0,2,4,'--device','cpu']
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
    stages=json.loads((out/'continuous/snapshots.json').read_text())
    other=json.loads((out/'resumed/snapshots.json').read_text())
    assert [e['update'] for e in stages['entries']]==[0,2,4]
    assert [e['state_sha256'] for e in stages['entries']]==[e['state_sha256'] for e in other['entries']]
    equality['stage_snapshots_equal']=True
    (out/'resume_equivalence.json').write_text(json.dumps(equality,indent=2))
    call('rank_weight_zero',common+['--out',out/'rank_zero','--rank-weight',0.])
    zero=torch.load(out/'rank_zero/last.pt',map_location='cpu',weights_only=False)
    assert zero['protocol']['loss_contract']['rank_weight']==0.
    assert not all(torch.equal(x['state'][k],zero['state'][k]) for k in x['state'])
    weights=[]
    for weight in [0.,8.,32.,64.]:
        path=out/f'runtime_lambda_{weight:g}.json'
        call(f'lambda_{weight:g}',eval_args+['--prepare-runtime',path,'--risk-weight',weight])
        weights.append(json.loads(path.read_text())['protocol']['risk_weight'])
    assert weights==[0.,8.,32.,64.]
    sys.path.insert(0,str(ROOT/'code'))
    from citf.contracts import file_hash
    from citf.opponents import OBSERVATION
    class ExportActor(torch.nn.Module):
        def forward(self,x):return torch.tanh(torch.stack((.02*x[:,0]+.03*x[:,1],-.01*x[:,0]+.04*x[:,1]),-1))
    actor=out/'opponent_interface_actor.pt'
    torch.jit.trace(ExportActor().eval(),torch.zeros(1,8)).save(str(actor))
    spec=out/'opponent_interface.json'
    spec.write_text(json.dumps({'kind':'torchscript','checkpoint':actor.name,'checkpoint_sha256':file_hash(actor),
         'observation':OBSERVATION,'stateful':False,'stochastic':False,'training_identity':'untrained-interface-test-only'},indent=2))
    call('opponent_behavior_tasks',['-m','citf.tasks','--project',project,'--registry',registry,'--out',out/'opponent_behavior_tasks.json',
         '--role','train','--seed-start',9625000,'--layouts',1])
    call('learned_opponent_collect',['-m','citf.collect','--project',project,'--source',source,'--registry',registry,
         '--tasks',out/'opponent_behavior_tasks.json','--out',out/'opponent_data.pt','--behavior-checkpoint',out/'continuous/best.pt',
         '--behavior-bank',2,'--behavior-budget',4,'--behavior-rounds',0,'--root-times',.2,'--max-per-layout',1,
         '--candidates',4,'--futures',2,'--max-roots',1,'--opponent-spec',spec])
    learned_opp=torch.load(out/'opponent_data.pt',map_location='cpu',weights_only=False)
    assert learned_opp['roots'][0]['opponent_identity']==learned_opp['roots'][0]['behavior_identity']['opponent']
    call('learned_opponent_closed_loop',eval_args+['--out',out/'opponent_controller_check','--opponent-spec',spec])
    call('stage_validation_curve',[ROOT/'tools/evaluate_stages.py','--run',out/'continuous','--project',project,'--source',source,
         '--tasks',out/'validation_tasks.json','--registry',registry,'--out',out/'stage_curve','--bank',2,'--budget',4,'--rounds',0])
    counts={r:json.loads((out/(r+'.json')).read_text()) for r in ['train','validation','behavior']}
    learned=torch.load(out/'behavior.pt',map_location='cpu',weights_only=False)
    assert learned['roots'][0]['behavior_identity']['method']=='CITF'
    status={'evidence_role':'bounded interface verification, not formal performance',
       'unit_tests':int(re.search(r'Ran (\d+) tests', (out/'unit_tests.log').read_text(encoding='utf-8')).group(1)),
       'online_risk_weights_verified':weights,'stage_updates':[0,2,4],
       'rank_weight_reaches_actual_objective':True,'learned_opponent_behavior_and_branch_identity_equal':True,'resume':equality,'train_roots':counts['train']['roots'],'validation_roots':counts['validation']['roots'],
       'train_branch_steps':counts['train']['branch_transitions'],'validation_branch_steps':counts['validation']['branch_transitions'],
       'learned_behavior_roots':counts['behavior']['roots'],'learned_behavior_branch_steps':counts['behavior']['branch_transitions'],
       'full_training_runs':0,'vrx_trials':0,'independent_closed_loop_interface_tasks':2,'stage_curve_validation_tasks':6,
       'raw_task_result':'controller_check/summary.json','formal_advantage_claim':False}
    (out/'verification_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
