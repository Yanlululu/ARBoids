"""One-shot dispatch of ready experiments in the authorized ten-hour campaign.

No scheduler, timer or background polling is installed. Each invocation fills
currently free remote compute slots and records exact launched commands.
"""
import pathlib,json,os,subprocess,datetime,collections
root=pathlib.Path('/root/autodl-tmp/citf-10h-20261102');r=root/'runs'
now=datetime.datetime.now(datetime.timezone.utc)
if now>=datetime.datetime.fromisoformat('2026-11-02T22:32:28+00:00'):raise SystemExit('Final bounded evaluation launch cutoff reached')
processes={};managers={}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if '-m' not in args or str(root) not in ' '.join(args):continue
  module=args[args.index('-m')+1]
  if not module.startswith('experiments.') or module=='experiments.continue_development':continue
  parent=int(next(line.split(':',1)[1] for line in (p/'status').read_text().splitlines() if line.startswith('PPid:')))
  processes[int(p.name)]=(module,parent)
  if module in ('experiments.parallel_evaluate','experiments.collect_development'):
   workers=int(args[args.index('--workers')+1])
   flag,filename=('--out','status.json') if module.endswith('parallel_evaluate') else ('--run','collection_status.json')
   status=pathlib.Path(args[args.index(flag)+1])/filename
   try:
    remaining=sum(j['status'] in ('pending','running') for j in json.loads(status.read_text())['jobs'])
    workers=min(workers,remaining)
   except (OSError,json.JSONDecodeError,KeyError):pass
   managers[int(p.name)]=workers
 except OSError:pass
count=sum(managers.values())+sum(pid not in managers and parent not in managers for pid,(_,parent) in processes.items())
print('UTC',now.isoformat(),'COMPUTE',count)
env=os.environ.copy();env.update(PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
source_records=json.loads((r/'v11_regularized'/'launches.json').read_text())
base_train=list(next(x['command'] for x in source_records if x['name']=='precise_boundary128_width48_nll1_s7101'))
base_eval=list(next(x['command'] for x in source_records if x['name']=='precise_boundary128_width48_nll1_s7103_w64_g16_m6'))
launched=0
waiting_for_slots=False
def registered(group,name):
 p=r/group/'launches.json'
 return p.exists() and any(x['name']==name for x in json.loads(p.read_text()))
def launch(group,name,cmd):
 global launched,waiting_for_slots
 slots=int(cmd[cmd.index('--workers')+1]) if '--workers' in cmd else 1
 if registered(group,name) or waiting_for_slots:return False
 if count+launched+slots>20:
  # Do not repeatedly consume one free slot with lower-priority work while
  # a ready two-worker full evaluation is waiting for its second slot.
  waiting_for_slots = count+launched<20
  return False
 g=r/group;g.mkdir(exist_ok=True);p=g/'launches.json'
 records=json.loads(p.read_text()) if p.exists() else []
 out=pathlib.Path(cmd[cmd.index('--out')+1])
 if out.exists():raise ValueError('Unregistered existing output '+str(out))
 with (g/(name+'.log')).open('w') as log:job=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 records.append(dict(name=name,pid=job.pid,command=cmd,started_utc=now.isoformat()));p.write_text(json.dumps(records,indent=2))
 launched+=slots;print('LAUNCHED',group,name,job.pid,'slots',slots);return True
def data_ready(group,names):
 p=r/group/'collection_status.json'
 if not p.exists():return False
 state={j['name']:j['status'] for j in json.loads(p.read_text())['jobs']}
 return all(state.get(x)=='completed' for x in names)
def model_ready(group,name):
 p=r/group/name/'status.json'
 return p.exists() and json.loads(p.read_text()).get('completed') is True
def train(group,name,module='experiments.synchronized_delta',seed=7101,append=None,replace_train=None,replace_val=None,updates=16000,overrides=None,extra=None):
 if now>=datetime.datetime.fromisoformat('2026-11-02T22:20:28+00:00'):
  return False
 cmd=list(base_train);cmd[cmd.index('-m')+1]=module;cmd[cmd.index('--seed')+1]=str(seed);cmd[cmd.index('--out')+1]=str(r/group/name)
 if replace_train is not None:
  i,j=cmd.index('--train')+1,cmd.index('--validation');cmd[i:j]=list(map(str,replace_train))
 if append:
  i=cmd.index('--validation');cmd[i:i]=list(map(str,append))
 if replace_val is not None:
  i,j=cmd.index('--validation')+1,cmd.index('--out');cmd[i:j]=list(map(str,replace_val))
 if updates!=16000:
  cmd[cmd.index('--updates')+1]=str(updates)
  for flag in ['--validate-every','--save-every']:cmd[cmd.index(flag)+1]='4000'
  i,j=cmd.index('--snapshot-updates')+1,cmd.index('--guard-weight');cmd[i:j]=list(map(str,sorted({0,min(16000,updates),min(32000,updates),updates})))
 if '--stop-file' not in cmd:cmd+=['--stop-file',str(r/'STOP_RESEARCH_TRAINING')]
 for flag,value in (overrides or {}).items():cmd[cmd.index(flag)+1]=str(value)
 if extra:cmd+=list(extra)
 return launch(group,name,cmd)
def evaluate(group,name,checkpoint=None,module='experiments.synchronized_delta',selection=False,long=False,extra=None):
 cmd=list(base_eval);cmd[cmd.index('--evaluator')+1]=module
 cmd[cmd.index('--out')+1]=str(r/group/name if selection else r/'extended'/name)
 if checkpoint is None:
  i=cmd.index('--checkpoint');del cmd[i:i+2]
 else:cmd[cmd.index('--checkpoint')+1]=str(checkpoint)
 if selection:
  cmd[cmd.index('--tasks')+1]=str(r/group/'tasks.json');cmd[cmd.index('--registry')+1]=str(r/group/'layout_registry.json')
 elif long:
  cmd[cmd.index('--tasks')+1]=str(r/'v18_long_recourse'/'closed_bank.json');cmd[cmd.index('--registry')+1]=str(r/'v18_long_recourse'/'layout_registry.json')
  cmd[cmd.index('--workers')+1]='2'
 if now>=datetime.datetime.fromisoformat('2026-11-02T21:55:00+00:00'):
  cmd[cmd.index('--workers')+1]='2'
 if now>=datetime.datetime.fromisoformat('2026-11-02T22:25:28+00:00'):
  # Complete already-trained recipes in the remaining time: prior two-worker
  # evaluations took 11--17 minutes, so four workers retain a bounded window.
  # The per-evaluation deadline still stops every child at 22:40:28 UTC.
  cmd[cmd.index('--workers')+1]='4'
 if extra:cmd+=list(map(str,extra))
 return launch(group,name,cmd)
# Priority: turn newly complete data into immutable experimental datasets.
for n in [16,32,64]:
 names=['train_%04d'%i for i in range(0,n,2)]
 if data_ready('v17_onpolicy',names):
  model='precise_boundary128_onpolicy%d_width48_nll1_s7101'%n
  train('v17_onpolicy',model,append=[r/'v17_onpolicy'/(x+'.pt') for x in names])
long_names=['train_%04d'%i for i in range(0,32,2)]+['validation_%04d'%i for i in range(0,24,2)]
if data_ready('v18_long_recourse',long_names):
 for stem,module in [('long32','long_recourse_guard'),('dual32','dual_interval_guard'),('multiscale32','multiscale_recourse_guard')]:
  train('v18_long_recourse',stem+'_width48_nll1_s7101',module='experiments.'+module,
        replace_train=[r/'v18_long_recourse'/('train_%04d.pt'%i) for i in range(0,32,2)],
        replace_val=[r/'v18_long_recourse'/('validation_%04d.pt'%i) for i in range(0,24,2)])
 train('v30_long_transfer','warmdual32_width48_nll1_s7101',module='experiments.warm_dual_interval_guard',
       replace_train=[r/'v18_long_recourse'/('train_%04d.pt'%i) for i in range(0,32,2)],
       replace_val=[r/'v18_long_recourse'/('validation_%04d.pt'%i) for i in range(0,24,2)],
       extra=['--initial-checkpoint',str(r/'v11_regularized'/'precise_boundary128_width48_nll1_s7101'/'best.pt')])
# The first complete long-interval pilot gained six tasks over its matched
# long nominal control. Replicate that fixed recipe and its learning controls.
long_pilot=r/'extended'/'long32_width48_nll1_s7101_w64_g16_m6'/'summary.json'
if long_pilot.exists() and json.loads(long_pilot.read_text()).get('completed'):
 for model,seed,extra in [
     ('long32_width48_nll1_s7102',7102,[]),
     ('long32_width48_nll1_s7103',7103,[]),
     ('long32_width48_nll1_nocontext_s7101',7101,['--without-fleet-context']),
     ('long32_width48_nll1_predictive_s7101',7101,['--mode','predictive_only'])]:
  train('v34_long_attribution',model,module='experiments.long_recourse_guard',seed=seed,
        replace_train=[r/'v18_long_recourse'/('train_%04d.pt'%i) for i in range(0,32,2)],
        replace_val=[r/'v18_long_recourse'/('validation_%04d.pt'%i) for i in range(0,24,2)],extra=extra)
  if model_ready('v34_long_attribution',model):
   evaluate('v34_long_attribution',model+'_w64_g16_m6',r/'v34_long_attribution'/model/'best.pt',
            module='experiments.long_recourse_guard',long=True)
new_names=['train_%04d'%i for i in range(0,256,2)]
# Complete the missing information-by-objective cell with a fixed first seed.
train('v33_objective_context','precise_boundary128_width48_nll1_nocontext_predictive_s7101',
      extra=['--without-fleet-context','--mode','predictive_only'])
if data_ready('v12_data',new_names):
 for seed in [7101,7102,7103]:
  train('v22_full384','precise_boundary384_width48_nll1_u64000_s%d'%seed,seed=seed,updates=64000,
        append=[r/'v12_data'/(x+'.pt') for x in new_names])
 # Expanded-data attribution is matched in both exposure and optimization.
 for suffix,extra in [('nocontext',['--without-fleet-context']),('predictive',['--mode','predictive_only'])]:
  train('v31_full384_controls','precise_boundary384_width48_nll1_u64000_'+suffix+'_s7101',
        updates=64000,append=[r/'v12_data'/(x+'.pt') for x in new_names],extra=extra)
 # Start the declared larger balanced run early enough to retain evaluation time.
 train('v29_family_balance','balanced384_width48_nll1_u64000_s7101',
       module='experiments.family_balanced_guard',updates=64000,
       append=[r/'v12_data'/(x+'.pt') for x in new_names])
# A fixed prefix gives a complete intermediate data/optimization recipe while
# the last full-pool trajectory is still being collected. No layouts are ranked.
prefix_names=['train_%04d'%i for i in range(0,128,2)]
if data_ready('v12_data',prefix_names):
 train('v25_data_scaling','precise_boundary256_width48_nll1_u32000_s7101',updates=32000,
       append=[r/'v12_data'/(x+'.pt') for x in prefix_names])
scaled='precise_boundary256_width48_nll1_u32000_s7101'
if model_ready('v25_data_scaling',scaled):
 evaluate('v25_data_scaling',scaled+'_w64_g16_m6',r/'v25_data_scaling'/scaled/'best.pt')
# Evaluate each completed new model once on the full common development bank.
evaluate('v18_long_recourse','dual_interval_phys_w64_g16_m6',
         module='experiments.dual_interval_guard',long=True)
factorial='precise_boundary128_width48_nll1_nocontext_predictive_s7101'
if model_ready('v33_objective_context',factorial):
 evaluate('v33_objective_context',factorial+'_w64_g16_m6',r/'v33_objective_context'/factorial/'best.pt')
 evaluate('v13_stage_selection','nocontext_predictive_s7101_best',r/'v33_objective_context'/factorial/'best.pt',selection=True)
anchor='anchored_boundary128_width48_nll1_s7101'
train('v24_anchored',anchor,module='experiments.anchored_team_guard')
if model_ready('v24_anchored',anchor):
 evaluate('v24_anchored',anchor+'_w64_g16_m6',r/'v24_anchored'/anchor/'best.pt',module='experiments.anchored_team_guard')
for n in [16,32,64]:
 model='precise_boundary128_onpolicy%d_width48_nll1_s7101'%n
 if model_ready('v17_onpolicy',model):evaluate('v17_onpolicy',model+'_w64_g16_m6',r/'v17_onpolicy'/model/'best.pt')
for stem,module in [('long32','long_recourse_guard'),('dual32','dual_interval_guard'),('multiscale32','multiscale_recourse_guard')]:
 model=stem+'_width48_nll1_s7101'
 if model_ready('v18_long_recourse',model):evaluate('v18_long_recourse',model+'_w64_g16_m6',r/'v18_long_recourse'/model/'best.pt',module='experiments.'+module,long=True)
warm='warmdual32_width48_nll1_s7101'
if model_ready('v30_long_transfer',warm):
 evaluate('v30_long_transfer',warm+'_w64_g16_m6',r/'v30_long_transfer'/warm/'best.pt',module='experiments.warm_dual_interval_guard',long=True)
# Evaluate the already-trained sampling comparison while replication time remains.
for balanced in ['balanced128_width48_nll1_s7101','balanced384_width48_nll1_u64000_s7101']:
 if model_ready('v29_family_balance',balanced):
  evaluate('v29_family_balance',balanced+'_w64_g16_m6',r/'v29_family_balance'/balanced/'best.pt',module='experiments.family_balanced_guard')
# The current leading recipe needs repeated, budget-matched attribution controls.
for seed in [7102,7103]:
 for suffix,extra in [('nocontext',['--without-fleet-context']),('predictive',['--mode','predictive_only'])]:
  model='precise_boundary128_width48_nll1_%s_s%d'%(suffix,seed)
  train('v19_matched_controls',model,seed=seed,extra=extra)
  if model_ready('v19_matched_controls',model):
   evaluate('v19_matched_controls',model+'_w64_g16_m6',r/'v19_matched_controls'/model/'best.pt')
   evaluate('v13_stage_selection','%s_s%d_best'%(suffix,seed),r/'v19_matched_controls'/model/'best.pt',selection=True)
for suffix in ['nocontext','predictive']:
 members=[r/'v19_matched_controls'/('precise_boundary128_width48_nll1_%s_s%d'%(suffix,seed))/'best.pt'
          for seed in [7101,7102,7103]]
 if all(path.is_file() for path in members):
  name='precise128_'+suffix+'_ensemble7101_7103'
  evaluate('v32_ensemble_controls',name+'_w64_g16_m6',members[0],
           module='experiments.ensemble_guard',extra=['--ensemble-checkpoints',*members])
  evaluate('v13_stage_selection',name,members[0],selection=True,
           module='experiments.ensemble_guard',extra=['--ensemble-checkpoints',*members])
model='history_boundary128_width48_nll1_s7101'
if model_ready('v20_history_reference',model):evaluate('v20_history_reference',model+'_w64_g16_m6',r/'v20_history_reference'/model/'best.pt',module='experiments.history_reference_guard')
for seed in [7101,7102,7103]:
 model='precise_boundary384_width48_nll1_u64000_s%d'%seed
 if model_ready('v22_full384',model):
  evaluate('v22_full384',model+'_w64_g16_m6',r/'v22_full384'/model/'best.pt')
for seed in [7101,7102,7103]:
 model='precise_boundary384_width48_nll1_u64000_s%d'%seed
 if model_ready('v22_full384',model):
  evaluate('v13_stage_selection','full384_s%d_best'%seed,r/'v22_full384'/model/'best.pt',selection=True)
for suffix in ['nocontext','predictive']:
 model='precise_boundary384_width48_nll1_u64000_'+suffix+'_s7101'
 if model_ready('v31_full384_controls',model):
  evaluate('v31_full384_controls',model+'_w64_g16_m6',r/'v31_full384_controls'/model/'best.pt')
  evaluate('v13_stage_selection','full384_'+suffix+'_s7101_best',r/'v31_full384_controls'/model/'best.pt',selection=True)
# Calibration was determined using labels only, independently of closed-loop outcomes.
calibration=r/'v21_uncertainty'/'calibration.json'
if calibration.exists():
 for row in json.loads(calibration.read_text())['models']:
  seed=pathlib.Path(row['checkpoint']).parent.name.rsplit('_s',1)[1]
  name='precise128_s'+seed+'_calibrated_floor_w64_g16_m6'
  evaluate('v21_uncertainty',name,pathlib.Path(row['checkpoint']),module='experiments.team_guard_uncertainty',
           extra=['--guard-combination','envelope','--guard-sigma-scale','1','--guard-sigma-floor',row['selected_floor_m']])
# Average every declared seed of the improved 128-layout recipe.
ensemble=[r/'v11_regularized'/('precise_boundary128_width48_nll1_s%d'%seed)/'best.pt' for seed in [7101,7102,7103]]
if all(path.is_file() for path in ensemble):
 evaluate('v23_regularized_ensemble','precise128_ensemble7101_7103_w64_g16_m6',ensemble[0],
          module='experiments.ensemble_guard',extra=['--ensemble-checkpoints',*ensemble])
 evaluate('v13_stage_selection','precise128_ensemble7101_7103',ensemble[0],selection=True,
          module='experiments.ensemble_guard',extra=['--ensemble-checkpoints',*ensemble])
# Complete every declared stage for seeds2 and3 before selecting checkpoints.
for stage in [8000,16000,'best']:
 for seed in [7102,7103]:
  model=r/'v11_regularized'/('precise_boundary128_width48_nll1_s'+str(seed))
  checkpoint=model/'best.pt' if stage=='best' else model/'snapshots'/('update_%06d.pt'%stage)
  name='s%d_best'%seed if stage=='best' else 's%d_u%06d'%(seed,stage)
  evaluate('v13_stage_selection',name,checkpoint,selection=True)
# The independent development bank also receives the same matched controls.
for suffix,label in [('nocontext','nocontext_s7101_best'),('predictive','predictive_s7101_best')]:
 model='precise_boundary128_width48_nll1_'+suffix+'_s7101'
 if model_ready('v19_matched_controls',model):
  evaluate('v13_stage_selection',label,r/'v19_matched_controls'/model/'best.pt',selection=True)
# Select and report complete independent stage comparisons.
g=r/'v13_stage_selection';selected_path=g/'selected_checkpoints.json'
selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
for seed in [7102,7103]:
 model=r/'v11_regularized'/('precise_boundary128_width48_nll1_s%d'%seed)
 best_update=json.loads((model/'status.json').read_text())['best']['update']
 stages=[('s7101_u000000',0,r/'v11_regularized'/'precise_boundary128_width48_nll1_s7101'/'snapshots'/'update_000000.pt')]
 stages += [('s%d_u%06d'%(seed,u),u,model/'snapshots'/('update_%06d.pt'%u)) for u in [4000,8000,16000]]
 stages += [('s%d_best'%seed,best_update,model/'best.pt')]
 if not all((g/name/'summary.json').exists() for name,_,_ in stages):continue
 candidates=[]
 for name,update,checkpoint in stages:
  d=json.loads((g/name/'summary.json').read_text())
  if not d.get('completed') or d.get('N')!=256:raise ValueError('Incomplete stage summary')
  candidates.append(dict(name=name,update=update,checkpoint=str(checkpoint),success=d['success'],collision=d['collision'],capped_capture_time=d['capped_capture_time']))
 if str(seed) not in selected:
  choice=min(candidates,key=lambda x:(-x['success'],x['collision'],x['capped_capture_time'],x['update']))
  selected[str(seed)]=dict(selection=choice,all_candidates=candidates,selected_utc=now.isoformat(),report_bank='extended/tasks.json')
  selected_path.write_text(json.dumps(selected,indent=2));print('SELECTED',seed,choice)
 choice=selected[str(seed)]['selection'];name='stage_selected_s%d_u%06d_w64_g16_m6'%(seed,choice['update'])
 if choice['name']=='s%d_best'%seed:
  existing=r/'extended'/('precise_boundary128_width48_nll1_s%d_w64_g16_m6'%seed)
  report=existing/'summary.json'
  if report.exists() and json.loads(report.read_text()).get('completed'):
   selected[str(seed)]['report_evaluation']=str(existing)
   selected[str(seed)]['report_reuse_reason']='identical checkpoint, full task bank and controller arguments already evaluated'
   selected_path.write_text(json.dumps(selected,indent=2))
   continue
 evaluate('v13_stage_selection',name,pathlib.Path(choice['checkpoint']))
# Train a geometric response readout without a competing absolute target.
response_recipes=[('relative_boundary128_width48_s7101',7101,[]),
                  ('relative_boundary128_width48_nocontext_s7101',7101,['--without-fleet-context'])]
first_response=r/'extended'/'relative_boundary128_width48_s7101_slack1_w64_g16_m6'/'summary.json'
if first_response.exists() and json.loads(first_response.read_text()).get('success',0)>245:
 response_recipes += [('relative_boundary128_width48_s%d'%seed,seed,[]) for seed in [7102,7103]]
for model,seed,extra in response_recipes:
 train('v28_relative_response',model,module='experiments.relative_team_response',seed=seed,
       overrides={'--guard-nll-weight':0,'--rank-weight':0},extra=extra)
 if model_ready('v28_relative_response',model):
  evaluate('v28_relative_response',model+'_slack1_w64_g16_m6',r/'v28_relative_response'/model/'best.pt',
           module='experiments.relative_team_response',extra=['--nominal-slack',1])
# Balance the exposure of long and short trajectories without changing data.
balanced_recipes=[('balanced128_width48_nll1_s7101',16000,None)]
if data_ready('v12_data',new_names):
 balanced_recipes.append(('balanced384_width48_nll1_u64000_s7101',64000,
                          [r/'v12_data'/(x+'.pt') for x in new_names]))
for model,updates,append in balanced_recipes:
 train('v29_family_balance',model,module='experiments.family_balanced_guard',updates=updates,append=append)
 if model_ready('v29_family_balance',model):
  evaluate('v29_family_balance',model+'_w64_g16_m6',r/'v29_family_balance'/model/'best.pt',module='experiments.family_balanced_guard')
# Select stages for the new mean-response selector from fixed physical labels.
response_selection=r/'v27_response_selection'/'selection.json'
validation=base_train[base_train.index('--validation')+1:base_train.index('--out')]
launch('v27_response_selection','label_selection',[base_train[0],'-u','-m','experiments.select_response_checkpoint',
    '--models-dir',str(r/'v11_regularized'),'--validation',*validation,
    '--out',str(response_selection),'--device','cuda','--nominal-slack','1'])
if response_selection.exists():
 for record in json.loads(response_selection.read_text())['models']:
  choice=record['selected']
  # Zero-initialized mean readouts yield the same analytic impulse geometry
  # for all seeds; the declared v26 zero-step control supplies this result.
  if choice['update']==0:continue
  name='response_selected_s%d_u%06d_slack1_w64_g16_m6'%(record['seed'],choice['update'])
  evaluate('v27_response_selection',name,pathlib.Path(choice['checkpoint']),
           module='experiments.trusted_team_response',extra=['--nominal-slack',1])
# Resolve physical near-ties with short-interval candidate response only.
# Every declared seed and the zero-step / matched-information controls stay.
response_models=[('s%d'%seed,r/'v11_regularized'/('precise_boundary128_width48_nll1_s%d'%seed)/'best.pt',1.)
                 for seed in [7101,7102,7103]]
response_models += [('zero',r/'v11_regularized'/'precise_boundary128_width48_nll1_s7101'/'snapshots'/'update_000000.pt',1.),
    ('nocontext',r/'v19_matched_controls'/'precise_boundary128_width48_nll1_nocontext_s7101'/'best.pt',1.),
    ('predictive',r/'v19_matched_controls'/'precise_boundary128_width48_nll1_predictive_s7101'/'best.pt',1.),
    ('s7101',r/'v11_regularized'/'precise_boundary128_width48_nll1_s7101'/'best.pt',.25)]
for label,checkpoint,slack in response_models:
 evaluate('v26_trusted_response','trusted_response_'+label+'_slack'+str(slack).replace('.','p')+'_w64_g16_m6',
          checkpoint,module='experiments.trusted_team_response',extra=['--nominal-slack',slack])
print('NEW_COUNT',count+launched)
