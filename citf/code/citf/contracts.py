"""CITF v3 execution contract; v2 tensor/model semantics retained."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import ast, hashlib, json, math, sys
import scipy
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'citf-stopped-aware-v2'
RUNTIME_REVISION = 'CITF-execution-contracts-v5'
COLLECTION_SCHEMA = 'citf-branch-events-v1'
BRANCH_EVENT_FIELDS = ['success','capture','collision','physical_collision','physical_breach',
                       'source_failure','radial_only_failure','timeout']
FEATURES = ['target_frame_x/60','target_frame_y/60','measured_vx/5','measured_vy/5',
            'sin_yaw','cos_yaw','measured_yaw_rate','attacker_flag']
CONTEXT = ['remaining_seconds/time_limit','known_prediction_steps/prediction_steps']


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    path=Path(path)
    return canonical_hash({str(p.relative_to(path)):file_hash(p) for p in sorted(path.rglob('*.py'))
                           if '__pycache__' not in p.parts})


def source_identity(project, source):
    project = Path(project)
    return dict(source_sha256=file_hash(source),
        source_implementation_sha256=file_hash(project / 'train/policy/networks.py'),
        environment_sha256=hashlib.sha256(
            (project / 'train/envs/TADgame.py').read_bytes()
            + (project / 'train/envs/modules.py').read_bytes()).hexdigest())


def python_dependencies(*entries):
    """Fingerprint the local import closure without importing or executing it.

    Include the actual CLI entry point: experiment adapters replace core objects
    at runtime, so hashing only the core package cannot identify their behavior.
    """
    root = Path(__file__).resolve().parents[1]
    pending = [Path(__file__), *(Path(p) for p in entries)]
    entry = Path(sys.argv[0]).resolve()
    if entry.suffix == '.py' and entry.is_relative_to(root) and entry.is_file():
        pending.append(entry)
    found = {}

    def enqueue(parts):
        for stop in range(1, len(parts) + 1):
            base = root.joinpath(*parts[:stop])
            for path in (base.with_suffix('.py'), base / '__init__.py'):
                if path.is_file():
                    pending.append(path)

    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(root):
            raise ValueError('Code dependency outside the declared Python source root: ' + str(path))
        name = path.relative_to(root).as_posix()
        if name in found:
            continue
        raw = path.read_bytes()
        found[name] = hashlib.sha256(raw).hexdigest()
        package = path.relative_to(root).parts[:-1]
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                base = list(package[:len(package) - node.level + 1]) if node.level else []
                base += node.module.split('.') if node.module else []
                enqueue(base)
                for alias in node.names:
                    if alias.name != '*':
                        enqueue(base + alias.name.split('.'))
    return dict(sorted(found.items()))


def immutable_run_contract(parameters, files=(), code=()):
    """Content identity shared by collection, training and resume readers."""
    paths = sorted({Path(p).resolve() for p in files})
    return dict(revision='citf-immutable-run-v1', parameters=parameters,
        artifacts={str(p): file_hash(p) for p in paths}, code=python_dependencies(*code))


def require_same_contract(recorded, actual, label):
    if recorded is None:
        raise ValueError(label + ' lacks a recorded immutable contract; retain the legacy run and use a new output')
    if canonical_hash(recorded) != canonical_hash(actual):
        raise ValueError(label + ' changed: data, source, protocol, configuration or implementation differs')


def validate_task_banks(*groups):
    """Validate roles, distinct tasks, physical-family separation and protocol."""
    assigned, families, physical = {}, {}, None
    for role, bank in groups:
        if role not in ('train', 'validation', 'test', 'final_test') or bank.get('role') != role:
            raise ValueError('Task bank role differs from its requested use')
        protocol = asdict(Protocol(**bank['protocol']))
        if physical is not None and protocol != physical:
            raise ValueError('Task banks use different physical protocols')
        physical = protocol
        tasks = bank['tasks']
        if not tasks:
            raise ValueError('Empty task bank')
        seen, task_ids = set(), set()
        families[role] = set()
        for task in tasks:
            key = (task['scene_seed'], task['future_seed'], task['defenders'], task['agility'])
            if key in seen:
                raise ValueError('Repeated physical task in bank')
            seen.add(key)
            if 'task_id' in task:
                if task['task_id'] in task_ids:
                    raise ValueError('Repeated task identifier in bank')
                task_ids.add(task['task_id'])
            family = task['geometry_family_id']
            if not family:
                raise ValueError('Missing physical geometry identity')
            if family in assigned and assigned[family] != role:
                raise ValueError('Physical layout appears in more than one split')
            assigned[family] = role
            families[role].add(family)
    return families


@dataclass(frozen=True)
class Protocol:
    name:str='citf-source80-v2'
    environment:str='source'
    time_limit:float=80.
    control_dt:float=.2
    integration_dt:float=.05
    target_radius:float=15.
    capture_radius:float=5.
    collision_radius:float=5.
    risk_clearance:float=7.
    radial_early_failure:bool=True
    prediction_steps:int=16
    history_steps:int=8
    risk_weight:float=32.
    prior_scale:float=.4
    def __post_init__(self):
        # The reference environment has a fixed integration/geometry contract.
        # Reject unsupported changes rather than silently recording false settings.
        locked={'environment':'source','control_dt':.2,'integration_dt':.05,
                'target_radius':15.,'capture_radius':5.,'collision_radius':5.,'radial_early_failure':True}
        for key,value in locked.items():
            if getattr(self,key)!=value:raise ValueError(f'Unsupported environment change {key}; explicit shared environment migration required')
        for key in ('time_limit','risk_clearance','prior_scale'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('Invalid '+key)
        if self.prediction_steps<1 or self.history_steps<1 or not math.isfinite(self.risk_weight) or self.risk_weight<0:raise ValueError('Invalid dimensions or risk weight')
        if abs(self.time_limit/self.control_dt-round(self.time_limit/self.control_dt))>1e-7:
            raise ValueError('Time limit must align with control ticks')
    def digest(self):return canonical_hash(asdict(self))
    def known_steps(self,elapsed):
        return min(self.prediction_steps,max(0,int(math.ceil((self.time_limit-elapsed)/self.control_dt-1e-7))))


def geometry_identity(initial_state,defenders,initialization='source-reset-v1'):
    a=np.asarray(initial_state,dtype=np.float64)
    if a.shape!=(defenders+1,6) or not np.isfinite(a).all():raise ValueError('Initial state must be finite [N+1,6]')
    # Fingerprint the actual geometry rather than a changeable layout name.
    payload={'initialization':initialization,'defenders':int(defenders),
             'initial_state':np.round(a,9).tolist()}
    payload['fingerprint']=canonical_hash(payload)
    return payload


def verify_geometry(meta):
    payload=meta['geometry_identity'];body={k:v for k,v in payload.items() if k!='fingerprint'}
    if canonical_hash(body)!=payload['fingerprint']:raise ValueError('Geometry content hash mismatch')
    if meta['geometry_family_id']!=payload['fingerprint']:raise ValueError('Geometry family alias mismatch')
    return payload['fingerprint']


class LayoutRegistry:
    """Persistent split allocation shared by generation, collection and training.
    Single writer only; multiworker runs use pre-created immutable registries.
    """
    def __init__(self,path):
        self.path=Path(path)
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {'schema':'citf-layout-registry-v2','families':{}}
        if self.data.get('schema')!='citf-layout-registry-v2':raise ValueError('Wrong registry schema')
    def register(self,identity,split,alias):
        if split not in ('train','validation','test','final_test'):raise ValueError('Unknown split')
        key=identity['fingerprint'];entry=self.data['families'].get(key)
        if entry and entry['split']!=split:raise ValueError('Physical geometry is already assigned to another split')
        if not entry:entry={'split':split,'aliases':[],'identity':identity};self.data['families'][key]=entry
        if alias not in entry['aliases']:entry['aliases'].append(alias)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(self.data,ensure_ascii=False,indent=2));tmp.replace(self.path)
        return key


def validate_dataset(blob):
    if blob.get('schema')!=SCHEMA or blob.get('history_features')!=FEATURES or blob.get('context_features')!=CONTEXT:
        raise ValueError('Input schema changed; legacy data/checkpoints require explicit conversion or recollection')
    data=blob['tensors'];required={'history','positions','candidate','nominal','entity_valid','history_valid',
                                 'candidate_valid','unary','real_positions','time_valid','time_context','known_time_valid'}
    if not required.issubset(data):raise ValueError('Missing tensors '+str(required-set(data)))
    n=data['history'].shape[0];meta=blob['roots'];p=Protocol(**blob['protocol'])
    if len(meta)!=n:raise ValueError('Root count mismatch')
    ids=set();mapping={}
    for idx,r in enumerate(meta):
        if r['root_id'] in ids:raise ValueError('Repeated root identifier')
        ids.add(r['root_id']);family=verify_geometry(r);split=r['split']
        if split not in ('train','validation','test'):raise ValueError('Wrong short-response role')
        if family in mapping and mapping[family]!=split:raise ValueError('Physical geometry leakage across splits')
        mapping[family]=split
        for key in ('source_sha256','source_implementation_sha256','environment_sha256','future_seeds','capture_stage','sampling_probability',
                    'elapsed_seconds','remaining_seconds','behavior_identity'):
            if key not in r:raise ValueError('Missing provenance '+key)
        if 'opponent_identity' in r:
            ident=r['opponent_identity'];body={k:v for k,v in ident.items() if k!='identity_sha256'}
            if canonical_hash(body)!=ident.get('identity_sha256'):raise ValueError('Opponent identity content mismatch')
            behavior=r['behavior_identity'].get('opponent')
            if behavior is not None and behavior!=ident:raise ValueError('Behavior and branch opponents differ')
        remaining=max(0.,p.time_limit-r['elapsed_seconds'])
        if abs(remaining-r['remaining_seconds'])>1e-6:raise ValueError('Time metadata mismatch')
        L=p.known_steps(r['elapsed_seconds'])
        expected=torch.arange(p.prediction_steps)<L
        if not torch.equal(data['known_time_valid'][idx].cpu(),expected):raise ValueError('Known interval differs from current time')
        context=torch.tensor([remaining/p.time_limit,L/p.prediction_steps],dtype=data['time_context'].dtype)
        if not torch.allclose(data['time_context'][idx].cpu(),context,atol=1e-6,rtol=1e-6):raise ValueError('Time context mismatch')
    for k,v in data.items():
        if not isinstance(v,torch.Tensor) or v.shape[0]!=n:raise ValueError('Invalid tensor '+k)
        if torch.is_floating_point(v) and not torch.isfinite(v).all():raise ValueError('Nonfinite '+k)
    tv=data['time_valid'];known=data['known_time_valid']
    if tv.dtype!=torch.bool or known.dtype!=torch.bool:raise ValueError('Boolean time masks required')
    if ((~tv[...,:-1])&tv[...,1:]).any() or not tv.any(-1).all():raise ValueError('Invalid stopped branch')
    if (tv&~known[:,None,None,:]).any():raise ValueError('Observed transition beyond known timeout')
    collection=blob.get('collection_schema')
    if collection is not None:
        if collection!=COLLECTION_SCHEMA or blob.get('branch_event_fields')!=BRANCH_EVENT_FIELDS:
            raise ValueError('Unknown branch event collection contract')
        validate_branch_events(data,meta,p)
    elif any(k.startswith('branch_') for k in data):
        raise ValueError('Branch events require an explicit collection schema')
    return {'roots':n,'geometry_families':len(mapping),'split_root_counts':{s:sum(r['split']==s for r in meta) for s in ('train','validation','test')}}


def validate_branch_events(data,roots,protocol):
    shape=data['time_valid'].shape[:-1]
    types={'branch_stop_step':torch.int64,'branch_outcome':torch.int64,
           'branch_terminated':torch.bool,'branch_prediction_truncated':torch.bool,
           'branch_stop_time':torch.float64,'branch_events':torch.bool}
    for name,dtype in types.items():
        expected=shape+(len(BRANCH_EVENT_FIELDS),) if name=='branch_events' else shape
        value=data.get(name)
        if value is None or value.shape!=expected or value.dtype!=dtype:
            raise ValueError('Invalid branch record '+name)
    steps=data['branch_stop_step'];codes=data['branch_outcome'];ended=data['branch_terminated']
    truncated=data['branch_prediction_truncated'];events=data['branch_events']
    if not torch.equal(steps,data['time_valid'].sum(-1)):raise ValueError('Branch stop step differs from observed prefix')
    if ((codes<0)|(codes>4)).any() or not torch.equal(ended,codes!=0):raise ValueError('Branch termination code mismatch')
    if not torch.equal(truncated,~ended) or (truncated&(steps!=protocol.prediction_steps)).any():
        raise ValueError('Prediction truncation is not a task terminal')
    elapsed=torch.tensor([r['elapsed_seconds'] for r in roots],dtype=torch.float64)[:,None,None]
    if not torch.allclose(data['branch_stop_time'],elapsed+steps*protocol.control_dt,atol=1e-5,rtol=0):
        raise ValueError('Branch stop timestamp differs from observed prefix')
    expected={'success':codes>2,'capture':codes==3,'collision':codes==2,
              'source_failure':codes==1,'timeout':codes==4,
              'radial_only_failure':(codes==1)&~events[...,BRANCH_EVENT_FIELDS.index('physical_breach')]}
    for name,value in expected.items():
        if not torch.equal(events[...,BRANCH_EVENT_FIELDS.index(name)],value):
            raise ValueError('Branch outcome/event mismatch '+name)
    if ((codes==4)&((data['branch_stop_time']-protocol.time_limit).abs()>1e-5)).any():
        raise ValueError('Timeout event before task time limit')


def check_split_families(groups):
    assigned={}
    for split,metas in groups:
        for meta in metas:
            family=verify_geometry(meta)
            if family in assigned and assigned[family]!=split:raise ValueError('Physical layout appears in more than one split')
            assigned[family]=split
    return assigned


def predictor_contract(protocol):
    """Only online risk_weight may vary for the preregistered one-factor study.
    It changes scoring, not the target D or the model input. All other fields
    (including prior normalization and clearance) remain compatibility checks.
    """
    value=asdict(protocol) if isinstance(protocol,Protocol) else dict(protocol)
    Protocol(**value)
    return {k:v for k,v in value.items() if k not in ('name','risk_weight')}


def validate_predictor_protocol(trained_protocol,online_protocol):
    if predictor_contract(trained_protocol)!=predictor_contract(online_protocol):
        raise ValueError('Predictor dynamics/input/label contract mismatch')
    return {'training_risk_weight':float(trained_protocol['risk_weight']),
            'online_risk_weight':float(online_protocol.risk_weight),
            'online_scoring_changed':float(trained_protocol['risk_weight'])!=float(online_protocol.risk_weight)}


def runtime_identity(project,source,checkpoint,protocol,control,device,method='CITF',opponent=None,code_entries=()):
    project=Path(project);here=Path(__file__).parent
    source_files=['train/envs/TADgame.py','train/envs/modules.py','train/policy/networks.py']
    if opponent is None:
        from .opponents import APFOpponent
        opponent=APFOpponent().identity
    dependencies=python_dependencies(*code_entries)
    return {'revision':RUNTIME_REVISION,'method':method,'schema':SCHEMA,
            'history_features':FEATURES,'context_features':CONTEXT,'protocol':asdict(protocol),
            'source_sha256':file_hash(source),'checkpoint_sha256':file_hash(checkpoint) if checkpoint else None,
            'project_code':{x:file_hash(project/x) for x in source_files},
            'runtime_code':canonical_hash({'core':tree_hash(here),'dependencies':dependencies}),
            'runtime_dependencies':dependencies,
            'controller':control,'opponent':opponent,'software':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'device':str(device)}


def verify_freeze(manifest,checkpoint,task_manifest,runtime=None):
    m=json.loads(Path(manifest).read_text());t=json.loads(Path(task_manifest).read_text())
    if runtime is None:raise ValueError('Full effective controller identity required')
    required={'frozen','runtime','runtime_sha256','tasks_sha256','selection_rule','method_matrix','primary_metric','registry_sha256'}
    if not required.issubset(m) or not m['frozen']:raise ValueError('Incomplete/nonfrozen manifest')
    if t['role']!='final_test' or m['tasks_sha256']!=file_hash(task_manifest):raise ValueError('Final task bank mismatch')
    if canonical_hash(m['runtime'])!=m['runtime_sha256'] or canonical_hash(runtime)!=m['runtime_sha256']:
        raise ValueError('Frozen controller changed: source/model/code/protocol/search/risk/software/device')
    if runtime['checkpoint_sha256']!=(file_hash(checkpoint) if checkpoint else None):raise ValueError('Checkpoint mismatch')
    return m
