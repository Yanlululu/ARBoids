"""Allocate physical layout families before collection/evaluation."""
import argparse,json
from pathlib import Path
from dataclasses import asdict,replace
from .bridge import World
from .contracts import Protocol,LayoutRegistry,canonical_hash,validate_task_banks,CURRENT_PROTOCOL,LEGACY_PROTOCOL

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project');ap.add_argument('--registry',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--role',choices=['train','validation','test','final_test'])
    ap.add_argument('--seed-start',type=int);ap.add_argument('--layouts',type=int)
    ap.add_argument('--from-bank',help='Migrate an existing bank while preserving tasks and split allocation')
    ap.add_argument('--protocol',choices=[CURRENT_PROTOCOL,LEGACY_PROTOCOL],default=CURRENT_PROTOCOL)
    ap.add_argument('--defenders',type=int,default=6);ap.add_argument('--agilities',type=float,nargs='+',default=[2.,2.5])
    ap.add_argument('--cross-agilities',action='store_true');ap.add_argument('--future-offset',type=int,default=10000000)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Refuse task bank overwrite')
    registry=LayoutRegistry(a.registry);p=Protocol(name=a.protocol);rows=[]
    if a.from_bank:
        original=json.loads(Path(a.from_bank).read_text(encoding='utf-8'))
        validate_task_banks((original['role'],original))
        if a.role and a.role!=original['role']:raise ValueError('Migration preserves the allocated split')
        for task in original['tasks']:
            allocation=registry.data['families'].get(task['geometry_family_id'])
            if not allocation or allocation['split']!=original['role']:
                raise ValueError('Migrated layout must retain its registered split')
        migrated=dict(original,protocol=asdict(replace(Protocol(**original['protocol']),name=p.name,
            radial_early_failure=p.radial_early_failure,apf_obstacle_rule=p.apf_obstacle_rule)),
            migration=dict(parent_bank=canonical_hash(original),parent_protocol=original['protocol'],
                changes=['APF obstacle aggregation','radial termination'],tasks_preserved=True))
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(migrated,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"Migrated {len(migrated['tasks'])} tasks to {p.name}")
        return
    if not a.project or not a.role or a.seed_start is None or a.layouts is None or a.layouts<1:
        ap.error('Generation requires --project, --role, --seed-start and positive --layouts')
    for i in range(a.layouts):
        seed=a.seed_start+i;levels=a.agilities if a.cross_agilities else [a.agilities[i%len(a.agilities)]]
        for agility in levels:
            world=World(a.project,p,a.defenders);world.reset(seed,a.future_offset+seed,agility)
            ident=world.initial_identity;alias=f'n{a.defenders}_scene{seed}'
            family=registry.register(ident,a.role,alias)
            row={'layout_id':alias,'geometry_family_id':family,'scene_seed':seed,'future_seed':a.future_offset+seed,
                 'defenders':a.defenders,'agility':agility}
            row['task_id']=canonical_hash(row)[:24];rows.append(row)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'schema':'citf-task-bank-v2','role':a.role,'protocol':asdict(p),'tasks':rows},ensure_ascii=False,indent=2))
    print(f'{len(rows)} tasks in {a.layouts} layout families')
if __name__=='__main__':main()
