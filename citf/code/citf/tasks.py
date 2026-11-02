"""Allocate physical layout families before collection/evaluation."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
from .bridge import World
from .contracts import Protocol,LayoutRegistry,canonical_hash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--registry',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--role',required=True,choices=['train','validation','test','final_test'])
    ap.add_argument('--seed-start',type=int,required=True);ap.add_argument('--layouts',type=int,required=True)
    ap.add_argument('--defenders',type=int,default=6);ap.add_argument('--agilities',type=float,nargs='+',default=[2.,2.5])
    ap.add_argument('--cross-agilities',action='store_true');ap.add_argument('--future-offset',type=int,default=10000000)
    a=ap.parse_args();out=Path(a.out)
    if out.exists() or a.layouts<1:raise ValueError('Refuse overwrite or invalid layout count')
    registry=LayoutRegistry(a.registry);p=Protocol();rows=[]
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
