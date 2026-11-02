"""Paired hierarchical bootstrap, retaining every run and physical-layout family."""
import argparse,json
from pathlib import Path
import numpy as np
from citf.contracts import file_hash,canonical_hash


def load_results(directory):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if not summary['completed']:raise ValueError('Incomplete evaluation '+str(directory))
    rows=[json.loads(line) for line in (directory/'tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=summary['N'] or len({r['task']['task_id'] for r in rows})!=len(rows):raise ValueError('Missing or repeated evaluation tasks')
    for r in rows:
        if file_hash(directory/r['trace'])!=r['trace_sha256']:raise ValueError('Trace differs from committed task')
    return summary,{r['task']['task_id']:r for r in rows}


def paired_difference(left,right,metric='success',repetitions=10000,seed=801):
    if len(left)!=len(right) or not left:raise ValueError('Every run needs a matched control')
    grouped=[];per_run=[];role=None;common_tasks=None
    for a,b in zip(left,right):
        sa,ra=load_results(a);sb,rb=load_results(b)
        if sa['phase']!=sb['phase'] or sa['tasks_sha256']!=sb['tasks_sha256'] or set(ra)!=set(rb):raise ValueError('Unpaired task banks or phases')
        if role is None:role=sa['phase'];common_tasks=sa['tasks_sha256']
        if role!=sa['phase'] or common_tasks!=sa['tasks_sha256']:raise ValueError('Run repetitions need the same task bank and role')
        for key in ('protocol','source_sha256','opponent'):
            if canonical_hash(sa['runtime'][key])!=canonical_hash(sb['runtime'][key]):raise ValueError('Unmatched runtime '+key)
        families={};rescues=harms=0
        for tid in sorted(ra):
            x,y=ra[tid],rb[tid]
            if x['task']!=y['task']:raise ValueError('Changed task content')
            family=x['task']['geometry_family_id'];families.setdefault(family,[]).append(float(x[metric])-float(y[metric]))
            rescues+=int(x['success']==1 and y['success']==0);harms+=int(x['success']==0 and y['success']==1)
        grouped.append(families)
        per_run.append({'left':str(a),'right':str(b),'N':len(ra),'left_success':sum(x['success'] for x in ra.values()),
                        'right_success':sum(x['success'] for x in rb.values()),'rescues':rescues,'harms':harms,
                        'difference':float(np.mean([v for xs in families.values() for v in xs]))})
    family_order=sorted(grouped[0])
    if any(sorted(g)!=family_order for g in grouped):raise ValueError('Different family coverage across runs')
    if len({len(xs) for g in grouped for xs in g.values()})!=1:raise ValueError('Unequal family condition coverage')
    values=np.array([[g[k] for k in family_order] for g in grouped]);rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(repetitions):
        run_ids=rng.integers(len(grouped),size=len(grouped));sampled=[]
        for i in run_ids:sampled.append(values[i,rng.integers(len(family_order),size=len(family_order))].mean())
        estimates.append(np.mean(sampled))
    return {'phase':role,'metric':metric,'runs':len(grouped),'families':len(family_order),'paired_tasks_per_run':sum(map(len,grouped[0].values())),
            'mean_difference':float(values.mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'bootstrap_repetitions':repetitions,'per_run':per_run,'tasks_sha256':common_tasks,
            'replication_scope':'fixed-source predictor repetitions' if len({load_results(p)[0]['runtime']['source_sha256'] for p in left})==1 else 'multiple sources'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--left',nargs='+',required=True);p.add_argument('--right',nargs='+',required=True)
    p.add_argument('--metric',default='success');p.add_argument('--out',required=True);a=p.parse_args()
    result=paired_difference(a.left,a.right,a.metric);out=Path(a.out)
    if out.exists():raise ValueError('Refuse statistical result overwrite')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
