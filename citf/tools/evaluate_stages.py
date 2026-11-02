"""Evaluate immutable current-stage snapshots for Fig. 5, using VALIDATION only.

Each stage runs the complete controller on the same task bank. Best-so-far
checkpoints are not substituted for current-stage weights. No automatic
selection using these results overrides the registered predictor selection.
"""
from pathlib import Path
import argparse,subprocess,json,hashlib,sys,csv,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    for key in ['run','project','source','tasks','registry','out']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--bank',type=int,default=12)
    ap.add_argument('--budget',type=int,default=96);ap.add_argument('--rounds',type=int,default=1)
    ap.add_argument('--risk-weight',type=float,default=32.);ap.add_argument('--opponent-spec')
    a=ap.parse_args();a=vars(a)
    for key in ['run','project','source','tasks','registry','out']:
        a[key]=str(Path(a[key]).resolve())
    if a['opponent_spec']:a['opponent_spec']=str(Path(a['opponent_spec']).resolve())
    tasks=json.loads(Path(a['tasks']).read_text())
    if tasks['role']!='validation':raise ValueError('Stage curves must not evaluate test/final_test banks')
    run=Path(a['run']);index=json.loads((run/'snapshots.json').read_text());out=Path(a['out'])
    if out.exists():raise ValueError('Refuse to overwrite stage curve results')
    out.mkdir(parents=True);rows=[]
    for stage in index['entries']:
        checkpoint=run/stage['path']
        if sha(checkpoint)!=stage['checkpoint_sha256']:raise ValueError('Stage snapshot changed')
        destination=out/f"update_{stage['update']:06d}"
        cmd=[sys.executable,'-m','citf.evaluate','--project',a['project'],'--source',a['source'],
             '--tasks',a['tasks'],'--registry',a['registry'],'--checkpoint',str(checkpoint),
             '--out',str(destination),'--device',a['device'],'--bank',str(a['bank']),
             '--budget',str(a['budget']),'--rounds',str(a['rounds']),'--risk-weight',str(a['risk_weight'])]
        if a['opponent_spec']:cmd+=['--opponent-spec',a['opponent_spec']]
        with (out/f"update_{stage['update']:06d}.log").open('w') as log:
            subprocess.run(cmd,cwd=ROOT/'code',stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((destination/'summary.json').read_text())
        rows.append({'update':stage['update'],'N':report['N'],'success':report['success'],
          'capture':report['capture'],'collision':report['collision'],'capped_capture_time':report['capped_capture_time'],
          'checkpoint_sha256':sha(checkpoint),'tasks_sha256':sha(a['tasks']),'result':str(destination/'summary.json')})
        (out/'curve.json').write_text(json.dumps({'phase':'validation-stage-curve','rows':rows,'arguments':a},ensure_ascii=False,indent=2))
    with (out/'curve.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
