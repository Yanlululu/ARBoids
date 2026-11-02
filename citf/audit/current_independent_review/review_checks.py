"""Read-only checks of the supplied CITF handoff; no performance evaluation."""
from pathlib import Path
import argparse,sys,json,re,hashlib
import numpy as np
import torch

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--package-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.package_root.resolve();sys.path.insert(0,str(root/'code'));torch.set_num_threads(1)
    from citf.bridge import Source,World,Controller,factorize
    from citf.collect import sample_root
    from citf.contracts import tree_hash
    mh=json.loads((root/'release/FILE_MANIFEST.json').read_text());bad=[]
    for row in mh['files']:
        p=root/row['path']
        if not p.is_file() or len(p.read_bytes())!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:bad.append(row['path'])
    manifest=json.loads((root/'release/experiment_manifest.json').read_text());reg=json.loads((root/'release/method_registry.json').read_text());paper=(root/'paper/CITF_中文独立论文_修订稿.md').read_text()
    checks={'file_manifest':{'count':len(mh['files']),'mismatches':bad},'roles':{'standalone_CI_in_paper':len(re.findall(r'(?<![A-Za-z])CI(?![A-Za-z])',paper)),'main_table_matches_registry':manifest['main_methods']==reg['paper_main_table'],'Phys_execution':reg['definitions']['CITF-Phys']['execution'],'attacker_reference':manifest['attacker']['reference_defender'],'vrx_methods':manifest['vrx']['methods']},'runtime_code_hash':tree_hash(root/'code/citf')}
    project=root/'source_archive';ck=next(project.rglob('best-m1-400.pth'));source=Source(project,ck)
    score_checks=[]
    for elapsed in (0.,79.8):
        w=World(project);w.reset(9621000,9621001,2.);w.env.Current_T=elapsed
        force,trace=Controller(source,model=None,bank=12,budget=96,rounds=1).act(w)
        f=factorize(w,source);plans=trace['configurations'];L=w.protocol.known_steps(elapsed);N=w.env.defender_num
        energies=[]
        for row in plans:
            trajectory=torch.stack([f['paths'][i,row[i],:L] for i in range(N)]).double()
            uni=sum(float(f['unary'][i,row[i]]) for i in range(N));risk=0.
            for i in range(N):
                for j in range(i+1,N):
                    d=float((trajectory[i]-trajectory[j]).norm(dim=-1).min());risk+=max(0.,w.protocol.risk_clearance-d)**2
            energies.append(uni+w.protocol.risk_weight*risk)
        score_key='scores' if 'scores' in trace else 'energies'
        if score_key not in trace: raise KeyError(list(trace))
        diff=float(np.max(np.abs(np.array(energies)-np.array(trace[score_key]))))
        score_checks.append({'elapsed':elapsed,'known_steps':L,'unique_configurations':trace['evaluated_unique'],'source_center_scored':trace['source_center_scored'],'max_scoring_difference':diff,'bounded_thrust':bool(np.isfinite(force).all() and (force>=-500.001).all() and (force<=1000.001).all())})
    checks['phys_definition_checks']=score_checks
    # Observe the real terminal event, then inspect what the collector actually returns.
    w=World(project);w.reset(9621000,9621001,2.);w.env.Current_T=79.8
    events=[];original=World.step
    def watch(self,*args,**kwargs):
        result=original(self,*args,**kwargs);events.append(result.copy());return result
    World.step=watch
    try:tensors,meta,count=sample_root(w,source,np.full((1,6),4),[9621011],{'root_id':'constructed-terminal-probe','layout_id':'review','split':'test'})
    finally:World.step=original
    fields=list(tensors)+list(meta)
    checks['branch_event_recording']={'constructed_probe':True,'actual_steps':count,'time_valid':tensors['time_valid'].sum(-1).tolist(),'observed_world_step_events':events,'returned_tensor_fields':list(tensors),'returned_metadata_fields':list(meta),'structured_terminal_event_saved':any(k in fields for k in ['terminal_events','terminal_outcome','outcome','outcomes','branch_events','events','terminal_flags'])}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False,indent=2))
    assert not bad
    assert all(s['source_center_scored'] and s['bounded_thrust'] and s['max_scoring_difference']<.01 for s in score_checks)
if __name__=='__main__':main()
