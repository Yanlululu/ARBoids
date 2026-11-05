"""Register an already selected method; no model evaluation or tuning."""
import argparse,json
from pathlib import Path
from .contracts import canonical_hash,file_hash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',required=True);ap.add_argument('--tasks',required=True)
    ap.add_argument('--registry',required=True);ap.add_argument('--out',required=True);ap.add_argument('--selection-rule',required=True)
    ap.add_argument('--method-matrix',required=True);ap.add_argument('--confirm-method-finalized',action='store_true')
    a=ap.parse_args();out=Path(a.out)
    if out.exists() or not a.confirm_method_finalized:raise ValueError('Explicit finalized-method authorization required; no overwrite')
    runtime=json.loads(Path(a.runtime).read_text());tasks=json.loads(Path(a.tasks).read_text())
    matrix=json.loads(Path(a.method_matrix).read_text())
    if tasks['role']!='final_test':raise ValueError('Wrong task role')
    required={'revision','schema','history_features','context_features','protocol','source_sha256','checkpoint_sha256','project_code','runtime_code','controller','software','device','method','opponent'}
    if not required.issubset(runtime):raise ValueError('Incomplete runtime identity')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'frozen':True,'runtime':runtime,'runtime_sha256':canonical_hash(runtime),
        'tasks_sha256':file_hash(a.tasks),'registry_sha256':file_hash(a.registry),'selection_rule':a.selection_rule,
        'method_matrix':matrix,'primary_metric':'defense_success'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
