"""Validate publication roles without renaming or changing archived evidence."""
from pathlib import Path
import json,re,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[1]
paper=(ROOT/'paper/CITF_中文独立论文_修订稿.md').read_text()
manifest=json.loads((ROOT/'release/experiment_manifest.json').read_text())
registry=json.loads((ROOT/'release/method_registry.json').read_text())
errors=[]
def check(ok,msg):
    if not ok: errors.append(msg)
check(not re.search(r'原\s*CI|旧\s*CI|CI-original|CI-physical|升级CI|改进CI',paper),'Unpublished development identity leaked into manuscript.')
check(not re.search(r'(?<![A-Za-z])CI(?![A-Za-z])',paper),'Standalone CI predecessor remains in paper.')
check('本文构造CITF-Phys' in paper,'Author-constructed physical control is not defined.')
check('保持推力' in paper and '名义同步最小间距' in paper,'Physical control definition missing.')
check(not re.search(r'\\tag\{|\\begin\{equation\}',paper),'Numbered equation found.')
check(paper.count('\\[')==paper.count('\\]'),'Unbalanced display math.')
check(paper.count('\\(')==paper.count('\\)'),'Unbalanced inline math.')
check('**待补**' in paper,'Pending empirical results were removed.')
start=paper.index('**表3')
end=paper.index('### 2',start)
rows=[line.split('|')[1].strip() for line in paper[start:end].splitlines() if line.startswith('|') and '**待补**' in line]
check(rows==manifest['main_methods'],'Main paper table and declared published-baseline comparison differ.')
check(registry['paper_main_table']==rows,'Method registry and main table differ.')
check('CI-original' not in manifest['main_methods'],'Internal CI still in formal main algorithms.')
check(manifest['attacker']['reference_defender'].startswith('fixed development CITF-Phys'),'Attacker reference not aligned.')
check(manifest['vrx']['methods']==['CITF','CITF-Phys','ARBoids'],'VRX role definitions not aligned.')
check(manifest['source_protocol']['time_limit']==80 and manifest['primary_training_team_size']==6,'Task protocol changed.')
check(manifest['training']['source_seeds']==[101,102,103,104,105],'Independent source seeds changed.')
checks={
    'checks_passed':15-len(errors),'checks_total':15,'errors':errors,
    'public_comparison':rows,'author_control':'CITF-Phys',
    'algorithm_code_changed':False,'new_training_results':False,
    'scope':'Document identity, experimental role and mathematical-format consistency only.'}
out=ROOT/'audit/narrative_revision/narrative_checks.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
if errors: sys.exit(1)
