from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import base64,io,cairosvg,csv,json,numpy as np
R=Path(Path('/mnt/data/civrl_revision_work/evidence_path.txt').read_text())
O=Path('/mnt/data/CIVRL_CI_学术定稿/figures');O.mkdir(parents=True,exist_ok=True)
im=Image.open('/mnt/data/a_clean_scientific_infographic_illustration_on_a_w.png').convert('RGB')
# New illustrative artwork is kept separate from measured data.
im.crop((450,0,1340,333)).save(O/'fig1_task.png')
im.crop((27,808,1514,1015)).save(O/'fig3_mechanism.png')
assets={
 'sac':im.crop((75,420,248,465)),
 'boids':im.crop((75,540,248,592)),
 'fleet0':im.crop((324,432,477,562)),
 'fleet1':im.crop((492,432,646,562)),
 'fleet2':im.crop((659,432,813,562)),
 'network':im.crop((899,442,1020,547)),
 'root':im.crop((306,670,439,753)),
 'branch0':im.crop((606,646,733,697)),
 'branch1':im.crop((606,722,733,774)),
}
def embed(name,x,y,w,h):
 b=io.BytesIO();assets[name].save(b,format='PNG')
 return f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"/>'
# Publication figure is manually typeset around the redrawn illustrative assets.
s=['''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1600" height="850" viewBox="0 0 1600 850">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#24364a"/></marker></defs>
<rect width="1600" height="850" fill="white"/>
<style>text{font-family:'Noto Sans CJK SC',sans-serif;fill:#172b40} .t{font-size:25px;font-weight:600} .s{font-size:22px} .m{font-family:'Liberation Serif',serif;font-size:29px;font-style:italic} .a{fill:none;stroke:#24364a;stroke-width:2.2;marker-end:url(#a)} .d{fill:none;stroke:#6c6875;stroke-width:2;stroke-dasharray:8 5;marker-end:url(#a)}</style>
<rect x="30" y="130" width="225" height="151" rx="9" fill="#f2f6fa" stroke="#3a5a78" stroke-width="1.8"/>
<rect x="30" y="311" width="225" height="151" rx="9" fill="#f3f8f6" stroke="#487265" stroke-width="1.8"/>
<rect x="315" y="112" width="570" height="368" rx="11" fill="#f3f7fb" stroke="#315b83" stroke-width="2.1"/>
<rect x="938" y="160" width="252" height="286" rx="10" fill="#fbf7f1" stroke="#9c7b57" stroke-width="1.8"/>
<rect x="1250" y="162" width="316" height="284" rx="10" fill="#f3f7f3" stroke="#567861" stroke-width="1.8"/>
<text class="t" x="142" y="165" text-anchor="middle">SAC 学习控制</text>
<text class="t" x="142" y="348" text-anchor="middle">Boids 规则控制</text>
<text class="t" x="600" y="153" text-anchor="middle">CI 候选交互联合融合</text>
<text class="s" x="600" y="190" text-anchor="middle">反馈运动响应与艇对相容性</text>
<text class="t" x="1064" y="201" text-anchor="middle">CV 条件价值学习</text>
<text class="t" x="1408" y="205" text-anchor="middle">联合融合配置选择</text>
<text class="s" x="600" y="420" text-anchor="middle">局部融合先验  ＋  艇对交互代价</text>
<text class="m" x="600" y="458" text-anchor="middle">E<tspan font-size="18" dy="7">CI</tspan><tspan dy="-7">(x,G)</tspan></text>
<text class="m" x="1064" y="393" text-anchor="middle">p₀(x), r(x,z), h(x,z)</text>
<text class="s" x="1064" y="425" text-anchor="middle">完整方案条件任务价值</text>
<text class="m" x="1408" y="279" text-anchor="middle">z*  →  G*</text>
<text class="s" x="1408" y="326" text-anchor="middle">保留参考  或  选择候选</text>
<text class="s" x="1408" y="390" text-anchor="middle">通道融合与艇队执行</text>
<text class="m" x="1408" y="426" text-anchor="middle">G*  →  u</text>
<path class="a" d="M255 206 H315"/><path class="a" d="M255 387 H315"/>
<path class="a" d="M885 293 H938"/><path class="a" d="M1190 293 H1250"/>
<path class="a" d="M880 128 V70 H1408 V160"/>
<text class="s" x="1090" y="57" text-anchor="middle">联合候选集合  Z_CI(x)</text>
<path class="a" d="M1566 386 H1580 V25 H142 V129"/>
<text class="s" x="610" y="48" text-anchor="middle">观测状态 x</text>
<rect x="305" y="541" width="887" height="266" rx="9" fill="#fcfcfc" stroke="#8b9299" stroke-width="1.5" stroke-dasharray="9 6"/>
<text class="t" x="748" y="578" text-anchor="middle">配对闭环监督</text>
<text class="m" x="416" y="758" text-anchor="middle">x, ω</text>
<rect x="550" y="613" width="251" height="68" rx="7" fill="white" stroke="#9d8f7b"/>
<rect x="550" y="717" width="251" height="68" rx="7" fill="white" stroke="#6b8a79"/>
<text class="m" x="574" y="654">z₀</text><text class="m" x="574" y="759">z</text>
<path class="a" d="M484 694 H515 V647 H548"/><path class="a" d="M515 694 V751 H548"/>
<path class="a" d="M802 648 H870"/><path class="a" d="M802 752 H870"/>
<text class="m" x="905" y="657" text-anchor="middle">Y₀</text><text class="m" x="905" y="761" text-anchor="middle">Yz</text>
<path class="d" d="M936 651 H970 V696 H1000"/><path class="d" d="M936 753 H970 V696"/>
<rect x="1004" y="658" width="151" height="79" rx="8" fill="#fbf2f1" stroke="#a07171"/>
<text class="m" x="1079" y="709" text-anchor="middle">L<tspan font-size="18" dy="7">CV</tspan></text>
<path class="d" d="M1079 657 V448"/>
<text class="s" x="1090" y="517">参数训练</text>
<text class="s" x="1280" y="616">实线  在线决策</text><text class="s" x="1280" y="658">虚线  价值训练</text>
</svg>''']
svg=s[0]
extra=embed('sac',58,182,169,56)+embed('boids',58,365,169,57)
extra+='<text class="m" x="143" y="267" text-anchor="middle">uᴸ</text><text class="m" x="143" y="449" text-anchor="middle">uᴮ</text>'
for j,name in enumerate(['fleet0','fleet1','fleet2']):
 extra+=embed(name,339+j*176,216,154,135)+f'<text class="m" x="{416+j*176}" y="380" text-anchor="middle">G{["₀","₁","₂"][j]}</text>'
extra+=embed('network',995,226,140,128)+embed('root',350,620,134,107)+embed('branch0',618,622,167,50)+embed('branch1',618,726,167,50)
svg=svg.replace('</svg>',extra+'</svg>')
(O/'fig2_architecture.svg').write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(O/'fig2_architecture.pdf'))
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(O/'fig2_architecture.png'),scale=2)
# Measured VRX scene. Geometric crop retains the entire interacting vessel group and dock context.
vrx=Image.open(R/'vrx-cuda/case000-CIV/frame-002.png').convert('RGB')
vrx.crop((370,230,990,835)).save(O/'fig4a_vrx_scene.png')
# Numerical audit of all main rows.
rows=list(csv.DictReader((R/'tables/per_task_outcomes.csv').open(encoding='utf-8-sig')))
main=list(csv.DictReader((R/'tables/Fig_5_eight_methods.csv').open(encoding='utf-8-sig')))
audit=[]
for row in main:
 rr=[x for x in rows if x['cell']=='confirmation' and x['method']==row['method']]
 for key in ['success','capture','collision','physical_breach','timeout']:
  assert sum(int(x[key]) for x in rr)==int(row[key]),(row['method'],key)
 assert len(rr)==int(row['episodes'])
 audit.append({'method':row['method'],'tasks':len(rr),'verified':True})
sel={m:{(x['scene_seed'],x['future']):x for x in rows if x['cell']=='confirmation' and x['method']==m} for m in ['CIV','CI','ordinary','independent']}
common=[k for k in sel['CIV'] if int(sel['CIV'][k]['capture']) and int(sel['CI'][k]['capture'])]
d=np.array([float(sel['CIV'][k]['duration'])-float(sel['CI'][k]['duration']) for k in common]);rng=np.random.default_rng(26027)
boot=d[rng.integers(0,len(d),(15000,len(d)))].mean(1)
report={'main_checks':audit,'paired_capture_CIV_minus_CI':{'n':len(d),'mean_seconds':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()},'VRX':{}}
for m in ['CIV','CI','arboids']:
 rs=[]
 for p in (R/'vrx-cuda').glob('*/result.json'):
  a=json.loads(p.read_text())
  if a.get('passed') and a.get('method')==m:rs.append(a)
 assert len(rs)==12,(m,len(rs))
 report['VRX'][m]={'n':len(rs),'success':sum(a['success'] for a in rs),'mean_capture_s':float(np.mean([a['simulation_seconds'] for a in rs if a['success']]))}
Path('/mnt/data/civrl_revision_work/data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
