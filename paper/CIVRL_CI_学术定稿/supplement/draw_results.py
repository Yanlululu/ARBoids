from pathlib import Path
import csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(_font)
_fontname=font_manager.FontProperties(fname=_font).get_name()
from PIL import Image,ImageOps,ImageDraw,ImageFont
R=Path(Path('/mnt/data/civrl_revision_work/evidence_path.txt').read_text());O=Path('/mnt/data/CIVRL_CI_学术定稿/figures')
plt.rcParams.update({'font.family':_fontname,'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
def rows(name):return list(csv.DictReader((R/'tables'/name).open(encoding='utf-8-sig')))
def axnew(w=4.8,h=3.5):
 f=plt.figure(figsize=(w,h));a=f.add_subplot(111);return f,a
def save(f,name):
 f.tight_layout(pad=1.0);f.savefig(O/(name+'.pdf'),bbox_inches='tight');f.savefig(O/(name+'.png'),bbox_inches='tight',dpi=320);plt.close(f)
# Recorded trajectories, without smoothing or imputation.
a=np.load(R/'vrx-cuda/case000-CIV/trajectory.npz');defs=a['DefPos'].reshape(-1,6,2)
f,ax=axnew(4.8,4.4)
for i in range(6):
 ax.plot(defs[:,i,0],defs[:,i,1],label=f'D{i+1}',lw=1.4)
 ax.plot(defs[0,i,0],defs[0,i,1],marker='o',ms=3,ls='none')
ax.plot(a['AttPos'][:,0],a['AttPos'][:,1],label='攻击艇',lw=1.7,ls='--')
ax.add_patch(plt.Circle((0,0),15,fill=False,lw=1,ls=':'))
ax.text(0,0,'保护区域',ha='center',va='center',fontsize=9)
ax.set(xlabel='x / m',ylabel='y / m',aspect='equal',xlim=(-20,50),ylim=(-20,50))
ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
save(f,'fig4b_trajectory')
# Source learning, actual recorded one-run histories.
rr=rows('Fig_5_learning.csv');f,ax=axnew(5,3.5)
for m,l,ls in [('arboids','ARBoids','-'),('rp','RP','--'),('sac','SAC',':')]:
 x=[r for r in rr if r['stage']=='base control' and r['method']==m]
 ax.plot([float(v['updates_or_steps'])/1e6 for v in x],[float(v['success_percent']) for v in x],label=l,ls=ls,lw=1.1)
ax.set(xlabel='环境交互步数 / 百万',ylabel='评价成功率 / %',xlim=(0,1),ylim=(0,100));ax.legend(frameon=False,ncol=3,fontsize=8);ax.grid(axis='y',alpha=.2)
save(f,'fig5a_source')
f,ax=axnew(5,3.5);r=[x for x in rr if x['stage']=='CV full policy']
ax.plot([0,1500,3000,4500],[float(x['success_percent']) for x in r],marker='o',lw=1.5,label='CIV-RL')
ax.axhline(90,ls='--',lw=1,label='CI 参考')
for x,y in zip([0,1500,3000,4500],[90,90,88.5,87.5]):ax.annotate(f'{y:g}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
ax.set(xlabel='每成员价值更新次数',ylabel='闭环成功率 / %',ylim=(0,100));ax.set_xticks([0,1500,3000,4500]);ax.legend(frameon=False,loc='lower left');ax.grid(axis='y',alpha=.2)
save(f,'fig5b_value')
f,ax=axnew(8.1,3.8);rr=rows('Fig_5_eight_methods.csv');labels=['CIV-RL','w/o CI','w/o CV','ARBoids','RP','SAC','Boids','MAPPO']
y=np.array([float(x['success_percent']) for x in rr]);lo=np.array([float(x['wilson95_low']) for x in rr]);hi=np.array([float(x['wilson95_high']) for x in rr]);xx=np.arange(8)
ax.bar(xx,y,width=.66,yerr=np.vstack((y-lo,hi-y)),capsize=4,linewidth=.8)
for x,v,u in zip(xx,y,hi):ax.text(x,u+1.5,f'{v:g}',ha='center',va='bottom',fontsize=9)
ax.set_xticks(xx,labels);ax.set(ylabel='防御成功率 / %',ylim=(0,104));ax.grid(axis='y',alpha=.2)
save(f,'fig5c_comparison')
# Matched condition sweeps.
scope=rows('Fig_6_robustness_and_fleet.csv')
for kind,name,xlabel,levels in [('agility','fig6a_agility','攻击者机动系数',[1.,1.5,2.,2.5,3.,3.5]),('fleet','fig6b_fleet','防御艇数量',[3,4,5,6,7,8])]:
 f,ax=axnew(5.2,3.7)
 for m,label,ls,marker in [('CIV','CIV-RL','-','o'),('CI','仅 CI','--','s'),('arboids','ARBoids','-.','^'),('rp','RP',':','d')]:
  d={float(x['cell'].split('-')[1]):float(x['success_percent']) for x in scope if x['cell'].startswith(kind+'-') and x['method']==m}
  ax.plot(levels,[d[v] for v in levels],label=label,ls=ls,marker=marker,lw=1.4,ms=5)
 ax.set(xlabel=xlabel,ylabel='防御成功率 / %',ylim=(0,104));ax.set_xticks(levels);ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,frameon=False,fontsize=8,loc='lower left')
 save(f,name)
# Actual continued attacker interactions and task retention, all stages retained.
rr=rows('Fig_7_learning_attacker.csv');stages=['attacker-0','attacker-8000','attacker-16000','attacker-final']
d={r['stage']:r for r in rr}
f,ax=axnew(5.4,3.6)
x=[0,8000,16000,32883]
for field,label,ls,marker in [('success','防御成功','-','o'),('attacker_win','攻击成功','--','s'),('collision','艇间碰撞',':','^')]:
 ax.plot(np.array(x)/1000,[100*int(d[s][field])/64 for s in stages],ls=ls,marker=marker,label=label,lw=1.5,ms=5)
ax.set(xlabel='攻击者新增交互步数 / 千',ylabel='任务比例 / %',ylim=(-2,104));ax.set_xticks(np.array(x)/1000,['0','8','16','32.883']);ax.legend(frameon=False,ncol=3,fontsize=8,loc='center right');ax.grid(axis='y',alpha=.2)
save(f,'fig7a_attacker')
f,ax=axnew(5.4,3.6);x=np.arange(3)
for vals,label,marker,ls in [([59,59,60],'学习型攻击者','o','-'),([63,61,63],'原人工势场攻击者','s','--')]:
 y=np.array(vals)/64*100;ax.plot(x,y,marker=marker,ls=ls,lw=1.5,label=label)
 for i,yy,n in zip(x,y,vals):ax.annotate(f'{n}/64',(i,yy),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
ax.set_xticks(x,['更新前','当前对手更新','加入原任务']);ax.set(ylabel='防御成功率 / %',ylim=(85,103));ax.legend(frameon=False,fontsize=8,loc='lower left');ax.grid(axis='y',alpha=.2)
save(f,'fig7b_retention')
# Composite only after each data chart has been rendered separately.
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',38)
def label(im,t):
 im=im.convert('RGB');out=Image.new('RGB',(im.width,im.height+65),'white');out.paste(im,(0,65));ImageDraw.Draw(out).text((16,10),t,font=font,fill='black');return out
# All measured plot panels retain their own axes and legends.
def side(files,titles,target):
 ims=[Image.open(O/p).convert('RGB') for p in files];height=max(i.height for i in ims)
 ims=[label(i.resize((round(i.width*height/i.height),height),Image.Resampling.LANCZOS),t) for i,t in zip(ims,titles)];height=ims[0].height
 out=Image.new('RGB',(sum(i.width for i in ims)+50,height),'white');x=0
 for i in ims:out.paste(i,(x,0));x+=i.width+50
 out.save(O/target)
side(['fig6a_agility.png','fig6b_fleet.png'],['(a) 攻击者机动性','(b) 未见艇队规模'],'fig6_scope.png')
side(['fig7a_attacker.png','fig7b_retention.png'],['(a) 攻击策略继续学习','(b) 防御更新与任务保持'],'fig7_adaptation.png')
side(['fig5a_source.png','fig5b_value.png'],['(a) 源策略学习','(b) 条件价值学习'],'fig5_top.png')
upper=Image.open(O/'fig5_top.png');lower=label(Image.open(O/'fig5c_comparison.png'),'(c) 独立任务性能')
lower=lower.resize((upper.width,round(lower.height*upper.width/lower.width)),Image.Resampling.LANCZOS)
image=Image.new('RGB',(upper.width,upper.height+lower.height+35),'white');image.paste(upper,(0,0));image.paste(lower,(0,upper.height+35));image.save(O/'fig5_performance.png')
side(['fig4a_vrx_scene.png','fig4b_trajectory.png'],['(a) 高保真场景  t = 6.95 s','(b) 实测闭环轨迹'],'fig4_vrx.png')
print('Saved results and panel compositions')
