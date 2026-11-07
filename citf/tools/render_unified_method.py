"""Render the manuscript's current control and learning schematic as vectors."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
fig, ax = plt.subplots(figsize=(17, 10))
fig.subplots_adjust(left=.012, right=.988, bottom=.015, top=.985)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis('off')
ink, blue, ochre = '#243744', '#2C708A', '#AA7425'
ax.add_patch(Rectangle((.005, .43), .99, .56, color='#F5F8FA', zorder=-2))
ax.add_patch(Rectangle((.005, .005), .99, .40, color='#FAF8F4', zorder=-2))


def text(x, y, content, size=15, weight='normal', color=ink, ha='center'):
    ax.text(x, y, content, fontproperties=FONT, fontsize=size, color=color,
            ha=ha, va='center', fontweight=weight, linespacing=1.7)


def box(x, y, width, height, content, colored=False, size=14):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle='round,pad=0.007,rounding_size=0.006',
                 linewidth=1.7 if colored else 1.0,
                 edgecolor=blue if colored else '#7C8B94',
                 facecolor='#EAF1F5' if colored else 'white'))
    text(x+width/2, y+height/2, content, size)


def arrow(points, learning=False):
    style = dict(color=ochre if learning else ink, linewidth=1.7,
                 linestyle=(0, (5, 4)) if learning else '-', zorder=0)
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), **style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='->',
                                mutation_scale=16, **style))


text(.03, .95, '联合融合控制：每个控制周期', 19, 'bold', ha='left')
box(.025, .705, .15, .17, '当前状态 $X$\n上一实际推力\n回合固定艇序')
box(.215, .715, .14, .15, '共享学习端点 $a_i$\nBoids端点 $b_i$\n同一可执行推力盒', True)
box(.395, .695, .17, .19,
    '自回归联合生成\n$G^{k,0}\\sim q_\\psi$\n$K$组样本＋2个锚点', True)
box(.605, .715, .16, .15,
    '实际联合动作\n$U(X,G)=\\mathcal{P}_X$\n$[B+G\\odot(A-B)]$')
box(.805, .67, .17, .24,
    '实际动作条件价值\nTransformer与艇对响应\n'
    '$\\widehat Q=\\frac{1}{n}\\sum_i\\min_m\\mathcal{Q}_{m,i}$\n'
    '近邻汇聚，整队评分', True, 13)

arrow([(.175, .79), (.215, .79)])
arrow([(.355, .79), (.395, .79)])
arrow([(.565, .79), (.605, .79)])
arrow([(.765, .79), (.805, .79)])

box(.715, .47, .20, .15,
    '连续联合改进\n有限步投影梯度\n保留每链最佳配置', True)
box(.445, .48, .20, .13, '选择整队最优配置\n$G^*$及实际动作$U^*$')
box(.205, .48, .17, .13, '施加$U^*$\n环境转移与任务奖励')
arrow([(.89, .67), (.89, .645), (.815, .645), (.815, .62)])
arrow([(.715, .545), (.645, .545)])
arrow([(.445, .545), (.375, .545)])
arrow([(.205, .545), (.10, .545), (.10, .705)])
arrow([(.75, .62), (.75, .65), (.685, .65), (.685, .715)])

text(.03, .375, '离策略学习：真实执行转移与同快照策略更新', 19, 'bold', ha='left')
box(.025, .17, .17, .13, '实际转移回放\n$(X,U,\\{r_i\\},r,X^\\prime,d)$', size=13)
box(.245, .145, .24, .18,
    '逐艇真实回报监督双Q\n'
    '$y_i=r_i+\\gamma(1-d)\\min_m\\overline{\\mathcal{Q}}_{m,i}$\n'
    '目标端完整生成、改进与选择', True, 13)
box(.535, .16, .20, .15,
    '生成器优化配置反馈\n加权联合似然\n＋独立基础生成学习', True)
box(.785, .16, .19, .15,
    '学习端点更新\n固定配置与价值网络\n沿实际动作求价值梯度', True)

arrow([(.29, .48), (.29, .425), (.11, .425), (.11, .30)], True)
arrow([(.195, .235), (.245, .235)], True)
arrow([(.485, .235), (.535, .235)], True)
arrow([(.365, .145), (.365, .105), (.88, .105), (.88, .16)], True)
arrow([(.635, .31), (.635, .405), (.48, .405), (.48, .695)], True)
arrow([(.88, .31), (.88, .345), (.985, .345), (.985, .925), (.285, .925), (.285, .865)], True)
arrow([(.365, .325), (.365, .355), (.995, .355), (.995, .79), (.975, .79)], True)

text(.5, .065, '实线：在线控制　虚线：学习数据与参数更新；端点和生成器共用更新前快照。', 11)
text(.5, .030, '两项核心消融分别删除当前候选动作交互、生成器优化配置反馈。', 11)

for extension in ('pdf', 'png'):
    fig.savefig(ROOT/'figures'/f'fig2_method.{extension}', dpi=220,
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print(ROOT/'figures'/'fig2_method.pdf')
