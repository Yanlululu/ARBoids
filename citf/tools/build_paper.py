"""Build the unnumbered-equation Chinese manuscript with native LaTeX math.

Requires system pandoc, xelatex, Noto CJK, Liberation fonts. No fonts bundled.
"""
from pathlib import Path
import re,shutil,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[1]
paper=ROOT/'paper'
for name in ('pandoc','xelatex'):
    if not shutil.which(name):raise SystemExit(f'Missing executable: {name}')
source=(paper/'CITF_中文独立论文_修订稿.md').read_text(encoding='utf8')
body=source[source.index('## 摘要'):]
for stem in ('fig1_task','fig2_method','fig3_geometry'):
    body=body.replace(stem+'.png',stem+'.pdf')
with tempfile.TemporaryDirectory() as tmp:
    md=Path(tmp)/'body.md';md.write_text(body,encoding='utf8')
    target=paper/'CITF_中文独立论文_修订稿.tex'
    subprocess.run(['pandoc',str(md),'-f','markdown+tex_math_single_backslash','-t','latex','-s',
        '--template='+str(paper/'template.tex'),'--lua-filter='+str(paper/'red_pending.lua'),'-o',str(target)],check=True)
    tex=target.read_text(encoding='utf8').replace('\n\n\\[\n','\n\\nopagebreak[4]\n\\[\n')
    tex=re.sub(r'(?m)^(\\textbf\{表[1-5].*)$',r'\\Needspace{8\\baselineskip}\n\1',tex)
    tex=re.sub(r'\\Needspace\{8\\baselineskip\}\n(?=\\textbf\{表4)',r'\\Needspace{14\\baselineskip}\n',tex)
    tex=re.sub(r'(?m)^(\{\[\}\d+\{\]\})',r'\\Needspace{3\\baselineskip}\n\1',tex)
    # Give the mechanism-comparison names adequate width, without splitting the table.
    if r'\textbf{表4' in tex:
        start=tex.index(r'\textbf{表4')
        head_end=tex.index(r'\toprule',start)
        head=tex[start:head_end]
        widths=iter(['0.3300','0.1400','0.1400','0.2100','0.1800'])
        head=re.sub(r'(?<=real\{)0\.2000(?=\})',lambda m:next(widths),head)
        tex=tex[:start]+head+tex[head_end:]
    # Compact journal-style bibliography without changing its text or order.
    tex=tex.replace(r'\section{参考文献}',r'\begingroup\small\setstretch{1.15}\setlength{\parskip}{0.2em}\sloppy\section{参考文献}')
    tex=tex.replace(r'\end{document}',r'\endgroup'+'\n'+r'\end{document}')
    target.write_text(tex,encoding='utf8')
for _ in range(2):
    subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','CITF_中文独立论文_修订稿.tex'],cwd=paper,check=True)
print(paper/'CITF_中文独立论文_修订稿.pdf')
