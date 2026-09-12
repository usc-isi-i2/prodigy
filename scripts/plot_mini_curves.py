"""Plot all completed mini-transfer histories and build a self-contained gallery."""
import base64
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from plot_mini_transfer import ROOT,ORDER,LABELS
HISTORIES=json.loads((ROOT/'curves.json').read_text())
TITLES={'fr':'Node-only feature reconstruction','node':'Node-only link prediction','node_neighbors_disjoint':'Node + 10-neighbor LP · disjoint context'}
for stage,title in TITLES.items():
    runs=HISTORIES[stage]
    summaries={s['sources'][0]:s for s in json.loads((ROOT/stage/'summaries.json').read_text())}
    assert len(runs)==9
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for source,label,ax in zip(ORDER,LABELS,axes.flat):
        rows=runs['ss_'+source]
        train=[r for r in rows if 'train/loss' in r]
        val=[r for r in rows if 'validation/loss' in r]
        assert train and val and train[-1]['optimizer_step']==summaries[source]['final_step']
        ax.plot([r['optimizer_step'] for r in train],[r['train/loss'] for r in train],color='#3274A1',lw=1.4,label='Train (100-update mean)')
        ax.plot([r['optimizer_step'] for r in val],[r['validation/loss'] for r in val],color='#D55E00',lw=1.5,marker='o',ms=2.8,label='Validation (every 2k)')
        best=summaries[source]['best_step'];bestrow=next(r for r in val if r['optimizer_step']==best)
        ax.scatter([best],[bestrow['validation/loss']],marker='*',s=105,color='#111111',zorder=5,label='Selected checkpoint')
        ax.set_title(f'{label} · best {best//1000}k / stop {summaries[source]["final_step"]//1000}k',fontsize=10)
        ax.grid(alpha=.18);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v/1000:g}k'))
        ax.set_xlabel('Optimizer updates',fontsize=9)
        ax.set_ylabel('Squared cosine error' if stage=='fr' else 'BCE',fontsize=9)
        if stage=='fr':ax.set_yscale('log')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(title+' — all nine source models\n'+('Logarithmic error axis; independent panel ranges' if stage=='fr' else 'Train and source validation use 1 positive : 5 negatives; independent panel ranges'),fontsize=15)
    fig.savefig(ROOT/'figures'/f'{stage}_curves.png',dpi=150,bbox_inches='tight',pad_inches=.15)
    fig.savefig(ROOT/'figures'/f'{stage}_curves.pdf',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

def img(name,caption):
    data=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
    return f'<figure><figcaption>{caption}</figcaption><img src="data:image/png;base64,{data}" alt="{caption}" loading="lazy"></figure>'
sections=[]
for stage,title in TITLES.items():
    content=img(stage+'_curves','Training and source-validation curves — nine models')
    if stage=='fr':content+=img('fr_matrix','Transfer matrix: squared cosine error ↓')
    else:content+='<div class="matrices">'+img(stage+'_auc','Transfer matrix: raw-dot AUC ↑')+img(stage+'_bce','Transfer matrix: raw-dot BCE ↓')+'</div>'
    if stage=='node_neighbors_disjoint':
        with (ROOT/'disjoint_comparison.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        table='<h3>Original vs corrected neighbor model</h3><p>Gaps are validation BCE minus training BCE at each run’s selected checkpoint. AUC/BCE below use the unchanged same-graph final-test pairs.</p><div style="overflow:auto"><table><thead><tr><th>Source</th><th>Old gap</th><th>New gap</th><th>Old AUC</th><th>New AUC</th><th>Old BCE</th><th>New BCE</th></tr></thead><tbody>'
        for row in rows:
            table+='<tr><td>'+row['label']+'</td>'+''.join('<td>'+format(float(row[k]),'.3f')+'</td>' for k in ['old_gap','new_gap','old_auc','new_auc','old_bce','new_bce'])+'</tr>'
        content+=table+'</tbody></table></div>'
    anchor='node_neighbors' if stage=='node_neighbors_disjoint' else stage
    sections.append(f'<section id="{anchor}"><h2>{title}</h2>{content}</section>')
html='''<!doctype html><html><head><meta charset="utf-8"><title>All mini-transfer curves and matrices</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>
table{border-collapse:collapse;background:white;width:100%}td,th{padding:9px 14px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#17202a}main{max-width:1500px;margin:auto;padding:24px}h1{margin-bottom:6px}p{max-width:1000px}nav{display:flex;gap:20px;flex-wrap:wrap;position:sticky;top:0;background:#fff;padding:14px;z-index:2;border-bottom:1px solid #ddd}a{color:#1769aa}section{scroll-margin-top:75px;margin-top:40px}figure{margin:18px 0;background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #0001}figcaption{font-weight:600;padding:0 10px 12px}img{width:100%;height:auto;display:block}.matrices{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:1000px){.matrices{grid-template-columns:1fr}}
</style></head><body><main><h1>All training curves and transfer matrices</h1><p>27 source models · 243 evaluation cells · seed 0 · nonzero features · Ukraine/COVID one-hop 500k-node minis.</p><p>Blue: training loss averaged over 100 updates. Orange: source-validation loss every 2,000 updates. ★: selected checkpoint. Checkpoint selection uses source validation; final and selected updates appear on each panel. Matrix rows are training sources; columns are evaluation targets; orange outlines mark same-graph cells.</p><p><b>Metrics:</b> FR uses squared cosine reconstruction error. LP curves use 1:5 positive/negative sampling; LP matrices use balanced degree-matched test pairs. Their BCE values are not directly comparable. AUC/BCE color scales are shared between the two LP matrices.</p><p><b>Corrected neighbor protocol:</b> The original 70% training-edge pool is split equally into context-only edges and positive supervision edges. No positive training pair appears in neighbor context. Validation/test sets and final evaluation pairs are unchanged. <a href="gallery_original.html">Original overlapping-context gallery</a>.</p><nav><a href="#fr">Node-only FR</a><a href="#node">Node-only LP</a><a href="#node_neighbors">Node + 10-neighbor LP</a></nav>'''+''.join(sections)+'''</main></body></html>'''
(ROOT/'gallery.html').write_text(html)
print('Created three curve figures and an eight-figure self-contained gallery.')
