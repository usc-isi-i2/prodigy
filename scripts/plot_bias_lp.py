"""Render completed bias LP matrices and histories alongside unchanged FR."""
import base64,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from plot_mini_transfer import ROOT,ORDER,LABELS
STAGES=['node_bias','node_neighbors_bias']
TITLES=['Node-only LP · learned bias','Node + 10-neighbor LP · learned bias · disjoint context']

def img(name,caption):
    data=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
    return f'<figure><figcaption>{caption}</figcaption><img src="data:image/png;base64,{data}" alt="{caption}" loading="lazy"></figure>'

def main():
    histories=json.loads((ROOT/'bias_curves.json').read_text())
    frames={s:pd.read_csv(ROOT/s/'matrix.csv') for s in STAGES}
    bce_limits=(min(.1,min(d.bce.min() for d in frames.values())),max(.5,max(d.bce.max() for d in frames.values())))
    sections=['<section id="fr"><h2>Node-only feature reconstruction · unchanged</h2>'+img('fr_curves','Training and source-validation curves — nine models')+img('fr_matrix','Feature reconstruction transfer matrix')+'</section>']
    comparisons=[]
    for stage,title,old in zip(STAGES,TITLES,['node_uniform','node_neighbors_disjoint_uniform']):
        df=frames[stage];summaries={s['sources'][0]:s for s in json.loads((ROOT/stage/'summaries.json').read_text())}
        fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
        for source,label,ax in zip(ORDER,LABELS,axes.flat):
            rows=histories[stage]['ss_'+source];tr=[r for r in rows if 'train/loss' in r];va=[r for r in rows if 'validation/loss' in r];s=summaries[source]
            assert tr[-1]['optimizer_step']==s['final_step']
            ax.plot([r['optimizer_step'] for r in tr],[r['train/loss'] for r in tr],label='Train (100-update mean)',lw=1.2)
            ax.plot([r['optimizer_step'] for r in va],[r['validation/loss'] for r in va],label='Validation (every 2k)',marker='.',lw=1.2)
            best=next(r for r in va if r['optimizer_step']==s['best_step']);ax.scatter([s['best_step']],[best['validation/loss']],marker='*',s=100,c='black',zorder=4,label='Selected checkpoint')
            ax.axhline(.450561,ls='--',lw=.7,c='gray',label='Constant p=1/6');ax.set_title(f'{label} · best {s["best_step"]//1000}k / stop {s["final_step"]//1000}k',fontsize=10)
            ax.set_xlabel('Optimizer updates');ax.set_ylabel('BCE');ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:g}k'));ax.grid(alpha=.15)
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
        fig.suptitle(title+'\nUniform negatives · 1 positive : 5 negatives · independent panel ranges')
        for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{stage}_curves.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
        for metric in ['auc','bce']:
            values=df.pivot(index='source',columns='target',values=metric).reindex(index=ORDER,columns=ORDER).to_numpy()
            fig,ax=plt.subplots(figsize=(11.8,8.4));lo,hi=(.5,1.) if metric=='auc' else bce_limits
            im=ax.imshow(values,cmap='viridis' if metric=='auc' else 'viridis_r',vmin=lo,vmax=hi)
            ax.set_xticks(range(9),LABELS,rotation=35,ha='right');ax.set_yticks(range(9),LABELS);ax.set_xlabel('Evaluation target');ax.set_ylabel('Training source');ax.set_title(title+'\nFrozen source bias · uniform 1:5 · '+metric.upper()+(' ↑' if metric=='auc' else ' ↓'))
            for i in range(9):
                for j in range(9):
                    c=im.cmap(im.norm(values[i,j]));lum=.2126*c[0]+.7152*c[1]+.0722*c[2]
                    ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='black' if lum>.55 else 'white',fontsize=10)
                ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='#ff9e00',lw=2))
            fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{stage}_{metric}.{ext}',dpi=170)
            plt.close(fig)
        prior=pd.read_csv(ROOT/old/'matrix.csv')
        for subset in ['all','same_graph','cross_graph']:
            mask=lambda d: d if subset=='all' else d[d.source==d.target] if subset=='same_graph' else d[d.source!=d.target]
            a,b=mask(prior),mask(df)
            comparisons.append(dict(view=stage,subset=subset,old_auc=a.auc.mean(),new_auc=b.auc.mean(),old_bce=a.bce.mean(),new_bce=b.bce.mean()))
        cap=sum(not s['converged'] for s in summaries.values());note=f'{9-cap}/9 runs stopped on validation plateau; {cap}/9 reached the safety cap.'
        anchor='node' if stage=='node_bias' else 'node_neighbors'
        sections.append(f'<section id="{anchor}"><h2>{title}</h2><p>{note}</p>'+img(stage+'_curves','Training and source-validation curves — nine models')+'<div class="matrices">'+img(stage+'_auc','Transfer AUC ↑')+img(stage+'_bce','Transfer BCE ↓')+'</div></section>')
    comparison=pd.DataFrame(comparisons);comparison.to_csv(ROOT/'bias_comparison.csv',index=False)
    display=comparison.replace({'view':{'node_bias':'Node-only','node_neighbors_bias':'Node + 10 neighbors'},'subset':{'all':'All pairs of graphs','same_graph':'Same graph','cross_graph':'Cross graph'}}).rename(columns={'view':'Model','subset':'Evaluation','old_auc':'Previous AUC','new_auc':'Bias AUC','old_bce':'Previous BCE','new_bce':'Bias BCE'})
    table=display.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Bias LP transfer matrices and curves</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.5 system-ui;margin:0;background:#f5f6f8;color:#17202a}main{max-width:1500px;margin:auto;padding:24px}nav{position:sticky;top:0;background:white;padding:14px;display:flex;gap:25px;z-index:2}section{scroll-margin-top:75px;margin-top:40px}figure{background:white;padding:14px;border-radius:10px;margin:18px 0}figcaption{font-weight:600;margin-bottom:12px}img{width:100%;display:block}.matrices{display:grid;grid-template-columns:1fr 1fr;gap:18px}table{border-collapse:collapse;background:white}td,th{padding:8px 14px;border-bottom:1px solid #ddd}a{color:#1769aa}@media(max-width:1000px){.matrices{grid-template-columns:1fr}}</style></head><body><main><h1>Training curves and transfer matrices</h1><p>LP rerun with a learned scalar bias: logit = dot(zᵤ,zᵥ) + b. Bias starts at −log(5), is trained jointly with the encoder, and stays frozen on every transfer target. No extra scale and no target calibration in headline metrics.</p><p>18 new LP source models · 162 LP evaluation cells · seed 0. Uniform independent-endpoint negatives, 1 positive : 5 negatives, rejecting self-loops and all known edges. Same evaluation pairs as the previous uniform matrices. Constant-p=1/6 BCE: 0.4506. Neighbor inputs use fixed up-to-10 neighbors from context edges disjoint from positive supervision. FR is unchanged.</p><p>Blue: mean training loss over 100 updates. Orange: source validation every 2,000 updates. ★: selected checkpoint. Matrix rows: source; columns: target; orange outlines: same graph. Color scales shared across LP variants. W&B histories are stored offline.</p><p><a href="gallery_uniform_raw.html">Previous uniform evaluation without bias</a> · <a href="gallery_degree_matched.html">Historical degree-matched evaluation</a></p><h2>Matched comparison with previous uniform matrices</h2><p>Same graph: 9 cells; cross graph: 72 cells; all: 81 cells per model type. Single seed; no uncertainty intervals.</p>'''+table+'''<nav><a href="#fr">Node-only FR</a><a href="#node">Node-only LP</a><a href="#node_neighbors">Node + 10-neighbor LP</a></nav>'''+''.join(sections)+'</main></body></html>'
    (ROOT/'gallery.html').write_text(html)
    print(comparison.to_string(index=False))
if __name__=='__main__':main()
