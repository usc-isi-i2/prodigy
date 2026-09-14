"""Render compact input-shift audit JSON; no raw graph access required."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL = dict(ukr_rus_twitter='Ukraine', covid19_twitter='COVID', midterm='Midterm',
             covid_political='Political', ukr_rus_suspended='Suspended',
             twibot20='TwiBot20', cp_hk_twitter='Hong Kong', facebook_page_reference='Facebook')


def matrix(ax, names, rows, field, title, lo=None, hi=None, diagonal=None):
    a = np.full((len(names), len(names)), np.nan)
    for r in rows:
        i, j = names.index(r['first']), names.index(r['second'])
        a[i,j] = a[j,i] = r[field]
    if diagonal:
        for i, s in enumerate(names):
            a[i,i] = diagonal[s]
    im = ax.imshow(a, cmap='viridis', vmin=lo, vmax=hi)
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=40, ha='right')
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
    for (i,j), v in np.ndenumerate(a):
        if np.isfinite(v):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,
                    color='black' if im.norm(v) > .65 else 'white')
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)


def run(args):
    d = json.loads(Path(args.input).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    figures, data = out/'figures', out/'data'
    figures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    names = list(d['graphs'])
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(15,13), constrained_layout=True)
    for ax, (v, title) in zip(axes.flat, [('node','Uniform nodes: raw embedding'),
            ('node_neighbors','Uniform nodes: node + mean neighbors'),
            ('pair_node','Task pairs: raw endpoint summary'),
            ('pair_node_neighbors','Task pairs: node + neighbor endpoint summary')]):
        table = d['distances'][v]
        matrix(ax,names,table['pairs'],'normalized_sliced_w1',title,lo=0,diagonal=table['within_graph'])
        pd.DataFrame(table['pairs']).to_csv(data/(v+'_distances.csv'),index=False)
    fig.suptitle('Input distributions across eight graphs\nNormalized sliced Wasserstein distance; diagonal = within-graph sampling reference',fontsize=15)
    fig.savefig(figures/'input_distances.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
    for ax,v in zip(axes,['node','node_neighbors']):
        matrix(ax,names,d['domain'][v]['pairs'],'domain_auc',
               'Raw embedding' if v=='node' else 'Node + mean neighbors',lo=.5,hi=1,diagonal=d['domain'][v]['within_graph'])
        pd.DataFrame(d['domain'][v]['pairs']).to_csv(data/(v+'_domain.csv'),index=False)
    fig.suptitle('Can a linear classifier identify the graph?\nHeld-out AUC; diagonal = within-graph pseudo-domain control',fontsize=15)
    fig.savefig(figures/'graph_discrimination.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True,constrained_layout=True)
    occupied = []
    for g in d['graphs'].values():
        h=g['cosine_histogram']; bins=np.asarray(h['bins'])
        active=np.flatnonzero(np.asarray(h['edge'])+np.asarray(h['nonedge']))
        occupied.extend([bins[active[0]],bins[active[-1]+1]])
    for ax,s in zip(axes.flat,names):
        g = d['graphs'][s]
        h = g['cosine_histogram']
        bins = np.asarray(h['bins'])
        for k,label,c in [('edge','Supervision edges','#187fa2'),('nonedge','Uniform nonedges','#c44f38')]:
            y = np.asarray(h[k]); y = y/y.sum()/np.diff(bins)
            ax.stairs(y,bins,label=label,color=c,linewidth=2)
        ax.set_xlim(max(-1,min(occupied)-.02),min(1,max(occupied)+.02))
        ax.set_title(f"{LABEL[s]}\nCosine edge-score AUC = {g['edge_similarity_auc']:.3f}")
        ax.set_xlabel('Endpoint embedding cosine')
        ax.set_ylabel('Density')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Do similar nodes connect? Compare edges with each graph’s own nonedge background',fontsize=15)
    fig.savefig(figures/'edge_similarity.png',dpi=170)
    plt.close(fig)
    fig,axes = plt.subplots(2,4,figsize=(16,10),constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.12,w_pad=.08,hspace=.12,wspace=.04)
    for ax,s in zip(axes.flat,names):
        m = d['mixing'][s]
        counts = np.asarray(m['counts'])
        observed = counts/counts.sum()
        expected = np.asarray(m['configuration_expected'])
        # Mask sparse cells; smooth with one expected/observed count before log ratio.
        lift = np.log2((counts+1)/(expected*counts.sum()+1))
        lift = np.ma.masked_where(expected*counts.sum()<5, lift)
        im=ax.imshow(lift,cmap='RdBu_r',vmin=-3,vmax=3)
        ax.set_title(f"{LABEL[s]}\nSame cluster: {m['observed_same_cluster']:.2f} vs expected {m['expected_same_cluster']:.2f}")
        ax.set_xlabel('Shared embedding cluster');ax.set_ylabel('Shared embedding cluster')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='log₂ edge mixing / configuration expectation')
    fig.suptitle('Which kinds of nodes connect? Shared raw-embedding clusters\nCells with fewer than five expected endpoint counts are masked',fontsize=15)
    fig.savefig(figures/'cluster_mixing.png',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    values=[100*d['graphs'][s]['sampled_isolate_fraction'] for s in names]
    bars=ax.bar([LABEL[s] for s in names],values,color='#277f98')
    ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set_ylim(0,70);ax.set_ylabel('Sampled nodes with zero context mean (%)')
    ax.set_title('Context availability differs substantially across graphs\nFixed context-only edges; original node features are nonzero')
    fig.savefig(figures/'context_availability.png',dpi=170)
    plt.close(fig)
    pd.DataFrame([dict(source=s,**{k:v for k,v in g.items() if isinstance(v,(int,float))}) for s,g in d['graphs'].items()]).to_csv(data/'graph_summary.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
