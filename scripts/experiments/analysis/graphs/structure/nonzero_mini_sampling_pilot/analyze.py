"""Regenerate structural comparison tables and scientific figures from pilot JSON."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = ['uniform', 'edge', 'walk1_s0', 'walk2_s0', 'walk4_s0', 'walk8_s0']
LABELS = ['Uniform\nnodes', 'Edge\nendpoints', '1 hop', '2 hops', '4 hops', '8 hops']
COLORS = ['#9ca3af', '#ed8936', '#2563eb', '#0891b2', '#16846d', '#805ad5']
reports = {name: json.loads((ROOT / 'data' / (name + '.json')).read_text()) for name in ['ukraine', 'covid']}
assert all(r['completed'] and r['parent_unchanged'] for r in reports.values())
rows = []
for graph, report in reports.items():
    for candidate in [{'name': 'parent', 'metrics': report['parent']}] + report['candidates']:
        m = candidate['metrics']
        rows.append({'graph': graph, 'method': candidate['name'], 'nodes': m['nodes'], 'edges': m['stored_edges'],
                     'edges_per_node': m['edges_per_node'], 'isolates_pct': 100*m['isolate_fraction'],
                     'degree_ks': m.get('degree_ks', 0), 'degree_p50': m['degree_quantiles']['0.5'],
                     'degree_p90': m['degree_quantiles']['0.9'], 'degree_p99': m['degree_quantiles']['0.99'],
                     'largest_component_pct': 100*m['largest_component_fraction'], 'components': m['components']})
with (ROOT / 'data' / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2), constrained_layout=True)
for i, (graph, report) in enumerate(reports.items()):
    candidates = {r['name']: r['metrics'] for r in report['candidates']}
    for j, (key, scale, title) in enumerate([('edges_per_node', 1, 'Stored edges per node'), ('isolate_fraction', 100, 'Isolated nodes (%)'), ('largest_component_fraction', 100, 'Nodes in largest weak component (%)')]):
        ax = axes[i,j]
        ax.bar(range(6), [candidates[n][key]*scale for n in NAMES], color=COLORS, width=.68)
        baseline = report['parent'][key]*scale
        ax.axhline(baseline, color='#111827', ls='--', lw=1.4)
        for k in [2,3]:
            other = NAMES[k].replace('_s0','_s1')
            if other in candidates:
                ax.scatter(k, candidates[other][key]*scale, c='black', s=19, zorder=5)
        ax.set_xticks(range(6), LABELS)
        ax.set_title(f'{graph.title()} · {title}', fontsize=10)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        if j > 0:
            ax.set_ylim(0,105)
        ax.text(.97,.95,f'Parent: {baseline:.2f}',ha='right',va='top',transform=ax.transAxes,fontsize=9,
                bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
fig.suptitle('500k-node induced minis: short walks reduce, but do not eliminate, structural distortion\nDashed line = full nonzero parent; black dots = confirmation seed 1', fontsize=13)
for ext in ['png','pdf']:
    fig.savefig(ROOT / 'figures' / ('structural_comparison.' + ext))
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
for ax, (graph, report) in zip(axes, reports.items()):
    cand = {r['name']:r['metrics'] for r in report['candidates']}
    series = [('Parent',report['parent'],'#111827','-',2.1), ('Uniform nodes',cand['uniform'],COLORS[0],'-',1.5),
              ('Edge endpoints',cand['edge'],COLORS[1],'-',1.5), ('1 hop, seed 0',cand['walk1_s0'],COLORS[2],'-',1.9),
              ('1 hop, seed 1',cand['walk1_s1'],COLORS[2],':',1.7), ('2 hops, seed 0',cand['walk2_s0'],COLORS[3],'--',1.5)]
    for name, m, color, style, width in series:
        h=m['degree_histogram']; x=np.asarray(h['values']); counts=np.asarray(h['counts']); y=np.cumsum(counts[::-1])[::-1]/counts.sum(); good=x>0
        ax.loglog(x[good],y[good],label=name,color=color,ls=style,lw=width)
    ax.set_title(graph.title());ax.set_xlabel('Unique undirected non-self neighbors (degree d)')
    ax.set_ylabel('Fraction of all nodes with degree ≥ d')
    ax.grid(alpha=.15,which='major');ax.legend(fontsize=8,frameon=False)
fig.suptitle('Degree distributions: one-hop walks improve the bulk, but still distort the tail',fontsize=13)
for ext in ['png','pdf']:
    fig.savefig(ROOT / 'figures' / ('degree_distributions.'+ext))
plt.close(fig)
selection = {graph: {'screening_selection':f'/dataMeR1/phil/gfm/prodigy-walk-mini-pilot/state/{graph}/walk1_s0.pt',
                     'confirmation_selection':f'/dataMeR1/phil/gfm/prodigy-walk-mini-pilot/state/{graph}/walk1_s1.pt',
                     'status':'closest tested compromise; not a representative or density-matched graph',
                     'complete_graph_artifact_created':False} for graph in reports}
(ROOT/'data'/'preferred_selections.json').write_text(json.dumps(selection,indent=2)+'\n')
