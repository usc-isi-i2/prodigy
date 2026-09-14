from pathlib import Path
import json,hashlib
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
root=args.runtime
panelpath=args.panel;panel=np.load(panelpath)
scores=np.load(root/'bce0/best_scores.npz')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((root/'bce0/results.json').read_text());cfg=json.loads((root/'bce0/config.json').read_text())
assert sha(root/'bce0/best_scores.npz')==r['files']['best_scores.npz']
assert sha(panelpath)==next(v for k,v in cfg['files'].items() if k.endswith('assessment/year2018.npz'))
cutoff=float(np.sort(scores['n'])[-50]);misses=np.flatnonzero(scores['p']<=cutoff)
assert 1-len(misses)/len(scores['p'])==r['best']['validation_hits']
seed=20260914
indices=np.random.Generator(np.random.PCG64(seed)).choice(misses,size=20,replace=False)
manifest=dict(model='fresh-negative joint, seed0, update150',year=2018,definition='All validation positives with score <= 50th largest validation negative score',population=int(len(misses)),total_positives=len(scores['p']),cutoff=cutoff,rng='numpy Generator(PCG64)',seed=seed,sampling='uniform without replacement, first draw, original draw order; no stratification or replacement',indices=indices.tolist(),panel_sha256=sha(panelpath),score_sha256=sha(root/'bce0/best_scores.npz'),test_access=False)
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
other=[np.load(root/f'bce{s}/best_scores.npz') for s in (1,2)]
warm=np.load(root/'warm0/best_scores.npz')
rows=[];g=panel['graph_edges']
for k,idx in enumerate(indices,1):
 f=panel['psymmetric'][idx];u,v=map(int,panel['pos'][idx]);neighbors=[]
 for node in (u,v):
  neighbors.append(set(g[g[:,0]==node,1].tolist()+g[g[:,1]==node,0].tolist()))
 row=dict(number=k,index=int(idx),pair=[u,v],score=float(scores['p'][idx]),margin=float(scores['p'][idx])-cutoff,negatives_scoring_at_least_as_high=int(np.count_nonzero(scores['n']>=scores['p'][idx])),cosine=float(f[7]),degree_min=int(round(np.expm1(f[5]))),degree_max=int(round(np.expm1(f[6]))),recent_min=int(round(np.expm1(f[10]))),recent_max=int(round(np.expm1(f[11]))),prior_events=int(round(np.expm1(f[8]))),common_neighbors=len(neighbors[0]&neighbors[1]),log_l3=float(f[1]),aa_zero=bool(f[12]==1),frozen_aa_hit=bool(panel['bp'][idx]>np.sort(panel['bn'])[-50]),official_calibrated_aa_hit=bool(panel['official_bp'][idx]>np.sort(panel['official_bn'])[-50]),other_seed_hits=[bool(a['p'][idx]>np.sort(a['n'])[-50]) for a in other],warm_seed0_hit=bool(warm['p'][idx]>np.sort(warm['n'])[-50]))
 assert row['aa_zero']==(row['common_neighbors']==0)
 assert sorted(map(len,neighbors))==[row['degree_min'],row['degree_max']]
 rows.append(row)
manifest['examples']=rows
(args.out/'uniform-20-misses.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
