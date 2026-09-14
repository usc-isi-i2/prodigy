"""Independent sorted-cutoff audit, archive replay, cohorts and error slices."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--panel',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
protocol=read(a.runtime/'protocol.json')
features=np.load(a.panel)['psymmetric']
strata={'all':np.ones(len(features),bool),'novel':features[:,8]==0,'repeat':features[:,8]>0,'zero_aa':features[:,12]==1,'nonzero_aa':features[:,12]==0}
rows=[];paired=[]
for seed in range(3):
    masks=[]
    for arm in ['bce','warm']:
        root=a.runtime/f'{arm}{seed}';r=read(root/'results.json');h=read(root/'history.json');cfg=read(root/'config.json');cohort=read(root/'cohort.json')
        assert r['revision']==cfg['revision']==protocol['revision']
        assert r['complete'] and not r['test_access'] and r['selected_replay_exact'] and r['official_evaluator_parity']
        assert len(h)==40 and [x['step'] for x in h]==list(range(50,2001,50))
        assert r['best']==max(h,key=lambda x:x['validation_hits'])
        assert r['neural_parameters']==496385 and r['total_inference_scalars_conservative']==496448
        for name,digest in r['files'].items(): assert sha(root/name)==digest,(root,name)
        panel_hash=next(v for k,v in cfg['files'].items() if k.endswith('assessment/year2018.npz'))
        assert sha(a.panel)==panel_hash
        assert cohort['trained_positives']==(119622 if arm=='bce' else 47546)
        scores=np.load(root/'best_scores.npz');cutoff=float(np.sort(scores['n'])[-50]);hit=scores['p']>cutoff
        assert float(hit.mean())==r['best']['validation_hits']
        if arm=='bce':
            old=read(a.archive/f'select{seed}/history.json')
            assert len(old)==len(h) and all(x['joint_hits']==y['validation_hits'] for x,y in zip(old,h))
        masks.append(hit)
        rows.append(dict(arm=arm,seed=seed,step=r['best']['step'],hits=float(hit.mean()),cutoff=cutoff,final_hits=h[-1]['validation_hits'],elapsed_seconds=r['elapsed_seconds'],strata={k:dict(count=int(v.sum()),hits=float(hit[v].mean())) for k,v in strata.items()}))
    delta=float(masks[1].mean()-masks[0].mean())
    slices={k:dict(recovered=int((v&masks[1]&~masks[0]).sum()),lost=int((v&~masks[1]&masks[0]).sum())) for k,v in strata.items()}
    paired.append(dict(seed=seed,difference=delta,strata=slices))
mean=float(np.mean([x['difference'] for x in paired]))
result=dict(complete=True,rows=rows,paired=paired,mean_difference=mean,advances=all(x['difference']>0 for x in paired) and mean>=.005,all120_controls_match_archive=True,all6_selected_scores_independently_recomputed=True,artifact_hashes_valid=True,test_access=False,prior_test_exploration_disclosed=True)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
