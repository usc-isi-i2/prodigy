"""Independent score, selection, source identity, and paired sampling audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True);a=p.parse_args()
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
f=np.load(a.panel)['psymmetric'];masks={'all':np.ones(len(f),bool),'novel':f[:,8]==0,'repeat':f[:,8]>0,'zero_aa':f[:,12]==1,'nonzero_aa':f[:,12]==0}
rows=[];pairs=[];revisions=set()
for seed in range(3):
    states=[];hitmasks={}
    for arm in ['control','observed','completion']:
        d=a.runtime/f'{arm}{seed}';r=read(d/'results.json');h=read(d/'history.json');cfg=read(d/'config.json')
        assert r['complete'] and not r['test_access'] and r['selected_replay_exact'] and r['official_evaluator_parity'] and not cfg['smoke']
        assert cfg['cache_manifest_sha256']==sha(a.cache_manifest)
        assert cfg['source_files']==read(a.cache_manifest)['files']
        assert r['arm']==cfg['arm']==arm and r['seed']==cfg['seed']==seed
        assert r['revision']==cfg['revision'];revisions.add(r['revision'])
        assert [x['step'] for x in h]==list(range(50,2001,50));assert r['best']==max(h,key=lambda x:x['validation_hits'])
        for name,digest in r['files'].items():assert sha(d/name)==digest,(d,name)
        assert sha(a.panel)==next(v for k,v in cfg['source_files'].items() if k.endswith('assessment/year2018.npz'))
        assert r['neural_parameters']==cfg['neural_parameters'][arm]
        assert r['total_inference_scalars_conservative']==cfg['conservative_inference_scalars'][arm]<1000000
        sc=np.load(d/'best_scores.npz');mask=sc['p']>np.sort(sc['n'])[-50]
        assert len(mask)==len(f) and float(mask.mean())==r['best']['validation_hits']
        if arm=='control':
            old=read(a.archive/f'select{seed}/history.json')
            assert len(old)==len(h) and all(x['joint_hits']==y['validation_hits'] for x,y in zip(old,h))
        states.append(r);hitmasks[arm]=mask
        rows.append(dict(arm=arm,seed=seed,step=r['best']['step'],hits=float(mask.mean()),final_hits=h[-1]['validation_hits'],seconds=r['elapsed_seconds'],max_gpu_bytes=r['max_gpu_bytes'],strata={k:dict(count=int(v.sum()),hits=float(mask[v].mean())) for k,v in masks.items()}))
    assert len({x['stream_sha256'] for x in states})==1
    assert len({x['initial_encoder_sha256'] for x in states})==1
    for reference,treatment in [('control','observed'),('control','completion'),('observed','completion')]:
        base,changed=hitmasks[reference],hitmasks[treatment]
        pairs.append(dict(seed=seed,reference=reference,treatment=treatment,difference=float(changed.mean()-base.mean()),strata={k:dict(recovered=int((v&changed&~base).sum()),lost=int((v&~changed&base).sum())) for k,v in masks.items()}))
assert len(revisions)==1
summary={}
for arm in ['observed','completion']:
    ds=[x['difference'] for x in pairs if x['reference']=='control' and x['treatment']==arm]
    summary[arm]=dict(mean_gain=float(np.mean(ds)),all_seeds_positive=all(x>0 for x in ds),advances=all(x>0 for x in ds) and np.mean(ds)>=.005)
qualified=[arm for arm in ['observed','completion'] if summary[arm]['advances']]
winner=max(qualified,key=lambda arm:summary[arm]['mean_gain']) if qualified else None
result=dict(complete=True,revision=next(iter(revisions)),all9_selected_scores_independently_recomputed=True,all120_controls_match_archive=True,all_artifact_hashes_valid=True,paired_rng_streams_exact=True,paired_encoder_initialization_exact=True,rows=rows,paired=pairs,summary=summary,selected_arm_for_replication=winner,test_access=False,prior_test_exploration_disclosed=True)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
