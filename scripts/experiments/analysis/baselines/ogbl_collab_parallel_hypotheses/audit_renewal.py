"""Parent independent audit of all six H1 score archives and selection histories."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[];paired=[]
for seed in range(3):
    states=[];masks=[]
    for arm in ('fixed','renew'):
        root=a.runtime/f'{arm}{seed}';r=json.loads((root/'results.json').read_text());h=json.loads((root/'history.json').read_text())
        assert r['complete'] and not r['test_scored'] and r['exact_checkpoint_replay']
        assert len(h)==40 and [x['step'] for x in h]==list(range(50,2001,50))
        assert r['best']==max(h,key=lambda x:x['hits'])
        for name,key in [('history.json','history_sha256'),('best.pt','checkpoint_sha256'),('best_scores.npz','score_sha256')]:
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==r[key]
        s=np.load(root/'best_scores.npz');mask=s['p']>np.sort(s['n'])[-50];assert float(mask.mean())==r['best']['hits']
        assert r['neural_parameters']==496385 and r['total_inference_scalars']<1000000
        if arm=='fixed':
            old=json.loads((a.archive/f'select{seed}/history.json').read_text())
            assert all(x['joint_hits']==y['hits'] for x,y in zip(old,h))
        rows.append(dict(seed=seed,arm=arm,step=r['best']['step'],hits=r['best']['hits'],final_hits=h[-1]['hits'],selected_probe_hits=r['best']['probe_hits'],final_probe_hits=h[-1]['probe_hits']))
        states.append(r);masks.append(mask)
    assert states[0]['sample_stream_sha256']==states[1]['sample_stream_sha256']
    paired.append(dict(seed=seed,difference=states[1]['best']['hits']-states[0]['best']['hits'],recovered=int((masks[1]&~masks[0]).sum()),lost=int((masks[0]&~masks[1]).sum())))
mean=sum(x['difference'] for x in paired)/3
result=dict(complete=True,all_six_scores_independently_recomputed=True,all120_controls_match_archive=True,artifact_hashes_valid=True,paired_rng_streams_match=True,rows=rows,paired=paired,mean_difference=mean,advances=all(x['difference']>0 for x in paired) and mean>=.005,test_access=False)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
