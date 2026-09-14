"""Validate frozen H2 six-cell evidence and compare full archived BCE curves."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--archive',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    cfg=read(a.runtime/'protocol.json'); rows=[]; comparisons=[]
    for seed in range(3):
        paired=[]
        for arm in ['bce','tail']:
            d=a.runtime/f'{arm}{seed}'; r=read(d/'results.json'); h=read(d/'history.json')
            assert r['complete'] and not r['test_access'] and r['revision']==cfg['revision']
            assert r['official_evaluator_parity'] and r['selected_replay_exact']
            assert [x['step'] for x in h]==list(range(50,2001,50))
            assert r['best']==max(h,key=lambda x:x['validation_hits'])
            assert r['neural_parameters']==496385 and r['total_inference_scalars_conservative']<1000000
            for name,expected in r['files'].items(): assert sha(d/name)==expected,(d,name)
            rows.append(dict(seed=seed,arm=arm,step=r['best']['step'],hits=r['best']['validation_hits'],selected_probe_hits=r['best']['probe_hits'],final_probe_hits=h[-1]['probe_hits'],final_validation_hits=h[-1]['validation_hits'],final_seen_hits=h[-1]['seen_hits'],elapsed_seconds=r['elapsed_seconds'],max_gpu_bytes=r['max_gpu_bytes'],wandb_directory=r['wandb_directory']))
            paired.append(r)
        assert paired[0]['stream_sha256']==paired[1]['stream_sha256']
        masks=[]
        for arm in ['bce','tail']:
            z=np.load(a.runtime/f'{arm}{seed}/best_scores.npz'); mask=z['p']>np.sort(z['n'])[-50]
            assert float(mask.mean())==paired[len(masks)]['best']['validation_hits']; masks.append(mask)
        rows[-1]['rescued_control_misses']=int((masks[1]&~masks[0]).sum())
        rows[-1]['lost_control_hits']=int((masks[0]&~masks[1]).sum())
        old=read(a.archive/f'select{seed}/history.json'); new=read(a.runtime/f'bce{seed}/history.json')
        assert [x['step'] for x in old]==[x['step'] for x in new]
        difference=max(abs(x['joint_hits']-y['validation_hits']) for x,y in zip(old,new))
        comparisons.append(dict(seed=seed,max_all40_absolute_difference=difference,all40_within_predeclared_1e6=difference<=1e-6))
    differences=[rows[2*s+1]['hits']-rows[2*s]['hits'] for s in range(3)]
    result=dict(complete=True,protocol=cfg,rows=rows,paired_differences=differences,mean_difference=sum(differences)/3,advances=all(x>0 for x in differences) and sum(differences)/3>=.005,archive_controls=comparisons,matched_draw_streams=True,checkpoint_hashes_verified=True,qualification='Three optimization seeds on one repeatedly explored validation panel, not independent datasets. Same-year probe uses training positives. No2019 access. Prior test exploration remains disclosed.')
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
