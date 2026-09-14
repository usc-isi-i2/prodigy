"""Descriptive validation-only diagnostics. No new model, fit, or test access."""
import argparse, hashlib, itertools, json, subprocess
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from ogb.linkproppred import Evaluator
import wandb


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def top(x,k=50):
    out=np.zeros(len(x),bool); out[np.argsort(-x,kind='stable')[:k]]=True; return out
def cut(x,k=50): return float(np.partition(x,-k)[-k])
def masks(s): return s['p']>cut(s['n'])
def metrics(s):
    n=np.sort(s['n']); p=s['p']
    hits={str(k):float((p>n[-k]).mean()) for k in [10,25,50,60,75,100,500,2000]}
    assert hits['50']==Evaluator(name='ogbl-collab').eval({'y_pred_pos':p,'y_pred_neg':s['n']})['hits@50']
    auc=float(((np.searchsorted(n,p,'left')+np.searchsorted(n,p,'right'))/2/len(n)).mean())
    return dict(hits=hits,auc=auc,cutoff=cut(n),positive_margin_quantiles=np.quantile(p-cut(n),[.1,.25,.5,.75,.9]).tolist())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--panel',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),classification='descriptive post-hoc validation diagnosis',test_access=False,source=str(a.source),panel=str(a.panel),no_fit=True)
    (a.out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    w=wandb.init(project='ogbl-collab-compact',group='fresh-diagnosis-v1',name='validation-diagnosis',mode='offline',dir=str(a.out),config=cfg)
    prep=read(a.source/'prepared.json'); assert sha(a.panel)==prep['files'][str(a.panel)]
    panel=np.load(a.panel); base={s:panel['official_b'+s] for s in ['p','n']}
    allscores={'aa':base}; dynamics=[]; identities={}
    for seed in range(3):
        d=a.source/f'select{seed}'; r=read(d/'results.json'); h=read(d/'history.json')
        assert sha(d/'history.json')==r['history_sha256']; assert sha(d/'best_scores.npz')==r['best_score_sha256']
        allscores[str(seed)]=dict(np.load(d/'best_scores.npz')); identities[str(seed)]=r['best_score_sha256']
        z=max(h,key=lambda x:x['joint_hits']); selected=next(x for x in h if x['step']==r['best']['step'])
        dynamics.append(dict(seed=seed,selected_step=selected['step'],selected_joint_hits=selected['joint_hits'],joint_best_step=z['step'],joint_best_hits=z['joint_hits'],last_joint_hits=h[-1]['joint_hits'],first_loss=h[0]['loss'],last_loss=h[-1]['loss'],curve=[{k:x[k] for k in ['step','loss','joint_hits']} for x in h]))
    ph={k:masks(s) for k,s in allscores.items()}; nt={k:top(s['n']) for k,s in allscores.items()}
    comparisons=[]
    for i,j in itertools.combinations(['0','1','2'],2):
        avg={s:(allscores[i][s].astype(float)+allscores[j][s].astype(float))/2 for s in ['p','n']}
        ah=masks(avg); at=top(avg['n']); union=ph[i]|ph[j]
        comparisons.append(dict(pair=[i,j],metrics=metrics(avg),positive_union=int(union.sum()),positive_intersection=int((ph[i]&ph[j]).sum()),union_missed_by_average=int((union&~ah).sum()),average_new_outside_union=int((ah&~union).sum()),top50_negative_intersection=int((nt[i]&nt[j]).sum()),top50_negative_union=int((nt[i]|nt[j]).sum()),average_top50_new=int((at&~(nt[i]|nt[j])).sum()),negative_spearman=float(spearmanr(allscores[i]['n'],allscores[j]['n']).statistic),recovered_vs_seed0=int((ah&~ph['0']).sum()),lost_vs_seed0=int((~ah&ph['0']).sum())))
    feature_names=['log_aa','log_l3','external_ratio','lcc_min','lcc_max','log_degree_min','log_degree_max','raw_cosine','log_pair_events','last_event_age','log_recent_min','log_recent_max','aa_zero','log_frozen_aadc']
    def profile(side,mask):
        f=panel[side+'symmetric'][mask]; edges=panel['pos' if side=='p' else 'neg'][mask]
        if not len(f): return {'count':0}
        return dict(count=len(f),medians=dict(zip(feature_names,map(float,np.median(f,axis=0)))),both_recent=float((f[:,10]>0).mean()),zero_aa=float((f[:,12]==1).mean()),repeat=float((f[:,8]>0).mean()),unique_endpoints=len(np.unique(edges)),max_endpoint_occurrences=int(np.unique(edges,return_counts=True)[1].max()))
    jp=np.stack([ph[str(i)] for i in range(3)]); jn=np.stack([nt[str(i)] for i in range(3)])
    positives={'all_seed_hits':jp.all(0),'all_seed_misses':~jp.any(0),'rescued_vs_aa_all':jp.all(0)&~ph['aa'],'lost_vs_aa_all':~jp.any(0)&ph['aa'],'missed_by_aa_and_all':~jp.any(0)&~ph['aa'],'seed0_rescues':ph['0']&~ph['aa'],'seed0_losses':~ph['0']&ph['aa']}
    negatives={'top50_all_seeds':jn.all(0),'top50_any_seed':jn.any(0),'promoted_vs_aa_all':jn.all(0)&~nt['aa'],'seed0_top50':nt['0'],'seed0_top2000':top(allscores['0']['n'],2000)}
    strata={}
    f=panel['psymmetric']
    for label,mask in {'repeat':f[:,8]>0,'novel':f[:,8]==0,'zero_aa':f[:,12]==1,'nonzero_aa':f[:,12]==0,'both_recent':f[:,10]>0,'not_both_recent':f[:,10]==0}.items():
        strata[label]=dict(count=int(mask.sum()),hits={k:float(v[mask].mean()) for k,v in ph.items()},consistent_misses=int((mask&~jp.any(0)).sum()))
    shifts={}
    for year in [2017,2018]:
        path=a.source/f'train{year}.npz'; assert sha(path)==prep['files'][str(path)]
        t=np.load(path)
        shifts[str(year)]={side:dict(count=len(t[side+'symmetric']),repeat=float((t[side+'symmetric'][:,8]>0).mean()),zero_aa=float((t[side+'symmetric'][:,12]==1).mean()),both_recent=float((t[side+'symmetric'][:,10]>0).mean())) for side in ['p','n']}
    out=dict(**cfg,panel_sha256=sha(a.panel),score_hashes=identities,metrics={k:metrics(s) for k,s in allscores.items()},dynamics=dynamics,pairs=comparisons,positive_profiles={k:profile('p',v) for k,v in positives.items()},negative_profiles={k:profile('n',v) for k,v in negatives.items()},strata=strata,training_panel_profiles=shifts,wandb_directory=w.dir,qualification='Oracles, AUC and alternate K are diagnostics only, not selected models. Previous test exploration remains disclosed; no test files opened by this analysis.')
    (a.out/'results.json').write_text(json.dumps(out,indent=2)+'\n'); w.log({'diagnosis/complete':1}); w.finish()
    print(json.dumps({k:out[k] for k in ['metrics','pairs','strata','training_panel_profiles']},indent=2))

if __name__=='__main__': main()
