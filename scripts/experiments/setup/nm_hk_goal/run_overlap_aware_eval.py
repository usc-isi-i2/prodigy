"""Correct HK NM evaluation and mask adjacency-contradicted negative messages."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.experiments.setup.nm_hk_mechanism.run import build_model
from scripts.experiments.setup.nm_support_resampling.run import digest_state


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
    return h.hexdigest()


def score_summary(d, prefix, baseline_assigned, baseline_multi):
    assigned=d[prefix+"_assigned"].astype(bool); multi=d[prefix+"_multi"].astype(bool)
    unique=d.valid_test_candidates.eq(1)
    out={"rows":len(d),"assigned_correct":int(assigned.sum()),"assigned_accuracy":float(assigned.mean()),
         "multi_correct":int(multi.sum()),"multi_accuracy":float(multi.mean()),
         "unique_rows":int(unique.sum()),"unique_accuracy":float(assigned[unique].mean()),
         "mean_valid_candidates":float(d.valid_test_candidates.mean())}
    if prefix+"_mrr" in d:
        out.update(mrr=float(d[prefix+"_mrr"].mean()),
            hits_at_1=float(d[prefix+"_hits1"].mean()), hits_at_3=float(d[prefix+"_hits3"].mean()),
            hits_at_5=float(d[prefix+"_hits5"].mean()))
    if baseline_assigned is not None:
        out["assigned_recovered"]=int((~baseline_assigned & assigned).sum())
        out["assigned_lost"]=int((baseline_assigned & ~assigned).sum())
        out["multi_recovered"]=int((~baseline_multi & multi).sum())
        out["multi_lost"]=int((baseline_multi & ~multi).sum())
    out["node_weighted_assigned_accuracy"]=float(d.assign(v=assigned).groupby("query").v.mean().mean())
    out["node_weighted_multi_accuracy"]=float(d.assign(v=multi).groupby("query").v.mean().mean())
    return out


def ranking(scores, positives):
    order=np.argsort(-scores,axis=1,kind="stable")
    ranked=np.take_along_axis(positives,order,axis=1)
    first=ranked.argmax(1)+1
    assert ranked.any(1).all()
    return 1/first, ranked[:,:1].any(1), ranked[:,:3].any(1), ranked[:,:5].any(1)


@torch.no_grad()
def forward(model,args,keep):
    pre,labels,edges,attrs,roles=args
    fedges, fattrs, froles=edges[:,keep],attrs[keep],roles[keep]
    x,l=model.forward_metagraph(model.layer_list[2],pre,labels,fedges,fattrs,froles,None,None,None)
    # Decode the complete candidate set, independent of which message edges were masked.
    return model.decode(model.final_input_mlp(x),model.final_label_mlp(l),edges).reshape(210,30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_source_stage_20260908_v4"))
    p.add_argument("--references",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908/cp_hk_twitter/paired_cluster_queries_private.tsv"))
    p.add_argument("--views",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/hk_canonical_views_private.pt"))
    p.add_argument("--views-receipt",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/receipt.json"))
    p.add_argument("--config",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908/effective_config.json"))
    p.add_argument("--out",type=Path,required=True); p.add_argument("--threads",type=int,default=2)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); assert 1<=a.threads<=2
    if a.dry_run:
        print(json.dumps({"target":"HK","rows":61440,"conditions":["canonical","train_mask","all_views_mask"],
            "device":"cpu","threads":a.threads,"training":False,"encoding":False,"sampling":False},indent=2)); return
    a.out.mkdir(parents=True,exist_ok=False); os.nice(10); torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    started=time.time(); rec=json.loads((a.cache/"receipt.json").read_text()); vrec=json.loads(a.views_receipt.read_text())
    assert rec["complete"] and rec["model_states_unchanged"] and digest(a.views)==vrec["canonical_views_sha256"]
    packed_views=torch.load(a.views,map_location="cpu",weights_only=False)
    views={k:{tuple(sorted(map(int,e))) for e in v.T.tolist()} for k,v in packed_views["views"].items()}
    unions={"train_mask":views["train"],"all_views_mask":set().union(*views.values())}
    refs=pd.read_csv(a.references,sep="\t"); refs=refs[refs.split.eq("test")].set_index(["episode","sample"])
    stages=pd.read_csv(a.cache/"cp_hk/stage_predictions_private.csv")
    stages=stages[stages.model.eq("hk")].set_index(["episode","sample"]); assert len(refs)==len(stages)==61440
    params=json.loads(a.config.read_text()); params["device"]="cpu"
    ckpt=rec["checkpoints"]["hk"]["path"]; model=build_model(params,ckpt,"cpu"); state=digest_state(model)
    assert state==rec["checkpoints"]["hk"]["state_sha256"] if "state_sha256" in rec["checkpoints"]["hk"] else True
    qslots=torch.tensor([i for i in range(210) if i%7>=3]); truth=(qslots//7).numpy()
    rows=[]; masked={k:[] for k in unions}; parity=[]
    for bi,item in enumerate(rec["targets"]["cp_hk"]["batches"]):
        path=a.cache/"cp_hk"/item["file"]; assert digest(path)==item["sha256"]
        cache=torch.load(path,map_location="cpu",weights_only=False)
        for ep,v in cache["models"]["hk"].items():
            episode=f"{bi}:{ep}"; args=tuple(x.clone() for x in v["inputs"]); edges,attrs=args[2],args[3]
            index=pd.MultiIndex.from_tuples([(episode,ep*210+int(q)) for q in qslots])
            rr=refs.loc[index]; ss=stages.loc[index]
            classes=rr.assign(class_slot=truth).drop_duplicates("class_slot").sort_values("class_slot")
            anchors=classes.anchor.to_numpy(dtype=int); assert len(anchors)==30
            support=np.full(210,-1,dtype=int)
            for c,row in classes.iterrows():
                slot=int(row.class_slot)
                for j in range(3): support[slot*7+j]=int(row[f"true_support_{j}"])
            positives=np.array([[tuple(sorted((int(q),int(anchor)))) in views["test"] for anchor in anchors]
                                for q in rr["query"].to_numpy(dtype=int)],dtype=bool)
            assert positives[np.arange(120),truth].all()
            base=v["native_logits"].numpy(); base_pred=ss.native_prediction.to_numpy(dtype=int)
            shortfall=base.max(1)-base[np.arange(120),base_pred]
            assert shortfall.max()<1e-4 and np.array_equal(base_pred==truth,ss.native_correct.to_numpy(dtype=bool))
            parity.append(float(shortfall.max()))
            mrr,h1,h3,h5=ranking(base,positives)
            common={"episode":[episode]*120,"sample":[x[1] for x in index],"query":rr["query"].to_numpy(),
                    "truth":truth,"valid_test_candidates":positives.sum(1),"canonical_prediction":base_pred,
                    "canonical_assigned":base_pred==truth,"canonical_multi":positives[np.arange(120),base_pred],
                    "canonical_mrr":mrr,"canonical_hits1":h1,"canonical_hits3":h3,"canonical_hits5":h5}
            frame=pd.DataFrame(common)
            for condition,known in unions.items():
                contradiction=np.zeros(edges.shape[1],dtype=bool)
                neg=(edges[0]<210)&(attrs[:,1]<0)
                for ei in torch.where(neg)[0].tolist():
                    src=int(edges[0,ei]); dest=int(edges[1,ei])-210
                    contradiction[ei]=tuple(sorted((int(support[src]),int(anchors[dest])))) in known
                keep=torch.from_numpy(~contradiction); masked[condition].append(int(contradiction.sum()))
                z=forward(model,args,keep)[qslots].numpy(); pred=z.argmax(1); rm,hh1,hh3,hh5=ranking(z,positives)
                frame[condition+"_prediction"]=pred; frame[condition+"_assigned"]=pred==truth
                frame[condition+"_multi"]=positives[np.arange(120),pred]
                frame[condition+"_mrr"]=rm; frame[condition+"_hits1"]=hh1; frame[condition+"_hits3"]=hh3; frame[condition+"_hits5"]=hh5
                frame[condition+"_masked_edges"]=int(contradiction.sum())
            rows.append(frame)
        print("BATCH",bi,"elapsed",round(time.time()-started,2),flush=True)
    d=pd.concat(rows,ignore_index=True); assert len(d)==61440 and not d.duplicated(["episode","sample"]).any()
    baseline_a=d.canonical_assigned.astype(bool); baseline_m=d.canonical_multi.astype(bool)
    report={"complete":True,"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "seconds":time.time()-started,"device":"cpu","threads":a.threads,"rows":len(d),"checkpoint":ckpt,
        "model_state_sha256":state,"unchanged_model":digest_state(model)==state,"conditions":{},
        "masked_edges":{k:{"total":sum(v),"episodes_with_any":sum(x>0 for x in v),"median_per_episode":float(np.median(v)),"max_per_episode":max(v)} for k,v in masked.items()},
        "max_canonical_winner_shortfall":max(parity),"sources":{str(x):digest(x) for x in [a.references,a.views,a.views_receipt,a.config,a.cache/"receipt.json",a.cache/"cp_hk/stage_predictions_private.csv"]},
        "training":False,"encoding":False,"sampling":False,
        "limits":["Train-mask is an inference-only intervention and cannot undo training gradients.","All-views mask uses validation/test adjacency and is a nondeployable diagnostic upper bound."]}
    report["conditions"]["canonical"]=score_summary(d,"canonical",None,None)
    for condition in unions: report["conditions"][condition]=score_summary(d,condition,baseline_a,baseline_m)
    d.to_csv(a.out/"rows_private.csv",index=False); report["rows_sha256"]=digest(a.out/"rows_private.csv")
    (a.out/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__": main()
