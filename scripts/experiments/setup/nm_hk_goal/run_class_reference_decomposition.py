"""Exact additive HK metagraph class-reference score decomposition from caches."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS","2"); os.environ.setdefault("MKL_NUM_THREADS","2")
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scripts.experiments.setup.nm_hk_mechanism.run import build_model
from scripts.experiments.setup.nm_hk_goal.metagraph import replay
from scripts.experiments.setup.nm_support_resampling.run import digest_state


def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
    return h.hexdigest()


def summarize(d):
    err=~d.native_correct
    e=d[err]
    components=["positive","negative","self","output_bias","label_residual","bn_offset"]
    return {"rows":len(d),"errors":len(e),"accuracy":float(d.native_correct.mean()),
        "error_gap": {"median":float(e.rival_minus_true.median()),"mean":float(e.rival_minus_true.mean())},
        "error_component_gap_means":{c:float(e[c+"_gap"].mean()) for c in components},
        "error_component_gap_medians":{c:float(e[c+"_gap"].median()) for c in components},
        "largest_pro_error_component":{str(k):{"count":int(v),"fraction":float(v/len(e))} for k,v in e.largest_pro_error_component.value_counts().items()},
        "positive_support_favors_wrong_winner":float(e.positive_gap.gt(0).mean()),
        "positive_support_gap_exceeds_total_gap":float(e.positive_gap.gt(e.rival_minus_true).mean()),
        "negative_support_favors_wrong_winner":float(e.negative_gap.gt(0).mean()),
        "positive_term_top1_accuracy":float(d.positive_prediction.eq(d.truth).mean()),
        "positive_plus_negative_top1_accuracy":float(d.support_prediction.eq(d.truth).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_source_stage_20260908_v4"))
    p.add_argument("--config",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908/effective_config.json"))
    p.add_argument("--checkpoint",type=Path,default=None,
                   help="Optional local copy of the selected checkpoint recorded by the cache receipt.")
    p.add_argument("--model", choices=("hk", "ukr"), default="hk")
    p.add_argument("--out",type=Path,required=True); p.add_argument("--threads",type=int,default=2); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); assert 1<=a.threads<=2
    if a.dry_run:
        print(json.dumps({"target":"HK","episodes":512,"queries":61440,"device":"cpu","threads":a.threads,"training":False,"encoding":False},indent=2)); return
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        os.nice(10)
    except PermissionError:
        pass
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    started=time.time(); receipt=json.loads((a.cache/"receipt.json").read_text()); assert receipt["complete"]
    stages=pd.read_csv(a.cache/"cp_hk/stage_predictions_private.csv"); stages=stages[stages.model.eq(a.model)].set_index(["episode","sample"])
    params=json.loads(a.config.read_text());params["device"]="cpu"
    ckpt=str(a.checkpoint) if a.checkpoint is not None else receipt["checkpoints"][a.model]["path"]
    model=build_model(params,ckpt,"cpu"); state=digest_state(model); query_slots=torch.tensor([i for i in range(210) if i%7>=3]);truth=(query_slots//7).numpy()
    components=["positive","negative","self","output_bias","label_residual","bn_offset"]
    rows=[]; max_error=0.; argmax_differences=0
    for bi,item in enumerate(receipt["targets"]["cp_hk"]["batches"]):
        path=a.cache/"cp_hk"/item["file"];assert digest(path)==item["sha256"]
        packed=torch.load(path,map_location="cpu",weights_only=False)
        for ep,v in packed["models"][a.model].items():
            episode=f"{bi}:{ep}"; args=tuple(x.clone() for x in v["inputs"]); z,audit=replay(model,args); scores=z[query_slots]
            ref=stages.loc[pd.MultiIndex.from_tuples([(episode,ep*210+int(q)) for q in query_slots])]
            canonical=ref.native_prediction.to_numpy(dtype=int); shortfall=scores.max(1).values-scores[torch.arange(120),torch.tensor(canonical)]
            assert float(shortfall.max())<1e-4; argmax_differences+=int((scores.argmax(1).numpy()!=canonical).sum())
            q=F.normalize(audit["inputs"][query_slots],dim=1); norms=audit["labels"].norm(dim=1); scale=float(model.logit_scale.exp())
            terms={k:(q@v.T/norms[None]*scale) for k,v in audit["parts"].items()}
            error=float((sum(terms.values())-scores).abs().max()); assert error<1e-4;max_error=max(max_error,error)
            rival=scores.clone();rival[torch.arange(120),torch.tensor(truth)]=-torch.inf;rival_idx=rival.argmax(1).numpy()
            ix=np.arange(120); data={"episode":[episode]*120,"sample":ep*210+query_slots.numpy(),"truth":truth,"native_correct":ref.native_correct.to_numpy(dtype=bool),
                "native_prediction":canonical,"best_rival":rival_idx,"rival_minus_true":(scores[ix,rival_idx]-scores[ix,truth]).numpy()}
            gaps=[]
            for k in components:
                gap=(terms[k][ix,rival_idx]-terms[k][ix,truth]).numpy();data[k+"_gap"]=gap;gaps.append(gap)
            data["largest_pro_error_component"]=np.array(components)[np.stack(gaps,axis=1).argmax(1)]
            data["positive_prediction"]=terms["positive"].argmax(1).numpy()
            data["support_prediction"]=(terms["positive"]+terms["negative"]).argmax(1).numpy()
            rows.append(pd.DataFrame(data))
        print("BATCH",bi,"elapsed",round(time.time()-started,2),flush=True)
    d=pd.concat(rows,ignore_index=True);assert len(d)==61440 and not d.duplicated(["episode","sample"]).any();assert digest_state(model)==state
    d.to_csv(a.out/"rows_private.csv",index=False)
    first=d[(d.episode=="0:0")&d["sample"].eq(3)].iloc[0]
    report={"complete":True,"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"seconds":time.time()-started,"device":"cpu","threads":a.threads,"model":a.model,
        "model_state_sha256":state,"unchanged_model":True,"max_additive_error":max_error,"canonical_argmax_differences_within_1e4":argmax_differences,
        "all":summarize(d),"episode_0_0":summarize(d[d.episode.eq("0:0")]),"first_query":{k:(bool(v) if isinstance(v,np.bool_) else int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else v) for k,v in first.to_dict().items()},
        "rows_sha256":digest(a.out/"rows_private.csv"),"cache_receipt_sha256":digest(a.cache/"receipt.json"),"config_sha256":digest(a.config),"training":False,"encoding":False,"sampling":False,
        "interpretation":"Exact additive decomposition of each full-model score using the realized normalized label reference. Component dominance is descriptive; components are not independent counterfactual causes."}
    (a.out/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":main()
