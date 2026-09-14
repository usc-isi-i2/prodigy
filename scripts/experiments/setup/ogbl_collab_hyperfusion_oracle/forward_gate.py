#!/usr/bin/env python3
"""Fit the frozen shallow gate on 2018 and apply it unchanged to 2019 test."""
import argparse, json
from pathlib import Path
import numpy as np
import run as oracle
from gate import fit_predict

def percentile(reference_pos, reference_neg, target):
    reference=np.sort(np.concatenate([reference_pos,reference_neg]))
    return np.searchsorted(reference,target,side="right").astype(float)/len(reference)

def feature_sets(vp,vn,tp,tn,validation,test):
    values=[[],[],[],[]]
    for i in range(3):
        for out,target in zip(values,(vp[i],vn[i],tp[i],tn[i])):
            out.append(percentile(vp[i],vn[i],target))
    def make(rows,panel,side):
        s=np.stack(rows); d=np.stack([s[0]-s[1],s[0]-s[2],s[1]-s[2]],axis=1)
        return np.column_stack([s.T,np.abs(d),panel[side+"features"],panel[side+"symmetric"]]).astype(np.float32)
    return make(values[0],validation,"p"),make(values[1],validation,"n"),make(values[2],test,"p"),make(values[3],test,"n")

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--oracle-root",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--dataset-root",type=Path,default=Path("/dataMeR1/phil/data/ogb")); ap.add_argument("--upstream",type=Path,default=Path("/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream")); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    protocol=dict(name="compact_forward_gate_v1",revision=oracle.revision(),train_year=2018,test_year=2019,
        arm="HistGB(iter80,leaves7,lr.05,l2=1,class-balanced,seed0)",selection="none",target=.713,
        score_calibration="2018 empirical CDF frozen for 2019",inputs="same33 oracle-gate features",
        parameter_upper_bound=535376,test_previously_opened=True,evidence="forward but post-hoc after test inspection")
    if args.dry_run: print(json.dumps(protocol,indent=2)); return
    args.out.mkdir(parents=True,exist_ok=False); oracle.save(args.out/"protocol.json",protocol)
    old=json.loads((args.oracle_root/"results.json").read_text()); z=np.load(args.oracle_root/"scores.npz"); assert old["scores_sha256"]==oracle.sha(args.oracle_root/"scores.npz")
    graph,split,_,version=oracle.joint.shared.load_official_dataset(args.dataset_root); assert oracle.joint.shared.audit_dataset(graph,split,version)["split_fingerprint"]==oracle.gate.FINGERPRINT
    aa=oracle.module(args.upstream/"aa_dc.py","aa_forward"); validation,_,cal=oracle.gate.make_year(aa,oracle.joint.shared,graph,split,2018,None,warm_only=True,symmetric_features=True); test,meta=oracle.test_panel(graph,split,aa,cal)
    np.testing.assert_array_equal(validation["bp"],z["validation_aa_p"]); np.testing.assert_array_equal(test["bp"],z["test_aa_p"])
    rows=[]
    for seed in oracle.SEEDS:
        vp=np.stack([z["validation_aa_p"],z[f"validation_structure{seed}_p"],z[f"validation_joint{seed}_p"]]); vn=np.stack([z["validation_aa_n"],z[f"validation_structure{seed}_n"],z[f"validation_joint{seed}_n"]])
        tp=np.stack([z["test_aa_p"],z[f"test_structure{seed}_p"],z[f"test_joint{seed}_p"]]); tn=np.stack([z["test_aa_n"],z[f"test_structure{seed}_n"],z[f"test_joint{seed}_n"]])
        xp,xn,yp,yn=feature_sets(vp,vn,tp,tn,validation,test); x=np.concatenate([xp,xn]); labels=np.concatenate([np.ones(len(xp)),np.zeros(len(xn))]); pred=fit_predict("histgb",x,labels,np.concatenate([yp,yn]))
        rows.append(dict(seed=seed,test_hits_at_50=oracle.hits(pred[:len(yp)],pred[len(yp):])))
    vals=[r["test_hits_at_50"] for r in rows]; mean=float(np.mean(vals)); result=dict(complete=True,revision=oracle.revision(),rows=rows,mean=mean,sample_sd=float(np.std(vals,ddof=1)),beats_hyperfusion=mean>.7129,target_reached=mean>=.713,test_scored=True,oracle_results_sha256=oracle.sha(args.oracle_root/"results.json"),test_metadata=meta)
    oracle.save(args.out/"results.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
