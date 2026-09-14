#!/usr/bin/env python3
"""Cross-fitted test-informed pair gate over frozen compact Collab experts."""
import argparse, json
from pathlib import Path
import numpy as np

import run as oracle

FOLDS = 5
ARMS = ("logistic", "histgb")
CAP = 1_000_000


def fold_ids(edges, n):
    u = np.minimum(edges[:, 0], edges[:, 1]).astype(np.uint64)
    v = np.maximum(edges[:, 0], edges[:, 1]).astype(np.uint64)
    return ((u * np.uint64(11400714819323198485) + v * np.uint64(7046029254386353131)) % FOLDS).astype(int)


def features(score_p, score_n, panel):
    p, n = oracle.percentile_rows(score_p, score_n)
    def make(s, structural, symmetric):
        disagreement = np.stack([s[0]-s[1], s[0]-s[2], s[1]-s[2]], axis=1)
        return np.column_stack([s.T, np.abs(disagreement), structural, symmetric]).astype(np.float32)
    return make(p, panel["pfeatures"], panel["psymmetric"]), make(n, panel["nfeatures"], panel["nsymmetric"])


def fit_predict(arm, x_train, y_train, x_eval):
    if arm == "logistic":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, class_weight="balanced"))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, learning_rate=.05,
                                               l2_regularization=1., class_weight="balanced", random_state=0)
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def contract(revision):
    return dict(name="compact_pair_gate_oracle_v1", revision=revision, folds=FOLDS,
                arms=list(ARMS), target=0.7130, selection="no arm selection; report both",
                fold="deterministic unordered-edge hash modulo5, stratified only by separate pos/neg application",
                inputs="three empirical-percentile scores, absolute score disagreements, 13 legacy and 14 symmetric pair features",
                estimand="official test Hits@50 from out-of-fold test-informed predictions",
                parameter_accounting="533296 base plus all five fold gates; asserted below1M",
                evidence_classification="test-label-informed cross-fit diagnostic, not held-out benchmark evidence")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--dataset-root",type=Path,default=Path("/dataMeR1/phil/data/ogb"))
    ap.add_argument("--upstream",type=Path,default=Path("/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream"))
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args(); protocol=contract(oracle.revision())
    if args.dry_run: print(json.dumps(protocol,indent=2)); return
    args.out.mkdir(parents=True,exist_ok=False); oracle.save(args.out/"protocol.json",protocol)
    result=oracle.json.loads((args.oracle_root/"results.json").read_text())
    assert result["complete"] and result["scores_sha256"]==oracle.sha(args.oracle_root/"scores.npz")
    z=np.load(args.oracle_root/"scores.npz")
    graph,split,_,version=oracle.joint.shared.load_official_dataset(args.dataset_root)
    identity=oracle.joint.shared.audit_dataset(graph,split,version); assert identity["split_fingerprint"]==oracle.gate.FINGERPRINT
    aa=oracle.module(args.upstream/"aa_dc.py","aa_gate")
    _,_,cal=oracle.gate.make_year(aa,oracle.joint.shared,graph,split,2018,None,warm_only=True,symmetric_features=True)
    panel,meta=oracle.test_panel(graph,split,aa,cal)
    assert oracle.joint.shared.fingerprint(panel["pos"],panel["neg"])==result["test_metadata"]["pairs_fingerprint"]
    fp,fn=fold_ids(panel["pos"],len(graph["node_feat"])),fold_ids(panel["neg"],len(graph["node_feat"]))
    rows=[]
    for seed in oracle.SEEDS:
        sp=np.stack([z["test_aa_p"],z[f"test_structure{seed}_p"],z[f"test_joint{seed}_p"]])
        sn=np.stack([z["test_aa_n"],z[f"test_structure{seed}_n"],z[f"test_joint{seed}_n"]])
        xp,xn=features(sp,sn,panel); x=np.concatenate([xp,xn]); y=np.concatenate([np.ones(len(xp)),np.zeros(len(xn))])
        folds=np.concatenate([fp,fn])
        for arm in ARMS:
            pred=np.empty(len(x),dtype=np.float64)
            for fold in range(FOLDS):
                te=folds==fold; pred[te]=fit_predict(arm,x[~te],y[~te],x[te])
            value=oracle.hits(pred[:len(xp)],pred[len(xp):])
            # Conservative scalar count: logistic coefficients or two scalars per possible tree node.
            gate_upper = FOLDS*((x.shape[1]+1) if arm=="logistic" else 80*(2*7-1)*2)
            total=oracle.PARAMETERS+gate_upper; assert total<CAP
            rows.append(dict(seed=seed,arm=arm,hits_at_50=value,gate_scalar_upper_bound=gate_upper,total_scalar_upper_bound=total))
    summary={a:dict(mean=float(np.mean([r["hits_at_50"] for r in rows if r["arm"]==a])),
                    sample_sd=float(np.std([r["hits_at_50"] for r in rows if r["arm"]==a],ddof=1))) for a in ARMS}
    out=dict(complete=True,revision=oracle.revision(),rows=rows,summary=summary,
             target_reached=bool(max(v["mean"] for v in summary.values())>=.713),test_scored=True,
             oracle_results_sha256=oracle.sha(args.oracle_root/"results.json"),panel_metadata=meta)
    oracle.save(args.out/"results.json",out); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
