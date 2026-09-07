"""Validate the full published-cue audit and summarize every declared stratum."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.experiments.setup.target_performance_mechanisms.annotation_cues import DETECTORS, SUBSETS


METRICS = ("roc_auc", "accuracy", "f1", "nll", "unique_account_auc")


def validate_and_summarize(protocol, done, rows, receipts, provenance, support_counts):
    import math
    models = set(protocol["models"])
    conditions = {tuple(x) for x in protocol["conditions"]}
    wanted_conditions = {("intact","intact",0),("intact","removed",0),("removed","intact",0),("removed","removed",0)} | {("intact","rewired",d) for d in range(3)}
    if len(models) != 6 or conditions != wanted_conditions or protocol["target"] != "covid_political":
        raise ValueError("Wrong model or condition panel")
    if protocol["detectors"] != list(DETECTORS) or protocol["subsets"] != list(SUBSETS):
        raise ValueError("Changed cue definitions or strata")
    expected = {(s,m,*c,d,g) for s in ("original","fresh") for m in models for c in conditions for d in DETECTORS for g in SUBSETS}
    expected |= {(s,"raw_prototype","intact","intact",0,d,g) for s in ("original","fresh") for d in DETECTORS for g in SUBSETS}
    key = lambda r: tuple(r[k] for k in ("stream","model_id","query_condition","support_condition","draw","detector","subset"))
    by_key = {key(r):r for r in rows}
    if len(rows) != 1376 or len(by_key) != len(rows) or set(by_key) != expected or done["cells"] != len(rows):
        raise ValueError("Incomplete or duplicate cue grid")
    if not all(done[k] for k in ("provenance_join_complete","all_cached_features_and_labels_match","historical_metric_parity")) or done["new_model_forwards"] != 0:
        raise ValueError("Failed provenance or frozen-prediction audit")
    if len(receipts) != 12 or {(r["stream"],r["model_id"]) for r in receipts} != {(s,m) for s in ("original","fresh") for m in models}:
        raise ValueError("Incomplete prediction receipts")
    for r in receipts:
        if r["batch_count"] != 32 or not r["all_input_hashes_and_features_match"] or not 0 <= r["metric_max_abs_error"] <= 1e-6:
            raise ValueError("Failed input or historical parity")
    if provenance["graph_rows"] != 78672 or provenance["unique_complete_row_matches"] != 78672 or not provenance["all_labels_match"]:
        raise ValueError("Incomplete original-profile recovery")
    if {(r["stream"],r["episode"],r["detector"]) for r in support_counts} != {(s,e,d) for s in ("original","fresh") for e in range(128) for d in DETECTORS} or len(support_counts) != 512:
        raise ValueError("Incomplete support-context census")
    counts = {}
    for r in rows:
        if r["class_0"] + r["class_1"] != r["queries"] or not 0 <= r["unique_queries"] <= r["queries"] or not 0 <= r["episodes"] <= 128:
            raise ValueError("Invalid subset counts")
        ck = r["stream"],r["detector"],r["subset"]
        signature = tuple(r[k] for k in ("queries","unique_queries","episodes","class_0","class_1"))
        if ck in counts and counts[ck] != signature: raise ValueError("Condition-dependent stratum")
        counts[ck] = signature
        evaluable = min(r["class_0"],r["class_1"]) > 0
        for metric in METRICS:
            value = r[metric]
            if evaluable != (value is not None) or (value is not None and (not math.isfinite(value) or value < 0 or (metric != "nll" and value > 1))):
                raise ValueError("Invalid/undefined subset metric")
    for stream in ("original","fresh"):
        for detector in DETECTORS:
            c = {g: counts[stream,detector,g][0] for g in SUBSETS}
            for prefix in ("query_center","query_subgraph","supports"):
                if c[prefix+"_absent"] + c[prefix+"_present"] != c["all"]:
                    raise ValueError("Strata do not partition queries")
            if not c["all_input_absent"] <= min(c["query_subgraph_absent"], c["supports_absent"]) or c["query_subgraph_absent"] > c["query_center_absent"]:
                raise ValueError("Cue-containment invariant failed")
    paired, groups = [], {}
    for r in rows:
        if r["model_id"] == "raw_prototype" or (r["query_condition"],r["support_condition"]) == ("intact","intact"): continue
        base = by_key[r["stream"],r["model_id"],"intact","intact",0,r["detector"],r["subset"]]
        out = dict(r)
        for metric in METRICS:
            out["delta_"+metric] = None if r[metric] is None else r[metric]-base[metric]
        paired.append(out)
        gk = tuple(r[k] for k in ("source","query_condition","support_condition","draw","detector","subset"))
        groups.setdefault(gk,[]).append(out)
    summary = []
    for k, group in sorted(groups.items()):
        out = dict(zip(("source","query_condition","support_condition","draw","detector","subset"),k))
        out["cells"] = len(group)
        for metric in METRICS:
            v = [r["delta_"+metric] for r in group if r["delta_"+metric] is not None]
            out[metric] = {"mean":mean(v),"min":min(v),"max":max(v),"positive":sum(x>0 for x in v),"negative":sum(x<0 for x in v),"evaluable":len(v)} if v else None
        summary.append(out)
    subset_counts = [dict(zip(("stream","detector","subset"),k), **dict(zip(("queries","unique_queries","episodes","class_0","class_1"),v))) for k,v in sorted(counts.items())]
    return {"paired":paired,"summary":summary,"subset_counts":subset_counts,
            "raw_prototype":[r for r in rows if r["model_id"]=="raw_prototype"]}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    names=("protocol","DONE","metrics","receipts","provenance","support_counts")
    values=[json.loads((args.input/(n+".json")).read_text()) for n in names]
    result=validate_and_summarize(*values)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        (args.output/(name+".json")).write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    validation={"cells":len(values[2]),"complete_grid":True,"runtime_revision":values[0]["revision"],
        "original_profiles_joined":values[4]["unique_complete_row_matches"],
        "sha256":{n:hashlib.sha256((args.input/(n+".json")).read_bytes()).hexdigest() for n in names},
        "interpretation":"Descriptive overlapping strata; seed/stream ranges are not confidence intervals. Detector absence does not establish independent annotations or remove cues from supports."}
    (args.output/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__": main()
