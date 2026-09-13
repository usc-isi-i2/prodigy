"""Subgroup accounting for the frozen overlap-aware HK diagnostic."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def stats(d, condition):
    a=d[condition+"_assigned"].astype(bool); m=d[condition+"_multi"].astype(bool)
    ba=d.canonical_assigned.astype(bool); bm=d.canonical_multi.astype(bool)
    return {"n":len(d),"assigned_accuracy":float(a.mean()),"multi_accuracy":float(m.mean()),
        "assigned_recovered":int((~ba&a).sum()),"assigned_lost":int((ba&~a).sum()),
        "multi_recovered":int((~bm&m).sum()),"multi_lost":int((bm&~m).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows",type=Path,required=True); p.add_argument("--classified",type=Path,required=True)
    p.add_argument("--residual",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); d=pd.read_csv(a.rows); c=pd.read_csv(a.classified)
    keys=["episode","sample","query"]; d=d.merge(c[keys+["category"]],on=keys,validate="one_to_one")
    r=pd.read_csv(a.residual); keep=keys+["source_outcome","hk_predicted_edge_split","hk_query_behavior","rank_group"]
    d=d.merge(r[keep],on=keys,how="left",validate="one_to_one")
    d["valid_count_group"]=pd.cut(d.valid_test_candidates,[0,1,2,30],labels=["1","2","3+"])
    d["canonical_error_type"]=np.select([d.canonical_assigned,d.canonical_multi],
        ["assigned_correct","other_test_neighbor"],default="no_test_neighbor_selected")
    out={"conditions":{},"by_valid_candidate_count":{},"by_original_outcome":{},
         "by_failure_stage":{},"residual_phenotypes":{}}
    for condition in ["train_mask","all_views_mask"]:
        out["conditions"][condition]=stats(d,condition)
        out["by_valid_candidate_count"][condition]={str(k):stats(z,condition) for k,z in d.groupby("valid_count_group",observed=True)}
        out["by_original_outcome"][condition]={str(k):stats(z,condition) for k,z in d.groupby("canonical_error_type")}
        errors=d[~d.canonical_assigned]
        out["by_failure_stage"][condition]={str(k):stats(z,condition) for k,z in errors.groupby("category")}
        residual=d[d.category.eq("unseparated_by_measured_input_summaries")]
        out["residual_phenotypes"][condition]={field:{str(k):stats(z,condition) for k,z in residual.groupby(field,observed=True)}
            for field in ["source_outcome","hk_predicted_edge_split","hk_query_behavior","rank_group"]}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n"); print(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__": main()
