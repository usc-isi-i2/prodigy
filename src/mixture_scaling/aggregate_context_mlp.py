from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .context_views import VIEWS
from .evaluate_lattice import LP_TARGETS
from .lattice import SOURCE_ORDER
from .node_only_transfer import singleton_rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--view",choices=VIEWS,required=True); p.add_argument("--objective",choices=("fp","lp"),required=True); p.add_argument("--state-root",required=True); p.add_argument("--results-root",required=True); p.add_argument("--output-root",required=True); a=p.parse_args()
    state=Path(a.state_root); results=Path(a.results_root); targets=SOURCE_ORDER if a.objective=="fp" else LP_TARGETS; rows=[]; missing=[]; fanouts=set()
    for run_id,(source,) in singleton_rows():
        summary=state/a.view/a.objective/run_id/"summary.json"
        if not summary.is_file(): missing.append(f"train:{run_id}"); continue
        payload=json.loads(summary.read_text())
        if payload.get("status")!="complete" or payload.get("input_view")!=a.view: raise ValueError(summary)
        fanout=int(payload["fanout"]); fanouts.add(fanout)
        for target in targets:
            path=results/a.view/a.objective/f"{run_id}__to__{target}.json"
            if not path.is_file(): missing.append(f"eval:{run_id}->{target}"); continue
            value=json.loads(path.read_text())
            if value.get("status")!="complete": missing.append(f"eval:{run_id}->{target}"); continue
            if "fanout" in value and int(value["fanout"]) != fanout: raise ValueError(f"fanout mismatch {path}")
            row={"view":a.view,"objective":a.objective,"fanout":fanout,"source":source,"target":target,"checkpoint_step":value["checkpoint_step"]}
            if a.objective=="fp": row.update(metric="scaled_cosine_error",value=value["scaled_cosine_error_mean"],replicate_std=value["scaled_cosine_error_std"])
            else:
                if value["gates"]["holdout_leakage_edges"]!=0 or value["gates"]["endpoint_sensitivity"]<=0: raise ValueError(f"failed gate {path}")
                row.update(metric="roc_auc",value=value["report"]["auc"],replicate_std="")
            rows.append(row)
    if missing: raise FileNotFoundError(f"missing {len(missing)}: {missing[:20]}")
    if len(fanouts) != 1: raise ValueError(f"mixed fanouts: {sorted(fanouts)}")
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); table=out/f"{a.view}_{a.objective}_transfer.tsv"
    with table.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    receipt={"status":"complete","view":a.view,"objective":a.objective,"fanout":next(iter(fanouts)),"training_models":9,"evaluation_cells":len(rows)}; (out/f"{a.view}_{a.objective}_COMPLETE.json").write_text(json.dumps(receipt,indent=2)+"\n"); print(json.dumps(receipt)); return 0
if __name__=="__main__": raise SystemExit(main())
