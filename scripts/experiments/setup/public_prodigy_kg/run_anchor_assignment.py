"""Fixed label-code assignment sensitivity, not graph permutation equivariance."""
import argparse
from contextlib import contextmanager
import json
import os
import subprocess
from pathlib import Path
from . import run_native as native

EPISODES = 32
WAYS = 20


@contextmanager
def assignment(table, reverse):
    import torch
    original = table.detach().clone()
    try:
        with torch.no_grad():
            if reverse:
                table[:WAYS].copy_(original[:WAYS].flip(0))
        yield
    finally:
        with torch.no_grad():
            table.copy_(original)


def main():
    p = argparse.ArgumentParser(__doc__)
    for field in ("source", "checkpoint", "upstream", "output"):
        p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=(2,3), default=3)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    protocol = json.loads((args.source/"protocol.json").read_text())
    records = json.loads((args.source/"paired_episodes/index.json").read_text())["records"]
    if args.checkpoint.name != "state_dict_8000.ckpt" or native.file_sha256(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Nominated8k checkpoint required")
    if json.loads((args.source/"execution_status.json").read_text())["status"] != "complete" or len(records)!=500 or [r["ordinal"] for r in records]!=list(range(500)):
        raise ValueError("Complete fresh source required")
    upstream = native.verify_upstream(args.upstream)
    plan = dict(episodes=EPISODES, source=str(args.source), checkpoint=str(args.checkpoint),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        effective_params_sha256=native.file_sha256(args.source/"effective_params.json"),
        checkpoint_sha256=native.file_sha256(args.checkpoint), upstream=upstream,
        source_index_sha256=native.file_sha256(args.source/"paired_episodes/index.json"),
        source_protocol_sha256=native.file_sha256(args.source/"protocol.json"),
        identity=list(range(WAYS)), reverse=list(reversed(range(WAYS))),
        intervention="Reverse only first20 learned_label_embedding rows; input/output semantic columns unchanged",
        interpretation="Label-code assignment invariance, NOT graph-node permutation equivariance",
        native_parity_atol=1e-4, u1_parity="bit-exact", tuning=False)
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),WANDB_MODE="offline")
    native.runtime_versions()
    native.isolate_native_path(args.upstream)
    os.chdir(args.upstream)
    import torch
    import numpy as np
    from .run_saved_readout import construct_model
    from .run_trajectory import forward_checkpoint
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
    torch.set_num_threads(1)
    params=json.loads((args.source/"effective_params.json").read_text())
    if params["batch_size"]!=1 or params["n_way"]!=20 or not params["ignore_label_embeddings"] or params["zero_label_embeddings"]:
        raise ValueError("Expected single20way learned label-table episode")
    plan["not_freeze_learned_label_embedding"]=params.get("not_freeze_learned_label_embedding")
    plan["native_training_freeze_rule"]="Pinned upstream experiments/trainer.py144-148 disables table gradients when not_freeze_learned_label_embedding is false and filters those parameters from AdamW"
    model=construct_model(params,args.checkpoint,"cuda:0")
    native.verify_imports(args.upstream)
    table=model.learned_label_embedding.weight
    if len(table)<WAYS:
        raise ValueError("Label table too small")
    plan["table_requires_grad_at_construction"]=table.requires_grad
    plan["requires_grad_note"]="Constructor flag is not evidence of whether this table was optimized during pretraining"
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    args.output.mkdir(parents=True,exist_ok=False)
    native.write_json(args.output/"protocol.json",plan)
    rows,receipts=[],[]
    try:
        for record in records[:EPISODES]:
            path=args.source/"paired_episodes"/record["file"]
            if path.resolve().parent!=(args.source/"paired_episodes").resolve() or native.file_sha256(path)!=record["sha256"]:
                raise ValueError("Source receipt failed")
            artifact=torch.load(path,map_location="cpu")["native_capture"]
            predictions,geometry={},{}
            for name,reverse in (("identity",False),("reverse",True)):
                with assignment(table,reverse):
                    yt,z,x=forward_checkpoint(model,artifact,"cuda:0")
                if z.shape!=(80,20) or not torch.isfinite(z).all() or not torch.equal(yt,artifact["output"]["y_true"]):
                    raise ValueError("Invalid or unaligned native query output")
                predictions[name]=z
                geometry[name]=x
            torch.testing.assert_close(predictions["identity"],artifact["output"]["logits"],atol=1e-4,rtol=0)
            if not torch.equal(yt,artifact["output"]["y_true"]):
                raise ValueError("Semantic label order changed")
            if not torch.equal(geometry["identity"],geometry["reverse"]):
                raise ValueError("U1 differs; label-code intervention not isolated")
            for key,value in model.state_dict().items():
                if not torch.equal(value.cpu(),original[key]):
                    raise ValueError("Model state not restored: "+key)
            scores={name:episode_metrics(yt.argmax(1).numpy(),z.numpy()) for name,z in predictions.items()}
            row=dict(ordinal=record["ordinal"],metrics=scores,
                reverse_minus_identity={k:scores["reverse"][k]-scores["identity"][k] for k in scores["identity"]},
                disagreements=int((predictions["identity"].argmax(1)!=predictions["reverse"].argmax(1)).sum()),
                max_logit_change=float((predictions["identity"]-predictions["reverse"]).abs().max()),
                max_native_parity_error=float((predictions["identity"]-artifact["output"]["logits"]).abs().max()),u1_bitexact=True)
            rows.append(row)
            out=args.output/record["file"]
            torch.save(dict(source_sha256=record["sha256"],y_true_onehot=yt,logits=predictions,u1=geometry["identity"],row=row),out)
            receipts.append(dict(file=record["file"],sha256=native.file_sha256(out),source_sha256=record["sha256"]))
            print(f"Anchor assignment {len(rows)}/{EPISODES}",flush=True)
        means={arm:{k:float(np.mean([r["metrics"][arm][k] for r in rows])) for k in rows[0]["metrics"][arm]} for arm in ("identity","reverse")}
        native.write_json(args.output/"summary.json",dict(rows=rows,means=means,receipts=receipts,disagreements=sum(r["disagreements"] for r in rows)))
        native.write_json(args.output/"execution_status.json",dict(status="complete"))
    except BaseException as error:
        native.write_json(args.output/"execution_status.json",dict(status="failed",error=repr(error)))
        raise


if __name__=="__main__":
    main()
