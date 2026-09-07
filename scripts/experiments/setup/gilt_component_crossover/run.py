"""Fixed GILT100/900 crossover on canonical cached TwiBot20 episodes."""
import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.experiments.setup.public_prodigy_kg.run_native import file_sha256, write_json
from scripts.experiments.setup.public_prodigy_kg.role_centering import predict
from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics

STEPS=(100,900)
CONDITIONS=((100,100),(100,900),(900,100),(900,900))


def assemble(early,late,encoder_step,predictor_step):
    if early.keys()!=late.keys():raise ValueError("Different checkpoint inventories")
    if not torch.equal(early["feature_projection"],late["feature_projection"]):
        raise ValueError("Feature projection must be bit-exact")
    result,ownership={}, {"encoder":[],"predictor":[],"fixed":[]}
    for key in early:
        if key=="feature_projection":owner="fixed";source=early
        elif key.startswith("encoder."):owner="encoder";source=early if encoder_step==100 else late
        elif key.startswith("predictor."):owner="predictor";source=early if predictor_step==100 else late
        else:raise ValueError("Unknown state key: "+key)
        if early[key].shape!=late[key].shape or early[key].dtype!=late[key].dtype:raise ValueError("State shape/dtype differs")
        ownership[owner].append(key);result[key]=source[key].detach().clone()
    if not all(ownership.values()) or (encoder_step,predictor_step) not in CONDITIONS:raise ValueError("Invalid partition/condition")
    return result,ownership


def clone_episode(episode,device):
    result=copy.copy(episode)
    for field in fields(episode):
        value=getattr(episode,field.name)
        if isinstance(value,torch.Tensor):setattr(result,field.name,value.clone().to(device))
    return result


def score(labels,logits,label_map):
    from sklearn.metrics import f1_score
    result=episode_metrics(labels.numpy(),logits.numpy())
    mapping=label_map.cpu()
    if sorted(mapping.tolist())!=[0,1]:raise ValueError("Expected binary global mapping")
    result["f1"]=float(f1_score(mapping[labels].numpy(),mapping[logits.argmax(1)].numpy(),zero_division=0))
    return result


def observed_forward(model,episode,device,seed):
    captured={}
    def encoder_hook(_module,_args,out):captured["items"]=out[episode.centers.to(out.device)].detach().cpu().clone()
    def predictor_hook(_module,args):
        captured["context"]=args[1].detach().cpu().clone()
        captured["query"]=args[2].detach().cpu().clone()
        captured["support_labels"]=args[3].detach().cpu().clone()
        captured["prototypes"]=args[4].detach().cpu().clone()
    handles=[model.encoder.register_forward_hook(encoder_hook),model.predictor.register_forward_pre_hook(predictor_hook)]
    try:
        with torch.random.fork_rng(devices=[torch.device(device).index] if str(device).startswith("cuda") else []),torch.no_grad():
            torch.manual_seed(seed)
            logits=model.episode_logits(clone_episode(episode,device)).detach().cpu()
    finally:
        for handle in handles:handle.remove()
    torch.testing.assert_close(captured["context"],captured["items"][episode.support_mask],atol=0,rtol=0)
    torch.testing.assert_close(captured["query"],captured["items"][episode.query_mask],atol=0,rtol=0)
    if not torch.equal(captured["support_labels"],episode.labels[episode.support_mask]):raise ValueError("Support labels differ")
    expected_prototypes=torch.stack([captured["context"][captured["support_labels"]==c].mean(0) for c in range(episode.n_way)])
    torch.testing.assert_close(captured["prototypes"],expected_prototypes,atol=1e-5,rtol=0)
    return logits,captured


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True,help="Canonical original TwiBot20 replay target directory with cache.json/batches")
    p.add_argument("--upstream",type=Path,default=Path("/dataMeR1/phil/gfm/upstream/inductnode"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=range(4),default=3)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    checkpoints={s:args.checkpoint_dir/f"state_dict_{s}.pt" for s in STEPS}
    plan=dict(conditions=[list(x) for x in CONDITIONS],source="covid_political",seed=0,target="twibot20",episodes=128,
        ways=2,shots=10,queries_per_class=12,cache=str(args.cache),
        checkpoint_sha256={str(s):file_sha256(path) for s,path in checkpoints.items()},
        cache_sha256=file_sha256(args.cache/"cache.json"),mode="eval",parity_atol=1e-5,
        prediction="Late predictor helps both encoders; late encoder hurts both predictors; E100I900 beats both native endpoints",
        scope="Source-supervised GILT native-source adaptation, not full published multitask recipe",
        fitting=False,readouts="Native; encoder support-centered ridge; raw text support-centered ridge; lambda1 scale1",
        upstream=str(args.upstream))
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import GILTAdapter,validate_upstream
    from scripts.experiments.setup.icl_arch_matrix.common_protocol import iter_episodes,new_fingerprint,update_episode_fingerprint
    from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash
    validate_upstream("gilt",args.upstream)
    if not (args.cache.parent/"DONE").is_file():raise ValueError("Source replay must be complete")
    source_protocol=json.loads((args.cache.parent/"protocol.json").read_text())
    if source_protocol["datasets"]!="twibot20" or source_protocol["eval_episode_seed_offset"]!=0:raise ValueError("Expected original TwiBot20 stream")
    cache=json.loads((args.cache/"cache.json").read_text())
    if cache["episodes"]!=128 or len(cache["batch_sha256"])!=32:raise ValueError("Expected canonical128 episodes")
    torch.set_num_threads(4);device=f"cuda:{args.gpu}"
    loaded={s:torch.load(path,map_location="cpu",weights_only=False) for s,path in checkpoints.items()}
    for s,checkpoint in loaded.items():
        expected=dict(architecture="gilt",model_id="ss_covid_political",sources="covid_political",seed=0,step=s,pretraining_protocol="gilt_native_source_classification_episodes")
        if any(checkpoint[k]!=v for k,v in expected.items()):raise ValueError("Nominated checkpoint metadata mismatch")
    states={s:c["model_state"] for s,c in loaded.items()}
    _,plan["ownership"]=assemble(states[100],states[900],100,100)
    plan.update(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),upstream_metadata=loaded[100]["upstream"])
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/"protocol.json",plan)
    rows,receipts=[],[]
    try:
        episodes=[];fingerprint=new_fingerprint();inputs=args.output/"inputs";inputs.mkdir()
        for i in range(32):
            path=args.cache/"batches"/f"batch_{i:03d}.pt"
            batch=torch.load(path,map_location="cpu",weights_only=False)
            if batch_hash(batch)!=cache["batch_sha256"][i]:raise ValueError("Cached batch changed")
            for episode in iter_episodes(batch,n_way=2,n_shot=10,n_query=12):
                if set(episode.global_centers[episode.support_mask].tolist()) & set(episode.global_centers[episode.query_mask].tolist()):raise ValueError("Support/query overlap")
                update_episode_fingerprint(fingerprint,episode)
                out=inputs/f"episode_{len(episodes):03d}.pt";torch.save(episode,out)
                receipts.append(dict(kind="input",file=str(out.relative_to(args.output)),sha256=file_sha256(out),source=str(path),source_sha256=file_sha256(path)))
                episodes.append(episode)
        if len(episodes)!=128 or fingerprint.hexdigest()!=cache["episode_fingerprint"]:raise ValueError("Episode fingerprint differs")
        encoders={}
        for e,i in CONDITIONS:
            model=GILTAdapter(args.upstream).to(device).eval()
            state,_=assemble(states[100],states[900],e,i);model.load_state_dict(state,strict=True)
            reference=None
            if e==i:
                reference=GILTAdapter(args.upstream).to(device).eval();reference.load_state_dict(states[e],strict=True)
            name=f"E{e}I{i}";directory=args.output/name;directory.mkdir()
            for ordinal,episode in enumerate(episodes):
                logits,captured=observed_forward(model,episode,device,910000+ordinal)
                if reference is not None:
                    with torch.no_grad(),torch.random.fork_rng(devices=[args.gpu]):
                        torch.manual_seed(910000+ordinal)
                        baseline=reference.episode_logits(clone_episode(episode,device)).cpu()
                    torch.testing.assert_close(logits,baseline,atol=1e-5,rtol=0)
                if (e,ordinal) in encoders:torch.testing.assert_close(captured["items"],encoders[(e,ordinal)],atol=1e-5,rtol=0)
                else:encoders[(e,ordinal)]=captured["items"]
                labels=torch.nn.functional.one_hot(episode.labels,num_classes=2).float();s,q=episode.support_mask,episode.query_mask
                predictions={"native":logits}
                for key,x in (("encoder",captured["items"]),("raw",episode.x[episode.centers])):
                    predictions[key]=predict(x[s],labels[s],x[q],"support")
                truth=episode.labels[q]
                metrics={key:score(truth,value,episode.label_map) for key,value in predictions.items()}
                row=dict(condition=name,ordinal=ordinal,metrics=metrics);rows.append(row)
                out=directory/f"episode_{ordinal:03d}.pt"
                torch.save(dict(logits=predictions,truth=truth,global_centers=episode.global_centers,query_mask=q,label_map=episode.label_map,pre_predictor=captured,metrics=metrics),out)
                receipts.append(dict(kind="output",file=str(out.relative_to(args.output)),sha256=file_sha256(out)))
                if (ordinal+1)%16==0:print(f"{name} {ordinal+1}/128",flush=True)
            for key,value in model.state_dict().items():
                if not torch.equal(value.cpu(),state[key]):raise ValueError("State changed: "+key)
            if any(m.training for m in model.modules()):raise ValueError("Mode changed")
            del model,reference
        means={f"E{e}I{i}":{arm:{metric:float(np.mean([r["metrics"][arm][metric] for r in rows if r["condition"]==f"E{e}I{i}"])) for metric in rows[0]["metrics"][arm]} for arm in rows[0]["metrics"]} for e,i in CONDITIONS}
        write_json(args.output/"summary.json",dict(rows=rows,means=means,receipts=receipts,episode_fingerprint=fingerprint.hexdigest()))
        write_json(args.output/"execution_status.json",dict(status="complete",episodes=128,conditions=4))
    except BaseException as error:
        write_json(args.output/"execution_status.json",dict(status="failed",error=repr(error)));raise


if __name__=="__main__":main()
