#!/usr/bin/env python3
"""Bounded canonical NM support replacement and same-identity context control."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("WANDB_MODE","disabled")
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts/experiments/setup/final_core")]
from scripts.experiments.setup.nm_error_audit_split.run_audit import full_hash, membership
from evaluate_fixed_grid import load_dataset, TrainerFS, seed_everything, reset_fixed_eval_rng, make_frozen_loader, load_checkpoint_strict, atomic_json, git_commit
from data.dataset import SeededNodeIndex

def digest_state(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def encode(model, graphs, device):
    g=graphs.clone().to(device)
    g.x=model.initial_input_mlp(g.x)
    if model.txt_dropout is not None: g.x=model.txt_dropout(g.x)
    original=g.x.clone()
    g.x=model.layer_list[0].forward(original,g.x,g.edge_index.long(),g.edge_attr if "edge_attr" in g else None,
                                    g.edge_index_supernode,g.ptr[:-1],g.batch)
    return model.layer_list[1].forward(g.x,g.edge_index_supernode,g.supernode+g.ptr[:-1],g.batch)

def capture(model,batch,device):
    saved={}
    old=model.forward_metagraph
    def wrapped(module,x,labels,*args):
        saved["pre"]=x.detach().clone(); saved["labels"]=labels.detach().clone()
        return old(module,x,labels,*args)
    model.forward_metagraph=wrapped
    try:
        with torch.no_grad():
            _,logits,_=model(*[v.clone().to(device) for v in batch])
    finally: model.forward_metagraph=old
    return saved,logits

def meta(model,pre,labels,edges,attrs,mask):
    x,l=model.forward_metagraph(model.layer_list[2],pre,labels,edges,attrs,mask,None,None,None)
    return model.decode(model.final_input_mlp(x),model.final_label_mlp(l),edges).reshape(210,30)

def episode_inputs(batch,saved,ep):
    start=ep*210; end=start+210; n=batch[2].shape[0]
    sl=slice(start*30,end*30)
    edges=batch[3][:,sl].clone().to(saved["pre"].device)
    edges[0]-=start; edges[1]-=n+ep*30-210
    assert int(edges.min())==0 and int(edges.max())==239
    return (saved["pre"][start:end].clone(),saved["labels"][ep*30:(ep+1)*30].clone(),
            edges,batch[4][sl].to(edges.device),batch[5][sl].to(edges.device))

def choose_cases(df,dataset,offset,count,number,seed,native):
    rng=np.random.default_rng(seed)
    s=df[df.split=="test"].copy()
    s["amb"]=s.groupby(["episode","query"]).anchor.transform("nunique")>1
    freq=s.groupby("query")["query"].transform("size")
    s["freq"]=pd.cut(freq,[0,1,4,19,np.inf],labels=False)
    s["deg"]=pd.cut(s.query_degree,[-1,9,99,999,9999,99999,np.inf],labels=False)
    allqueries={e:set(z["query"]+offset) for e,z in s.groupby("episode")}
    rp,col,_=dataset.nm_test_neighbor_sampler.whole_adj.csr()
    rp,col=rp.numpy(),col.numpy()
    pools={}
    def pool(r):
        key=(r.episode,int(r.anchor))
        if key not in pools:
            anchor=int(r.anchor)+offset
            banned=allqueries[r.episode]|{int(r["true_support_"+str(j)])+offset for j in range(3)}|{anchor}
            a=np.unique(col[rp[anchor]:rp[anchor+1]])
            pools[key]=np.array([v for v in a if offset<=v<offset+count and v not in banned],dtype=np.int64)
        return pools[key]
    failed=s[(~s.amb)&(s[native+"_correct"]==0)]
    correct=s[(~s.amb)&(s[native+"_correct"]==1)]
    selected=[]; used=set(); rejected=0
    for idx in rng.permutation(failed.index):
        r=s.loc[idx]
        if int(r["query"]) in used: continue
        p=pool(r)
        if len(p)<3: rejected+=1; continue
        candidates=correct[(correct.deg==r.deg)&(correct.freq==r.freq)&(~correct["query"].isin(used|{int(r["query"])}))]
        match=None
        for ci in rng.permutation(candidates.index):
            c=s.loc[ci]
            if len(pool(c))>=3: match=c;break
        if match is None: rejected+=1;continue
        for row,cohort in [(r,"native_failed"),(match,"native_correct")]:
            case={k:(v.item() if isinstance(v,np.generic) else v) for k,v in row.to_dict().items()}
            case.update(cohort=cohort,case=len(selected),alternative_pool_size=len(pool(row)))
            case["trials"]=[]
            anchor=int(row.anchor)+offset
            for draw in range(5):
                ids=rng.choice(pool(row),3,replace=False).tolist()
                pairs=[(anchor,int(v)) for v in ids]
                assert membership(dataset.nm_test_neighbor_sampler,pairs).all()
                assert not membership(dataset.neighbor_sampler,pairs).any()
                assert not membership(dataset.nm_validation_neighbor_sampler,pairs).any()
                case["trials"].append({"draw":draw,"alternative":ids,
                    "same_context":[int(row["true_support_"+str(j)])+offset for j in range(3)],
                    "seeds":[int(rng.integers(1,2**31-1)) for _ in range(3)]})
            selected.append(case);used.add(int(row["query"]))
        if len(selected)==2*number:break
    assert len(selected)==2*number,(len(selected),number)
    return selected,{"candidate_failures":len(failed),"rejected_while_selecting":rejected,"single_anchor_only":True,"degree_and_frequency_exact_bin_matching":True,"unique_selected_queries":len(used)}

def summarize(rows):
    d=pd.DataFrame(rows); result=[]
    for (target,model,cohort,condition),s in d.groupby(["target","model","cohort","condition"]):
        by=s.groupby("case").agg(mean_correct=("correct","mean"),any_correct=("correct","max"),
             all_correct=("correct","min"),baseline=("baseline_correct","first"))
        result.append(dict(target=target,model=model,cohort=cohort,condition=condition,cases=len(by),
             trial_accuracy=float(s.correct.mean()),baseline_accuracy=float(by.baseline.mean()),
             any_draw_correct_fraction=float(by.any_correct.mean()),all_draw_correct_fraction=float(by.all_correct.mean()),
             baseline_failed_cases=int((by.baseline==0).sum()),
             any_rescue_fraction=float(by.loc[by.baseline==0,"any_correct"].mean()) if (by.baseline==0).any() else None,
             any_break_fraction=float((1-by.loc[by.baseline==1,"all_correct"]).mean()) if (by.baseline==1).any() else None,
             mean_true_rank=float(s.true_rank.mean()),mean_rank_change=float((s.true_rank-s.baseline_rank).mean())))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908"))
    p.add_argument("--bio-root",type=Path,default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cases",type=int,default=100)
    p.add_argument("--targets",nargs="+",default=["ukr_rus","cp_hk"],choices=["ukr_rus","cp_hk"])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=90831)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    protocol=json.loads((args.audit_root/"protocol_summary.json").read_text())
    params=json.loads((args.audit_root/"effective_config.json").read_text())
    params.update(device=args.device,log_dir=str(args.out/"logs"),state_dir=str(args.out/"state"),override_log=True)
    assert params["layers"]=="S,U,M" and params["ignore_label_embeddings"] and not params["skip_path"]
    if args.dry_run:
        print(json.dumps({"targets":args.targets,"cases_per_outcome":args.cases,"draws":5,"conditions":["alternative","same_context"],"source_revision":git_commit(),"device":args.device,"params":params}));return
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    seed_everything(params)
    print("Loading canonical training graph",flush=True)
    dataset=load_dataset(params);trainer=TrainerFS(dataset,params)
    assert dataset.nm_background_edge_view=="static_train" and dataset.nm_holdout_edge_view=="static_test"
    records=[];audits=[];started=time.time()
    for target in args.targets:
        info=protocol["targets"][target];offset=info["source_node_offset"];count=info["source_node_count"]
        native="ukr" if target=="ukr_rus" else "hk"
        data_name="ukr_rus_twitter" if native=="ukr" else "cp_hk_twitter"
        df=pd.read_csv(args.bio_root/data_name/"paired_cluster_queries_private.tsv",sep="\t")
        cases,selection=choose_cases(df,dataset,offset,count,args.cases,args.seed,native)
        atomic_json(args.out/(target+"_cases_private.json"),cases)
        print("Selected",target,len(cases),selection,flush=True)
        reset_fixed_eval_rng(target)
        trainer.parameter.update(neighbor_sampling_source_subset=target,eval_only_split="test")
        loader=trainer._build_dataloaders(dataset,trainer.dataset_name)[3]
        plans=torch.load(args.audit_root/target/"test_plan_private.pt",map_location="cpu")
        reset_fixed_eval_rng(target)
        cache=[]
        for i,b in enumerate(make_frozen_loader(dataset,loader.collate_fn,plans)):
            cache.append(b);print("Materialized",target,i,flush=True)
        digest=full_hash(cache)
        assert digest==info["splits"]["test"]["all_input_tensors_sha256"],(target,digest,info["splits"]["test"]["all_input_tensors_sha256"])
        print("Original input tensor hash matches",target,flush=True)
        # Draw support graphs once for paired model comparisons.
        graphs=[];spec=[];changes=[]
        for c in cases:
            bi,ep=map(int,c["episode"].split(":"));sample=int(c["sample"]);local=sample-ep*210
            assert 0<=local<210 and local%7>=3
            b=cache[bi];truth=int(b[2][sample].argmax())
            assert int(b[0].task_label_map[ep,truth])==int(c["anchor"])+offset
            assert int(b[0].global_node_ids[b[0].ptr[sample]])==int(c["query"])+offset
            slots=[ep*210+truth*7+j for j in range(3)]
            original=[b[0].get_example(slot) for slot in slots]
            assert [int(g.global_node_ids[0]) for g in original]==[int(c["true_support_"+str(j)])+offset for j in range(3)]
            c.update(batch=bi,ep=ep,local=local,truth=truth,slots=slots)
            for trial in c["trials"]:
                for condition in ["alternative","same_context"]:
                    start=len(graphs)
                    for j,center in enumerate(trial[condition]):
                        g=dataset[SeededNodeIndex(int(center),trial["seeds"][j])]
                        assert int(g.global_node_ids[0])==center
                        graphs.append(g)
                        if condition=="same_context":changes.append(full_hash([[g]])!=full_hash([[original[j]]]))
                    spec.append(dict(case=c["case"],condition=condition,draw=trial["draw"],start=start))
        torch.save({"graphs":graphs,"spec":spec},args.out/(target+"_support_draws_private.pt"))
        atomic_json(args.out/(target+"_cases_private.json"),cases)
        for model_name,modelkey in [("ukr","ukr_rus"),("hk","cp_hk")]:
            checkpoint=Path(info["splits"]["test"]["models"][modelkey]["checkpoint"])
            load_checkpoint_strict(trainer,checkpoint);model=trainer.model;model.eval()
            assert all(not m.training for m in model.modules())
            before=digest_state(model)
            episode_cache={};base_cases={}
            with torch.no_grad():
                for bi in sorted({c["batch"] for c in cases}):
                    b=cache[bi];saved,logits=capture(model,b,args.device)
                    qrows=torch.where(b[5].reshape(-1,30)[:,0].bool())[0].tolist()
                    positions={v:i for i,v in enumerate(qrows)}
                    if not episode_cache:
                        check=encode(model,b[0],args.device)
                        torch.testing.assert_close(check,saved["pre"],rtol=0,atol=1e-5)
                    for c in [x for x in cases if x["batch"]==bi]:
                        key=(bi,c["ep"])
                        if key not in episode_cache:episode_cache[key]=episode_inputs(b,saved,c["ep"])
                        z=meta(model,*episode_cache[key])[c["local"]]
                        ref=logits[positions[int(c["sample"])]]
                        torch.testing.assert_close(z,ref,rtol=0,atol=1e-4)
                        assert int(z.argmax()==c["truth"])==int(c[model_name+"_correct"]),("baseline decision mismatch",target,model_name,c["case"])
                        base_cases[c["case"]]=z.detach().cpu()
                    print("Baseline matched",target,model_name,bi,flush=True)
                encoded=[]
                for start in range(0,len(graphs),128):
                    encoded.append(encode(model,Batch.from_data_list(graphs[start:start+128]),args.device).detach())
                encoded=torch.cat(encoded)
                direct_checks=[]
                for ix,s in enumerate(spec):
                    c=cases[s["case"]];base=base_cases[c["case"]]
                    pre,labels,edges,attrs,mask=episode_cache[c["batch"],c["ep"]]
                    changed=pre.clone();slots=[c["truth"]*7+j for j in range(3)]
                    changed[slots]=encoded[s["start"]:s["start"]+3]
                    keep=torch.ones(210,dtype=torch.bool,device=pre.device);keep[slots]=False
                    assert torch.equal(changed[keep],pre[keep])
                    z=meta(model,changed,labels,edges,attrs,mask)[c["local"]]
                    if c["case"]==0 and s["draw"]==0:
                        # Independent whole-batch witness for both conditions.
                        original=cache[c["batch"]];ds=original[0].to_data_list()
                        for slot,g in zip(c["slots"],graphs[s["start"]:s["start"]+3]):ds[slot]=g.clone()
                        bg=Batch.from_data_list(ds)
                        for k,v in original[0]:
                            if k not in original[0]._slice_dict and k not in ("batch","ptr"):
                                bg[k]=v.clone() if isinstance(v,torch.Tensor) else v
                        altered=[bg,*original[1:]]
                        _,direct=capture(model,altered,args.device)
                        qrows=torch.where(original[5].reshape(-1,30)[:,0].bool())[0].tolist()
                        dz=direct[qrows.index(int(c["sample"]))]
                        torch.testing.assert_close(z,dz,rtol=0,atol=1e-4)
                        assert int(z.argmax())==int(dz.argmax())
                        direct_checks.append(float((z-dz).abs().max()))
                    records.append(dict(target=target,model=model_name,case=c["case"],cohort=c["cohort"],condition=s["condition"],draw=s["draw"],
                        baseline_correct=int(base.argmax()==c["truth"]),correct=int(z.argmax()==c["truth"]),
                        baseline_rank=int(1+(base>base[c["truth"]]).sum()),true_rank=int(1+(z>z[c["truth"]]).sum()),
                        baseline_probability=float(base.softmax(0)[c["truth"]]),true_probability=float(z.softmax(0)[c["truth"]])))
                assert digest_state(model)==before
                assert full_hash(cache)==digest
                audits.append(dict(target=target,model=model_name,model_state_sha256=before,checkpoint=str(checkpoint),
                    original_inputs_hash=digest,selected_baselines_matched=len(cases),fixed_label_vectors=True,
                    all_nonselected_premeta_rows_bit_exact=True,whole_forward_max_errors=direct_checks,
                    same_identity_graph_change_fraction=float(np.mean(changes)),selection=selection))
                atomic_json(args.out/"audits.json",audits)
                atomic_json(args.out/"summary.json",summarize(records))
                pd.DataFrame(records).to_csv(args.out/"results_private.csv",index=False)
                print("MODEL COMPLETE",target,model_name,flush=True)
            del episode_cache,encoded
        del cache,graphs,plans,loader
        gc.collect()
    atomic_json(args.out/"DONE.json",dict(complete=True,cases_per_cohort=args.cases,targets=args.targets,draws=5,rows=len(records),
        commit=git_commit(),seconds=time.time()-started,smoke=args.cases<100))
if __name__=="__main__": main()
