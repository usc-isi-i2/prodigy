"""HK-only frozen encoder/metagraph versus prototype diagnostic with reusable cache."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('WANDB_MODE','disabled')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/experiments/setup/final_core')]
from experiments.layers import get_module_list
from models.general_gnn import SingleLayerGeneralGNN
from scripts.experiments.setup.nm_support_resampling.run import encode, capture, meta, digest_state
from scripts.experiments.setup.nm_complete_input_audit.run import update_hash
from scripts.experiments.setup.nm_support_geometry.run import digest
from evaluate_fixed_grid import git_commit


def build_model(p,checkpoint,device):
    assert p['layers']=='S,U,M' and p['input_dim']==768 and p['ignore_label_embeddings']
    assert not p['skip_path'] and not p.get('use_edge_features',False)
    layers=get_module_list(p['layers'],p['emb_dim'],edge_attr_dim=None,input_dim=p['input_dim'],
        dropout=p['dropout'],reset_after_layer=p['reset_after_layer'],attention_mask_scheme=p['attention_mask_scheme'],
        has_final_back=p['has_final_back'],msg_pos_only=p.get('meta_gnn_pos_only',False),
        batch_norm_metagraph=not p['no_bn_metagraph'],batch_norm_encoder=not p['no_bn_encoder'],
        encoder_gnn_type=p['gnn_type'],gnn_use_relu=False)
    model=SingleLayerGeneralGNN(torch.nn.ModuleList(layers),initial_label_mlp=torch.nn.Linear(768,p['emb_dim']),
        params=p,text_dropout=torch.nn.Dropout(p['text_features_dropout'])).to(device)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state['model'],strict=True);model.eval()
    return model


def meta_args(batch,ep,pre,model):
    start=ep*210;end=start+210;n=batch[2].shape[0];sl=slice(start*30,end*30)
    edges=batch[3][:,sl].clone();edges[0]-=start;edges[1]-=n+ep*30-210
    labels=model.learned_label_embedding(torch.arange(ep*30,(ep+1)*30,device=pre.device))
    return pre,labels,edges.to(pre.device),batch[4][sl].to(pre.device),batch[5][sl].to(pre.device)


def heads(pre,local,model):
    support=pre.reshape(30,7,-1)[:,:3]
    query=F.normalize(pre[local],dim=0)
    prototype=F.normalize(support.mean(dim=1),dim=1)@query
    average=F.normalize(support,dim=2).mean(dim=1)@query
    scale=model.logit_scale.exp()
    return prototype*scale,average*scale


def metrics(z,truth,prefix):
    z=z.float();logp=z.log_softmax(0);other=z.clone();other[truth]=-torch.inf
    return {prefix+'_correct':int(z.argmax()==truth),prefix+'_prediction':int(z.argmax()),
        prefix+'_probability':float(logp[truth].exp()),prefix+'_nll':float(-logp[truth]),
        prefix+'_rank':int(1+(z>z[truth]).sum()),prefix+'_margin':float(z[truth]-other.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2'))
    p.add_argument('--intervention',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_support_resampling_20260908_v2'))
    p.add_argument('--audit',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908'))
    p.add_argument('--features',type=Path,default=Path('/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    started=time.time()
    receipt=json.loads((args.inputs/'receipt.json').read_text());assert receipt['complete']
    info=receipt['targets']['cp_hk'];offset=info['source_node_offset']
    protocol=json.loads((args.audit/'protocol_summary.json').read_text())
    checkpoint=protocol['targets']['cp_hk']['splits']['test']['models']['cp_hk']['checkpoint']
    params=json.loads((args.audit/'effective_config.json').read_text());params['device']=args.device
    model=build_model(params,checkpoint,args.device);before=digest_state(model)
    raw=torch.load(args.features,map_location='cpu',weights_only=False);features=raw['x'];del raw
    assert tuple(features.shape)==(333800,768)
    cases=json.loads((args.intervention/'cp_hk_cases_private.json').read_text())
    trials=pd.read_csv(args.intervention/'results_private.csv')
    trials=trials[(trials.target=='cp_hk')&(trials.model=='hk')].set_index(['case','condition','draw'])
    assert len(cases)==200 and len(trials)==2000 and trials.index.is_unique
    epochs={};h=hashlib.sha256();witness=[];allrows=[];logits=[]
    with torch.inference_mode():
        for bi in range(16):
            path=args.inputs/'cp_hk'/f'batch_{bi:03d}_private.pt'
            assert digest(path)==next(f['sha256'] for f in info['files'] if f['file']==path.name)
            batch=torch.load(path,map_location='cpu',weights_only=False)['batch'];g=batch[0]
            ids=g.global_node_ids;real=ids>=0
            assert ((ids[real]>=offset)&(ids[real]<offset+len(features))).all()
            local=(ids-offset).clamp_min(0)
            g.x=features[local].clone();g.x[~real]=0;update_hash(h,batch)
            selected=sorted({c['ep'] for c in cases if c['batch']==bi})
            if not selected:continue
            graphs=[g.get_example(ep*210+j) for ep in selected for j in range(210)]
            embeddings=[]
            for k in range(0,len(graphs),128):
                embeddings.append(encode(model,Batch.from_data_list(graphs[k:k+128]),args.device).cpu())
            embeddings=torch.cat(embeddings)
            for i,ep in enumerate(selected):
                pre=embeddings[i*210:(i+1)*210].to(args.device)
                a=meta_args(batch,ep,pre,model)
                epochs[bi,ep]=tuple(v.cpu() for v in a)
            # One independent full-model/full-batch witness checks manual encoding,
            # original within-batch label indices, and cached metagraph replay.
            if not witness:
                full,zfull=capture(model,batch,args.device)
                for ep in selected:
                    cached=epochs[bi,ep][0].to(args.device)
                    err=float((cached-full['pre'][ep*210:(ep+1)*210]).abs().max())
                    assert err<1e-5,err
                    replay=meta(model,*[v.to(args.device) for v in epochs[bi,ep]])
                    q=[j for j in range(210) if j%7>=3]
                    err2=float((replay[q]-zfull[ep*120:(ep+1)*120]).abs().max())
                    assert err2<1e-4,err2;witness.append([err,err2])
                del full,zfull
            del batch,g,graphs,embeddings;gc.collect()
            print('encoded original batch',bi,'selected episodes',len(selected),flush=True)
        assert h.hexdigest()==info['expected_hash'],(h.hexdigest(),info['expected_hash'])
        saved=torch.load(args.intervention/'cp_hk_support_draws_private.pt',map_location='cpu',weights_only=False)
        support_embeddings=[]
        for k in range(0,len(saved['graphs']),128):
            part=saved['graphs'][k:k+128]
            for g in part:
                ids=g.global_node_ids;real=ids>=0
                assert torch.equal(g.x[real],features[ids[real]-offset])
            support_embeddings.append(encode(model,Batch.from_data_list(part),args.device).cpu())
        support_embeddings=torch.cat(support_embeddings)
        assert len(support_embeddings)==6000
        max_prob_error=0.; baseline={}
        for c in cases:
            a=[v.to(args.device) for v in epochs[c['batch'],c['ep']]]
            z=meta(model,*a)[c['local']];pr,av=heads(a[0],c['local'],model)
            row=dict(case=c['case'],cohort=c['cohort'],condition='original',draw=-1)
            for name,values in [('native',z),('prototype',pr),('mean_cosine',av)]:row.update(metrics(values,c['truth'],name))
            assert row['native_correct']==c['hk_correct']
            error=abs(row['native_probability']-float(c['hk_true_probability']));assert error<1e-5,error
            max_prob_error=max(max_prob_error,error);allrows.append(row);baseline[c['case']]=row
            logits.append(torch.stack([z,pr,av]).cpu())
        for spec in saved['spec']:
            c=cases[spec['case']];a=[v.to(args.device) for v in epochs[c['batch'],c['ep']]]
            changed=a[0].clone();slots=[c['truth']*7+j for j in range(3)]
            changed[slots]=support_embeddings[spec['start']:spec['start']+3].to(args.device)
            keep=torch.ones(210,dtype=torch.bool,device=args.device);keep[slots]=False
            assert torch.equal(changed[keep],a[0][keep])
            z=meta(model,changed,*a[1:])[c['local']];pr,av=heads(changed,c['local'],model)
            row=dict(case=c['case'],cohort=c['cohort'],condition=spec['condition'],draw=spec['draw'])
            for name,values in [('native',z),('prototype',pr),('mean_cosine',av)]:row.update(metrics(values,c['truth'],name))
            old=trials.loc[c['case'],spec['condition'],spec['draw']]
            assert row['native_correct']==int(old.correct),(c['case'],spec,row,old)
            error=abs(row['native_probability']-float(old.true_probability));assert error<1e-5,error
            max_prob_error=max(max_prob_error,error)
            allrows.append(row);logits.append(torch.stack([z,pr,av]).cpu())
        assert digest_state(model)==before
    df=pd.DataFrame(allrows);assert len(df)==2200
    df.to_csv(args.out/'head_comparison_private.csv',index=False)
    torch.save(dict(episodes=epochs,support_embeddings=support_embeddings,spec=saved['spec'],cases=cases,
        logits=torch.stack(logits),logit_row_keys=df[['case','condition','draw']].to_dict('records'),
        head_order=['native','prototype','mean_cosine'],logit_scale=float(model.logit_scale.exp()),
        checkpoint=checkpoint,model_state_sha256=before),args.out/'embeddings_and_logits_private.pt')
    report=dict(complete=True,revision=git_commit(),seconds=time.time()-started,selected_cases=200,
        cached_episodes=len(epochs),cached_support_embeddings=6000,rows=len(df),
        standalone_features=str(args.features),full_input_hash=h.hexdigest(),expected_input_hash=info['expected_hash'],
        checkpoint=checkpoint,model_state_sha256=before,model_unchanged=True,
        replayed_original_decisions=200,replayed_intervention_decisions=2000,
        max_true_probability_error=max_prob_error,whole_batch_witness_errors=witness,
        prototype_temperature='native learned logit scale reused without calibration',
        feature_dimensions=768,embedding_dimensions=256,logit_scale=float(model.logit_scale.exp()),
        cache_sha256=digest(args.out/'embeddings_and_logits_private.pt'),csv_sha256=digest(args.out/'head_comparison_private.csv'))
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',report['seconds'],flush=True)


if __name__=='__main__':main()
