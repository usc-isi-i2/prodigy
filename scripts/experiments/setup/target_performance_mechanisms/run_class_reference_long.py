"""One nominated longer-trained checkpoint, retaining recorded model settings."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml
from experiments.layers import get_module_list
from experiments.run_shared_graph import write_json
from models.general_gnn import SingleLayerGeneralGNN
from .analyze_mixture_predictions import evaluate_logits
from .contrast_metrics import ranking_metrics
from .message_content import aggregation_audit
from .message_scale import radial_swap
from .replay import batch_hash,clone_batch,episode_probe,raw_stages
from .role_context import query_mask,intervene_roles
from .run_class_reference_contrast import TARGETS,final_forward,summarize_contrasts
from .verify_member_training import model_digest
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    classification_targets,build_classification_loader,reset_episode_rng,
    iter_episodes,new_fingerprint,update_episode_fingerprint)


def recorded_model(params,state=None):
    """No final-core config: construct directly from the recorded model recipe."""
    layers=get_module_list(params['layers'],params['emb_dim'],edge_attr_dim=None,input_dim=params['input_dim'],
        dropout=params['dropout'],reset_after_layer=params['reset_after_layer'],attention_mask_scheme=params['attention_mask_scheme'],
        has_final_back=params['has_final_back'],msg_pos_only=params['meta_gnn_pos_only'],
        batch_norm_metagraph=not params['no_bn_metagraph'],batch_norm_encoder=not params['no_bn_encoder'],encoder_gnn_type=params['gnn_type'])
    model=SingleLayerGeneralGNN(torch.nn.ModuleList(layers),initial_label_mlp=torch.nn.Linear(768,params['emb_dim']),
        params=dict(params),text_dropout=torch.nn.Dropout(params['text_features_dropout']))
    if state is not None:model.load_state_dict(state,strict=True)
    return model.eval()


def native_dataset(name,target,data_root):
    if name=='covid_political':
        from data.covid_political import get_covid_political_dataset as get_dataset,get_covid_political_dataloader as loader
    elif name=='facebook_page_reference':
        from data.facebook_page_reference import get_facebook_page_reference_dataset as get_dataset,get_facebook_page_reference_dataloader as loader
    else:
        from data.social_llm_dataset import get_twibot20_dataset as get_dataset,get_twibot20_dataloader as loader
    path=data_root/target['relative_path']
    dataset=get_dataset(root=str(path.parent),graph_filename=path.name,n_hop=1,task_name='classification',
        feature_subset='emb_only',original_features=True,edge_view='default',target_edge_view='default',seed=0,
        neighbor_sampling_hop_sizes='',neighbor_sampling_node_limit=2000)
    sampler=dataset.neighbor_sampler
    if sampler.num_hops!=1 or sampler.size!=100 or sampler.limit!=2000 or sampler.hop_sizes is not None:
        raise ValueError('historical one-hop sampler defaults not preserved')
    if dataset.graph.x.shape[1]!=768:raise ValueError('unexpected raw feature dimension')
    return dataset,loader,path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--expected-weights-sha256',required=True)
    p.add_argument('--data-root',type=Path,default=Path('/dataMeR1/phil/data'))
    p.add_argument('--catalog',default='docs/graph_catalog.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or args.episodes not in (2,128) or not 1<=args.threads<=4:
        raise ValueError('new output, hidden GPUs, 2 smoke or 128 full episodes required')
    torch.set_num_threads(args.threads)
    cfg_path=args.training_run/'files/effective_config.yaml';params=yaml.safe_load(cfg_path.read_text())['params']
    expected={'layers':'S,U,M','emb_dim':256,'input_dim':768,'n_hop':1,'n_shots':3,'n_way':3,'n_query':4,'batch_size':1,
        'ignore_label_embeddings':True,'gnn_type':'sage','no_bn_encoder':False,'no_bn_metagraph':False,'skip_path':False,
        'has_final_back':False,'reset_after_layer':None,'dropout':0,'text_features_dropout':0,'use_edge_features':False}
    if any(params.get(k)!=v for k,v in expected.items()):raise ValueError('nominated recorded training recipe differs')
    ckpt=args.training_run/'checkpoint/state_dict_50000.ckpt'
    state=torch.load(ckpt,map_location='cpu',weights_only=True)['model']
    best=torch.load(args.training_run/'state_dict',map_location='cpu',weights_only=False)
    if isinstance(best,dict) and 'model' in best:best=best['model']
    if model_digest(state)!=args.expected_weights_sha256 or model_digest(best)!=args.expected_weights_sha256:
        raise ValueError('nominated checkpoint or source-validation-selected best identity changed')
    del best
    protocol={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'checkpoint':str(ckpt),
        'weights_sha256':args.expected_weights_sha256,'source':'cp_hk','training_step':50000,'selection':'Previously nominated source-validation-selected best; no target-based checkpoint search.',
        'training_config':str(cfg_path),'training_config_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'recorded_training_params':params,'targets':list(TARGETS),'streams':['original','fresh'],'episodes_per_stream':args.episodes,
        'smoke_only':args.episodes!=128,'new_training':False,'device':'cpu','threads':args.threads,
        'eval_protocol':'Binary diagnostic episodes on the same target test splits; native one-hop fanout100/limit2000, three supports/class, batch1. Target-specific native query counts. This is not the recent 2-hop fanout9 10-shot protocol or the original all-class Facebook benchmark.',
        'label_interface':'Training ignore_label_embeddings=True; classification uses False, as in the original repository classification evaluation. Existing deterministic label vectors with original node features; no new text embedding model.',
        'conditions':['native_intact','native_removed','query_removed','support_direction_only','support_norm_only','raw_prototype','trained_prototype','untrained_prototype'],
        'contrast_conditions':['intact','removed','orientation_only','strength_only','unit_strength_intact','unit_strength_removed'],
        'prospective_predictions':'CP/FB support removal improves within-episode as well as pooled ranking, BOT removal hurts; final contrast orientation carries more of those changes than strength. FB query removal hurts while support removal helps. Assess every declared target and both streams; any failure narrows the claim, not a trigger to search checkpoints.',
        'baseline_interpretation':'Raw and encoder prototypes use fixed cosine readouts; untrained is one deterministic seed0 initialization, not the saved historical initialization. NLL temperatures differ, so these baselines primarily situate discrimination.',
        'generality_limit':'Longer-trained checkpoint with its original computational and neighborhood recipe. Neither an independent implementation nor public-benchmark replication, and not a training-duration-only causal comparison.'}
    if args.dry_run:print(json.dumps(protocol,indent=2));return
    args.output.mkdir(parents=True);write_json(args.output/'protocol.json',protocol)
    targets=classification_targets(args.catalog,include_facebook=True);rows=[];geometry=[];receipts=[];start=time.monotonic()
    eval_params={**params,'task_name':'classification','ignore_label_embeddings':False,'zero_shot':False,'n_way':2,'n_shots':3,'batch_size':1}
    torch.manual_seed(0);random_model=recorded_model(eval_params);random_digest=model_digest(random_model.state_dict())
    model=recorded_model(eval_params,state);aggr=aggregation_audit(model)
    if aggr[0]['aggregation_module']!='SumAggregation':raise ValueError('original aggregation behavior changed')
    with torch.no_grad():
        for target in TARGETS:
            dataset,loader,path=native_dataset(target,targets[target],args.data_root)
            for stream,offset in [('original',0),('fresh',100003)]:
                root=args.output/'private_inputs'/target/stream;root.mkdir(parents=True)
                batches=build_classification_loader(dataset_name=target,data_root=args.data_root,target=targets[target],
                    dataset=dataset,get_dataloader=loader,graph_path=path,workers=0,eval_episode_seed_offset=offset,
                    batch_count=args.episodes,batch_size=1,n_way=2,n_shot=3)
                reset_episode_rng();fingerprint=new_fingerprint();counts=[];ys=[];maps=[];ids=[];hashes=[]
                values={name:[] for name in protocol['conditions']};parts=[[],[]];errors=[];checks=0
                for bi,batch in enumerate(batches):
                    batch=clone_batch(batch);q=query_mask(batch);g=batch[0]
                    for episode in iter_episodes(batch,n_way=2,n_shot=3,n_query=targets[target]['n_query'],equal_query_counts=not targets[target]['eval_random_query']):
                        update_episode_fingerprint(fingerprint,episode)
                    if g.task_id_per_sample.unique().numel()!=1:raise ValueError('native batch1 violated')
                    task=g.task_id_per_sample[q];ys.append(batch[2][q].argmax(1));maps.append(g.task_label_map[task]);ids.append(torch.full((int(q.sum()),),bi));counts.append(int(q.sum()))
                    hashes.append(batch_hash(batch));torch.save(batch,root/f'batch_{bi:03d}.pt')
                    a,ta,pa,ea=final_forward(model,batch,alpha=1.);b,tb,pb,eb=final_forward(model,batch,alpha=0.)
                    values['native_intact'].append(a);values['native_removed'].append(b);parts[0].append(pa);parts[1].append(pb);errors.extend([ea,eb])
                    for stage in ('U1_pre_meta','M2_post_meta','final_input'):
                        torch.testing.assert_close(ta[stage][q],tb[stage][q],rtol=0,atol=0);checks+=1
                    query_logits,_,_,_=final_forward(model,intervene_roles(batch,'edges_query'))
                    values['query_removed'].append(query_logits)
                    for kind in ('direction','norm'):
                        direction,norm=(tb['U1_pre_meta'],ta['U1_pre_meta']) if kind=='direction' else (ta['U1_pre_meta'],tb['U1_pre_meta'])
                        replacement=ta['U1_pre_meta'].clone();replacement[~q],audit=radial_swap(direction[~q],norm[~q])
                        if audit['nonzero_norm_unassignable']:raise ValueError('undefined support radial swap')
                        logits,tr,_,_=final_forward(model,batch,replace_stage='pre_meta',replacement=replacement)
                        torch.testing.assert_close(tr['final_input'][q],ta['final_input'][q],rtol=0,atol=0);checks+=1
                        values['support_'+kind+'_only'].append(logits)
                    values['raw_prototype'].append(episode_probe(raw_stages(g)['raw_center'],batch,'prototype'))
                    values['trained_prototype'].append(episode_probe(ta['U1_pre_meta'],batch,'prototype'))
                    _,random_trace,_,_=final_forward(random_model,batch)
                    values['untrained_prototype'].append(episode_probe(random_trace['U1_pre_meta'],batch,'prototype'))
                    if batch_hash(batch)!=hashes[-1]:raise ValueError('input mutated')
                    if bi%32==31:print(json.dumps({'target':target,'stream':stream,'episodes_done':bi+1,'elapsed_seconds':time.monotonic()-start}),flush=True)
                if len(counts)!=args.episodes:raise ValueError('incomplete episode stream')
                labels={'local_y':torch.cat(ys),'mapping':torch.cat(maps),'episode_ids':torch.cat(ids),'use_global':target!='facebook_page_reference','batch_counts':counts}
                common={'target':target,'stream':stream,'source':'cp_hk','step':50000,'model_id':'source_selected_hk_50000','seed':0}
                saved={}
                for name,pieces in values.items():
                    logits=torch.cat(pieces);saved[name]=logits
                    metrics=evaluate_logits(logits,labels)
                    ranks,_=ranking_metrics((logits[:,1].double()-logits[:,0].double()).numpy(),labels['local_y'].numpy(),labels['mapping'].numpy(),labels['episode_ids'].numpy(),labels['use_global'])
                    rows.append({**common,'family':'readout','condition':name,**metrics,**ranks})
                combined=[{k:np.concatenate([v[k] for v in items]) for k in items[0]} for items in parts]
                contrast,geo,margins=summarize_contrasts(*combined,labels,float(model.logit_scale.exp()))
                rows.extend({**common,'family':'contrast','condition':c,**m} for c,m in contrast.items());geometry.extend({**common,**r} for r in geo)
                if model_digest(model.state_dict())!=args.expected_weights_sha256 or model_digest(random_model.state_dict())!=random_digest:raise ValueError('weights or buffers changed')
                receipts.append({**common,'episode_fingerprint':fingerprint.hexdigest(),'batch_sha256':hashes,'query_checks':checks,
                    'max_decoder_margin_error':max(errors),'weights_sha256':args.expected_weights_sha256,'untrained_weights_sha256':random_digest,'aggregation':aggr,
                    'sampler':{'hops':1,'fanout':100,'node_limit':2000},'episodes':len(counts)})
                dest=args.output/'predictions'/target;dest.mkdir(parents=True,exist_ok=True)
                torch.save({'logits':saved,'contrast_margins':margins,'labels':labels},dest/f'{stream}.pt')
                for name,objects in [('metrics',rows),('geometry',geometry),('receipts',receipts)]:write_json(args.output/f'{name}.json',objects)
                print(json.dumps({'target':target,'stream':stream,'complete':True,'cells':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
            del dataset,batches;gc.collect()
    if len(rows)!=84 or len(receipts)!=6:raise ValueError('incomplete declared test')
    write_json(args.output/'DONE.json',{'cells':len(rows),'target_streams':len(receipts),'smoke_only':args.episodes!=128,
        'query_checks':sum(r['query_checks'] for r in receipts),'all_weights_unchanged':True,'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
