"""Frozen scale-one deployment on existing five-target paired episode caches."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score
from .verify import verify
from experiments.run_shared_graph import write_json
from scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions import input_labels,evaluate_logits
from scripts.experiments.setup.target_performance_mechanisms.replay import clone_batch,trace_stages,raw_stages,batch_hash,episode_probe

TARGETS=('election2020','ukr_rus_suspended','twibot20','covid_political','facebook_page_reference')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def centered(x,batch):
    labels=batch[2];q=batch[5].reshape(len(labels),-1)[:,0].bool();tasks=batch[0].task_id_per_sample
    result=x.new_zeros(labels.shape)
    for task in tasks.unique(sorted=True):
        s=(tasks==task)&~q;t=(tasks==task)&q
        mean=x[s].mean(0,keepdim=True)
        xs,xq=F.normalize(x[s]-mean,dim=1),F.normalize(x[t]-mean,dim=1)
        result[t]=xq@xs.T@torch.linalg.solve(xs@xs.T+torch.eye(int(s.sum()),device=x.device,dtype=x.dtype),labels[s].to(x.dtype))
    return result[q]


def metrics(logits,labels):
    result=evaluate_logits(logits,labels)
    truth,pred=labels['local_y'],logits.argmax(1)
    if labels['use_global']:
        truth=labels['mapping'][torch.arange(len(truth)),truth]
        pred=torch.zeros_like(logits).scatter_add_(1,labels['mapping'],logits.softmax(1)).argmax(1)
    result.update(accuracy=float(accuracy_score(truth,pred)),f1=float(f1_score(truth,pred,zero_division=0)),
                  macro_f1=float(f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0)))
    return result


def build_model(params,state,device):
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN
    if params['layers']!='S,U,M' or params['input_dim']!=768:raise ValueError('Unexpected architecture')
    params=dict(params,encoder_solver_objective='native')
    layers=get_module_list(params['layers'],params['emb_dim'],edge_attr_dim=None,input_dim=768,
        dropout=params['dropout'],reset_after_layer=params['reset_after_layer'],attention_mask_scheme=params['attention_mask_scheme'],
        has_final_back=params['has_final_back'],msg_pos_only=params.get('meta_gnn_pos_only',False),
        batch_norm_metagraph=not params['no_bn_metagraph'],batch_norm_encoder=not params['no_bn_encoder'],
        encoder_gnn_type=params['gnn_type'],gnn_use_relu=False)
    model=SingleLayerGeneralGNN(torch.nn.ModuleList(layers),initial_label_mlp=torch.nn.Linear(768,params['emb_dim']),
        params=params,text_dropout=torch.nn.Dropout(params['text_features_dropout']))
    model.load_state_dict(state,strict=True);model.encoder_solver_training=False
    return model.to(device).eval()


def forward(model,batch,device,parity=False):
    current=clone_batch(batch,device)
    raw=raw_stages(current[0])['raw_center']
    with torch.no_grad():
        if parity:plain=model(*clone_batch(batch,device))[1]
        with trace_stages(model,current[0]) as stages:yt,z,_=model(*current)
        if parity:torch.testing.assert_close(z,plain,atol=1e-5,rtol=0)
        u1=stages['U1_pre_meta']
        return {k:v.cpu() for k,v in {'native_diagnostic':z,'u1_centered':centered(u1,current),'raw_centered':centered(raw,current),
            'u1_uncentered':episode_probe(u1,current,'ridge')}.items()},u1.cpu(),raw.cpu()


def main():
    p=argparse.ArgumentParser(__doc__)
    for field in ('root','reference','replay','output'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=range(4),default=3);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    print(json.dumps(dict(targets=TARGETS,streams=['original','fresh'],schedules=['blocked','interleaved'],deployment_scale=1,fit='support-only centered ridge'),indent=2))
    if not args.execute:return
    done=json.loads((args.root/'DONE.json').read_text())
    if done['steps']!=2500 or done['smoke']:raise ValueError('Completed full training required')
    verification=verify(args.root,args.reference)
    from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import resolved_params
    from scripts.experiments.setup.icl_arch_matrix.common_protocol import classification_targets
    torch.set_num_threads(4);device=f'cuda:{args.gpu}'
    plan=json.loads((args.root/'plan.json').read_text())
    checkpoints={e['arm']['schedule']:args.root/'state'/(e['name']+'_isolation_v1')/'checkpoint/state_dict_2500.ckpt' for e in plan}
    old=args.reference/'state/r4_blocked_s0_native_isolation_v1/checkpoint/state_dict_2500.ckpt'
    states={k:torch.load(path,map_location='cpu',weights_only=True)['model'] for k,path in checkpoints.items()}
    oldstate=torch.load(old,map_location='cpu',weights_only=True)['model']
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(verification=verification,training_root=str(args.root),reference=str(args.reference),
        replay_root=str(args.replay),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        stream_offsets=dict(original=0,fresh=100003),ridge_lambda=1,support_centering=True,
        checkpoint_sha256={k:sha(v) for k,v in checkpoints.items()},old_checkpoint_sha256=sha(old),deployment_scale=1,
        full_model='Diagnostic only; native solver not trained by this objective',targets=TARGETS))
    rows,receipts=[],[]
    try:
        specs=classification_targets('docs/graph_catalog.json',include_facebook=True)
        for stream in ('original','fresh'):
            for target in TARGETS:
                directory=args.replay/stream/target/target
                if not (directory.parent/'DONE').is_file():raise ValueError('Incomplete original evaluation')
                labels=input_labels(directory,target)
                cache=labels['cache'];graph=Path(cache['graph_path'])
                options=SimpleNamespace(config='scripts/experiments/setup/final_core/training.yaml',eval_state_root=str(args.output/'state'),
                    state_root=str(args.output/'state'),training_seed=0,eval_episode_seed_offset=0 if stream=='original' else 100003,
                    device=str(args.gpu),run_stamp='centered_eval',log_root=str(args.output/'log'))
                params=resolved_params(options,target,specs[target],graph,None,'centered_eval',2500)
                params.update(input_dim=768,encoder_solver_objective='native')
                provenance=args.output/stream/target
                provenance.mkdir(parents=True,exist_ok=True)
                write_json(provenance/'resolved_params.json',params)
                write_json(provenance/'cache_metadata.json',cache)
                batches=[torch.load(directory/'batches'/f'batch_{i:03d}.pt',map_location='cpu',weights_only=False) for i in range(32)]
                reference_path=directory/'r4_blocked_s0_native__baseline.pt'
                records=torch.load(reference_path,map_location='cpu',weights_only=False)
                control=build_model(params,oldstate,device)
                predicted,u1,raw=forward(control,batches[0],device,True)
                ref=records[0]
                if ref['batch_sha256']!=cache['batch_sha256'][0]:raise ValueError('Old reference cache differs')
                torch.testing.assert_close(predicted['native_diagnostic'],ref['logits']['full_model'],atol=1e-5,rtol=0)
                torch.testing.assert_close(predicted['u1_uncentered'],ref['logits']['U1_pre_meta/ridge'],atol=1e-5,rtol=0)
                torch.testing.assert_close(u1,ref['embeddings']['U1_pre_meta'],atol=1e-5,rtol=0)
                torch.testing.assert_close(predicted['u1_centered'],centered(ref['embeddings']['U1_pre_meta'],batches[0]),atol=1e-5,rtol=0)
                del control
                for schedule,state in states.items():
                    model=build_model(params,state,device);collected={};destination=args.output/stream/target/schedule;destination.mkdir(parents=True)
                    for i,batch in enumerate(batches):
                        if batch_hash(batch)!=cache['batch_sha256'][i]:raise ValueError('Cached input changed')
                        outputs,u1,raw=forward(model,batch,device,i==0)
                        for k,v in outputs.items():collected.setdefault(k,[]).append(v)
                        out=destination/f'batch_{i:03d}.pt';torch.save(dict(logits=outputs,u1=u1,raw_center=raw,source_batch_hash=cache['batch_sha256'][i]),out)
                        receipts.append(dict(file=str(out.relative_to(args.output)),sha256=sha(out),source=str(directory/'batches'/f'batch_{i:03d}.pt'),source_sha256=sha(directory/'batches'/f'batch_{i:03d}.pt'),episode_fingerprint=cache['episode_fingerprint']))
                    for key,value in model.state_dict().items():
                        if not torch.equal(value.cpu(),state[key]):raise ValueError('Evaluation mutated model state')
                    for decoder,parts in collected.items():rows.append(dict(stream=stream,target=target,schedule=schedule,decoder=decoder,**metrics(torch.cat(parts),labels)))
                    del model
                print(stream,target,'complete',flush=True)
        means=[]
        for stream in ('original','fresh'):
            for schedule in ('blocked','interleaved'):
                for panel in ('all5','fixed4'):
                    for decoder in ('native_diagnostic','u1_centered','raw_centered','u1_uncentered'):
                        selected=[r for r in rows if r['stream']==stream and r['schedule']==schedule and r['decoder']==decoder and (panel=='all5' or r['target']!='ukr_rus_suspended')]
                        if len(selected)!=(5 if panel=='all5' else 4):raise ValueError('Incomplete panel')
                        means.append(dict(stream=stream,schedule=schedule,panel=panel,decoder=decoder,**{m:float(np.mean([r[m] for r in selected])) for m in ('accuracy','macro_f1','f1','roc_auc','nll')}))
        write_json(args.output/'summary.json',dict(rows=rows,means=means,receipts=receipts))
        write_json(args.output/'execution_status.json',dict(status='complete',targets=5,streams=2,schedules=2))
    except BaseException as error:
        write_json(args.output/'execution_status.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
