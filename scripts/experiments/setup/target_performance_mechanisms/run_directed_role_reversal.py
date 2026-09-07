"""Fixed-checkpoint directed-context matching test; no training or selection."""
import argparse
import json
from pathlib import Path
import subprocess
import torch

from .directed_role_reversal import reverse_roles, CONDITIONS
from .analyze_mixture_predictions import input_labels
from .evaluate_matched_relations import metrics
from .replay import batch_hash, trace_stages, episode_probe
from .role_context import query_mask
from .role_topology import assert_background_attributes_unused
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    done=json.loads((args.reference_root/'DONE.json').read_text())
    if done['models']!=36 or not done['all_cached_inputs_match']:
        raise ValueError('Completed corrected replay required')
    args.output.mkdir(parents=True)
    protocol_record=dict(conditions=CONDITIONS,steps=[100,2500],streams=['original','fresh'],
        sources='all nine corrected single-source checkpoints',primary='mean within-episode AUC',
        contrast='(intact + both - support - query)/2',
        prediction='Early one-sided reversals hurt; both exceeds each one-sided arm; matching contrast weakens late.',
        recovery_gate='Early source-mean both AUC minus .5 >= half of intact AUC minus .5, in both streams.',
        preserved='nodes, features, labels, undirected edge slots, pooling edges, metagraph, checkpoint and frozen normalization',
        interpretation='Directed correspondence, not isolated degree mediation or semantic substitution',
        new_training=False,revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.output/'protocol.json').write_text(json.dumps(protocol_record,indent=2)+'\n')
    rows,receipts=[],[]
    for stream in ('original','fresh'):
        ref=args.reference_root/stream/'twibot20'
        labels=input_labels(ref,'twibot20')
        protocol=json.loads((args.reference_root/stream/'protocol.json').read_text())
        protocol.update(device='0',catalog='docs/graph_catalog.json',config='scripts/experiments/setup/final_core/training.yaml')
        meta=[json.loads(x) for x in (ref/'metrics.jsonl').read_text().splitlines()]
        arms=[r for r in meta if r['decoder']=='full_model' and r['model_id'].rsplit('_step',1)[1] in ('100','2500')]
        if len(arms)!=18 or len({r['sources'][0] for r in arms})!=9:
            raise ValueError('Nine sources and both endpoints required')
        first=torch.load(ref/'batches/batch_000.pt',map_location='cpu',weights_only=False)
        for record in arms:
            mid=record['model_id']
            state=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)['model']
            if model_digest(state)!=record['weights_sha256']: raise ValueError('Checkpoint changed')
            model=make_model(protocol,'twibot20',labels['cache']['graph_path'],first[0].x.shape[1],state)
            if any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
                raise ValueError('All model tensors must remain on CPU')
            if any(m.training for m in model.modules()): raise ValueError('Evaluation mode required')
            for m in model.modules():
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats:
                    raise ValueError('Frozen normalization required')
            assert_background_attributes_unused(model)
            saved=torch.load(ref/(mid+'__baseline.pt'),map_location='cpu',weights_only=False)
            if len(saved)!=32: raise ValueError('Incomplete reference predictions')
            dest=args.output/stream/mid; dest.mkdir(parents=True)
            all_logits={c:{} for c in CONDITIONS}
            with torch.no_grad():
                for bi,h in enumerate(labels['cache']['batch_sha256']):
                    batch=torch.load(ref/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                    if batch_hash(batch)!=h or saved[bi]['batch_sha256']!=h or saved[bi]['batch']!=bi:
                        raise ValueError('Input/prediction identity mismatch')
                    outcomes={}
                    for condition in CONDITIONS:
                        altered,audit=reverse_roles(batch,condition)
                        final_labels=[]
                        handle=model.final_label_mlp.register_forward_hook(lambda m,a,v:final_labels.append(v.detach().clone()))
                        try:
                            with trace_stages(model,altered[0]) as trace:
                                _,logits,_=model(*altered)
                        finally: handle.remove()
                        if len(final_labels)!=1: raise ValueError('One final label transformation required')
                        q=query_mask(batch)
                        pred={'full_model':logits}
                        for stage in ('S0_pool','U1_pre_meta'):
                            pred[stage+'/ridge']=episode_probe(trace[stage],batch,'ridge')
                        if condition=='intact':
                            torch.testing.assert_close(logits,saved[bi]['logits']['full_model'],rtol=0,atol=0)
                        for decoder,value in pred.items():
                            all_logits[condition].setdefault(decoder,[]).append(value)
                        outcomes[condition]=dict(logits=pred,final_queries=trace['final_input'][q],
                            final_labels=final_labels[0],pre_meta=trace['U1_pre_meta'],edge_audit=audit)
                    for a,b in (('intact','support'),('query','both')):
                        torch.testing.assert_close(outcomes[a]['final_queries'],outcomes[b]['final_queries'],rtol=0,atol=0)
                    if batch_hash(batch)!=h: raise ValueError('Original batch mutated')
                    torch.save(dict(batch=bi,batch_sha256=h,conditions=outcomes),dest/f'batch_{bi:03d}.pt')
            if model_digest(model.state_dict())!=record['weights_sha256']:
                raise ValueError('Weights/buffers mutated')
            for condition,preds in all_logits.items():
                for decoder,parts in preds.items():
                    row=dict(stream=stream,source=record['sources'][0],model_id=mid,
                        step=int(mid.rsplit('_step',1)[1]),condition=condition,decoder=decoder,
                        **metrics(torch.cat(parts),labels))
                    rows.append(row)
                    with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            receipts.append(dict(stream=stream,model_id=mid,weights_sha256=record['weights_sha256'],
                native_all_batches_bit_exact=True,query_role_checks_bit_exact=64,all_inputs_verified=True,weights_buffers_unchanged=True))
            print(json.dumps(dict(completed_models=len(receipts),stream=stream,model_id=mid)),flush=True)
    if len(rows)!=432 or len(receipts)!=36: raise ValueError('Incomplete grid')
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'DONE.json').write_text(json.dumps(dict(rows=len(rows),receipts=receipts,
        revision=protocol_record['revision'],new_training=False),indent=2)+'\n')


if __name__=='__main__': main()
