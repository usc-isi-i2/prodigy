"""Evaluate verified endpoint checkpoints on fixed-identity replacement caches."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .analyze_mixture_predictions import input_labels
from .replay import batch_hash, clone_batch, trace_stages, episode_probe, raw_stages
from .run_episode_cardinality import make_model
from .role_topology import assert_background_attributes_unused
from .verify_member_training import model_digest


def metrics(logits, labels):
    y, mapping = labels['local_y'], labels['mapping']
    p = logits.softmax(1)
    semantic = torch.zeros_like(p).scatter(1, mapping, p)
    truth = mapping[torch.arange(len(y)), y]
    pred = semantic.argmax(1)
    aucs = []
    for ep in labels['episode_ids'].unique(sorted=True):
        mask = labels['episode_ids'] == ep
        aucs.append(roc_auc_score(y[mask], p[mask,1]))
    return dict(roc_auc=float(roc_auc_score(truth,semantic[:,1])),
        mean_episode_auc=float(np.mean(aucs)), accuracy=float(accuracy_score(truth,pred)),
        f1=float(f1_score(truth,pred,zero_division=0)),
        nll=float(torch.nn.functional.cross_entropy(logits,y)))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    done=json.loads((args.cache_root/'DONE.json').read_text())
    if len(done['receipts'])!=2 or not all(r['exact_centers_features_labels_task_tensors'] for r in done['receipts']):
        raise ValueError('Verified complete matched caches required')
    args.output.mkdir(parents=True)
    rows,receipts=[],[]
    for stream in ('original','fresh'):
        ref=args.reference_root/stream/'twibot20'
        directory=args.cache_root/stream/'twibot20'
        labels=input_labels(directory,'twibot20')
        old_labels=input_labels(ref,'twibot20')
        for key in ('local_y','mapping','episode_ids'):
            if not torch.equal(labels[key],old_labels[key]): raise ValueError('Task identities changed')
        protocol=json.loads((args.reference_root/stream/'protocol.json').read_text())
        # Legacy parameter parsing always constructs a CUDA device string. This
        # model-only factory does not transfer tensors to that device: follow the
        # established cached-replay convention, hide GPUs and assert CPU tensors.
        protocol.update(device='0',catalog='docs/graph_catalog.json',
            config='scripts/experiments/setup/final_core/training.yaml')
        all_ref=[json.loads(x) for x in (ref/'metrics.jsonl').read_text().splitlines()]
        arms=[r for r in all_ref if r['decoder']=='full_model' and
              r['model_id'].rsplit('_step',1)[1] in ('100','2500')]
        if len(arms)!=18 or len({r['sources'][0] for r in arms})!=9:
            raise ValueError('All nine sources at both endpoints required')
        first=torch.load(ref/'batches/batch_000.pt',map_location='cpu',weights_only=False)
        for record in arms:
            mid=record['model_id']
            state=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)['model']
            if model_digest(state)!=record['weights_sha256']: raise ValueError('Checkpoint changed')
            model=make_model(protocol,'twibot20',labels['cache']['graph_path'],first[0].x.shape[1],state)
            if any(t.device.type != 'cpu' for t in list(model.parameters())+list(model.buffers())):
                raise ValueError('CPU-only model construction required')
            assert_background_attributes_unused(model)
            saved=torch.load(ref/(mid+'__baseline.pt'),map_location='cpu',weights_only=False)
            with torch.no_grad():
                if saved[0]['batch_sha256']!=batch_hash(first): raise ValueError('Reference input mismatch')
                _,baseline,_=model(*clone_batch(first))
                torch.testing.assert_close(baseline,saved[0]['logits']['full_model'],rtol=0,atol=0)
                exports=[]
                for bi, fingerprint in enumerate(labels['cache']['batch_sha256']):
                    batch=torch.load(directory/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                    if batch_hash(batch)!=fingerprint: raise ValueError('Matched cache changed')
                    raw=raw_stages(batch[0])
                    with trace_stages(model,batch[0]) as traces:
                        _,yp,_=model(*batch)
                    predictions={'full_model':yp}
                    for stage,x in (raw|traces).items():
                        for method in ('prototype','ridge'):
                            predictions[stage+'/'+method]=episode_probe(x,batch,method)
                    exports.append(dict(batch=bi,batch_sha256=fingerprint,logits=predictions))
            if model_digest(model.state_dict())!=record['weights_sha256']:
                raise ValueError('Weights or normalization buffers changed')
            out=args.output/stream/'twibot20'; out.mkdir(parents=True,exist_ok=True)
            torch.save(exports,out/(mid+'__baseline.pt'))
            for decoder in exports[0]['logits']:
                logits=torch.cat([r['logits'][decoder] for r in exports])
                row=dict(stream=stream,source=record['sources'][0],model_id=mid,
                    checkpoint=record['checkpoint'],weights_sha256=record['weights_sha256'],
                    decoder=decoder,step=int(mid.rsplit('_step',1)[1]),
                    episode_fingerprint=labels['cache']['episode_fingerprint'],**metrics(logits,labels))
                rows.append(row)
                with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            receipts.append(dict(stream=stream,model_id=mid,baseline_first_batch_bit_exact=True,
                                 weights_buffers_unchanged=True,all_input_hashes_verified=True))
            print(json.dumps({'stream':stream,'model_id':mid,'completed_models':len(receipts)}),flush=True)
    if len(rows)!=612 or len(receipts)!=36: raise ValueError('Incomplete evaluation')
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'DONE.json').write_text(json.dumps(dict(rows=len(rows),receipts=receipts,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        new_training=False),indent=2)+'\n')


if __name__=='__main__': main()
