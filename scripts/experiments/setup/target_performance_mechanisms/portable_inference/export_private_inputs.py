"""One-time trusted-checkout adapter; private tensors stay separate from code.

This adapter reads existing trusted PyG object caches. The exported runtime input
uses only tensors and primitives, and can subsequently be loaded weights-only.
No training, sampling, parameter update, original-cache write or upload occurs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from ..replay import batch_hash as original_batch_hash, clone_batch as original_clone
from ..role_topology import encode as original_encode
from ..run_episode_cardinality import make_model as original_model
from ..analyze_mixture_predictions import input_labels
from ..verify_member_training import model_digest as original_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--topology', type=Path, required=True)
    p.add_argument('--messages', type=Path, required=True)
    p.add_argument('--dose', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 8:
        raise ValueError('New private output and bounded CPU threads required')
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.package.resolve()))
    from graph_role.model import make_model, PARAM_KEYS
    from graph_role.io import pack_batch, unpack_batch, sha, write_json
    from graph_role.predict import expected_conditions
    from graph_role.role_topology import encode
    from graph_role.run import verify_code
    code_sha = verify_code()
    read = lambda folder, n: json.loads((folder/(n+'.json')).read_text())
    for folder in (args.topology, args.messages, args.dose):
        done = read(folder, 'DONE')
        if done['smoke_only']:
            raise ValueError('Completed non-smoke references required')
    protocol = read(args.topology, 'protocol')
    topology_rows, message_rows, dose_rows = [read(f, 'metrics') for f in (args.topology, args.messages, args.dose)]
    receipts = read(args.topology, 'receipts')
    if len(topology_rows) != 1140 or len(message_rows) != 960 or len(dose_rows) != 1140 or len(receipts) != 60:
        raise ValueError('Complete previously validated panels required')
    arms = [a for a in json.loads(Path(protocol['arms']).read_text()) if a['policy'] == 'lowest_sorted']
    if len(arms) != 6 or {(a['source'],a['seed']) for a in arms} != {(s,n) for s in ('cp_hk','ukr_rus') for n in range(3)}:
        raise ValueError('All six final source/seed controls required')
    args.output.mkdir(parents=True)
    (args.output/'weights').mkdir()
    models, datasets, audits = [], [], []
    started = time.monotonic()
    states = {}
    for arm in arms:
        state = torch.load(arm['checkpoint'], map_location='cpu', weights_only=True)['model']
        if original_digest(state) != arm['final_sha256']:
            raise ValueError('Reference checkpoint changed')
        states[arm['model_id']] = state
        dest = args.output/'weights'/(arm['model_id']+'.pt')
        torch.save(state, dest)
        models.append({'id':arm['model_id'], 'source':arm['source'], 'seed':arm['seed'], 'step':2500,
                       'path':str(dest.relative_to(args.output)), 'sha256':sha(dest), 'tensor_sha256':arm['final_sha256']})
    for target in protocol['targets']:
        for stream in ('original','fresh'):
            cache_root = Path(protocol['cache_roots'][f'{target}/{stream}'])
            cached_protocol = json.loads((cache_root/'protocol.json').read_text())
            source = cache_root/target
            labels = input_labels(source, target)
            dataset = {'id':f'{target}/{stream}', 'target':target, 'stream':stream,
                       'use_global':labels['use_global'], 'batches':[], 'references':{}}
            out = args.output/'batches'/target/stream
            out.mkdir(parents=True)
            for bi in range(32):
                original = torch.load(source/'batches'/f'batch_{bi:03d}.pt', map_location='cpu', weights_only=False)
                before = original_batch_hash(original)
                if before != labels['cache']['batch_sha256'][bi]:
                    raise ValueError('Historical batch fingerprint mismatch')
                packed = pack_batch(original)
                restored = unpack_batch(packed)
                dest = out/f'batch_{bi:03d}.pt'
                torch.save(packed, dest)
                dataset['batches'].append({'path':str(dest.relative_to(args.output)), 'sha256':sha(dest)})
                if bi == 0:
                    for model_record, arm in zip(models, arms):
                        mid = arm['model_id']
                        old = original_model(cached_protocol, target, labels['cache']['graph_path'], original[0].x.shape[1], states[mid])
                        config = {'params':{k:old.params[k] for k in PARAM_KEYS},
                                  'feature_dim':original[0].x.shape[1], 'label_dim':original[1].shape[1]}
                        if 'config' in model_record and model_record['config'] != config:
                            raise ValueError('Model configuration varies across target inputs')
                        model_record['config'] = config
                        new = make_model(config, states[mid])
                        with torch.no_grad():
                            old_pre, old_logits = original_encode(old, original)
                            new_pre, new_logits = encode(new, restored)
                        torch.testing.assert_close(old_pre, new_pre, rtol=0, atol=0)
                        torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
                        audits.append({'target':target,'stream':stream,'model_id':mid,'constructor_and_tensor_adapter_bit_exact':True})
                if original_batch_hash(original) != before:
                    raise ValueError('Original cached input mutated')
            reference_out = args.output/'references'/target/stream
            reference_out.mkdir(parents=True)
            for arm in arms:
                mid=arm['model_id']
                tr = torch.load(args.topology/'predictions'/target/stream/(mid+'.pt'), map_location='cpu', weights_only=False)
                mr = torch.load(args.messages/'predictions'/target/stream/(mid+'.pt'), map_location='cpu', weights_only=False)
                dr = torch.load(args.dose/'predictions'/target/stream/(mid+'__step2500.pt'), map_location='cpu', weights_only=False)
                logits = {f'topology/{q}/{s}/{d}':v for (q,s,d),v in tr['logits'].items()}
                logits.update({f'message/{c}/{r}':v for (c,r),v in mr['logits'].items() if c != 'intact'})
                logits.update({f'dose/{d}/{draw}':v for (d,draw),v in dr['logits'].items() if d in (25,50,75)})
                metrics = {}
                for panel,rows in [('topology',topology_rows),('message',message_rows),('dose',dose_rows)]:
                    for r in rows:
                        if (r['target'],r['stream'],r['model_id']) != (target,stream,mid): continue
                        if panel == 'topology': key=f"topology/{r['query_condition']}/{r['support_condition']}/{r['draw']}"
                        elif panel == 'message':
                            if r['condition']=='intact': continue
                            key=f"message/{r['condition']}/{r['role']}"
                        else:
                            if r['step']!=2500 or r['suppression_percent'] not in (25,50,75): continue
                            key=f"dose/{r['suppression_percent']}/{r['draw']}"
                        if key in metrics: raise ValueError('Duplicate reference')
                        metrics[key]={k:r[k] for k in ('roc_auc','accuracy','f1','nll')}
                if set(logits)!=expected_conditions()-{'prototype/raw','prototype/encoder'} or set(metrics)!=set(logits):
                    raise ValueError('Missing reference condition')
                for r in (tr,mr,dr):
                    for k in ('local_y','mapping'):
                        torch.testing.assert_close(r['labels'][k],labels[k],rtol=0,atol=0)
                dest=reference_out/(mid+'.pt')
                torch.save({'logits':logits,'metrics':metrics},dest)
                dataset['references'][mid]={'path':str(dest.relative_to(args.output)),'sha256':sha(dest)}
            datasets.append(dataset)
            print(json.dumps({'exported_dataset':dataset['id'],'batches':32,'models':6,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    write_json(args.output/'inputs.json',{'format':1,'scope':'private_final_checkpoint_replay','models':models,'datasets':datasets})
    write_json(args.output/'export_receipt.json',{'code_manifest_sha256':code_sha,'datasets':len(datasets),'models':len(models),
        'batches':sum(len(d['batches']) for d in datasets),'account_ids_removed':True,'features_remain_private':True,
        'constructor_checks':audits,'source_replay_revisions':{str(f.name):read(f,'protocol')['revision'] for f in (args.topology,args.messages,args.dose)},
        'elapsed_seconds':time.monotonic()-started,'public_upload':False})


if __name__=='__main__': main()
