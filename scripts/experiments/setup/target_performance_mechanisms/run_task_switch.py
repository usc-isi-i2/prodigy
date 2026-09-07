"""Paired fixed-input task-switch discovery, two saved checkpoints, no training."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .task_switch_batch import build_batch, relabel_batch
from .replay import batch_hash, clone_batch, trace_stages, episode_probe
from .role_context import query_mask
from .probe_cached_inputs import standardized_probe
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest
from .evaluate_matched_relations import metrics
from .support_exceptions import exception_labels


def load_bank(root, entries):
    grouped = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(entry['batch'], []).append((index, entry))
    result = [None] * len(entries)
    for bi, requests in grouped.items():
        batch = torch.load(root / 'batches' / f'batch_{bi:03d}.pt',
                           map_location='cpu', weights_only=False)
        digest = batch_hash(batch)
        graphs = batch[0].to_data_list()
        for index, entry in requests:
            if digest != entry['batch_sha256']:
                raise ValueError('Original cache changed')
            graph = graphs[entry['subgraph']]
            if int(graph.global_node_ids[0]) != entry['node']:
                raise ValueError('Center identity mismatch')
            result[index] = graph
    return result


def episode_indices(entries, seed):
    cells = np.array([e['joint_cell'] for e in entries])
    if len({e['node'] for e in entries}) != len(entries):
        raise ValueError('Bank identities must be unique')
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(128):
        draws = [rng.choice(np.flatnonzero(cells == c), 11, replace=False) for c in range(4)]
        indices = np.concatenate([np.concatenate([d[:5] for d in draws]),
                                  np.concatenate([d[5:] for d in draws])])
        if len(set(indices)) != 44:
            raise ValueError('Episode repeats a center')
        episodes.append(indices)
    return np.stack(episodes)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--bank-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--support-exceptions', action='store_true')
    p.add_argument('--clean-root', type=Path)
    args = p.parse_args()
    if args.support_exceptions != (args.clean_root is not None):
        raise ValueError('Exception mode requires the completed clean reference')
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    summary = json.loads((args.bank_root / 'summary.json').read_text())
    if not all(r['feasible'] for r in summary['reports'].values()):
        raise ValueError('Bank feasibility failed')
    manifest = json.loads((args.bank_root / 'private_manifest.json').read_text())
    rule = np.load(args.bank_root / 'private_rule.npz')
    table = torch.from_numpy(rule['label_table'].copy())
    args.output.mkdir(parents=True)
    protocol = dict(source='cp_hk', steps=[100, 2500], episodes=128,
        input_seeds=dict(original=20260907, fresh=20260908),
        rules=['receive_edge', 'content_feature'], polarities=[0, 1],
        primary='mean episode AUC averaged over polarity, separately in each stream',
        gate=dict(structural_change_max=-.03, content_change_min=.03,
                  native_early_structural_min=.60, native_late_content_min=.60,
                  pool_late_structural_min=.70, pool_structural_change_min=-.02),
        new_training=False, revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if args.support_exceptions:
        clean_done=json.loads((args.clean_root/'DONE.json').read_text())
        if clean_done['rows']!=48 or len(clean_done['receipts'])!=4:
            raise ValueError('Completed clean discovery required')
        protocol.update(rules=['receive_edge','exceptions20','exceptions40'],
            gate=dict(primary='exceptions40',late_minus_early_auc_max=-.03,required_streams=2),
            exception_mask_seed='550000 + batch + 1000*(stream is fresh)',
            exception_design='Nested1/2 flips per five supports within each joint rule cell; query labels unchanged')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows, receipts = [], []
    for stream in ('original', 'fresh'):
        ref = args.reference_root / stream / 'twibot20'
        entries = manifest[stream]
        graphs = load_bank(ref, entries)
        for graph, entry in zip(graphs, entries):
            feature = graph.x[0].numpy().astype(np.float64)
            score = float((feature / np.linalg.norm(feature) - rule['mean']) @ rule['direction'])
            degree = int((graph.edge_index[1] == 0).sum())
            cell = 2 * int(degree > 0) + int(score > float(rule['threshold']))
            if degree != entry['degree'] or cell != entry['joint_cell'] or abs(score-entry['content_score'])>1e-10:
                raise ValueError('Frozen task rule does not match retained graph')
        selections = episode_indices(entries, protocol['input_seeds'][stream])
        if args.support_exceptions:
            if not np.array_equal(selections,np.load(args.clean_root/stream/'private_selections.npy')):
                raise ValueError('Clean episode identities changed')
        dest = args.output / stream
        dest.mkdir()
        np.save(dest / 'private_selections.npy', selections)
        source_protocol = json.loads((args.reference_root / stream / 'protocol.json').read_text())
        source_protocol.update(device='0', catalog='docs/graph_catalog.json',
                               config='scripts/experiments/setup/final_core/training.yaml')
        cache = json.loads((ref / 'cache.json').read_text())
        all_records = [json.loads(x) for x in (ref/'metrics.jsonl').read_text().splitlines()]
        records = [r for r in all_records if r['decoder']=='full_model' and r['sources']==['cp_hk']
                   and r['model_id'].rsplit('_step',1)[1] in ('100','2500')]
        if len(records)!=2 or {r['model_id'].rsplit('_step',1)[1] for r in records}!={'100','2500'}:
            raise ValueError('Exactly two Hong Kong checkpoints required')
        for record in records:
            mid = record['model_id']
            state = torch.load(record['checkpoint'],map_location='cpu',weights_only=True)['model']
            if model_digest(state)!=record['weights_sha256']:
                raise ValueError('Checkpoint changed')
            model = make_model(source_protocol,'twibot20',cache['graph_path'],768,state)
            if any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
                raise ValueError('CPU only')
            if any(m.training for m in model.modules()):
                raise ValueError('Frozen eval normalization required')
            if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats
                   for m in model.modules()):
                raise ValueError('Batch-dependent normalization is out of scope')
            old = torch.load(ref/'batches/batch_000.pt',map_location='cpu',weights_only=False)
            saved = torch.load(ref/(mid+'__baseline.pt'),map_location='cpu',weights_only=False)
            clean_saved = (torch.load(args.clean_root/stream/(mid+'.pt'),map_location='cpu',weights_only=False)
                           if args.support_exceptions else None)
            with torch.no_grad():
                _, pred, _ = model(*clone_batch(old))
            if saved[0]['batch_sha256']!=batch_hash(old):
                raise ValueError('Reference hash mismatch')
            torch.testing.assert_close(pred,saved[0]['logits']['full_model'],rtol=0,atol=0)
            parts, truths = {}, {}
            exports = []
            with torch.no_grad():
                for bi in range(32):
                    indices = selections[bi*4:bi*4+4].reshape(-1)
                    chosen = [graphs[i] for i in indices]
                    cells = torch.tensor([entries[i]['joint_cell'] for i in indices])
                    labels_by_rule = {'receive_edge':cells//2, 'content_feature':cells%2}
                    base = build_batch(chosen, labels_by_rule['receive_edge'], table)
                    original_hash = batch_hash(base)
                    q = query_mask(base)
                    quantities = {'receive_edge':torch.tensor([float(entries[i]['degree']>0) for i in indices])[:,None],
                                  'content_feature':torch.tensor([entries[i]['content_score'] for i in indices],dtype=torch.float32)[:,None]}
                    if args.support_exceptions:
                        labels_by_rule={'receive_edge':cells//2}
                        for count in (1,2):
                            name=f'exceptions{count*20}'
                            labels_by_rule[name],_=exception_labels(cells,count,550000+bi+1000*(stream=='fresh'))
                            quantities[name]=quantities['receive_edge']
                    fixed_queries = None
                    outcomes = {}
                    for name, labels in labels_by_rule.items():
                        for polarity in (0,1):
                            altered = relabel_batch(base, labels ^ polarity)
                            fingerprint = batch_hash(altered)
                            # A forward must not consume query scoring labels.
                            if bi == 0:
                                leak_labels = (labels ^ polarity).clone()
                                leak_labels[q] = 1-leak_labels[q]
                                leak = relabel_batch(base,leak_labels,require_balance=False)
                                _, leak_pred, _ = model(*leak)
                            with trace_stages(model,altered[0]) as trace:
                                _, pred, _ = model(*altered)
                            if bi == 0:
                                torch.testing.assert_close(pred,leak_pred,rtol=0,atol=0)
                            queries = trace['final_input'][q]
                            if fixed_queries is None:
                                fixed_queries = queries.clone()
                            else:
                                torch.testing.assert_close(queries,fixed_queries,rtol=0,atol=0)
                            preds = {'full_model':pred,
                                     'S0_pool/ridge':episode_probe(trace['S0_pool'],altered,'ridge'),
                                     'scalar_rule/ridge':standardized_probe(quantities[name],altered)}
                            if args.support_exceptions and name=='receive_edge':
                                old_outcome=clean_saved[bi]['outcomes'][f'receive_edge/{polarity}']
                                if old_outcome['batch_sha256']!=fingerprint:
                                    raise ValueError('Clean task input changed')
                                for decoder,value in preds.items():
                                    torch.testing.assert_close(value,old_outcome['logits'][decoder],rtol=0,atol=0)
                            key = (name,polarity)
                            truths.setdefault(key,[]).append((labels ^ polarity)[q])
                            for decoder, logits in preds.items():
                                parts.setdefault((name,polarity,decoder),[]).append(logits)
                            outcomes[f'{name}/{polarity}'] = dict(logits=preds,batch_sha256=fingerprint)
                    if batch_hash(base)!=original_hash:
                        raise ValueError('Paired base mutated')
                    exports.append(dict(batch=bi,base_sha256=original_hash,outcomes=outcomes))
            if model_digest(model.state_dict())!=record['weights_sha256']:
                raise ValueError('Weights/buffers changed')
            torch.save(exports,dest/(mid+'.pt'))
            for (name,polarity,decoder), logits in parts.items():
                y = torch.cat(truths[(name,polarity)])
                labels = dict(local_y=y,mapping=torch.tensor([0,1]).repeat(len(y),1),
                              episode_ids=torch.arange(128).repeat_interleave(24))
                rows.append(dict(stream=stream,source='cp_hk',step=int(mid.rsplit('_step',1)[1]),
                                 rule=name,polarity=polarity,decoder=decoder,**metrics(torch.cat(logits),labels)))
            receipts.append(dict(stream=stream,model_id=mid,weights_sha256=record['weights_sha256'],
                                 native_reference_exact=True,query_label_leak_checks=2*len(protocol['rules']),
                                 fixed_query_comparisons=32*(2*len(protocol['rules'])-1),weights_unchanged=True,
                                 clean_all_batches_exact=args.support_exceptions))
            print(json.dumps(receipts[-1]),flush=True)
    if len(rows)!=4*len(protocol['rules'])*2*3 or len(receipts)!=4:
        raise ValueError('Incomplete discovery grid')
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'DONE.json').write_text(json.dumps(dict(rows=len(rows),receipts=receipts),indent=2)+'\n')


if __name__=='__main__':
    main()
