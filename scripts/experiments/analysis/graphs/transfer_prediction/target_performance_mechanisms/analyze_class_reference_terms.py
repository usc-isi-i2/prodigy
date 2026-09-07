"""Decompose already captured label vectors and margins; zero new model forwards."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from .class_reference_terms import decompose_class_reference
from scripts.experiments.setup.target_performance_mechanisms.verify_member_training import model_digest


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs_for_queries(audit, episode_ids):
    edges = audit['edge_index']; query_nodes = audit['query_mask'].nonzero().flatten()
    candidate = audit['edge_roles']['query_labels']; n = len(audit['query_mask'])
    pairs = []; query_episode = []
    unique_ids = list(dict.fromkeys(episode_ids.tolist()))
    for ep in unique_ids:
        members = query_nodes[episode_ids == ep]
        choices = [edges[0, candidate & (edges[1] == int(node))] - n for node in members]
        if any(len(pair) != 2 or not torch.equal(pair, choices[0]) for pair in choices):
            raise ValueError('Query candidate labels differ inside one episode.')
        pairs.append(choices[0]); query_episode.append(episode_ids == ep)
    return torch.stack(pairs), query_episode, unique_ids


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--discovery-root', type=Path, required=True)
    p.add_argument('--long-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('Hidden GPUs and a new output directory required.')
    torch.set_num_threads(2)
    rows = []; receipts = []; examples = []; verification = []
    with torch.no_grad():
        for phase, root in [('discovery', args.discovery_root), ('long', args.long_root)]:
            done = read(root / 'DONE.json')
            expected = 5 if phase == 'discovery' else 10
            if done['cells'] != expected or not done['all_joint_endpoints_bit_exact']:
                raise ValueError('Requires complete K/V source run.')
            for receipt in read(root / 'receipts.json'):
                stream = receipt['stream']
                state = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)['model']
                if model_digest(state) != receipt['weights_sha256']:
                    raise ValueError('Selected checkpoint state differs.')
                path = root / 'private_predictions' / (stream + '.pt')
                predictions = torch.load(path, map_location='cpu', weights_only=False)
                lab = predictions['labels']; counts = lab['batch_counts']; offset = 0
                for bi, count in enumerate(counts):
                    capture_path = root / 'private_activations' / stream / f'batch_{bi:03d}.pt'
                    audits = torch.load(capture_path, map_location='cpu', weights_only=False)
                    ep_ids = lab['episode_ids'][offset:offset+count]
                    if not torch.is_tensor(ep_ids): ep_ids = torch.as_tensor(ep_ids)
                    maps = lab['mapping'][offset:offset+count]
                    ylocal = lab['local_y'][offset:offset+count]
                    y = maps[torch.arange(count), ylocal] if lab['use_global'] else ylocal
                    sign = torch.where(maps[:, 1] == 1, 1., -1.).double() if lab['use_global'] else torch.ones(count).double()
                    for condition, audit in audits.items():
                        pair, memberships, eids = pairs_for_queries(audit, ep_ids)
                        terms = decompose_class_reference(audit, state, label_pairs=pair)
                        qhat = F.normalize(audit['final_inputs'][audit['query_mask']].double(), dim=1)
                        if len(qhat) != count: raise ValueError('Prediction/capture cardinality differs.')
                        tau = state['logit_scale'].double().exp()
                        actual_logits = predictions['logits'][condition][offset:offset+count].double()
                        native_margin = (actual_logits[:, 1]-actual_logits[:, 0])*sign
                        max_margin_error = 0.
                        for ei, (members, episode) in enumerate(zip(memberships, eids)):
                            yp = y[members]; native = native_margin[members]
                            dpair = pair[ei]
                            raw = terms['raw_labels_observed'][dpair].double()
                            raw_diff = raw[1]-raw[0]
                            total_d = terms['normalized_contrast_observed'][ei]
                            component_margins = {}
                            common = {'phase':phase, 'stream':stream, 'batch':bi, 'episode':int(episode),
                                      'condition':condition, 'query_count':int(members.sum()),
                                      'raw_total_difference_norm':float(raw_diff.norm()),
                                      'total_contrast_norm':float(total_d.norm()),
                                      'native_mean_margin':float(native.mean()),
                                      'native_positive_decision_fraction':float((native > 0).double().mean())}
                            for name, component in terms['raw_components'].items():
                                raw_part = component[dpair[1]]-component[dpair[0]]
                                d = terms['contrast_components'][name][ei]
                                margins = (tau*(qhat[members]@d))*sign[members]
                                component_margins[name] = margins
                                cosine = lambda a,b: float(F.cosine_similarity(a[None],b[None],dim=1))
                                rows.append({**common, 'component':name,
                                    'raw_component_difference_norm':float(raw_part.norm()),
                                    'raw_component_cos_total':cosine(raw_part,raw_diff),
                                    'normalized_component_norm':float(d.norm()),
                                    'normalized_component_cos_total':cosine(d,total_d),
                                    'mean_margin':float(margins.mean()),
                                    'rms_margin':float(margins.square().mean().sqrt()),
                                    'positive_minus_negative_mean_margin':float(margins[yp==1].mean()-margins[yp==0].mean())})
                            reconstructed = sum(component_margins.values())
                            error = float((reconstructed-native).abs().max()); max_margin_error = max(max_margin_error,error)
                            if error > 5e-5: raise ValueError(f'Native margin reconstruction mismatch: {error}')
                            if bi == 0 and ei == 0:
                                examples.append({**common, 'selection':'First saved episode, no outcome selection.',
                                    'true_labels':yp.tolist(), 'native_margins':native.tolist(),
                                    'component_margins':{k:v.tolist() for k,v in component_margins.items()},
                                    'max_margin_error':error})
                        verification.append({'phase':phase, 'stream':stream, 'batch':bi, 'condition':condition,
                            'max_margin_error':max_margin_error, **terms['discrepancies']})
                    receipts.append({'phase':phase, 'stream':stream, 'capture':str(capture_path), 'sha256':digest(capture_path)})
                    offset += count
                if offset != len(lab['local_y']): raise ValueError('Incomplete query coverage.')
                receipts.append({'phase':phase, 'stream':stream, 'predictions':str(path), 'sha256':digest(path),
                                 'weights_sha256':receipt['weights_sha256']})
    grouped = defaultdict(list)
    for row in rows: grouped[(row['phase'],row['stream'],row['condition'],row['component'])].append(row)
    numerical = [k for k,v in rows[0].items() if isinstance(v,float)]
    summary = [{**dict(zip(['phase','stream','condition','component'],key)), 'episodes':len(group),
                **{field:float(np.mean([r[field] for r in group])) for field in numerical}} for key,group in grouped.items()]
    if len(summary) != 75 or len(rows) != 9600: raise ValueError('Incomplete experiment panel.')
    args.output.mkdir(parents=True)
    protocol = {'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_roots':[str(args.discovery_root),str(args.long_root)], 'new_model_forwards':0,
                'interpretation':'Additive decomposition of observed vectors/scores with their native normalization denominators; not causal component removal, not new readouts.',
                'bn_eps':1e-5, 'native_margin_max_tolerance':5e-5,
                'first_case':'First saved episode per configuration/stream; no outcome or text selection.'}
    for name,value in [('protocol',protocol),('summary',summary),('episode_terms',rows),('verification',verification),('receipts',receipts),('numeric_examples',examples)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    finished={'summary_cells':len(summary),'episode_component_rows':len(rows),'batches_conditions':len(verification),
              'new_model_forwards':0,'max_margin_error':max(r['max_margin_error'] for r in verification)}
    (args.output/'DONE.json').write_text(json.dumps(finished,indent=2)+'\n')
    print(json.dumps(finished,indent=2))


if __name__ == '__main__':
    main()
