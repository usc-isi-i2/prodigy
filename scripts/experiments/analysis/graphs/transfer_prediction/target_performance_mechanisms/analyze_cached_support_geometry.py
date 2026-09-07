"""Read existing nine-source embeddings; no model forward or new intervention.

Exploratory descriptors chosen from the preceding Hong Kong/Ukraine geometry
contrast. All outcomes have already been seen: excluding Hong Kong below is
NOT prospective validation. Query labels do not enter any descriptor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash
from scripts.experiments.setup.target_performance_mechanisms.role_context import query_mask


def describe(values, labels):
    norms = values.double().norm(dim=1)
    if not torch.isfinite(values).all() or (norms == 0).any():
        raise ValueError('finite nonzero support vectors required')
    unit = F.normalize(values.double(), dim=1)
    classes = labels.unique(sorted=True)
    if classes.tolist() != [0, 1]:
        raise ValueError('two support classes required')
    means = torch.stack([unit[labels == c].mean(0) for c in classes])
    dispersion = ((unit - means[labels]) ** 2).sum(1).mean()
    return {'support_norm_mean':float(norms.mean()),
            'support_norm_cv':float(norms.std(unbiased=False)/norms.mean()),
            'support_unit_dispersion':float(dispersion),
            'support_class_cosine':float(F.cosine_similarity(means[:1],means[1:])[0])}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('new output and hidden GPUs required')
    torch.set_num_threads(1)
    inputs=json.loads((args.inventory/'input_inventory.json').read_text())
    models=json.loads((args.inventory/'inventory.json').read_text())
    done=json.loads((args.inventory/'DONE.json').read_text())
    if done['cells']!=720 or not done['all_weights_unchanged'] or not done['all_baseline_metrics_reproduced']:
        raise ValueError('completed historical nine-source grid required')
    rows=[];receipts=[]
    with torch.no_grad():
        for entry in inputs:
            root=Path(entry['root']); target=entry['target']; stream=entry['stream']
            meta=[]
            for bi in range(32):
                batch=torch.load(root/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                if batch_hash(batch)!=entry['batch_sha256'][bi]:raise ValueError('input hash differs')
                q=query_mask(batch)
                meta.append((~q,batch[0].task_id_per_sample,batch[2].argmax(1)))
            for model in [m for m in models if m['target']==target and m['stream']==stream]:
                path=root/f"{model['model_id']}__baseline.pt"
                records=torch.load(path,map_location='cpu',weights_only=False)
                if len(records)!=32:raise ValueError('incomplete saved embeddings')
                for bi,(record,(support,tasks,labels)) in enumerate(zip(records,meta)):
                    if record['batch']!=bi or record['batch_sha256']!=entry['batch_sha256'][bi]:
                        raise ValueError('saved representation order differs')
                    values=record['embeddings']['U1_pre_meta']
                    if len(values)!=len(support):raise ValueError('representation shape differs')
                    for task in tasks.unique(sorted=True).tolist():
                        mask=support&(tasks==task)
                        rows.append({'target':target,'stream':stream,'model_id':model['model_id'],
                                     'source':model['source'],'batch':bi,'episode_in_batch':task,
                                     'supports':int(mask.sum()),**describe(values[mask],labels[mask])})
                receipts.append({'target':target,'stream':stream,'model_id':model['model_id'],
                                 'embedding_file':str(path),'bytes':path.stat().st_size,
                                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'weights_sha256':model['weights_sha256'],
                                 'episode_fingerprint':entry['episode_fingerprint']})
                print(f'{target}/{stream}/{model["model_id"]}: saved embeddings only',flush=True)
    if len(rows)!=11520 or len(receipts)!=90:raise ValueError('incomplete geometry grid')
    args.output.mkdir(parents=True)
    for name,value in [('episodes',rows),('receipts',receipts),('DONE',{
            'episodes':len(rows),'model_target_streams':len(receipts),'new_model_forwards':0,
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'scope':'Exploratory support-only geometry descriptors of already-observed outcomes, not held-out prediction.',
            'directional_expectation':'Higher within-class unit-support dispersion accompanies larger political suppression benefits; check all sources and all targets.',
            'descriptor_set':['support_norm_mean','support_norm_cv','support_unit_dispersion','support_class_cosine']} )]:
        (args.output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
