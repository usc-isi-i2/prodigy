"""Label-free output sensitivity from the existing 720-cell role replay only."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('new output required')
    torch.set_num_threads(1)
    inventory=json.loads((args.inventory/'inventory.json').read_text())
    if len(inventory)!=90:raise ValueError('complete historical inventory required')
    rows=[]
    for item in inventory:
        path=Path(item['prediction_file'])
        if not path.is_absolute():path=args.runtime_root/path
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['weights_sha256']!=item['weights_sha256']:raise ValueError('model identity differs')
        # Deliberately do not access saved labels, query correctness or outcomes.
        logits=saved['logits']; base=logits['baseline']; removed=logits['edges_support']
        if base.shape!=removed.shape or len(base)!=item['queries'] or not torch.isfinite(base).all() or not torch.isfinite(removed).all():
            raise ValueError('invalid saved logits')
        tv=.5*(base.double().softmax(1)-removed.double().softmax(1)).abs().sum(1)
        rows.append({k:item[k] for k in ['source','target','stream','model_id','queries','weights_sha256']}|
                    {'mean_probability_total_variation':float(tv.mean()),
                     'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'prediction_file':str(path)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'rows':rows,'new_model_forwards':0,
        'definition':'Mean total-variation distance between saved intact and support-edge-removed class probabilities. No labels used.',
        'status':'Exploratory association analysis; all intervention outcomes were previously observed.'},indent=2)+'\n')
    print(f'Analyzed {len(rows)} already-saved cells; no model forwards.')


if __name__=='__main__':main()
