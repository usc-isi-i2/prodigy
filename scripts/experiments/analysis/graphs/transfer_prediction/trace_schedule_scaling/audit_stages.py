"""Direct readout baselines and exact decision accounting on saved schedules."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from .analyze_results import TARGETS, evaluate_logits
from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_contrast import softmax, to_numpy, validate_pair
from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_multi_health import health_probability_fusion


def score(record, logits):
    out = evaluate_logits(record, logits)
    y = to_numpy(record['labels']['local_y']).astype(int)
    pred = np.asarray(logits).argmax(1)
    if record['labels']['use_global']:
        mapping = to_numpy(record['labels']['mapping']).astype(int)
        rows = np.arange(len(y))
        y, pred = mapping[rows, y], mapping[rows, pred]
    out['macro_f1'] = float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0))
    return out


def audit(record, target, stream):
    y = to_numpy(record['labels']['local_y']).astype(int)
    cells, ensembles = [], []
    for name, model in record['models'].items():
        full = to_numpy(model['logits']['full_model'])
        u1 = to_numpy(model['logits']['U1_pre_meta/ridge'])
        f, u = full.argmax(1) == y, u1.argmax(1) == y
        corrected, corrupted = int((f & ~u).sum()), int((u & ~f).sum())
        assert abs(float(f.mean()-u.mean())-(corrected-corrupted)/len(y)) < 1e-12
        cells.append(dict(target=target, stream=stream, model=name,
            rung=model['rung'], seed=model['seed'], schedule=model['schedule'],
            queries=len(y), corrected=corrected, corrupted=corrupted,
            full=score(record, full), u1=score(record, u1)))
    for rung in (2, 3, 4):
        for seed in (0, 1, 2):
            names = sorted(n for n,m in record['models'].items() if m['rung']==rung and m['seed']==seed)
            if len(names) != 3:
                raise ValueError('three schedules required')
            full = [to_numpy(record['models'][n]['logits']['full_model']) for n in names]
            u1 = [to_numpy(record['models'][n]['logits']['U1_pre_meta/ridge']) for n in names]
            healthy = {n:f.argmax(1)==u.argmax(1) for n,f,u in zip(names,full,u1)}
            competent = max(float(to_numpy(record['models'][n]['support_health']['u1_loo_prototype_accuracy']).mean()) for n in names) >= .55
            probs = {'full_equal': np.mean([softmax(z) for z in full], axis=0),
                     'u1_equal': np.mean([softmax(z) for z in u1], axis=0),
                     'trace': health_probability_fusion(record,names,healthy)}
            for method,p in probs.items():
                ensembles.append(dict(target=target, stream=stream, rung=rung, seed=seed,
                    competent=competent, method=method, **score(record,np.log(np.clip(p,1e-300,1)))))
    return cells, ensembles


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('new output required')
    cells, ensembles, receipts=[],[],[]
    for target in TARGETS:
        pair=[]
        for stream in ('original','fresh'):
            path=args.input/target/(stream+'.pt')
            record=torch.load(path,map_location='cpu',weights_only=False)
            pair.append(record)
            if len(record['models'])!=27: raise ValueError('27 models required')
            c,e=audit(record,target,stream); cells+=c; ensembles+=e
            receipts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        validate_pair(*pair)
    summary={}
    for stream in ('original','fresh'):
        summary[stream]={method:{metric:float(np.mean([r[metric] for r in ensembles if r['stream']==stream and r['method']==method and r['competent']])) for metric in ('accuracy','macro_f1','roc_auc','nll')} for method in ('full_equal','u1_equal','trace')}
    args.output.mkdir(parents=True)
    for name,value in [('cells',cells),('ensembles',ensembles),('summary',summary),('receipts',receipts)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
