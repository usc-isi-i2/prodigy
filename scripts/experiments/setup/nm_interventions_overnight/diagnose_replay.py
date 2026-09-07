"""Training-validation-only numerical replay diagnosis; never evaluates test data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.evaluate import discover,model_from_params


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--run-dirs',nargs='+',required=True)
    a.add_argument('--model',required=True);a.add_argument('--source',required=True)
    a.add_argument('--repeats',type=int,default=20)
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    jobs=[j for j in discover(args.run_dirs) if j['params']['prefix']==args.model]
    if len(jobs)!=1:raise ValueError('Expected exactly one completed model')
    job=jobs[0];p=job['params'];selection=job['selection'];state=Path(job['state'])
    if args.source not in selection['sources']:raise ValueError('Only active training-source validation is allowed')
    if args.repeats<2:raise ValueError('At least two repeats are required')
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from experiments.nm_campaign import materialize,atomic_json
    from experiments.params import get_params
    from experiments.run_single_experiment import load_dataset
    from experiments.run_shared_graph import prepare_shared_dataset
    torch.set_num_threads(2);torch.autograd.set_detect_anomaly(False)
    if torch.cuda.device_count()!=1:raise ValueError('Set CUDA_VISIBLE_DEVICES to one owned GPU')
    device=torch.device('cuda:0')
    dataset=prepare_shared_dataset(load_dataset(get_params(['--config',p['config']])))
    protocol=json.loads((state/'validation_protocol.json').read_text())
    batches,fingerprint=materialize(dataset,p,args.source,'val',protocol['episodes_per_source'])
    if fingerprint!=protocol['fingerprints'][args.source]:raise ValueError('Validation fingerprint mismatch')
    history=json.loads((state/'validation_history.json').read_text())
    expected=next(r for r in history if r['step']==selection['best_step'])['per_source'][args.source]
    model=model_from_params(p,selection['checkpoint'],device)
    probabilities=[];rows=[]
    started=time.time()
    with torch.no_grad():
        for repeat in range(args.repeats):
            ys=[];ps=[];losses=[]
            for batch in batches:
                yt,yp,_=model(*[x.clone().to(device) for x in batch])
                if not torch.isfinite(yp).all():raise ValueError('Non-finite predictions')
                losses.append(float(F.cross_entropy(yp,yt.float())))
                ys.append(yt.cpu());ps.append(torch.softmax(yp,dim=1).cpu())
            y=torch.cat(ys).numpy();prob=torch.cat(ps).numpy()
            probabilities.append(prob)
            prediction=prob.argmax(1);truth=y.argmax(1)
            reference=probabilities[0];first_prediction=reference.argmax(1)
            changed=np.flatnonzero(prediction!=first_prediction)
            margins=np.sort(prob,axis=1)[:,-1]-np.sort(prob,axis=1)[:,-2]
            first_margins=np.sort(reference,axis=1)[:,-1]-np.sort(reference,axis=1)[:,-2]
            row=dict(repeat=repeat,roc_auc=float(roc_auc_score(y,prob,average='macro')),
                     accuracy=float((prediction==truth).mean()),loss=float(np.mean(losses)),
                     max_probability_delta=float(np.max(np.abs(prob-reference))),
                     changed_predictions=int(len(changed)),
                     changed_cases=[dict(query=int(i),truth=int(truth[i]),first=int(first_prediction[i]),
                         current=int(prediction[i]),first_margin=float(first_margins[i]),
                         current_margin=float(margins[i])) for i in changed])
            rows.append(row);print(json.dumps(row),flush=True)
    np.savez_compressed(args.output/'predictions.npz',truth=y,probabilities=np.stack(probabilities))
    result=dict(model_id=args.model,source=args.source,split='training_validation_only',
        checkpoint=selection['checkpoint'],checkpoint_sha256=hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest(),
        fingerprint=fingerprint,expected=expected,replays=rows,queries=len(truth),
        episodes=len(batches),seconds=time.time()-started,
        diagnosis_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    atomic_json(args.output/'diagnosis.json',result)


if __name__=='__main__':main()
