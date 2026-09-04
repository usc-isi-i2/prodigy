"""Persistent shared-graph, target-major NM evaluation of selected campaign models."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS, HOLDOUT


def discover(run_dirs):
    jobs=[]
    for run in run_dirs:
        manifest=json.loads((Path(run)/'manifest.json').read_text())
        for path in sorted(Path(run).glob('job_*/result.json')):
            result=json.loads(path.read_text())
            if result.get('status')!='complete':
                continue
            params=json.loads((path.parent/'effective_config.json').read_text())
            params['campaign_revision']=manifest['revision']
            if params['exp_name'].startswith('smoke_'):
                continue
            state=Path(result['checkpoint_dir']).parent
            selection=json.loads((state/'selection.json').read_text())
            if selection['status']!='complete':
                continue
            jobs.append(dict(params=params,selection=selection,state=str(state)))
    keys=[j['params']['prefix'] for j in jobs]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate completed model IDs; explicitly resolve retry provenance')
    return jobs


def model_from_params(params, checkpoint, device):
    import torch
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN
    p=dict(params,device=device)
    layers=get_module_list(p['layers'],p['emb_dim'],edge_attr_dim=None,input_dim=p['input_dim'],
        dropout=p['dropout'],reset_after_layer=p['reset_after_layer'],
        attention_mask_scheme=p['attention_mask_scheme'],has_final_back=p['has_final_back'],
        msg_pos_only=p.get('meta_gnn_pos_only',False),batch_norm_metagraph=not p['no_bn_metagraph'],
        batch_norm_encoder=not p['no_bn_encoder'],encoder_gnn_type=p['gnn_type'],gnn_use_relu=False)
    model=SingleLayerGeneralGNN(torch.nn.ModuleList(layers),
        initial_label_mlp=torch.nn.Linear(768,p['emb_dim']),params=p,
        text_dropout=torch.nn.Dropout(p['text_features_dropout']))
    payload=torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval()


def worker(dataset,jobs,output,worker_id,episodes,targets,validation_only=False,replay_repeats=1,invocation_id=None):
    os.setsid()
    import torch
    from experiments.nm_campaign import atomic_json,materialize,evaluate
    output=Path(output)
    stream=(output/f'worker_{worker_id}.log').open('a',buffering=1)
    os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
    torch.set_num_threads(2)
    torch.autograd.set_detect_anomaly(False)
    device=torch.device('cuda:0')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    worker_record=dict(invocation_id=invocation_id,pid=os.getpid(),started=time.time())
    atomic_json(output/f'worker_{worker_id}_status.json',dict(status='running',**worker_record))
    print(f'INVOCATION {invocation_id}: {len(jobs)} models; targets={targets}',flush=True)
    try:
        for target in targets:
            # Replay the trainer's exact validation panel after strict checkpoint
            # reload. This checks model reconstruction AND metric/data parity.
            validation_cache={}
            for job in jobs:
                if target not in job['selection']['sources']:continue
                p=job['params'];selection=job['selection'];state=Path(job['state'])
                protocol=json.loads((state/'validation_protocol.json').read_text())
                count=protocol['episodes_per_source']
                if count not in validation_cache:
                    validation_cache[count]=materialize(dataset,p,target,'val',count)
                batches,fp=validation_cache[count]
                if fp!=protocol['fingerprints'][target]:
                    raise ValueError(f'Validation fingerprint mismatch: {p["prefix"]} {target}')
                model=model_from_params(p,selection['checkpoint'],device)
                replays=[evaluate(model,batches,device) for _ in range(replay_repeats)]
                metrics=replays[0]
                history=json.loads((state/'validation_history.json').read_text())
                expected=next(r for r in history if r['step']==selection['best_step'])['per_source'][target]
                errors={key:max(abs(m[key]-expected[key]) for m in replays) for key in ('roc_auc','accuracy','loss')}
                # CUDA scatter rounding can swap nearly tied ranks while loss and
                # decisions agree. AUC tolerance is 100x below the .001 effect
                # threshold; keep episode identity, accuracy and loss gates strict.
                tolerances=dict(roc_auc=1e-5,accuracy=1e-6,loss=1e-6)
                if any(errors[key]>tolerances[key] for key in errors):
                    raise ValueError(f'Validation checkpoint replay mismatch: {p["prefix"]} {target}: {errors}')
                atomic_json(output/'validation_replay'/p['prefix']/f'{target}.json',
                    dict(status='passed',checkpoint=selection['checkpoint'],fingerprint=fp,
                         errors=errors,tolerances=tolerances,replays=replays,metrics=metrics,
                         invocation_id=invocation_id,evaluation_revision=revision))
                print(f'VALIDATION REPLAY PASS {p["prefix"]} {target}',flush=True)
                del model
            validation_cache.clear()
            if validation_only:continue
            pending=[]
            for job in jobs:
                path=output/'cells'/job['params']['prefix']/f'{target}.json'
                if path.exists():
                    old=json.loads(path.read_text())
                    if old['checkpoint']!=job['selection']['checkpoint'] or old['episodes']!=episodes or old['protocol']!='nmi_fixed_nm_v1':
                        raise ValueError(f'Resume mismatch: {path}')
                else:
                    pending.append(job)
            if not pending:
                continue
            print(f'TARGET {target}: {len(pending)} models; materializing {episodes} episodes',flush=True)
            batches,fingerprint=materialize(dataset,pending[0]['params'],target,'test',episodes)
            for job in pending:
                p=job['params'];selection=job['selection'];checkpoint=selection['checkpoint']
                model=model_from_params(p,checkpoint,device)
                metrics=evaluate(model,batches,device)
                with open(checkpoint,'rb') as f:
                    sha=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
                payload=dict(protocol='nmi_fixed_nm_v1',model_id=p['prefix'],target=target,
                    invocation_id=invocation_id,
                    sources=selection['sources'],holdout=HOLDOUT,seed=p['seed'],
                    flags=selection['flags'],checkpoint=checkpoint,checkpoint_sha256=sha,
                    checkpoint_step=selection['best_step'],training_steps=selection['training_steps'],
                    training_revision=p.get('campaign_revision','recorded in training manifest'),
                    evaluation_revision=revision,fingerprint=fingerprint,**metrics)
                atomic_json(output/'cells'/p['prefix']/f'{target}.json',payload)
                print(f"DONE {p['prefix']} {target} auc={metrics['roc_auc']:.6f} seconds={metrics['seconds']:.1f}",flush=True)
                del model
                gc.collect()
            del batches
            gc.collect()
        atomic_json(output/f'worker_{worker_id}_status.json',dict(status='complete',completed=time.time(),**worker_record))
    except BaseException:
        atomic_json(output/f'worker_{worker_id}_status.json',dict(status='failed',error=traceback.format_exc(),**worker_record))
        traceback.print_exc()
        raise


def main():
    a=argparse.ArgumentParser();a.add_argument('--run-dirs',nargs='+',required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3],choices=range(4))
    a.add_argument('--workers-per-gpu',type=int,default=2);a.add_argument('--episodes',type=int,default=512)
    a.add_argument('--targets',nargs='+',choices=TARGETS,default=list(TARGETS));a.add_argument('--dry-run',action='store_true')
    a.add_argument('--validation-only',action='store_true',help='Replay selected checkpoints on training-source validation only')
    a.add_argument('--models',nargs='+',help='Exact completed model IDs to evaluate')
    a.add_argument('--replay-repeats',type=int,default=1,help='Repeated validation forwards to quantify numerical variation')
    args=a.parse_args()
    if args.workers_per_gpu<1:a.error('--workers-per-gpu must be positive')
    if args.episodes<1:a.error('--episodes must be positive')
    jobs=discover(args.run_dirs)
    if args.models:
        missing=set(args.models)-{j['params']['prefix'] for j in jobs}
        if missing:raise ValueError(f'Requested models are not complete: {sorted(missing)}')
        jobs=[j for j in jobs if j['params']['prefix'] in args.models]
    if args.replay_repeats<1:raise ValueError('replay-repeats must be positive')
    if not jobs: raise ValueError('No completed production models')
    if args.validation_only:
        print(f'{len(jobs)} models: training-source checkpoint replay only; no test results',flush=True)
    else:
        print(f'{len(jobs)} models x {len(args.targets)} targets = {len(jobs)*len(args.targets)} cells',flush=True)
    if args.dry_run:return
    import torch
    from experiments.params import get_params
    from experiments.run_single_experiment import load_dataset
    from experiments.run_shared_graph import prepare_shared_dataset,start_on_gpu
    from experiments.nm_campaign import atomic_json
    args.output.mkdir(parents=True,exist_ok=True)
    invocation_id=f'{time.time_ns()}_{os.getpid()}'
    previous=[args.output/'plan.json',args.output/'status.json',*args.output.glob('worker_*_status.json')]
    for path in previous:
        if path.exists():
            atomic_json(args.output/'history'/f'before_{invocation_id}'/path.name,json.loads(path.read_text()))
    atomic_json(args.output/'plan.json',dict(models=[j['params']['prefix'] for j in jobs],targets=args.targets,
        episodes=args.episodes,validation_only=args.validation_only,replay_repeats=args.replay_repeats,
        invocation_id=invocation_id,gpus=args.gpus,workers_per_gpu=args.workers_per_gpu,
        evaluation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    started=time.time()
    atomic_json(args.output/'status.json',dict(status='running',invocation_id=invocation_id,
        pid=os.getpid(),started=started))
    processes=[]
    try:
        p=get_params(['--config',jobs[0]['params']['config']])
        torch.set_num_threads(4);torch.autograd.set_detect_anomaly(False)
        dataset=prepare_shared_dataset(load_dataset(p))
        slots=[g for g in args.gpus for _ in range(args.workers_per_gpu)]
        context=torch.multiprocessing.get_context('spawn')
        for i,gpu in enumerate(slots):
            selected=jobs[i::len(slots)]
            if not selected:continue
            process=context.Process(target=worker,args=(dataset,selected,str(args.output),i,args.episodes,args.targets,args.validation_only,args.replay_repeats,invocation_id))
            start_on_gpu(process,gpu);processes.append(process)
        for process in processes:process.join()
        failures=[p.exitcode for p in processes if p.exitcode!=0]
        if failures:raise RuntimeError(f'Worker failures: {failures}')
        atomic_json(args.output/'status.json',dict(status='complete',invocation_id=invocation_id,
            started=started,completed=time.time(),failures=[]))
    except BaseException:
        atomic_json(args.output/'status.json',dict(status='failed',invocation_id=invocation_id,
            started=started,completed=time.time(),error=traceback.format_exc()))
        for process in processes:
            if process.is_alive():process.terminate();process.join()
        raise

if __name__=='__main__':main()
