"""Two-graph asynchronous convergence with exact sampling-state rewind."""
import argparse,copy,csv,json,random,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import interleaved_mlp_pairs as base
from . import gradient_diagnostic as diagnostic
from .config import load_config
from .evaluation_artifacts import atomic_json
from .node_only_transfer import seed_everything
from .ladder_tracking import tracked_run

SOURCES=('ukr_rus_twitter','facebook_page_reference')
ARMS=('extended_bce','async_kd','rewind_bce')

def cpu_clone(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)

def initialize_sampling(graphs,seed,device):
    for g in graphs:
        g.update(generator=torch.Generator(device=device).manual_seed(seed+49979687),
                 order_generator=torch.Generator(device=device).manual_seed(seed+7919),
                 order=torch.empty(0,dtype=torch.long,device=device),offset=g['positive'].shape[1])

def next_batch(g):
    if g['offset']>=g['positive'].shape[1]:
        g['order']=torch.randperm(g['positive'].shape[1],generator=g['order_generator'],device=g['x'].device)
        g['offset']=0
    pos=g['positive'][:,g['order'][g['offset']:g['offset']+1024]]
    g['offset']+=pos.shape[1]
    return (*base.lp.pair_batch(pos,g['sampler'],g['generator']),pos.shape[1])

def snapshot(model,opt,graphs,runtime,metadata):
    device=next(model.parameters()).device
    return dict(model=cpu_clone(model.state_dict()),optimizer=cpu_clone(opt.state_dict()),
                step=runtime['logical_step'],metadata=copy.deepcopy(metadata),
                runtime=copy.deepcopy(runtime),
                samplers=[dict(order=cpu_clone(g['order']),offset=g['offset'],
                               generator=cpu_clone(g['generator'].get_state()),
                               order_generator=cpu_clone(g['order_generator'].get_state())) for g in graphs],
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),
                         cuda=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None),
                exact_sampling_resume=True)

def restore(ck,model,opt,graphs):
    model.load_state_dict(ck['model']);opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    device=next(model.parameters()).device
    for g,s in zip(graphs,ck['samplers']):
        g['order']=s['order'].to(device).clone();g['offset']=s['offset']
        g['generator'].set_state(s['generator'].cpu());g['order_generator'].set_state(s['order_generator'].cpu())
    random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
    torch.set_rng_state(ck['rng']['cpu'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(ck['rng']['cuda'].cpu(),device)
    return copy.deepcopy(ck['runtime'])

def save(path,ck):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(path)

def task_loss(logits,y,teacher_logits=None):
    if teacher_logits is None:return F.binary_cross_entropy_with_logits(logits,y)
    return F.binary_cross_entropy_with_logits(logits,teacher_logits.detach().sigmoid())

def update_tracker(t,auc,path,min_delta):
    if not np.isfinite(auc):raise FloatingPointError('nonfinite validation AUC')
    if auc>t['best_auc']:t.update(best_auc=auc,best_path=str(path))
    if auc>t['reference']+min_delta:t.update(reference=auc,stale=0)
    else:t['stale']+=1

def empty_counts():
    return dict(supervised_updates=0,kd_updates=0,positive_examples=0,negative_examples=0,
                supervised_positive_examples=0,supervised_negative_examples=0,kd_positive_examples=0,kd_negative_examples=0)

def empty_runtime():
    return dict(logical_step=0,counts=[empty_counts() for _ in SOURCES],
                trackers=[dict(best_auc=-1.,best_path=None,reference=-1.,stale=0) for _ in SOURCES],
                converged=[False,False],teacher_paths=[None,None],mean_bce_best=None,mean_bce_value=float('inf'))

def fixed_probe(g,seed):
    gen=torch.Generator(device=g['x'].device).manual_seed(seed)
    order=torch.randperm(g['positive'].shape[1],generator=gen,device=g['x'].device)[:20480]
    pos=g['positive'][:,order]
    return [base.lp.pair_batch(pos[:,start:start+1024],g['sampler'],gen) for start in range(0,pos.shape[1],1024)]

def measure(model,graphs,validation,probes):
    mode=model.training;model.eval()
    try:
        return dict(validation=[diagnostic.evaluate(model,g,p)[0] for g,p in zip(graphs,validation)],
                    training_probe=[diagnostic.evaluate(model,g,p)[0] for g,p in zip(graphs,probes)])
    finally:model.train(mode)

def frozen_teacher(path,device):
    # Construction must not advance the student RNG stream.
    cpu_rng=torch.get_rng_state()
    model=base.BiasMLP('node_neighbors').to(device)
    torch.set_rng_state(cpu_rng)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    return model.eval().requires_grad_(False)

def train(args,device):
    root=Path(args.root);run_dir=root/'node_neighbors/lp'/args.arm
    if run_dir.exists():raise FileExistsError(run_dir)
    if args.arm=='rewind_bce' and not (root/'first_rewind.pt').exists():raise FileNotFoundError('KD arm has not produced a first rewind')
    config=load_config(args.config);base.preflight(config)
    graphs=[base.load_graph(s,config,args.seed,device) for s in SOURCES]
    initialize_sampling(graphs,args.seed,device)
    validation=[diagnostic.validation_pairs(g) for g in graphs]
    probes=[fixed_probe(g,813719+args.seed+1009*i) for i,g in enumerate(graphs)]
    seed_everything(args.seed)
    model=base.BiasMLP('node_neighbors').to(device);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    runtime=empty_runtime();physical=0;events=[];teachers={};started=time.monotonic()
    metadata=dict(run_id=args.arm,sources=list(SOURCES),seed=args.seed,view='node_neighbors',
                  architecture='node_mlp',objective='lp',code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  protocol=dict(learning_rate=args.learning_rate,max_physical_steps=args.max_steps,max_logical_steps=args.max_steps,
                                validation_interval=args.validation_interval,patience=args.patience,min_delta_auc=args.min_delta,
                                minimum_steps=args.minimum_steps,source_schedule='alternating_1_to_1',kd_temperature=1.,kd_weight=1.,
                                checkpoint_selection='all-converged endpoint for KD; fixed-exposure endpoints for controls'),
                  graph_split_receipts={s:g['receipt'] for s,g in zip(SOURCES,graphs)})
    if args.arm=='rewind_bce':
        ck=torch.load(root/'first_rewind.pt',map_location='cpu',weights_only=False)
        runtime=restore(ck,model,opt,graphs)
        # Sibling inherits the raw rewind state, before the first source is declared converged.
        runtime['converged']=[False,False];runtime['teacher_paths']=[None,None]
        metadata['imported_logical_step']=runtime['logical_step'];metadata['prefix_compute_shared_with']='async_kd'
    run_dir.mkdir(parents=True)
    for i,p in enumerate(probes):
        q=cpu_clone(p);save(run_dir/f'probe_{SOURCES[i]}.pt',q)
    atomic_json(run_dir/'metadata.json',metadata)
    physical_counts=[empty_counts() for _ in SOURCES]
    history=[];loss_sums=[0.,0.];loss_counts=[0,0];stop='safety_cap'
    with tracked_run(run_dir,metadata,args) as (wandb,log):
        def validate_save(initial=False):
            reports=measure(model,graphs,validation,probes)
            path=run_dir/'checkpoints'/f'physical_{physical:06d}_logical_{runtime["logical_step"]:06d}.pt'
            for i,report in enumerate(reports['validation']):
                if not runtime['converged'][i]:
                    update_tracker(runtime['trackers'][i],report['auc'],path,args.min_delta)
            mean_bce=sum(r['bce'] for r in reports['validation'])/2
            if mean_bce<runtime['mean_bce_value']:
                runtime['mean_bce_value']=mean_bce;runtime['mean_bce_best']=str(path)
            row=dict(physical_step=physical,logical_step=runtime['logical_step'],counts=copy.deepcopy(runtime['counts']),
                     checkpoint=str(path),converged=list(runtime['converged']),**reports)
            save(path,snapshot(model,opt,graphs,runtime,metadata))
            atomic_json(run_dir/'validation'/f'physical_{physical:06d}.json',row);history.append(row)
            with (run_dir/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            fields={'validation/mean_bce':mean_bce,'check/logical_step':runtime['logical_step']}
            for i,s in enumerate(SOURCES):
                fields.update({f'validation/{s}/auc':reports['validation'][i]['auc'],
                               f'validation/{s}/bce':reports['validation'][i]['bce'],
                               f'check/{s}/hard_label_probe_bce':reports['training_probe'][i]['bce']})
            log(physical,fields)
            print(json.dumps(dict(arm=args.arm,physical=physical,logical=runtime['logical_step'],validation=reports['validation'])),flush=True)
            if args.arm!='async_kd' or initial or runtime['logical_step']<args.minimum_steps:return False
            candidates=[i for i in range(2) if not runtime['converged'][i] and runtime['trackers'][i]['stale']>=args.patience]
            if not candidates:return False
            # If simultaneous, choose the earliest best checkpoint, then revalidate the other source on the restored path.
            selected=min(candidates,key=lambda i:(torch.load(runtime['trackers'][i]['best_path'],map_location='cpu',weights_only=False)['step'],i))
            trigger=runtime['logical_step'];path=runtime['trackers'][selected]['best_path']
            ck=torch.load(path,map_location='cpu',weights_only=False)
            first=not any(runtime['converged'])
            if first:save(root/'first_rewind.pt',ck)
            previous_converged=list(runtime['converged']);previous_teachers=list(runtime['teacher_paths'])
            restored=restore(ck,model,opt,graphs)
            previous_converged[selected]=True;previous_teachers[selected]=str(path)
            restored['converged']=previous_converged;restored['teacher_paths']=previous_teachers
            runtime.clear();runtime.update(restored)
            teachers[selected]=frozen_teacher(path,device)
            event=dict(physical_step=physical,trigger_logical_step=trigger,restored_logical_step=runtime['logical_step'],
                       source=SOURCES[selected],teacher_checkpoint=str(path),discarded_logical_steps=trigger-runtime['logical_step'],
                       simultaneous_candidates=[SOURCES[i] for i in candidates])
            events.append(event);atomic_json(run_dir/'events.json',events)
            # Discard any training-window totals from the abandoned tail.
            loss_sums[:]=[0.,0.];loss_counts[:]=[0,0]
            print(json.dumps(dict(event='rewind_and_switch',**event)),flush=True)
            return all(runtime['converged'])
        validate_save(initial=True)
        while physical<args.max_steps and runtime['logical_step']<args.max_steps:
            index=runtime['logical_step']%2
            pairs,y,npos=next_batch(graphs[index]);opt.zero_grad(set_to_none=True)
            logits=base.score(model,graphs[index]['x'],pairs)
            teacher_logits=None
            if runtime['converged'][index]:
                with torch.no_grad():teacher_logits=base.score(teachers[index],graphs[index]['x'],pairs)
            loss=task_loss(logits,y,teacher_logits)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            physical+=1;runtime['logical_step']+=1
            kind='kd_updates' if teacher_logits is not None else 'supervised_updates'
            for counts in (runtime['counts'],physical_counts):
                counts[index][kind]+=1;counts[index]['positive_examples']+=npos;counts[index]['negative_examples']+=5*npos
                prefix='kd' if teacher_logits is not None else 'supervised'
                counts[index][prefix+'_positive_examples']+=npos;counts[index][prefix+'_negative_examples']+=5*npos
            loss_sums[index]+=float(loss.detach());loss_counts[index]+=1
            if physical%args.log_interval==0:
                log(physical,{f'train/{SOURCES[i]}/objective':loss_sums[i]/loss_counts[i] for i in range(2) if loss_counts[i]})
                loss_sums[:]=[0.,0.];loss_counts[:]=[0,0]
            if physical%args.validation_interval==0 or physical==args.max_steps or runtime['logical_step']==args.max_steps:
                if validate_save():stop='all_tasks_converged';break
        if args.arm!='async_kd' and runtime['logical_step']>=args.max_steps:stop='exposure_horizon'
        endpoint=snapshot(model,opt,graphs,runtime,metadata);save(run_dir/'endpoint.pt',endpoint)
        # Existing evaluator consumes best.pt; this is the DECLARED endpoint, not a test-selected checkpoint.
        save(run_dir/'best.pt',endpoint)
        final_reports=measure(model,graphs,validation,probes)
        summary=dict(metadata,status='complete',stop_reason=stop,physical_steps=physical,logical_step=runtime['logical_step'],
                     surviving_counts=runtime['counts'],physical_counts=physical_counts,converged_sources=[SOURCES[i] for i,x in enumerate(runtime['converged']) if x],
                     events=events,endpoint=final_reports,elapsed_seconds=time.monotonic()-started,
                     mean_bce_best_checkpoint=runtime['mean_bce_best'],
                     teacher_forward_batches=sum(c['kd_updates'] for c in physical_counts),
                     teacher_forward_pairs=sum(c['kd_positive_examples']+c['kd_negative_examples'] for c in physical_counts))
        atomic_json(run_dir/'summary.json',summary);wandb.summary.update(dict(physical_steps=physical,logical_step=runtime['logical_step'],stop_reason=stop))

def singleton_probes(args,device):
    from .ranking_preservation import SOURCE_ROOT
    root=Path(args.root);graphs=[base.load_graph(s,load_config(args.config),args.seed,device) for s in SOURCES]
    parts=[fixed_probe(g,813719+args.seed+1009*i) for i,g in enumerate(graphs)]
    rows=[]
    for i,source in enumerate(SOURCES):
        run=SOURCE_ROOT/f'ss_{source}'
        for name in ('best.pt','latest.pt'):
            path=run/name
            ck=torch.load(path,map_location='cpu',weights_only=False);model=base.BiasMLP('node_neighbors').to(device).eval()
            model.load_state_dict(ck['model'])
            report,_=diagnostic.evaluate(model,graphs[i],parts[i])
            rows.append(dict(source=source,checkpoint=str(path),step=ck['step'],**report))
    atomic_json(root/'singleton_probes.json',rows)

def prepare_eval(args):
    root=Path(args.root);kd=root/'node_neighbors/lp/async_kd'
    summary=json.loads((kd/'summary.json').read_text());target=summary['logical_step']
    models=[dict(run_id=a,selection='declared endpoint') for a in ARMS];unavailable=[]
    for arm in ('extended_bce','rewind_bce'):
        run=root/'node_neighbors/lp'/arm
        history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        matches=[r for r in history if r['logical_step']==target]
        if not matches:
            unavailable.append(dict(arm=arm,logical_step=target,reason='no checkpoint at the KD endpoint exposure'));continue
        row=matches[0]
        if row['counts'][0]['positive_examples']!=summary['surviving_counts'][0]['positive_examples'] or row['counts'][1]['positive_examples']!=summary['surviving_counts'][1]['positive_examples']:
            raise ValueError('logical-step match failed actual per-source exposure match')
        alias=arm+'_matched';dest=root/'node_neighbors/lp'/alias;dest.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(row['checkpoint'],dest/'best.pt')
        models.append(dict(run_id=alias,selection='matched to KD surviving per-source exposure',original_checkpoint=row['checkpoint'],logical_step=target))
    replay_check=None
    alias_paths=[root/'node_neighbors/lp'/a/'best.pt' for a in ('extended_bce_matched','rewind_bce_matched')]
    if all(p.exists() for p in alias_paths):
        first,second=[torch.load(p,map_location='cpu',weights_only=False) for p in alias_paths]
        maximum=max(float((first['model'][k]-second['model'][k]).abs().max()) for k in first['model'])
        replay_check=dict(max_parameter_difference=maximum,within_1e_6=maximum<=1e-6,
                          interpretation='rewind-only should reproduce unchanged BCE at equal surviving exposure')
    atomic_json(root/'evaluation_manifest.json',dict(models=models,unavailable=unavailable,replay_check=replay_check,selection_data='source validation only; frozen before downstream evaluation'))

def aggregate(args):
    root=Path(args.root);results=[]
    manifest=json.loads((root/'evaluation_manifest.json').read_text())
    for entry in manifest['models']:
        arm=entry['run_id']
        for target in base.SOURCES:
            path=root/'results/node_neighbors_bias'/f'{arm}__to__{target}.json'
            d=json.loads(path.read_text())
            results.append(dict(arm=arm,target=target,checkpoint_step=d['checkpoint_step'],**{k:d['metrics']['test'][k] for k in ('roc_auc','bce','average_precision')}))
    with (root/'results/matrix.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    atomic_json(root/'results/COMPLETE.json',dict(status='complete',models=len(manifest['models']),evaluation_cells=len(results),unavailable=manifest['unavailable'],replay_check=manifest.get('replay_check')))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','train','eval','probes','prepare_eval','aggregate'])
    p.add_argument('--root',required=True);p.add_argument('--arm',choices=ARMS,default='extended_bce');p.add_argument('--device',type=int,choices=range(4),default=0)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=.0001);p.add_argument('--minimum-steps',type=int,default=2500)
    p.add_argument('--learning-rate',type=float,default=.0005);p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer');p.add_argument('--wandb-group',default='async-convergence')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot');args=p.parse_args()
    if any(n<=0 or n%2 for n in (args.max_steps,args.validation_interval,args.log_interval)) or args.patience<1:p.error('positive even step counts and positive patience required')
    if not np.isfinite(args.learning_rate) or args.learning_rate<=0 or not np.isfinite(args.min_delta) or args.min_delta<0 or args.minimum_steps<0:p.error('invalid LR, minimum improvement, or minimum steps')
    if args.phase=='plan':print(json.dumps(dict(sources=SOURCES,arms=ARMS,inputs=base.preflight(load_config(args.config))),indent=2));return
    if args.phase=='aggregate':aggregate(args);return
    if args.phase=='prepare_eval':prepare_eval(args);return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(f'cuda:{args.device}')
    if args.phase=='eval':
        args.views=('node_neighbors',);args.targets=base.SOURCES;args.model_rows=[(m['run_id'],m['run_id']) for m in json.loads((Path(args.root)/'evaluation_manifest.json').read_text())['models']];base.evaluate(args,device)
    elif args.phase=='probes':singleton_probes(args,device)
    else:train(args,device)
if __name__=='__main__':main()
