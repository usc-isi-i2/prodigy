"""Fixed-prototype localization of the nominated matched page crossover.

Only Hong Kong seeds 0/1/2, original/fresh episodes, steps 100 and 2500.
Private tensors and metrics must remain outside public Git transport.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from experiments.run_shared_graph import write_json
from .contrast_metrics import ranking_metrics
from .message_scale import forward_scaled
from .replay import batch_hash, episode_probe
from .role_context import query_mask
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--scale-root',type=Path,required=True)
    p.add_argument('--dose-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(); read=lambda p:json.loads(p.read_text())
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('fresh output and hidden GPUs required')
    torch.set_num_threads(4)
    topology=read(Path(read(args.scale_root/'protocol.json')['reference_replay'])/'protocol.json')
    if read(args.scale_root/'DONE.json')['cells']!=900: raise ValueError('complete scale parent required')
    selected=[r for r in read(args.dose_root/'checkpoint_inventory.json')
              if r['source']=='cp_hk' and r['step'] in [100,2500]]
    if {(r['seed'],r['step']) for r in selected}!={(s,t) for s in [0,1,2] for t in [100,2500]} or len(selected)!=6:
        raise ValueError('exact six-checkpoint contrast required')
    protocol={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scale_root':str(args.scale_root),'dose_root':str(args.dose_root),
        'target':'facebook_page_reference','streams':['original','fresh'],
        'checkpoints':selected,'batches_per_cell':32,'paired_cells':12,
        'hypothesis':'Prototype context utility remains positive at both steps while native utility reverses.',
        'decision':'If prototype utility reverses too, reject constructor-specific crossover. Mixed results are inconclusive, not permission for checkpoint search.',
        'readout':'Mean unit supports within class, normalize means, score fixed intact unit U1 queries. Float64. No fitting.',
        'new_training':False}
    if args.dry_run: print(json.dumps(protocol,indent=2)); return
    args.output.mkdir(parents=True);write_json(args.output/'protocol.json',protocol)
    metrics=[];receipts=[];started=time.monotonic()
    try:
        with torch.no_grad():
            for stream in protocol['streams']:
                for rec in sorted(selected,key=lambda r:(r['seed'],r['step'])):
                    target=protocol['target'];mid=rec['model_id'];step=rec['step']
                    directory=Path(topology['cache_roots'][target+'/'+stream])/target
                    if step==2500:
                        path=args.scale_root/'predictions'/target/stream/(mid+'.pt'); names=['scale_1','scale_0']
                    else:
                        path=args.dose_root/'predictions'/target/stream/(mid+'__step100.pt'); names=[(0,0),(100,0)]
                    old=torch.load(path,map_location='cpu',weights_only=False); labels=old['labels']
                    state=torch.load(rec['checkpoint'],map_location='cpu',weights_only=True)['model']
                    if model_digest(state)!=rec['weights_sha256']: raise ValueError('checkpoint identity changed')
                    first=torch.load(directory/'batches/batch_000.pt',map_location='cpu',weights_only=False)
                    model=make_model(read(directory.parent/'protocol.json'),target,labels['cache']['graph_path'],first[0].x.shape[1],state)
                    model.eval(); values={k:[] for k in ['native_intact','native_suppressed','prototype_intact','prototype_suppressed']}
                    for bi in range(32):
                        batch=torch.load(directory/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                        expected=labels['cache']['batch_sha256'][bi]
                        if batch_hash(batch)!=expected: raise ValueError('input identity changed')
                        a,ta=forward_scaled(model,batch,alpha=1.)
                        b,tb=forward_scaled(model,batch,alpha=0.)
                        q=query_mask(batch)
                        for stage in ['U1_pre_meta','final_input']:
                            torch.testing.assert_close(ta[stage][q],tb[stage][q],rtol=0,atol=0)
                        pa=episode_probe(ta['U1_pre_meta'].double(),batch,'prototype')
                        mixed=tb['U1_pre_meta'].double().clone();mixed[q]=ta['U1_pre_meta'][q].double()
                        pb=episode_probe(mixed,batch,'prototype')
                        for key,value in zip(values,[a,b,pa,pb]):values[key].append(value)
                        if batch_hash(batch)!=expected: raise ValueError('input mutated')
                    logits={k:torch.cat(v) for k,v in values.items()}
                    for key,name in zip(['native_intact','native_suppressed'],names):
                        torch.testing.assert_close(logits[key],old['logits'][name],rtol=0,atol=0)
                    common={'seed':rec['seed'],'step':step,'stream':stream,'model_id':mid}
                    for condition,value in logits.items():
                        ranks,episodes=ranking_metrics((value[:,1].double()-value[:,0].double()).numpy(),
                            labels['local_y'].numpy(),labels['mapping'].numpy(),labels['episode_ids'].numpy(),labels['use_global'])
                        metrics.append({**common,'condition':condition,**ranks,
                                        'local_accuracy':float((value.argmax(1)==labels['local_y']).double().mean())})
                    if model_digest(model.state_dict())!=rec['weights_sha256']: raise ValueError('weights changed')
                    receipts.append({**common,'weights_sha256':rec['weights_sha256'],'batch_sha256':labels['cache']['batch_sha256'],
                                     'native_endpoints_bitexact':True,'queries_bitexact':True})
                    torch.save({'logits':logits,'labels':labels},args.output/f'{mid}_{step}_{stream}.pt')
                    write_json(args.output/'metrics.json',metrics);write_json(args.output/'receipts.json',receipts)
                    print(json.dumps({**common,'completed':len(receipts),'elapsed':time.monotonic()-started}),flush=True)
        write_json(args.output/'DONE.json',{'paired_cells':len(receipts),'metric_cells':len(metrics),'elapsed':time.monotonic()-started})
    except BaseException as exc:
        write_json(args.output/'FAILED.json',{'error':repr(exc)})
        raise


if __name__=='__main__':main()
