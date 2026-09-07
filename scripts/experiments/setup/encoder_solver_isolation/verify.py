"""Check completed isolation arms, source-private consumption, and encoder parity."""
import argparse
import json
from pathlib import Path
import torch
from scripts.experiments.setup.trace_schedule_scaling.verify_training import find_audit, read_audit, payload_digest


def verify(root):
    done=json.loads((root/'DONE.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    if len(plan)!=8 or done['arms']!=8:raise ValueError('eight completed arms required')
    records=[]; states={}; reference_sources=None
    initial_reference=None; initial_count=0
    for entry in plan:
        name=entry['name']+'_isolation_v1'
        directory=root/'state'/name/'checkpoint'
        initial_path=directory/'state_dict_0.ckpt'
        if initial_path.exists():
            initial=torch.load(initial_path,map_location='cpu',weights_only=True)['model']
            if initial_reference is None:initial_reference=initial
            if initial.keys()!=initial_reference.keys() or any(
                not torch.equal(initial[k],initial_reference[k]) for k in initial
            ):raise ValueError('initial model states differ')
            initial_count+=1
        elif done['steps']==2500:
            raise ValueError('full experiment requires captured initial weights')
        training=torch.load(directory/f'training_state_{done["steps"]}.ckpt',map_location='cpu',weights_only=False)
        metadata=training['_training_checkpoint']
        if metadata['completed_steps']!=done['steps']:raise ValueError('wrong completed step')
        if metadata['parameter_contract']['encoder_solver_objective']!=entry['mode']:raise ValueError('wrong objective')
        rows=read_audit(find_audit(root/'native_log',name))
        if len(rows)!=done['steps'] or [r['step'] for r in rows]!=list(range(1,done['steps']+1)):
            raise ValueError('incomplete consumed episodes')
        grouped={}
        for row in rows:
            source=tuple(sorted(set(row['source_ids'])))
            if len(source)!=1:raise ValueError('source-private episodes required')
            grouped.setdefault(str(source[0]),[]).append(row)
        sources={k:payload_digest(v) for k,v in grouped.items()}
        if reference_sources is None:reference_sources=sources
        if sources!=reference_sources:raise ValueError('source-private examples differ')
        state=torch.load(directory/f'state_dict_{done["steps"]}.ckpt',map_location='cpu',weights_only=True)['model']
        states[entry['name']]=state
        records.append(dict(name=name,sources=sources,steps=len(rows)))
    parity=[]
    for schedule in ('blocked','interleaved'):
        a=next(e for e in plan if e['arm']['schedule']==schedule and e['mode']=='isolated')
        b=next(e for e in plan if e['arm']['schedule']==schedule and e['mode']=='ridge_only')
        sa,sb=states[a['name']],states[b['name']]
        # S and U modules constitute the encoder; label/solver heads excluded.
        keys=[k for k in sa if k.startswith(('layer_list.0.','layer_list.1.','initial_input_mlp.'))]
        if not keys:raise ValueError('no encoder tensors')
        errors={k:float((sa[k].double()-sb[k].double()).abs().max()) for k in keys if sa[k].numel()}
        parity.append(dict(schedule=schedule,exact=all(torch.equal(sa[k],sb[k]) for k in keys),
                           max_abs_error=max(errors.values()),tensor_count=len(keys)))
    return dict(arms=records,source_private_examples_exact=True,encoder_parity=parity,
                initialization_exact=initial_count==8,initial_checkpoints=initial_count,
                note='GPU parity measured, not silently rounded; legacy smoke may lack initial checkpoints.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
