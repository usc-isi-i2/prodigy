"""Export small, provenance-bearing evidence from cluster run directories."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.evaluate import discover
from scripts.experiments.setup.nm_interventions_overnight.plan import HOLDOUT, TARGETS


def audit_exposure(exposure, episodes, sources, flags):
    """Check actual consumed episodes, including the terminal partial log interval."""
    if HOLDOUT in sources or set(exposure)!=set(TARGETS):
        raise ValueError('Invalid source exposure panel')
    if any(not math.isfinite(v) or v<0 for v in exposure.values()):
        raise ValueError('Non-finite or negative source exposure')
    if any(v!=0 for source,v in exposure.items() if source not in sources):
        raise ValueError('Exposure outside active training sources')
    if abs(sum(exposure.values())-episodes)>1e-5:
        raise ValueError('Source exposure does not sum to consumed episodes')
    if 'blocked' in flags:
        cycles,remainder=divmod(episodes,64*len(sources))
        expected={s:cycles*64+max(0,min(64,remainder-i*64)) for i,s in enumerate(sources)}
        if any(abs(v-expected.get(s,0))>1e-8 for s,v in exposure.items()):
            raise ValueError('Consumed exposure disagrees with 64-episode source blocks')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=sorted({k for row in rows for k in row})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows)


def main():
    a=argparse.ArgumentParser();a.add_argument('--run-dirs',nargs='+',required=True)
    a.add_argument('--eval-dirs',nargs='*',default=[]);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();jobs=discover(args.run_dirs);args.output.mkdir(parents=True,exist_ok=True)
    runtime={}
    for root in args.run_dirs:
        for path in Path(root).glob('job_*/result.json'):
            result=json.loads(path.read_text())
            if result.get('status')!='complete':continue
            config=json.loads((path.parent/'effective_config.json').read_text())
            match=re.search(r'Number of trainable parameters of the model:\s*(\d+)',
                            (path.parent/'console.log').read_text())
            if not match:raise ValueError(f'Missing model parameter count: {path.parent}')
            runtime[config['prefix']]=dict(result=result,model_parameters_before_label_freeze=int(match.group(1)))
    records=[];curves=[];vals=[];cells=[];resources=[];exposures=[];exposure_audits=[]
    for job in jobs:
        p=job['params'];selection=job['selection'];model=p['prefix']
        state=Path(job['state']);hist=json.loads((state/'validation_history.json').read_text())
        resource=runtime[model];result=resource['result'];dimension=p['emb_dim']
        # TrainerFS prints model count before freezing the registered 1000-label
        # table and before adding its two-linear-layer reconstruction head.
        frozen_label=0 if p['not_freeze_learned_label_embedding'] else 1000*dimension
        aux=dimension*dimension+dimension+dimension*p['input_dim']+p['input_dim']
        resources.append(dict(model_id=model,model_parameters=resource['model_parameters_before_label_freeze'],
            frozen_label_parameters=frozen_label,auxiliary_parameters=aux,
            optimizer_parameter_slots=resource['model_parameters_before_label_freeze']-frozen_label+aux,
            auxiliary_loss_active='aux_reconstruction' in selection['flags'],
            physical_gpu=result['physical_gpu'],peak_allocated_bytes=result['peak_allocated_bytes'],
            job_seconds=result['completed']-result['started'],
            initialization_seconds=result['training_started']-result['started'],
            training_and_cache_seconds=result['completed']-result['training_started'],
            loop_seconds=selection['seconds'],training_steps=selection['training_steps'],
            selected_step=selection['best_step'],stop_reason=selection['stop_reason'],
            validation_inference_seconds=sum(v['seconds'] for h in hist for v in h['per_source'].values())))
        records.append(dict(model_id=model,params=p,selection=selection,validation_history=hist,runtime=resource,
                            validation_protocol=json.loads((state/'validation_protocol.json').read_text())))
        curve_checks=0
        for line in (state/'training_curve.jsonl').read_text().splitlines():
            r=json.loads(line)
            audit_exposure(r.pop('exposure'),r['episodes'],selection['sources'],selection['flags'])
            curve_checks+=1
            curves.append(dict(model_id=model,**r))
        final_episodes=selection['training_steps']*p['batch_size']
        audit_exposure(selection['exposure'],final_episodes,selection['sources'],selection['flags'])
        exposures.extend(dict(model_id=model,source=source,training_steps=selection['training_steps'],
                              consumed_episodes=final_episodes,exposure_episodes=value)
                         for source,value in selection['exposure'].items())
        exposure_audits.append(dict(model_id=model,status='passed',curve_records_checked=curve_checks,
                                   terminal_episodes_checked=final_episodes,
                                   exact_block_schedule_checked='blocked' in selection['flags']))
        for check in hist:
            for source,row in check['per_source'].items():
                vals.append(dict(model_id=model,step=check['step'],source=source,**row))
    for root in args.eval_dirs:
        for path in Path(root).glob('cells/*/*.json'):
            row=json.loads(path.read_text());row['sources']=','.join(row['sources']);row['flags']=','.join(row['flags']);cells.append(row)
    keys=[(r['model_id'],r['target']) for r in cells]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate NM evaluation cells')
    fingerprints={}
    for row in cells:
        key=row['target'];fp=row['fingerprint']
        if key in fingerprints and fingerprints[key]!=fp:raise ValueError(f'Target episode mismatch: {key}')
        fingerprints[key]=fp
    (args.output/'model_records.json').write_text(json.dumps(records,indent=2)+'\n')
    write_csv(args.output/'training_curves.csv',curves)
    write_csv(args.output/'validation.csv',vals)
    write_csv(args.output/'resources.csv',resources)
    write_csv(args.output/'source_exposure.csv',exposures)
    (args.output/'exposure_audit.json').write_text(json.dumps(exposure_audits,indent=2)+'\n')
    write_csv(args.output/'nm_results.csv',cells)
    (args.output/'coverage.json').write_text(json.dumps(dict(completed_models=len(records),nm_cells=len(cells),fingerprints=fingerprints),indent=2)+'\n')
    print(f'Exported {len(records)} completed models; {len(cells)} NM cells')

if __name__=='__main__':main()
