"""Matched gradient-path experiment; dry-run by default, one owned GPU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan

MODES = ('native', 'joint', 'isolated', 'ridge_only')


def commands(root, output, gpu, steps, seed):
    if gpu not in range(4) or steps not in (20, 2500):
        raise ValueError('owned GPU0-3 and smoke20/full2500 required')
    entries=[]
    for arm in build_plan(total_steps=steps,rungs=(4,),seeds=(seed,),schedules=('blocked','interleaved')):
        for mode in MODES:
            name=f'{arm.model_id}_{mode}'
            cmd=[sys.executable,'-u','experiments/run_single_experiment.py',
                '--config',str(root/'scripts/experiments/setup/trace_schedule_scaling/training.yaml'),
                '--device',str(gpu),'--seed',str(seed),'--prefix',name,'--timestamp','isolation_v1',
                '--state_dir',str(output/'state'),'--log_dir',str(output/'native_log'),
                '--dataset_len_cap',str(steps),'--checkpoint_steps',f'0,{steps}',
                '--neighbor_sampling_source_subset',','.join(arm.sources),
                '--neighbor_sampling_source_schedule',','.join(arm.segment_sources),
                '--neighbor_sampling_source_schedule_steps',','.join(map(str,arm.segment_steps)),
                '--neighbor_matching_member_seed',str(380100+seed),
                '--neighbor_sampling_source_schedule_seed',str(480100+seed),
                '--encoder_solver_objective',mode]
            entries.append(dict(name=name,mode=mode,arm=asdict(arm),command=cmd))
    return entries


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=range(4),required=True)
    p.add_argument('--steps',type=int,choices=(20,2500),default=20)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    output=a.output.resolve()
    plan=commands(root,output,a.gpu,a.steps,a.seed)
    if output.exists(): raise ValueError('new output required; no implicit resume')
    print(json.dumps(plan,indent=2))
    if not a.execute:return
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'revision.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
    for entry in plan:
        with (output/(entry['name']+'.log')).open('w') as stream:
            subprocess.run(entry['command'],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        checkpoint=output/'state'/(entry['name']+'_isolation_v1')/'checkpoint'/f'state_dict_{a.steps}.ckpt'
        if not checkpoint.is_file():raise ValueError(f'missing terminal checkpoint: {checkpoint}')
    (output/'DONE.json').write_text(json.dumps(dict(arms=len(plan),steps=a.steps,smoke=a.steps==20))+'\n')


if __name__=='__main__':main()
