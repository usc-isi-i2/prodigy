"""Prepare the existing replay harness for completed isolation checkpoints."""
import argparse
import json
from pathlib import Path
import sys

TARGETS=('election2020','ukr_rus_suspended','twibot20','covid_political','facebook_page_reference')


def prepare(root, output, gpu):
    if gpu not in range(4):raise ValueError('owned GPU0-3 required')
    done=json.loads((root/'DONE.json').read_text())
    if done['steps']!=2500 or done['smoke'] or done['arms']!=8:
        raise ValueError('full eight-arm training required for outcome evaluation')
    plan=json.loads((root/'plan.json').read_text())
    expected={(schedule,mode) for schedule in ('blocked','interleaved') for mode in ('native','joint','isolated','ridge_only')}
    if len(plan)!=8 or len({e['name'] for e in plan})!=8 or {(e['arm']['schedule'],e['mode']) for e in plan}!=expected:
        raise ValueError('complete unique objective/schedule inventory required')
    manifest=['model_id\tcheckpoint\tsources']
    for entry in plan:
        checkpoint=root/'state'/(entry['name']+'_isolation_v1')/'checkpoint'/'state_dict_2500.ckpt'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        manifest.append('\t'.join((entry['name'],str(checkpoint),','.join(entry['arm']['sources']))))
    commands=[]
    for stream,offset in [('original',0),('fresh',100003)]:
        for target in TARGETS:
            commands.append([sys.executable,'-B','-u','-m','scripts.experiments.setup.target_performance_mechanisms.replay',
                '--model-list',str(output/'models.tsv'),'--output',str(output/stream/target),
                '--datasets',target,'--variants','baseline','--device',str(gpu),'--threads','4',
                '--batch-count','32','--save-embeddings','--training-label-count','30',
                '--trace-parity-atol','0.00001','--eval-episode-seed-offset',str(offset)])
    return '\n'.join(manifest)+'\n',commands


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=range(4),required=True)
    p.add_argument('--write-plan',action='store_true')
    a=p.parse_args();manifest,commands=prepare(a.root.resolve(),a.output.resolve(),a.gpu)
    print(json.dumps(commands,indent=2))
    if a.write_plan:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'models.tsv').write_text(manifest)
        (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        (a.output/'protocol_note.txt').write_text('Reuses native deployment replay construction (objective defaults to native); training-mode provenance remains in parent plan/checkpoints. Original/fresh streams are reused for fixed comparison, not newly unseen data. No target-based hyperparameter tuning.\n')
