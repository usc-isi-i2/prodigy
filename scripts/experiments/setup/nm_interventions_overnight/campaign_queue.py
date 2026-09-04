"""Resume at completed-model granularity; never relabel partial runs as complete."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import generate, HERE


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[1,2,3]);p.add_argument('--models-per-gpu',type=int,default=4)
    p.add_argument('--arms',nargs='+');p.add_argument('--rungs',nargs='+',type=int)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    rows=generate(arms=args.arms,rungs=args.rungs)
    done={}
    for run in sorted(args.root.glob('train_*')):
        for result_path in run.glob('job_*/result.json'):
            result=json.loads(result_path.read_text())
            if result.get('status')!='complete':continue
            config=json.loads((result_path.parent/'effective_config.json').read_text())
            selection_path=Path(result['checkpoint_dir']).parent/'selection.json'
            if not selection_path.exists():continue
            selection=json.loads(selection_path.read_text())
            if selection.get('status')=='complete' and Path(selection['checkpoint']).exists():
                if config['prefix'] in done:raise ValueError('Duplicate completed model: '+config['prefix'])
                done[config['prefix']]=str(result_path)
    todo=[r for r in rows if f'nmi_{r["arm"]}_r{r["rung"]}_s0' not in done]
    print(f'{len(done)} completed; {len(todo)} requested models pending',flush=True)
    if not todo:return
    run=args.root/time.strftime('train_%Y%m%d_%H%M%S')
    cmd=[sys.executable,'-u','experiments/run_shared_graph.py','--configs',*[r['config'] for r in todo],
         '--gpus',*[str(g) for g in args.gpus],'--models-per-gpu',str(args.models_per_gpu),
         '--worker-budget',str(len(args.gpus)*args.models_per_gpu*4),'--threads-per-model','2','--run-dir',str(run)]
    print(' '.join(cmd),flush=True)
    if args.dry_run:return
    args.root.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
