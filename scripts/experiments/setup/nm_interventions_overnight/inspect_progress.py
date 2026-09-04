"""Read-only compact live curve/status report for manual inspection."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--snapshot',action='store_true',help='Include full curves for local visual inspection')
    args=p.parse_args()
    counts={};rows=[]
    for path in sorted(args.root.glob('train_*/job_*/result.json')):
        result=json.loads(path.read_text());status=result['status'];counts[status]=counts.get(status,0)+1
        config=json.loads((path.parent/'effective_config.json').read_text())
        state=Path(config['state_dir'])/config['exp_name']
        row=dict(model=config['prefix'],status=status,gpu=result['physical_gpu'])
        curve=state/'training_curve.jsonl'
        if curve.exists():
            lines=curve.read_text().splitlines()
            if args.snapshot:
                records=[]
                for line in lines:
                    try: records.append(json.loads(line))
                    except json.JSONDecodeError: pass  # Last line may still be being written.
                row['training_curve']=records
            try:last=json.loads(lines[-1])
            except (IndexError,json.JSONDecodeError):last={}
            row.update(step=last.get('step'),loss=last.get('loss'),curve_age_seconds=round(time.time()-curve.stat().st_mtime))
        history=state/'validation_history.json'
        if history.exists():
            checks=json.loads(history.read_text());row['validation']=[(r['step'],round(r['macro_roc_auc'],5)) for r in checks]
            if args.snapshot:row['validation_history']=checks
        selection=state/'selection.json'
        if selection.exists():
            chosen=json.loads(selection.read_text());row['selected_step']=chosen['best_step'];row['stop']=chosen.get('stop_reason')
        rows.append(row)
    print(json.dumps(dict(captured_at_unix=time.time(),counts=counts,models=rows),indent=2))
if __name__=='__main__':main()
