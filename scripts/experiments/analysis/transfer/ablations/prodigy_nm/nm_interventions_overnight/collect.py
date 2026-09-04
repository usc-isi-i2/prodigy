"""Export small, provenance-bearing evidence from cluster run directories."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.evaluate import discover


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=sorted({k for row in rows for k in row})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows)


def main():
    a=argparse.ArgumentParser();a.add_argument('--run-dirs',nargs='+',required=True)
    a.add_argument('--eval-dirs',nargs='*',default=[]);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();jobs=discover(args.run_dirs);args.output.mkdir(parents=True,exist_ok=True)
    records=[];curves=[];vals=[];cells=[]
    for job in jobs:
        p=job['params'];selection=job['selection'];model=p['prefix']
        state=Path(job['state']);hist=json.loads((state/'validation_history.json').read_text())
        records.append(dict(model_id=model,params=p,selection=selection,validation_history=hist,
                            validation_protocol=json.loads((state/'validation_protocol.json').read_text())))
        for line in (state/'training_curve.jsonl').read_text().splitlines():
            r=json.loads(line);r.pop('exposure',None);curves.append(dict(model_id=model,**r))
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
    write_csv(args.output/'nm_results.csv',cells)
    (args.output/'coverage.json').write_text(json.dumps(dict(completed_models=len(records),nm_cells=len(cells),fingerprints=fingerprints),indent=2)+'\n')
    print(f'Exported {len(records)} completed models; {len(cells)} NM cells')

if __name__=='__main__':main()
