"""Aggregate fixed-bank HK support selection; private case IDs stay outside repo."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def aggregate(root):
    receipt=json.loads((root/'receipt.json').read_text())
    path=root/'selection_results_private.csv'
    assert receipt['complete'] and hashlib.sha256(path.read_bytes()).hexdigest()==receipt['csv_sha256']
    d=pd.read_csv(path)
    assert len(d)==1600 and not d.duplicated(['case','method','draw']).any()
    result={'receipt':receipt,'methods':{}}
    for method,s in d.groupby('method'):
        out={}
        for cohort,z in [('all',s),*list(s.groupby('cohort')),('previously_persistent',s[s.previously_persistent])]:
            out[cohort]={'draws':len(z),'cases':int(z.case.nunique()),
                'correct':int(z.native_correct.sum()),'accuracy':float(z.native_correct.mean()),
                'mean_nll':float(z.native_nll.mean()),
                'mean_selected_cosine':float(z.selected_mean_cosine.mean()),
                'cases_correct_at_least_once':int(z.groupby('case').native_correct.max().sum())}
        result['methods'][method]=out
    near=d[d.method=='nearest'].set_index('case')
    rand=d[d.method=='random'].groupby('case').native_correct.mean()
    result['nearest_minus_random_per_case_accuracy']={cohort:float((z.native_correct-rand.loc[z.index]).mean()) for cohort,z in near.groupby('cohort')}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(aggregate(a.input_root),indent=2,allow_nan=False))
