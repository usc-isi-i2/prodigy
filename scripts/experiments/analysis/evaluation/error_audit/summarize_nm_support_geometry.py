"""Aggregate paired support geometry; draws are repeated observations within cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def desc(x):
    x=pd.to_numeric(x,errors='coerce').dropna()
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
                median=float(x.median()) if len(x) else None)


def aggregate(root):
    receipt=json.loads((root/'receipt.json').read_text());assert receipt['complete']
    assert hashlib.sha256((root/'paired_geometry_private.csv').read_bytes()).hexdigest()==receipt['geometry_sha256']
    d=pd.read_csv(root/'paired_geometry_private.csv')
    assert len(d)==8000 and not d.duplicated(['target','model','case','condition','draw']).any()
    result=dict(protocol='paired_support_geometry_v1',receipt=receipt,targets={})
    for target,model in [('ukr_rus','ukr'),('cp_hk','hk')]:
        s=d[(d.target==target)&(d.model==model)].copy()
        assert len(s)==2000
        assert s.groupby(['condition','case']).size().eq(5).all()
        assert s.loc[s.cohort=='native_failed','baseline_correct'].eq(0).all()
        assert s.loc[s.cohort=='native_correct','baseline_correct'].eq(1).all()
        s['flipped']=s.correct!=s.baseline_correct
        r={}
        for condition,c in s.groupby('condition'):
            section={}
            for metric,mask,old,new,delta in [
                ('full_mean',c.paired_all_valid,'baseline_margin','new_margin','margin_change'),
                # Sensitivity only: ignore fixed rival classes with undefined
                # zero-vector cosine, but require finite old/new true-class scores.
                ('full_mean_available_rivals',np.isfinite(c.margin_change),'baseline_margin','new_margin','margin_change'),
                ('node_jaccard',pd.Series(True,index=c.index),'baseline_jaccard_margin','new_jaccard_margin','jaccard_margin_change')]:
                a=c[mask].copy()
                stats=dict(total_draws=len(c),valid_draws=len(a),valid_cases=int(a.case.nunique()),
                           excluded_flips=int(c.loc[~mask,'flipped'].sum()),cohorts={})
                for cohort,z in a.groupby('cohort'):
                    groups={}
                    for outcome,y in z.groupby('correct'):
                        groups[str(outcome)]=dict(draws=len(y),cases=int(y.case.nunique()),
                            baseline_margin=desc(y[old]),new_margin=desc(y[new]),change=desc(y[delta]),
                            improved=int(y[delta].gt(1e-6).sum()),worsened=int(y[delta].lt(-1e-6).sum()),
                            numerically_unchanged=int(y[delta].abs().le(1e-6).sum()),
                            true_strictly_best_after=int(y[new].gt(1e-7).sum()),
                            rival_strictly_better_after=int(y[new].lt(-1e-7).sum()))
                    paired=z.groupby(['case','correct'])[new].mean().unstack('correct').dropna()
                    dif=paired[1]-paired[0] if {0,1}.issubset(paired.columns) else pd.Series(dtype=float)
                    groups['within_case_correct_minus_wrong']=dict(**desc(dif),positive=int(dif.gt(1e-6).sum()),
                        negative=int(dif.lt(-1e-6).sum()),numerically_tied=int(dif.abs().le(1e-6).sum()))
                    stats['cohorts'][cohort]=groups
                stats['small_change_sensitivity']=[]
                for eps in [1e-4,1e-3,1e-2]:
                    small=a[a[delta].abs()<=eps]
                    stats['small_change_sensitivity'].append(dict(threshold=eps,draws=len(small),
                        flips=int(small.flipped.sum()),flip_fraction=float(small.flipped.mean()) if len(small) else None,
                        rescues=int(((small.baseline_correct==0)&(small.correct==1)).sum()),
                        breaks=int(((small.baseline_correct==1)&(small.correct==0)).sum())))
                a['transition']=pd.Series(np.where(a[old]>1e-7,'true','rival_or_tie'),index=a.index)+'->'+np.where(a[new]>1e-7,'true','rival_or_tie')
                stats['transitions']={}
                for (cohort,transition),z in a.groupby(['cohort','transition']):
                    stats['transitions'][cohort+'/'+transition]=dict(draws=len(z),accuracy=float(z.correct.mean()))
                section[metric]=stats
            r[condition]=section
        result['targets'][target]=r
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(aggregate(args.input_root),indent=2,allow_nan=False))
