"""Aggregate frozen HK native/prototype head comparisons and loss changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def describe(x):
    x=pd.to_numeric(x,errors='coerce').dropna()
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(x.median()) if len(x) else None)


def correlation(x,y):
    s=pd.DataFrame({'x':x,'y':y}).dropna()
    if len(s)<3 or s.x.nunique()<2 or s.y.nunique()<2:return None
    return float(s.x.corr(s.y,method='spearman'))


def aggregate(root,geometry):
    receipt=json.loads((root/'receipt.json').read_text());assert receipt['complete']
    path=root/'head_comparison_private.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['csv_sha256']
    d=pd.read_csv(path);assert len(d)==2200 and not d.duplicated(['case','condition','draw']).any()
    base=d[d.condition=='original'].set_index('case');assert len(base)==200
    g=pd.read_csv(geometry);g=g[(g.target=='cp_hk')&(g.model=='hk')]
    result=dict(protocol='hk_frozen_head_comparison_v1',receipt=receipt,conditions={})
    for condition,s0 in d.groupby('condition'):
        section={}
        for cohort,s in [('all',s0),*list(s0.groupby('cohort'))]:
            r=dict(rows=len(s),heads={})
            for head in ['native','prototype','mean_cosine']:
                r['heads'][head]=dict(accuracy=float(s[head+'_correct'].mean()),nll=describe(s[head+'_nll']),
                    rank=describe(s[head+'_rank']))
                if condition!='original':
                    orig=s['case'].map(base[head+'_correct'])
                    origpred=s['case'].map(base[head+'_prediction'])
                    r['heads'][head].update(correctness_flips=int((orig!=s[head+'_correct']).sum()),
                        prediction_changes=int((origpred!=s[head+'_prediction']).sum()))
            r['native_vs_prototype']={f'{int(a)}_{int(b)}':len(z)
                for (a,b),z in s.groupby(['native_correct','prototype_correct'])}
            r['native_vs_mean_cosine']={f'{int(a)}_{int(b)}':len(z)
                for (a,b),z in s.groupby(['native_correct','mean_cosine_correct'])}
            if condition!='original':
                both=s['case'].map(base.native_correct).eq(1)&s['case'].map(base.prototype_correct).eq(1)
                z=s[both]
                r['originally_both_correct']=dict(draws=len(z),
                    native_break_prototype_holds=int(((z.native_correct==0)&(z.prototype_correct==1)).sum()),
                    prototype_break_native_holds=int(((z.native_correct==1)&(z.prototype_correct==0)).sum()))
                joined=s.merge(g,on=['case','condition','draw','cohort'],validate='one_to_one')
                assert len(joined)==len(s)
                joined['native_loss_change']=joined.native_nll-joined['case'].map(base.native_nll)
                joined['encoded_margin_change']=(joined.mean_cosine_margin-joined['case'].map(base.mean_cosine_margin))/receipt['logit_scale']
                v=joined[joined.paired_all_valid]
                r['loss_geometry']=dict(paired_valid_rows=len(v),
                    raw_margin_vs_native_loss_change=correlation(v.margin_change,v.native_loss_change),
                    encoded_margin_vs_native_loss_change=correlation(v.encoded_margin_change,v.native_loss_change),
                    all_rows_encoded_margin_vs_native_loss_change=correlation(joined.encoded_margin_change,joined.native_loss_change),
                    within_case_raw_correlations=describe(pd.Series([correlation(z.margin_change,z.native_loss_change) for _,z in v.groupby('case')],dtype=float)),
                    within_case_encoded_correlations=describe(pd.Series([correlation(z.encoded_margin_change,z.native_loss_change) for _,z in v.groupby('case')],dtype=float)))
                r['input_vs_encoded_separation']={}
                for label,z in v.groupby(v.new_margin>1e-7):
                    r['input_vs_encoded_separation'][str(bool(label))]=dict(rows=len(z),
                        encoded_true_strictly_best=int((z.mean_cosine_margin>1e-7*receipt['logit_scale']).sum()),
                        native_accuracy=float(z.native_correct.mean()),mean_cosine_accuracy=float(z.mean_cosine_correct.mean()))
            section[cohort]=r
        result['conditions'][condition]=section
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(aggregate(a.input_root,a.geometry),indent=2,allow_nan=False))
