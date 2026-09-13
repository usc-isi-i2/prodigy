"""Cached-score comparison of successful/unsuccessful support draws; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def describe(x):
    x=x.dropna()
    return dict(n=len(x),higher=int((x>1e-6).sum()),lower=int((x< -1e-6).sum()),
        tied=int((x.abs()<=1e-6).sum()),mean=float(x.mean()) if len(x) else None,
        median=float(x.median()) if len(x) else None)


def paired(d,col):
    p=d.groupby(['case','native_correct'])[col].mean().unstack().dropna()
    return describe(p[1]-p[0]) if {0,1}.issubset(p.columns) else describe(pd.Series(dtype=float))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    a=p.parse_args()
    receipt=json.loads(a.receipt.read_text());assert receipt['complete']
    assert hashlib.sha256(a.heads.read_bytes()).hexdigest()==receipt['csv_sha256']
    d=pd.read_csv(a.heads);base=d[d.condition=='original'].set_index('case')
    g=pd.read_csv(a.geometry);g=g[(g.target=='cp_hk')&(g.model=='hk')]
    result=dict(protocol='hk_rescued_draw_cosine_v1',head_csv_sha256=receipt['csv_sha256'],
        geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        definition='Mean cosine between fixed pre-metagraph query and three support embeddings. Rival supports are fixed, so differences of stored margins equal differences of true-class similarities.',
        tolerance=1e-6,conditions={})
    for cond in ['alternative','same_context']:
        s=d[(d.condition==cond)&(d.cohort=='native_failed')].copy()
        assert len(s)==500 and s.groupby('case').size().eq(5).all()
        s['learned']=s.mean_cosine_margin/receipt['logit_scale']
        s['delta']=(s.mean_cosine_margin-s['case'].map(base.mean_cosine_margin))/receipt['logit_scale']
        assert s['case'].map(base.native_correct).eq(0).all()
        rescued=s[s.native_correct==1]
        r=dict(rescued_queries=int(rescued.case.nunique()),rescued_draws_vs_original=describe(rescued.delta),
            within_query_successful_minus_unsuccessful=paired(s,'learned'))
        s=s.merge(g,on=['case','condition','draw','cohort'],validate='one_to_one')
        assert len(s)==500 and s.paired_all_valid.notna().all()
        valid=s[s.paired_all_valid];rescued=valid[valid.native_correct==1]
        r['matched_raw_valid_subset']=dict(draws=len(valid),
            rescued_draws_learned=describe(rescued.delta),rescued_draws_raw=describe(rescued.margin_change),
            within_query_learned=paired(valid,'learned'),within_query_raw=paired(valid,'new_margin'))
        result['conditions'][cond]=r
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
