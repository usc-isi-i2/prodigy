"""Aggregate complete sampled-input geometry without publishing node identities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def describe(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return dict(n=len(s), mean=float(s.mean()) if len(s) else None,
                median=float(s.median()) if len(s) else None)


def summarize(root, reference_root=None):
    receipt = json.loads((root/'receipt.json').read_text())
    assert receipt['complete']
    result = dict(protocol='canonical_nm_complete_input_geometry_v1', receipt=receipt, targets={})
    for target,native in [('ukr_rus','ukr'),('cp_hk','hk')]:
        path = root/target/'query_geometry_private.csv'
        d = pd.read_csv(path)
        assert len(d)==61440 and not d.duplicated(['episode','sample']).any()
        r = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), models={})
        if reference_root is not None:
            refpath=reference_root/(native+'.tsv')
            ref=pd.read_csv(refpath,sep='\t')
            ref=ref[ref.split=='test'].merge(d,on=['episode','sample'],validate='one_to_one',suffixes=('_old',''))
            assert len(ref)==61440
            for key in ['query','anchor','ukr_correct','hk_correct']:
                assert ref[key].equals(ref[key+'_old']),key
            valid=ref.cos_true_support.notna() & ref.center_true.notna()
            error=float((ref.loc[valid,'cos_true_support']-ref.loc[valid,'center_true']).abs().max())
            assert error<1e-6,error
            r['independent_center_check']=dict(valid_rows=int(valid.sum()),max_absolute_error=error,
                reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                all_query_anchor_ids_and_both_model_outcomes_match=True)
        d['degree_group'] = pd.cut(d.query_degree,[-1,9,99,999,9999,99999,np.inf],labels=False)
        d['ambiguous'] = d.groupby(['episode','query']).anchor.transform('nunique')>1
        for model in ['ukr','hk']:
            col=model+'_correct'
            metrics = ['query_nodes','query_edges','query_zero_fraction','true_support_nodes_mean',
                       'true_shared_nodes','query_center_in_true_support','query_center_in_any_rival_support']
            metrics += [c for c in d if c.endswith(('_true','_margin','_true_support_pair_cosine'))]
            cell = dict(accuracy=float(d[col].mean()),outcomes={})
            for outcome,s in d.groupby(col):
                cell['outcomes'][str(outcome)] = dict(rows=len(s), descriptors={m:describe(s[m]) for m in metrics})
            cell['separation']={}
            for name in ['center','full_mean','context_mean','center_context_concat','node_jaccard']:
                # Fair all-candidate comparisons require valid query and all supports.
                s=d[d[name+'_all_valid']].copy()
                margin=name+'_margin'
                s['separation']=np.where(s[margin]>1e-7,'true_strictly_best',
                    np.where(s[margin]<-1e-7,'rival_strictly_better','tied'))
                desc=dict(valid_rows=len(s), groups={})
                for g,z in s.groupby('separation'):
                    desc['groups'][g]=dict(rows=len(z),accuracy=float(z[col].mean()),
                        fraction_of_valid_errors=float((z[col]==0).sum()/max(1,(s[col]==0).sum())))
                desc['single_anchor_groups']={g:dict(rows=len(z),accuracy=float(z[col].mean()))
                    for g,z in s[~s.ambiguous].groupby('separation')}
                # Equal-node comparison among nodes occurring both correct/wrong.
                paired=s.groupby(['query',col])[margin].mean().unstack(col).dropna()
                delta=paired[1]-paired[0] if {0,1}.issubset(paired.columns) else pd.Series(dtype=float)
                desc['within_query_correct_minus_wrong_margin']=describe(delta)
                desc['degree_strata']=[]
                for deg,z in s.groupby('degree_group'):
                    for g,y in z.groupby('separation'):
                        desc['degree_strata'].append(dict(degree_bin=int(deg),group=g,rows=len(y),accuracy=float(y[col].mean())))
                cell['separation'][name]=desc
            r['models'][model]=cell
        result['targets'][target]=r
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,help='Optional original ukr.tsv/hk.tsv prediction tables for independent checks')
    args=p.parse_args()
    print(json.dumps(summarize(args.input_root,args.reference_root),indent=2,allow_nan=False))
