"""Read-only corrected NM query/support/episode audit; prints aggregates only."""
import json, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('NM_AUDIT_ROOT', '/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908'))
def describe(x):
    x=np.asarray(x,dtype=float)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,
            'quantiles':{str(q):float(np.quantile(x,q)) for q in [0,.1,.25,.5,.75,.9,1]} if len(x) else {}}
def corr(a,b):
    return float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)),method='spearman')) if len(a)>2 else None
out={'protocol':'canonical_nm_query_support_episode_v1','seed':827,'permutations':200,'targets':{}}
for target in ['ukr_rus_twitter','cp_hk_twitter']:
    path=ROOT/target/'paired_cluster_queries_private.tsv'
    raw=path.read_bytes()
    d=pd.read_csv(path,sep='\t')
    assert not d.duplicated(['split','episode','sample','query']).any()
    d['ambiguous']=d.groupby(['split','episode','query']).anchor.transform('nunique')>1
    d['degree_group']=pd.cut(d.query_degree,[-1,9,99,999,9999,99999,np.inf],labels=False)
    n=d.groupby(['split','query']).query.transform('size')
    d['frequency_group']=pd.cut(n,[0,1,4,19,np.inf],labels=False)
    for split,s in d.groupby('split'):
        assert len(s)==61440 and s.episode.nunique()==512
        assert (s.groupby('episode').size()==120).all()
        assert (s.groupby(['episode','anchor']).size()==4).all()
        for k in range(3):
            assert (s.groupby(['episode','anchor'])['true_support_'+str(k)].nunique()==1).all()
    val=d[d.split=='val'].copy(); test=d[d.split=='test'].copy()
    result={'input_sha256':hashlib.sha256(raw).hexdigest(),'models':{}}
    for model in ['ukr','hk']:
        col=model+'_correct'
        vq=val.groupby('query')[col].agg(['mean','size'])
        tq=test.groupby('query')[col].agg(['mean','size'])
        common=vq.join(tq,lsuffix='_val',rsuffix='_test',how='inner')
        stable=common[(common.size_val>=5)&(common.size_test>=5)]
        repeated=tq[tq['size']>=5]
        # Validation-defined cohorts, evaluated on test; selection never uses test outcomes.
        frequent=vq[vq['size']>=5]
        query_cohorts={}
        for label,ids in [('val_hard',frequent.index[frequent['mean']<=.2]),
                          ('val_easy',frequent.index[frequent['mean']>=.8])]:
            z=test[test['query'].isin(ids)]
            query_cohorts[label]={'selected_val_nodes':len(ids),'test_nodes':int(z['query'].nunique()),'test_occurrences':len(z),'test_accuracy':float(z[col].mean()) if len(z) else None}
        query={'repeated_5plus_nodes':len(repeated),'repeated_accuracy':describe(repeated['mean']),
               'always_wrong_5plus_nodes':int((repeated['mean']==0).sum()),
               'always_correct_5plus_nodes':int((repeated['mean']==1).sum()),
               'shared_val_test_5plus_nodes':len(stable),'val_test_spearman':corr(stable.mean_val,stable.mean_test),
               'validation_selected':query_cohorts}
        # Residual correctness relative to the same query in OTHER episodes.
        # Sparse queries shrink toward coarse degree/ambiguity/frequency stratum mean.
        supports={}
        enriched={}
        for split,s in [('val',val.copy()),('test',test.copy())]:
            groups=['degree_group','ambiguous','frequency_group']
            baseline=s.groupby(groups,observed=True)[col].transform('mean')
            total=s.groupby('query')[col].transform('sum')
            count=s.groupby('query')[col].transform('size')
            esum=s.groupby(['episode','query'])[col].transform('sum')
            ecount=s.groupby(['episode','query'])[col].transform('size')
            s['expected']=(total-esum+10*baseline)/(count-ecount+10)
            s['residual']=s[col]-s.expected
            cl=s.groupby(['episode','anchor']).agg(accuracy=(col,'mean'),residual=('residual','mean'),
                s0=('true_support_0','first'),s1=('true_support_1','first'),s2=('true_support_2','first')).reset_index()
            long=cl.melt(id_vars=['episode','anchor','accuracy','residual'],value_vars=['s0','s1','s2'],value_name='support')
            assert len(long)==46080
            supports[split]=long.groupby('support').agg(exposures=('accuracy','size'),accuracy=('accuracy','mean'),residual=('residual','mean'))
            enriched[split]=(s,cl)
        eligible=supports['val'][supports['val'].exposures>=10]
        k=max(1,int(np.ceil(len(eligible)*.2)))
        # Fixed low residual tail, deterministic support-ID tie-breaking.
        bad=set(eligible.sort_values(['residual'],kind='stable').head(k).index) if len(eligible) else set()
        support_common=supports['val'].join(supports['test'],lsuffix='_val',rsuffix='_test',how='inner')
        support_common=support_common[(support_common.exposures_val>=10)&(support_common.exposures_test>=10)]
        s,cl=enriched['test']
        s['selected_support']=s[['true_support_0','true_support_1','true_support_2']].isin(bad).any(axis=1)
        support_groups={}
        for flag,z in s.groupby('selected_support'):
            support_groups[str(bool(flag))]={'occurrences':len(z),'nodes':int(z['query'].nunique()),'accuracy':float(z[col].mean()),'query_adjusted_residual':float(z.residual.mean())}
        matched=s.groupby(['query','selected_support'])[col].agg(['mean','size']).unstack('selected_support')
        if False in matched['mean'].columns and True in matched['mean'].columns:
            both=matched.dropna()
            differences=both['mean'][True]-both['mean'][False]
            matched_summary=describe(differences)
        else: matched_summary=describe([])
        pair=s.groupby(['query','anchor','selected_support'])[col].mean().unstack('selected_support')
        if False in pair.columns and True in pair.columns:
            pair=pair.dropna()
            pair_summary=describe(pair[True]-pair[False])
        else: pair_summary=describe([])
        support={'within_same_query_anchor_selected_minus_other_accuracy':pair_summary,'val_eligible_supports_10plus':len(eligible),'val_selected_low_residual_supports':len(bad),
                 'common_supports_10plus':len(support_common),
                 'val_test_accuracy_spearman':corr(support_common.accuracy_val,support_common.accuracy_test),
                 'val_test_residual_spearman':corr(support_common.residual_val,support_common.residual_test),
                 'test_exposure_groups':support_groups,'within_same_query_selected_minus_other_accuracy':matched_summary}
        # Episode difficulty and class-wide failures.
        ep=s.groupby('episode').agg(accuracy=(col,'mean'),expected=('expected','mean'),residual=('residual','mean'),
            ambiguity=('ambiguous','mean'),log_degree=('query_degree',lambda x:np.log1p(x).mean()))
        class_counts=s.groupby(['episode','anchor'])[col].sum()
        # Shuffle correctness within each query: preserve query frequency and exact total correctness.
        # Singleton outcomes cannot move; test checks extra clustering over this fixed query mix.
        order=np.argsort(s['query'].to_numpy(),kind='stable')
        sorted_queries=s['query'].to_numpy()[order]
        bounds=np.r_[0,np.flatnonzero(sorted_queries[1:]!=sorted_queries[:-1])+1,len(s)]
        groups=[order[a:b] for a,b in zip(bounds[:-1],bounds[1:]) if b-a>1]
        episode_codes=pd.factorize(s.episode)[0]
        y=s[col].to_numpy(dtype=float)
        rng=np.random.default_rng(827)
        null=[]
        null_class_wrong=[]
        class_codes=pd.factorize(pd.MultiIndex.from_frame(s[['episode','anchor']]))[0]
        for rep in range(200):
            yp=y.copy()
            for ix in groups: yp[ix]=rng.permutation(y[ix])
            acc=np.bincount(episode_codes,weights=yp)/120
            null.append(float(np.std(acc)))
            null_class_wrong.append(float((np.bincount(class_codes,weights=yp)==0).mean()))
        observed=float(ep.accuracy.std(ddof=0))
        episode={'accuracy':describe(ep.accuracy),'query_adjusted_residual':describe(ep.residual),
                 'all_wrong_class_fraction':float((class_counts==0).mean()),
                 'within_query_permutation_all_wrong_class_fraction':describe(null_class_wrong),
                 'all_correct_class_fraction':float((class_counts==4).mean()),
                 'accuracy_vs_log_degree_spearman':corr(ep.accuracy,ep.log_degree),
                 'accuracy_vs_ambiguity_spearman':corr(ep.accuracy,ep.ambiguity),
                 'observed_accuracy_std':observed,'within_query_permutation_std':describe(null),
                 'permutation_upper_tail_fraction':float((1+np.sum(np.asarray(null)>=observed))/201)}
        result['models'][model]={'test_accuracy':float(test[col].mean()),'query':query,'support':support,'episode':episode}
    ep=test.groupby('episode')[['ukr_correct','hk_correct']].mean()
    result['between_model_episode_accuracy_spearman']=corr(ep.ukr_correct,ep.hk_correct)
    out['targets'][target]=result
print(json.dumps(out,allow_nan=False))
