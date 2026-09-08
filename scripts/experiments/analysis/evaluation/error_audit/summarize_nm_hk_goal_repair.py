"""Independent row/hash checks and compact aggregates for the bounded NM repair."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root, canonical_root):
    receipt = json.loads((root/'receipt.json').read_text())
    assert receipt['complete'] and not receipt['fitting'] and not receipt['encoding']
    out = {k:v for k,v in receipt.items() if k not in {'targets','selected'}}
    out['receipt_sha256'] = digest(root/'receipt.json')
    out['private_output_path'] = str(root)
    out['targets'] = {}
    for target,info in receipt['targets'].items():
        path = root/(target+'_predictions_private.csv')
        assert digest(path)==info['csv_sha256']
        d = pd.read_csv(path)
        oldpath = canonical_root/target/'stage_predictions_private.csv'
        assert digest(oldpath)==info['canonical_reference_sha256']
        old = pd.read_csv(oldpath)
        assert len(d)==122880 and not d.duplicated(['model','episode','sample']).any()
        j = d.merge(old[['model','episode','sample','query','anchor','native_prediction',
            'native_correct','encoded_prediction','encoded_correct']],
            on=['model','episode','sample'],validate='one_to_one',suffixes=('','_old'))
        for col in ['query','anchor','native_prediction','native_correct']:
            assert j[col].equals(j[col+'_old'])
        assert np.isfinite(d.select_dtypes('number')).all().all()
        verified={}
        for model,s in j.groupby('model'):
            assert len(s)==61440 and s.episode.nunique()==512
            assert s.groupby('episode').size().eq(120).all()
            expected=info['models'][model]['all']
            for head in ['geometry','residual']:
                assert s[head+'_correct'].eq(s[head+'_prediction'].eq(s.truth)).all()
            for head in ['native','geometry','residual']:
                now=expected[head]
                assert int(s[head+'_correct'].sum())==now['correct']
                assert int((s[head+'_correct'] & ~s.native_correct).sum())==now['recovered']
                assert int((~s[head+'_correct'] & s.native_correct).sum())==now['lost']
                assert now['correct']==int(s.native_correct.sum())+now['recovered']-now['lost']
                assert abs(float(s.groupby('query')[head+'_correct'].mean().mean())-now['node_weighted_accuracy'])<1e-12
            e=s[~s.native_correct]
            verified[model]=dict(geometry_prediction_differences=int(s.geometry_prediction.ne(s.encoded_prediction).sum()),
                geometry_correctness_differences=int(s.geometry_correct.ne(s.encoded_correct).sum()),
                repaired_despite_both_original_heads_wrong=int((~e.geometry_correct & e.residual_correct).sum()),
                recovered_fraction_of_native_errors=float(e.residual_correct.mean()),
                lost_fraction_of_native_successes=float((~s[s.native_correct].residual_correct).mean()))
        numeric=info['numerical']
        section={k:v for k,v in info.items() if k!='numerical'}
        section['verification']=verified
        section['numerical_summary']=dict(episode_cells=len(numeric),
            max_geometry_error=max(v['max_geometry_error'] for v in numeric),
            geometry_argmax_differences=sum(v['geometry_argmax_differences'] for v in numeric),
            canonical_native_argmax_differences=sum(v['canonical_native_argmax_differences'] for v in numeric),
            residual_ties_1e6=sum(v['residual_ties_1e6'] for v in numeric))
        native,foreign=('hk','ukr') if target=='cp_hk' else ('ukr','hk')
        section['source_gaps']={head:{weight:section['models'][native]['all'][head][weight]-section['models'][foreign]['all'][head][weight]
            for weight in ['accuracy','node_weighted_accuracy']} for head in ['native','geometry','residual']}
        out['targets'][target]=section
    path=root/'selected_predictions_private.csv'
    assert digest(path)==receipt['selected']['csv_sha256']
    d=pd.read_csv(path)
    assert len(d)==1800 and not d.duplicated(['case','condition','draw']).any()
    out['selected']=receipt['selected']
    for condition,s in d.groupby('condition'):
        for cohort,frame in [('failed',s[s.cohort=='native_failed']),('correct',s[s.cohort=='native_correct']),('persistent',s[s.persistent])]:
            exp=out['selected']['groups'][condition][cohort]
            assert len(frame)==exp['rows'] and frame.case.nunique()==exp['cases']
            for head in ['native','geometry','residual']:
                x=exp[head]
                assert int(frame[head+'_correct'].sum())==x['correct']
                assert int((~frame.native_correct & frame[head+'_correct']).sum())==x['recovered']
                assert int((frame.native_correct & ~frame[head+'_correct']).sum())==x['lost']
    return out


def verify_selected_scores(path, root, result):
    """Independent float64 NumPy formulation using the prior small score export."""
    cache=np.load(path)
    saved=pd.read_csv(root/'selected_predictions_private.csv').set_index(['case','condition','draw'])
    checked=0
    for kind,key in [('original','original_cache_sha256'),('extremes','bank_sha256')]:
        assert str(cache[kind+'_cache_sha256'])==result['selected'][key]
        logits=cache[kind+'_logits'].astype(np.float64)
        keys=json.loads(str(cache[kind+'_keys']))
        heads=json.loads(str(cache[kind+'_heads']))
        z=[]
        for head in ['native','mean_cosine']:
            x=logits[:,heads.index(head)]
            x=x-x.mean(1,keepdims=True)
            z.append(x/np.maximum(np.sqrt((x*x).mean(1,keepdims=True)),1e-8))
        pred=((z[0]+z[1])/2).argmax(1)
        for i,k in enumerate(keys):
            if kind=='original' and k['condition']!='original':
                continue
            condition='original' if kind=='original' else k['method']
            assert pred[i]==saved.loc[(k['case'],condition,k['draw']),'residual_prediction']
            checked+=1
    assert checked==len(saved)==1800
    return dict(predictions_checked=checked,prediction_differences=0,input_sha256=digest(path),
        method='Independent float64 NumPy score standardization and averaging')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--canonical-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected-logits',type=Path)
    a=p.parse_args()
    result=aggregate(a.input_root,a.canonical_root)
    if a.selected_logits:
        result['selected_numpy_verification']=verify_selected_scores(a.selected_logits,a.input_root,result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for target,t in result['targets'].items():
        for model,m in t['models'].items():
            s=m['all'];print(target,model,'native',s['native']['correct'],'residual',s['residual']['correct'],
                'recovered',s['residual']['recovered'],'lost',s['residual']['lost'])
