"""Join fixed-query support interventions to complete-input geometry; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def norm_mean(x):
    v=x.mean(axis=0)
    n=np.linalg.norm(v)
    return v/n if n>0 else np.zeros_like(v), bool(n>0)


def score(q,s):
    return float(np.dot(q,np.stack(s).mean(axis=0)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2'))
    p.add_argument('--intervention',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_support_resampling_20260908_v2'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    receipt=json.loads((args.inputs/'receipt.json').read_text());assert receipt['complete']
    # Memory-map the immutable source; access only the required feature rows.
    raw=torch.load(receipt['feature_artifact'],map_location='cpu',mmap=True,weights_only=False)
    features=raw['x'];assert list(features.shape)==receipt['feature_shape']
    results=pd.read_csv(args.intervention/'results_private.csv')
    assert len(results)==8000 and not results.duplicated(['target','model','case','condition','draw']).any()
    out=[];checks={};started=time.time()
    for target in ['ukr_rus','cp_hk']:
        info=receipt['targets'][target]
        assert info['original_hash']==info['reloaded_hash']==info['expected_hash']
        casespath=args.intervention/(target+'_cases_private.json')
        drawspath=args.intervention/(target+'_support_draws_private.pt')
        cases=json.loads(casespath.read_text())
        saved=torch.load(drawspath,map_location='cpu',weights_only=False)
        assert len(cases)==200 and len(saved['spec'])==2000 and len(saved['graphs'])==6000
        geometries=pd.read_csv(args.inputs/target/'query_geometry_private.csv').set_index(['episode','sample'])
        archives={};cache={};bases={};errors=[];feature_checks=0
        for c in cases:
            bi=c['batch'];ep=c['ep'];truth=c['truth'];sample=c['sample']
            if bi not in archives:
                path=args.inputs/target/f'batch_{bi:03d}_private.pt'
                expected=next(f['sha256'] for f in info['files'] if f['file']==path.name)
                assert digest(path)==expected
                archives[bi]=torch.load(path,map_location='cpu',weights_only=False)['batch'][0]
            g=archives[bi]
            def extract(slot):
                nonlocal feature_checks
                if (bi,slot) not in cache:
                    ids=g.global_node_ids[g.ptr[slot]:g.ptr[slot+1]]
                    ids=ids[ids>=0]
                    v,valid=norm_mean(features[ids].numpy())
                    cache[bi,slot]=(v,valid,set(ids.tolist()))
                    feature_checks+=len(ids)
                return cache[bi,slot]
            assert int(g.center_node_idx[sample])==int(c['query'])+info['source_node_offset']
            assert int(g.task_label_map[ep,truth])==int(c['anchor'])+info['source_node_offset']
            q,qvalid,qids=extract(sample)
            supports=[[extract(ep*210+label*7+j) for j in range(3)] for label in range(30)]
            sc=np.array([score(q,[v[0] for v in triple]) if qvalid and all(v[1] for v in triple) else np.nan for triple in supports])
            union=set().union(*(v[2] for v in supports[truth]))
            jac=len(qids&union)/len(qids|union)
            old=geometries.loc[(c['episode'],sample)]
            rival=np.delete(sc,truth);best=np.nanmax(rival) if np.isfinite(rival).any() else np.nan
            for value,reference in [(sc[truth],old.full_mean_true),(sc[truth]-best,old.full_mean_margin)]:
                assert np.isfinite(value)==np.isfinite(reference)
                if np.isfinite(value):errors.append(abs(value-reference));assert abs(value-reference)<1e-6
            assert bool(np.isfinite(sc).all())==bool(old.full_mean_all_valid)
            assert abs(jac-old.node_jaccard_true)<1e-12
            bases[c['case']]=dict(q=q,qvalid=qvalid,qids=qids,true=sc[truth],rival=best,
                margin=sc[truth]-best,all_valid=bool(np.isfinite(sc).all()),
                jac=jac,jac_rival=float(old.node_jaccard_true-old.node_jaccard_margin))
        assert len(bases)==200
        for spec in saved['spec']:
            c=cases[spec['case']];assert c['case']==spec['case']
            b=bases[spec['case']]
            triple=saved['graphs'][spec['start']:spec['start']+3]
            expected=c['trials'][spec['draw']][spec['condition']]
            assert [int(g.global_node_ids[0]) for g in triple]==expected
            vs=[];valid=[];node_sets=[]
            for g in triple:
                ids=g.global_node_ids;real=ids>=0
                # Saved intervention x must match the same backing features.
                assert torch.equal(g.x[real],features[ids[real]])
                v,ok=norm_mean(g.x[real].numpy());vs.append(v);valid.append(ok)
                node_sets.append(set(ids[real].tolist()))
            true=score(b['q'],vs) if b['qvalid'] and all(valid) else np.nan
            margin=true-b['rival']
            union=set().union(*node_sets);jac=len(b['qids']&union)/len(b['qids']|union)
            out.append(dict(target=target,case=c['case'],cohort=c['cohort'],condition=spec['condition'],draw=spec['draw'],
                query_degree=c['query_degree'],baseline_margin=b['margin'],new_margin=margin,
                margin_change=margin-b['margin'],baseline_all_valid=b['all_valid'],
                paired_all_valid=b['all_valid'] and all(valid) and b['qvalid'],
                baseline_jaccard_margin=b['jac']-b['jac_rival'],new_jaccard_margin=jac-b['jac_rival'],
                jaccard_margin_change=jac-b['jac']))
        checks[target]=dict(cases=200,geometry_draws=2000,original_feature_rows_accessed=feature_checks,
            max_baseline_geometry_error=max(errors),all_archives_match_receipt=True,
            all_saved_support_features_match_backing_graph=True,cases_sha256=digest(casespath),draws_sha256=digest(drawspath))
        print(target,'complete',flush=True)
        del saved,archives,cache,bases
    geometry=pd.DataFrame(out)
    assert len(geometry)==4000 and not geometry.duplicated(['target','case','condition','draw']).any()
    joined=results.merge(geometry,on=['target','case','cohort','condition','draw'],validate='many_to_one',how='left')
    assert len(joined)==8000 and joined.paired_all_valid.notna().all()
    joined.to_csv(args.out/'paired_geometry_private.csv',index=False)
    report=dict(complete=True,no_model_forwards=True,checks=checks,rows=len(joined),seconds=time.time()-started,
        results_sha256=digest(args.intervention/'results_private.csv'),
        input_receipt_sha256=digest(args.inputs/'receipt.json'),geometry_sha256=digest(args.out/'paired_geometry_private.csv'))
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',report['seconds'],flush=True)


if __name__=='__main__':main()
