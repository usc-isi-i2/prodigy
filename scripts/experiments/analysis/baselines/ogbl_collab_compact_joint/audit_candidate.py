"""Independent audit of frozen validation selection and final candidate scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_scores import read,sha,fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--validation-panel',type=Path,required=True)
    p.add_argument('--test-panel',type=Path,required=True)
    args=p.parse_args(); e=args.evidence; runtime=args.runtime
    va=np.load(args.validation_panel); te=np.load(args.test_panel)
    frozen=read(e/'frozen.json'); prepared=read(e/'prepared.json')
    def cut(x): return np.sort(x)[-50]
    def metric(p,n): return float(np.mean(p>cut(n)))
    def keys(edges): return np.minimum(edges[:,0],edges[:,1])*235868+np.maximum(edges[:,0],edges[:,1])
    overlap={}
    for year in (2017,2018):
        tr=np.load(runtime/f'train{year}.npz')
        expected=next(v for k,v in prepared['files'].items() if k.endswith(f'train{year}.npz'))
        assert sha(runtime/f'train{year}.npz')==expected
        assert len(np.unique(keys(tr['neg'])))==100000
        assert not np.intersect1d(keys(tr['pos']),keys(tr['neg'])).size
        rng=np.random.default_rng(20260000+year); seen={}
        while len(seen)<100100:
            draws=rng.integers(0,235868,size=(100100,2),dtype=np.int64)
            values=keys(draws)
            values=values[~np.isin(values,keys(tr['pos']))]
            seen.update(dict.fromkeys(map(int,values)))
        candidates=np.array(list(seen)[:100100],dtype=np.int64)
        candidates=candidates[~np.isin(candidates,keys(va['neg']))][:100000]
        np.testing.assert_array_equal(keys(tr['neg']),candidates)
        overlap[str(year)]={label:int(len(np.intersect1d(keys(tr['neg']),keys(panel['neg']))))
                           for label,panel in [('validation',va),('test',te)]}
        assert all(v==0 for v in overlap[str(year)].values())
        if year==2018:
            np.testing.assert_array_equal(tr['pos'],va['pos'])
            np.testing.assert_array_equal(tr['graph_edges'],va['graph_edges'])
            np.testing.assert_array_equal(tr['psymmetric'],va['psymmetric'])
    results=[]
    for seed in range(3):
        selected=read(e/f'select{seed}/results.json')
        history=read(e/f'select{seed}/history.json')
        assert selected==frozen['selections'][seed] and selected['exact_selected_replay']
        assert selected['history_sha256']==sha(e/f'select{seed}/history.json')
        assert [r['step'] for r in history]==list(range(50,2001,50))
        winner=max((dict(step=r['step'],**g) for r in history for g in r['grid']),key=lambda g:g['hits'])
        best=selected['best']
        assert all(best[k]==v for k,v in winner.items())
        path=runtime/f'select{seed}/best_scores.npz'
        assert sha(path)==selected['best_score_sha256']
        raw=dict(np.load(path))
        prefix='official_b' if best['base']=='validation2018' else 'b'
        assert best['scale']==cut(va[prefix+'n']) and best['offset']==cut(raw['n'])
        def fuse(panel,scores):
            return {s:np.maximum(panel[prefix+s]/best['scale'],best['alpha']*np.exp(np.clip(scores[s].astype(np.float64)-best['offset'],-30,30)))
                    if best['alpha'] else panel[prefix+s]/best['scale'] for s in ('p','n')}
        assert metric(**fuse(va,raw))==best['hits']
        final=read(e/f'refit{seed}/results.json')
        assert final['complete'] and final['steps']==best['step'] and final['frozen_settings']==best
        assert final['frozen_sha256']==sha(e/'frozen.json')
        assert final['test_panel_sha256']==sha(args.test_panel)
        assert final['history_sha256']==sha(e/f'refit{seed}/history.json')
        path=runtime/f'refit{seed}/test_scores.npz'
        assert sha(path)==final['test_score_sha256']
        raw=dict(np.load(path)); fusion=fuse(te,raw)
        assert metric(**raw)==final['joint_test_hits']
        assert metric(**fusion)==final['test_hits']
        for s in ('p','n'):
            edge=te['pos' if s=='p' else 'neg']
            assert np.isfinite(raw[s]).all() and len(raw[s])==len(edge)
            assert np.all(raw[s][edge[:,0]==edge[:,1]]==-1e9)
        base=te['official_bp']>cut(te['official_bn']); mask=fusion['p']>cut(fusion['n'])
        results.append(dict(seed=seed,validation=best,test_hits=final['test_hits'],
            joint_test_hits=final['joint_test_hits'],recovered=int((mask&~base).sum()),lost=int((base&~mask).sum())))
    values=[r['test_hits'] for r in results]
    output=dict(valid=True,complete=True,rows=results,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),
        negative_overlaps=overlap,total_inference_scalars=496448,test_used_for_selection=False,
        independent_negative_generation_verified=True,
        official_aadc_test=metric(te['official_bp'],te['official_bn']),
        qualification='Prior test exploration disclosed; acceptance unconfirmed. Selected raw validation scores and all final metrics independently audited; full selection grid checked from history, not replayed from every checkpoint.')
    (e/'audit_summary.json').write_text(json.dumps(output,indent=2)+'\n'); print(json.dumps(output,indent=2))


if __name__=='__main__': main()
