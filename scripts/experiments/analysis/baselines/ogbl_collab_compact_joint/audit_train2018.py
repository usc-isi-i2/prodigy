"""Independent score-grid audit and summary for2018-trained joint campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_scores import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--test-panel',type=Path,required=True)
    p.add_argument('--train-panel',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    args = p.parse_args()
    panel = np.load(args.test_panel)
    train = np.load(args.train_panel)
    def keys(edges):
        return np.minimum(edges[:,0],edges[:,1])*235868+np.maximum(edges[:,0],edges[:,1])
    overlaps = {f'train_{a}_test_{b}':int(len(np.intersect1d(keys(train[a]),keys(panel[b]))))
                for a in ('pos','neg') for b in ('pos','neg')}
    np.testing.assert_array_equal(np.unique(keys(panel['graph_edges'])),
        np.union1d(keys(train['graph_edges']),keys(train['pos'])))
    def cut(x): return np.sort(x)[-50]
    def metric(p,n): return float(np.mean(p>cut(n)))
    winners, checked = [], 0
    for seed in range(3):
        e = args.evidence/f'seed{seed}'
        result, cfg, history, sources = [read(e/name) for name in ('results.json','protocol.json','history.json','sources.json')]
        assert result['complete'] and result['steps_run']==2000 and result['exact_selected_checkpoint_replay']
        assert result['history_sha256']==sha(e/'history.json')
        assert cfg['seed']==seed and cfg['test_used_for_selection']
        assert result['total_inference_scalars']==496448
        testsha = next(v for k,v in sources['files'].items() if k.endswith('year2019.npz'))
        assert testsha == sha(args.test_panel)
        trainsha = next(v for k,v in sources['files'].items() if k.endswith('year2018.npz'))
        assert trainsha == sha(args.train_panel)
        assert [r['step'] for r in history]==list(range(50,2001,50))
        best = None
        for r in history:
            path = args.runtime/f'seed{seed}'/f"scores{r['step']}.npz"
            assert sha(path)==r['score_sha256']
            scores = np.load(path)
            assert metric(scores['p'],scores['n'])==r['joint_hits']
            for side in ('p','n'):
                edge = panel['pos' if side=='p' else 'neg']
                assert len(edge)==len(scores[side]) and np.isfinite(scores[side]).all()
                assert (scores[side][edge[:,0]==edge[:,1]]==-1e9).all()
            expected = [(b,a) for b in cfg['bases'] for a in cfg['alphas']]
            assert [(g['base'],g['alpha']) for g in r['grid']]==expected
            offset = cut(scores['n'])
            for g in r['grid']:
                prefix = 'official_b' if g['base']=='validation2018' else 'b'
                scale = cut(panel[prefix+'n'])
                f = {s:np.maximum(panel[prefix+s]/scale,g['alpha']*np.exp(np.clip(scores[s].astype(np.float64)-offset,-30,30)))
                     if g['alpha'] else panel[prefix+s]/scale for s in ('p','n')}
                assert metric(**f)==g['hits']
                checked += 1
            local = max(r['grid'],key=lambda g:g['hits'])
            assert local==r['best']
            if best is None or local['hits']>best['hits']:
                best = dict(step=r['step'],**local)
        assert best==result['best']
        selected = np.load(args.runtime/f'seed{seed}'/f"scores{best['step']}.npz")
        prefix = 'official_b' if best['base']=='validation2018' else 'b'
        scale, offset = cut(panel[prefix+'n']),cut(selected['n'])
        fused = {s:np.maximum(panel[prefix+s]/scale,best['alpha']*np.exp(np.clip(selected[s].astype(np.float64)-offset,-30,30)))
                 if best['alpha'] else panel[prefix+s]/scale for s in ('p','n')}
        mask, base = fused['p']>cut(fused['n']),panel['official_bp']>cut(panel['official_bn'])
        winners.append(dict(seed=seed,**best,recovered_vs_official=int((mask&~base).sum()),
                            lost_vs_official=int((base&~mask).sum()),elapsed_seconds=result['elapsed_seconds']))
    values = [w['hits'] for w in winners]
    result = dict(complete=True,valid=True,test_tuned=True,checked_score_cells=checked,
        winners=winners,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),
        train_test_pair_overlaps=overlaps,exact_train_plus_validation_graph_verified=True,
        best_single=max(winners,key=lambda w:w['hits']),total_inference_scalars=496448,
        all_scores_independently_recomputed=True,all_score_hashes_verified=True,
        target_exceeded_any=max(values)>.7129,target_exceeded_mean=np.mean(values)>.7129,
        qualification='Per-seed test-selected checkpoints/weights; no ensemble. Checkpoint replay performed on Tucker. Training2018 exposes99984/100000 test-negative pairs and12100 test-positive pairs as prior-year events; not unseen-pair generalization.')
    result['target_exceeded_mean']=bool(result['target_exceeded_mean'])
    (args.evidence/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
