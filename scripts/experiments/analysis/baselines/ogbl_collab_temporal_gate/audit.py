#!/usr/bin/env python3
"""Validate frozen selection, checkpoints and cached assessment counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=Path(__file__).parent/'data')
    p.add_argument('--control-runtime',type=Path)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data=args.evidence
    receipt=json.loads((data/'validation_receipt.json').read_text())
    result=json.loads((data/'results.json').read_text())
    r=args.runtime
    assert sha(r/'selection_frozen.json')==receipt['selection_frozen_sha256']
    assert sha(r/'assessment_scores.npz')==receipt['assessment_scores_sha256']
    frozen=json.loads((r/'selection_frozen.json').read_text())
    assert frozen['assessment_2018_scored'] is False
    score=np.load(r/'assessment_scores.npz')
    year=np.load(r/'year2018.npz')
    yearsel=np.load(r/'year2017.npz')
    def hitmask(pos,neg):
        return pos>np.sort(neg)[-50]
    base=hitmask(year['bp'],year['bn'])
    reference=hitmask(year['official_bp'],year['official_bn'])
    assert float(base.mean())==result['frozen_2015_aadc_2018_hits_at_50']
    assert float(reference.mean())==result['official_2018_calibrated_aadc_hits_at_50']
    rows=[]
    for seed,selection in enumerate(frozen['selections']):
        assert selection['seed']==seed
        assert sha(r/f'gate_seed{seed}.pt')==selection['sha256']
        assert selection['sha256']==receipt['checkpoints'][seed]['sha256']
        baseline2017=float(hitmask(yearsel['bp'],yearsel['bn']).mean())
        candidates=[(baseline2017,0,0.)]+[(c['hits_at_50'],h['step'],c['strength'])
                    for h in selection['history'] for c in h['selection']]
        if result.get('direct_score'):
            candidates=candidates[1:]
            assert all(c[2]==1. for c in candidates)
        best=min(candidates,key=lambda x:(-x[0],x[1],x[2]))
        assert best==(selection['selection_2017_hits_at_50'],selection['step'],selection['strength'])
        mask=hitmask(score[f'seed{seed}_positive'],score[f'seed{seed}_negative'])
        assert float(mask.mean())==result['seeds'][seed]['hits_at_50']
        selfmask=year['neg'][:,0]==year['neg'][:,1]
        assert np.all(score[f'seed{seed}_negative'][selfmask]==-1e9)
        rows.append({'seed':seed,'checkpoint_hash_verified':True,'selection_argmax_verified':True,
                     'assessment_hits_verified':True,'recovered_vs_frozen_aadc':int((mask&~base).sum()),
                     'lost_vs_frozen_aadc':int((~mask&base).sum()),
                     'recovered_vs_official_aadc':int((mask&~reference).sum()),
                     'lost_vs_official_aadc':int((~mask&reference).sum())})
        if result.get('direct_score'):
            zero=(year['bp']==0)
            rows[-1].update({'frozen_base_zero_positives':int(zero.sum()),
                'frozen_base_zero_hits':int((zero&mask).sum()),
                'zero_base_official_misses_recovered':int((zero&~reference&mask).sum())})
    panel_shift=[]
    for y,z in [(2017,yearsel),(2018,year)]:
        panel_shift.append({'year':y,'positive_raw_aa_nonzero_fraction':float((z['pfeatures'][:,0]>0).mean()),
                            'negative_raw_aa_nonzero_count':int((z['nfeatures'][:,0]>0).sum()),
                            'positive_prior_pair_seen_fraction':float((z['pfeatures'][:,8]>0).mean()),
                            'aadc_negative50':float(np.sort(z['bn'])[-50])})
    report={'complete':True,'test_scored':False,'all_selections_and_checkpoint_hashes_verified':True,
            'all_assessment_metrics_recomputed':True,'self_pair_rule_verified':True,
            'seed_audits':rows,'panel_shift_diagnostic':panel_shift,
            'source_selection_sha256':sha(r/'selection_frozen.json'),
            'source_assessment_sha256':sha(r/'assessment_scores.npz'),
            'audit_source_sha256':sha(Path(__file__))}
    if result.get('constrained_rescue'):
        protected_checks=[]
        scale=frozen['scale']
        baseline_logs={side:np.log(np.maximum(year['b'+side],1e-6)/scale) for side in ('p','n')}
        # Cross-platform NumPy log can differ in the last float32 bits. If present,
        # use a hash-verified zero-strength seed as an exact producer-side baseline.
        zero_seeds=[s['seed'] for s in frozen['selections'] if s['strength']==0.]
        if zero_seeds:
            producer_baseline={side:score[f'seed{zero_seeds[0]}_{key}']
                               for side,key in [('p','positive'),('n','negative')]}
            for seed in zero_seeds:
                for side,key in [('p','positive'),('n','negative')]:
                    assert np.array_equal(score[f'seed{seed}_{key}'],producer_baseline[side])
            assert np.array_equal(hitmask(producer_baseline['p'],producer_baseline['n']),base)
            report['protected_baseline_reference']={'zero_strength_seeds':zero_seeds,
                'zero_strength_hitmask_exactly_matches_raw_frozen_aadc':True,
                'local_log_max_difference_nonself':{
                    side:float(np.max(np.abs(producer_baseline[side][year[edge][:,0]!=year[edge][:,1]]-
                        baseline_logs[side][year[edge][:,0]!=year[edge][:,1]])))
                    for side,edge in [('p','pos'),('n','neg')]}}
            baseline_logs=producer_baseline
        eligible={side:(year[side+'features'][:,0]==0)&(year[side+'features'][:,8]==0)
                  for side in ('p','n')}
        frozen_hits=hitmask(baseline_logs['p'],baseline_logs['n'])
        for seed in range(3):
            for side,key in [('p','positive'),('n','negative')]:
                edge=year['pos' if side=='p' else 'neg']
                protected=~eligible[side] & (edge[:,0]!=edge[:,1])
                assert np.array_equal(score[f'seed{seed}_{key}'][protected],baseline_logs[side][protected])
            mask=hitmask(score[f'seed{seed}_positive'],score[f'seed{seed}_negative'])
            protected_checks.append({'seed':seed,'protected_scores_exactly_equal':True,
                'negative_cutoff_log_change':float(np.sort(score[f'seed{seed}_negative'])[-50]-np.sort(baseline_logs['n'])[-50]),
                'protected_positive_hits_lost_vs_frozen':int((~eligible['p']&frozen_hits&~mask).sum())})
        report['constrained_score_checks']=protected_checks
    if args.control_runtime:
        checks=[]
        for y in (2015,2016,2017,2018):
            current=np.load(r/f'year{y}.npz')
            control=np.load(args.control_runtime/f'year{y}.npz')
            for name in ('neg','nfeatures'):
                assert np.array_equal(current[name],control[name]),(y,name)
            keep=control['pfeatures'][:,5]>0 if y<2018 else np.ones(len(control['pos']),dtype=bool)
            for name in ('pos','pfeatures'):
                assert np.array_equal(current[name],control[name][keep]),(y,name)
            checks.append({'year':y,'negative_pairs_and_features_identical':True,
                           'positive_pairs_and_features_exact_expected_subset':True,
                           'original_positives':len(control['pos']),'retained_positives':len(current['pos'])})
        report['matched_panel_checks']=checks
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
