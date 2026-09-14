"""Post-hoc validation error overlap; never selects models or reads test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    r = args.runtime
    receipt = json.loads((args.evidence/'validation_receipt.json').read_text())
    result = json.loads((args.evidence/'results.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(r/'assessment_scores.npz') == receipt['assessment_scores_sha256']
    z = np.load(r/'year2018.npz'); s = np.load(r/'assessment_scores.npz')
    assert fingerprint(z['pos'], z['neg']) == result['metadata'][-1]['pairs_fingerprint']
    assert fingerprint(z['pfeatures'], z['nfeatures']) == result['metadata'][-1]['features_fingerprint']
    cutoff = lambda a: np.sort(a)[-50]
    hit = lambda a,b: a > cutoff(b)
    # Ties broken by fixed pair index for top-50 membership only; Hits uses strict >.
    def top(a):
        m = np.zeros(len(a), bool)
        m[np.lexsort((np.arange(len(a)), -a))[:50]] = True
        return m
    base = hit(z['official_bp'], z['official_bn'])
    assert base.mean() == result['official_2018_calibrated_aadc_hits_at_50']
    hs = np.stack([hit(s[f'seed{i}_positive'],s[f'seed{i}_negative']) for i in range(3)])
    for i in range(3):
        assert hs[i].mean() == result['seeds'][i]['hits_at_50']
    recover = hs & ~base; lost = ~hs & base
    nt = np.stack([top(s[f'seed{i}_negative']) for i in range(3)])
    bt = top(z['official_bn']); promoted = nt & ~bt
    def describe(mask, side='p'):
        x = z[side+'features']; n = int(mask.sum())
        return {'count': n, 'repeat_pairs': int((mask & (x[:,8]>0)).sum()),
                'aa_zero': int((mask & (x[:,0]==0)).sum()),
                'median_raw_cosine': float(np.median(x[mask,7])) if n else None,
                'median_min_degree': float(np.median(np.expm1(x[mask,5]))) if n else None}
    groups={}
    for label, mask in [('all',np.ones(len(base),bool)),('repeat',z['pfeatures'][:,8]>0),('new',z['pfeatures'][:,8]==0)]:
        groups[label]={'positives':int(mask.sum()),'baseline_hits':int((base&mask).sum()),
                       'recovered':[int((v&mask).sum()) for v in recover],
                       'lost':[int((v&mask).sum()) for v in lost],
                       'net':[int((recover[i]&mask).sum()-(lost[i]&mask).sum()) for i in range(3)]}
    profiles={label:describe(mask) for label,mask in {
        'baseline_misses':~base,'recovered_all_three':recover.all(0),
        'recovered_any_seed':recover.any(0),'lost_all_three':lost.all(0),
        'missed_by_baseline_and_all_seeds':~base & ~hs.any(0)}.items()}
    entrants=[]
    for j in np.flatnonzero(promoted.any(0)):
        row={'negative_index':int(j),'promoted_seed_count':int(promoted[:,j].sum()),
             'baseline_rank':int((z['official_bn']>z['official_bn'][j]).sum()+1),
             'seed_ranks':[int((s[f'seed{i}_negative']>s[f'seed{i}_negative'][j]).sum()+1) for i in range(3)],
             'repeat':bool(z['nfeatures'][j,8]>0),'aa_zero':bool(z['nfeatures'][j,0]==0),
             'raw_cosine':float(z['nfeatures'][j,7])}
        entrants.append(row)
    out={'classification':'post_hoc_validation_diagnostic','test_read_or_scored':False,
         'source_revision':result['revision'],'scores_sha256':sha(r/'assessment_scores.npz'),
         'features_and_pairs_fingerprints_verified':True,'all_metrics_recomputed':True,
         'groups':groups,'positive_profiles':profiles,
         'recovery_seed_count_histogram':np.bincount(recover.sum(0)[~base],minlength=4).tolist(),
         'loss_seed_count_histogram':np.bincount(lost.sum(0)[base],minlength=4).tolist(),
         'negative_top50_overlap_with_baseline':[int((v&bt).sum()) for v in nt],
         'negative_cutoff_tie_counts':[int((s[f'seed{i}_negative']==cutoff(s[f'seed{i}_negative'])).sum()) for i in range(3)],
         'promoted_negative_profiles':describe(promoted.any(0),'n'),
         'promoted_negative_all_three':describe(promoted.all(0),'n'),
         'promoted_negatives':entrants,
         'qualification':'Correlational slices, not feature attribution; new means no earlier pair event, not a new author. Negative promotion is top-50 entry with fixed-index tie breaking.'}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
