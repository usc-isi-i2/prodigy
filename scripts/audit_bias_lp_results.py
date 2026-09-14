"""Check paired comparison, bias application, and reported held-out BCE."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'results/nonzero_mini_transfer'
count=0
for stage in ['node_bias','node_neighbors_bias']:
    files=list((ROOT/stage).glob('*.scores.npz'));assert len(files)==81
    for file in files:
        a=np.load(file);b=np.load(ROOT/'node_neighbors_disjoint_uniform'/file.name)
        for key in ['u','v','labels','validation_mask','balanced_test_indices']:
            assert np.array_equal(a[key],b[key]),(file,key)
        report=json.loads(file.with_name(file.name.replace('.scores.npz','.json')).read_text())
        assert report['score_kind']=='dot_plus_frozen_source_bias'
        assert np.allclose(a['dot_logits'],a['raw_dot_scores']+report['decoder_bias'],atol=1e-6)
        y=a['labels'];test=~a['validation_mask'];l=a['dot_logits'].astype(float)
        assert np.isfinite(l).all() and sum(test&(y==1))==1400 and sum(test&(y==0))==7000
        bce=np.mean(np.logaddexp(0,l[test])-y[test]*l[test]);assert abs(bce-report['metrics']['test']['bce'])<1e-6
        assert report['known_edges_in_negatives']==0
        count+=1
print(f'Passed: {count} paired cells, fixed masks, bias application, finite logits, 1:5 counts, recomputed BCE.')
