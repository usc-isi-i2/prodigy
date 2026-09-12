import json
import numpy as np
import pytest
from mixture_scaling.evaluation_artifacts import atomic_json, destination, save_scores, signature


def test_legacy_results_preserved_and_scores_roundtrip(tmp_path):
    legacy=tmp_path/'result.json';legacy.write_text('{"report":{"auc":0.7}}')
    output=destination(legacy)
    assert output.name=='result.metrics-v2.json'
    atomic_json(output,{'schema_version':2,'metrics':{'accuracy':0.6}})
    assert json.loads(legacy.read_text())=={'report':{'auc':0.7}}
    scores=output.with_suffix('.scores.npz')
    save_scores(scores,labels=np.array([0,1]),dot_logits=np.array([-2.,3.]),validation_mask=np.array([True,False]))
    with np.load(scores) as data:
        assert data['labels'].tolist()==[0,1]
        assert data['dot_logits'].tolist()==[-2.,3.]
    with pytest.raises(ValueError):atomic_json(output,{'invalid':float('nan')})
    assert json.loads(output.read_text())['schema_version']==2


def test_checkpoint_changes_invalidate_provenance(tmp_path):
    paths=[tmp_path/name for name in ['checkpoint','graph','split']]
    for path in paths:path.write_bytes(b'initial')
    before=signature(*paths,0,2000)
    paths[0].write_bytes(b'updated')
    after=signature(*paths,0,2000)
    assert before['checkpoint']['sha256']!=after['checkpoint']['sha256']
