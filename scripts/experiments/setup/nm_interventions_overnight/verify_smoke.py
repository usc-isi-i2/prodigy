"""Fail before production if any arm, checkpoint or validation stream is missing."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import ARMS,ORDER,HOLDOUT
from scripts.experiments.setup.nm_interventions_overnight.evaluate import model_from_params


def main():
    import torch
    a=argparse.ArgumentParser();a.add_argument('run',type=Path);args=a.parse_args()
    status=json.loads((args.run/'status.json').read_text())
    assert status['status']=='complete'
    seen=set();fingerprints={};rows=[]
    for path in sorted(args.run.glob('job_*/result.json')):
        result=json.loads(path.read_text());assert result['status']=='complete'
        params=json.loads((path.parent/'effective_config.json').read_text())
        name=params['prefix'][4:].rsplit('_r',1)[0];seen.add(name)
        state=Path(result['checkpoint_dir']).parent
        selection=json.loads((state/'selection.json').read_text())
        assert selection['status']=='complete' and selection['training_steps']==20
        assert set(selection['sources'])==set(ORDER) and HOLDOUT not in selection['sources']
        model=model_from_params(params,selection['checkpoint'],torch.device('cpu'))
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        protocol=json.loads((state/'validation_protocol.json').read_text())
        for source,fp in protocol['fingerprints'].items():
            assert source not in fingerprints or fingerprints[source]==fp,(name,source)
            fingerprints[source]=fp
        rows.append(dict(arm=name,checkpoint=selection['checkpoint'],validation_auc=selection['best_val']))
    assert seen==set(ARMS),(seen,set(ARMS))
    output=dict(status='passed',arms=len(rows),fixed_source_fingerprints=fingerprints,models=rows)
    (args.run/'verified.json').write_text(json.dumps(output,indent=2)+'\n')
    print(f'PASS: {len(rows)} arms, strict checkpoint reload, finite weights, eight matching validation streams')
if __name__=='__main__':main()
