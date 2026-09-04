"""A completed result must remain bound to its checkpoint and prior replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.setup.nm_interventions_overnight.evaluate import completed_cell


class ResumeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);checkpoint=self.root/'checkpoint';checkpoint.write_bytes(b'weights')
        self.model='nmi_baseline_r1_s0';self.target='ukr_rus'
        self.selection=dict(checkpoint=str(checkpoint),best_step=2000,training_steps=6000,
                            sources=[self.target],flags=[])
        self.job=dict(params=dict(prefix=self.model,seed=0,campaign_revision='training-revision'),
                      selection=self.selection,state=str(self.root))
        (self.root/'validation_protocol.json').write_text(json.dumps(dict(fingerprints={self.target:'validation-fingerprint'})))
        self.replay=self.root/'validation_replay'/self.model/f'{self.target}.json'
        self.replay.parent.mkdir(parents=True)
        self.replay.write_text(json.dumps(dict(status='passed',checkpoint=str(checkpoint),fingerprint='validation-fingerprint')))
        cell=dict(protocol='nmi_fixed_nm_v1',model_id=self.model,target=self.target,episodes=512,
                  checkpoint=str(checkpoint),checkpoint_step=2000,training_steps=6000,sources=[self.target],
                  flags=[],holdout='twibot20',seed=0,training_revision='training-revision',
                  checkpoint_sha256=hashlib.sha256(b'weights').hexdigest(),roc_auc=.8,accuracy=.3,loss=2.)
        path=self.root/'cells'/self.model/f'{self.target}.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(cell))

    def test_same_path_changed_weights_are_not_reused(self):
        self.assertTrue(completed_cell(self.root,self.job,self.target,512))
        Path(self.selection['checkpoint']).write_bytes(b'different weights')
        with self.assertRaisesRegex(ValueError,'checkpoint content mismatch'):
            completed_cell(self.root,self.job,self.target,512)

    def test_missing_prior_gate_is_not_treated_as_complete(self):
        self.replay.unlink()
        with self.assertRaisesRegex(ValueError,'Missing prior validation replay'):
            completed_cell(self.root,self.job,self.target,512)


if __name__=='__main__':unittest.main()
