"""A resumed evaluator must preserve evidence and cannot retain a stale success."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.experiments.setup.nm_interventions_overnight import evaluate


class EvaluationLifecycleTests(unittest.TestCase):
    def test_setup_failure_archives_previous_success_and_keeps_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old={'status':'complete','invocation_id':'old'}
            (root/'status.json').write_text(json.dumps(old))
            (root/'plan.json').write_text(json.dumps({'models':['previous']}))
            cell=root/'cells/model/source.json';cell.parent.mkdir(parents=True)
            cell.write_text('{"existing":"evidence"}')
            job={'params':{'prefix':'nmi_baseline_r8_s0','config':'unused.yaml'}}
            argv=['evaluate','--run-dirs','unused','--output',str(root),'--validation-only']
            with patch('sys.argv',argv),patch.object(evaluate,'discover',return_value=[job]), \
                    patch('experiments.params.get_params',side_effect=RuntimeError('setup failure')):
                with self.assertRaisesRegex(RuntimeError,'setup failure'):evaluate.main()
            current=json.loads((root/'status.json').read_text())
            self.assertEqual(current['status'],'failed')
            self.assertNotEqual(current['invocation_id'],'old')
            prior=list((root/'history').glob('*/status.json'))
            self.assertEqual(len(prior),1)
            self.assertEqual(json.loads(prior[0].read_text()),old)
            self.assertEqual(json.loads(cell.read_text()),{'existing':'evidence'})

    def test_zero_workers_cannot_claim_success(self):
        argv=['evaluate','--run-dirs','unused','--output','unused','--workers-per-gpu','0']
        with patch('sys.argv',argv),patch.object(evaluate,'discover') as discover:
            with self.assertRaises(SystemExit):evaluate.main()
            discover.assert_not_called()

if __name__=='__main__':unittest.main()
