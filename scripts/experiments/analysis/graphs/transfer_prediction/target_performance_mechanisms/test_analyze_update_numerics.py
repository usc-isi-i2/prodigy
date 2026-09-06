import json
from pathlib import Path
import unittest

from .analyze_update_numerics import validate


class UpdateNumericsAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads((Path(__file__).parent / 'data/control_decoder_numerics.json').read_text())

    def test_complete_receipt_and_observed_stabilization(self):
        rows, summary, validation = validate(self.receipt)
        self.assertEqual(len(rows), 108)
        self.assertTrue(validation['valid'])
        self.assertEqual([r['all_four_updates_bit_exact'] for r in summary], [False, True, True])

    def test_missing_comparison_is_not_complete(self):
        self.receipt['comparisons'].pop()
        with self.assertRaisesRegex(ValueError, 'comparison grid'):
            validate(self.receipt)

    def test_hash_and_difference_must_agree(self):
        self.receipt['comparisons'][0]['steps'][0]['gradients']['bit_exact'] = True
        with self.assertRaisesRegex(ValueError, 'digest evidence'):
            validate(self.receipt)

    def test_different_input_and_forward_failure_rejected(self):
        self.receipt['replays'][1]['steps'][0]['input_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            validate(self.receipt)
        self.setUp()
        self.receipt['replays'][-1]['decoder_forward_parity'][0] = False
        with self.assertRaisesRegex(ValueError, 'forward-parity'):
            validate(self.receipt)

    def test_synthetic_cannot_be_relabelled_real_without_evidence(self):
        self.receipt['synthetic_workload'] = False
        with self.assertRaisesRegex(ValueError, 'unverified real input'):
            validate(self.receipt)


if __name__ == '__main__':
    unittest.main()
