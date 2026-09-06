import copy
import unittest

import torch

from .verify_readout_replay import check_predictions, UNCHANGED, DECODERS


class ReadoutReplayTests(unittest.TestCase):
    def setUp(self):
        self.hashes = [f"batch-{i}" for i in range(32)]
        self.reference = [dict(batch=i, batch_sha256=h, logits={d: torch.tensor([[.2, .8], [.7, .3]]) for d in DECODERS})
                          for i, h in enumerate(self.hashes)]

    def test_all_ten_upstream_probes_are_bit_exact(self):
        changed = copy.deepcopy(self.reference)
        for batch in changed:
            for decoder in DECODERS - UNCHANGED:
                batch["logits"][decoder] += .1
        self.assertEqual(check_predictions(changed, self.reference, self.hashes), 320)
        changed[17]["logits"]["S0_pool/ridge"][0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, "upstream"):
            check_predictions(changed, self.reference, self.hashes)

    def test_missing_batch_changed_inputs_nonfinite_and_missing_decoder(self):
        with self.assertRaises(ValueError):
            check_predictions(self.reference[:-1], self.reference, self.hashes)
        bad = copy.deepcopy(self.reference)
        bad[0]["batch_sha256"] = "same-nodes-different-features"
        with self.assertRaises(ValueError):
            check_predictions(bad, self.reference, self.hashes)
        bad = copy.deepcopy(self.reference)
        bad[0]["logits"]["full_model"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            check_predictions(bad, self.reference, self.hashes)
        bad = copy.deepcopy(self.reference)
        bad[0]["logits"].pop("full_model")
        with self.assertRaises(ValueError):
            check_predictions(bad, self.reference, self.hashes)


if __name__ == "__main__":
    unittest.main()
