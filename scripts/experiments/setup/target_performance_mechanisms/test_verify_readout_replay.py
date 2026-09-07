import copy
from pathlib import Path
import tempfile
import unittest

import torch

from .verify_readout_replay import check_predictions, check_saved_hybrid, UNCHANGED, DECODERS
from .build_readout_interventions import READOUT_KEYS, MODES, hybrid_state
from .verify_member_training import model_digest


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

    def test_persisted_tensor_donors_not_just_claimed_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = {k: torch.ones(2) for k in READOUT_KEYS}
            initial["untouched_buffer"] = torch.tensor(0)
            terminal = {k: v + 1 for k, v in initial.items()}
            hybrid, changed = hybrid_state(initial, terminal, MODES[0])
            for name, state in (("initial", initial), ("terminal", terminal), ("hybrid", hybrid)):
                torch.save({"model": state}, root / f"{name}.pt")
            row = dict(model_id="test", intervention=MODES[0], changed_keys=changed,
                       initial_checkpoint=str(root / "initial.pt"), terminal_checkpoint=str(root / "terminal.pt"),
                       checkpoint=str(root / "hybrid.pt"), initial_sha256=model_digest(initial),
                       terminal_sha256=model_digest(terminal), weights_sha256=model_digest(hybrid))
            self.assertTrue(check_saved_hybrid(row)["exact_donor_tensors"])
            hybrid["untouched_buffer"] += 7
            torch.save({"model": hybrid}, root / "hybrid.pt")
            row["weights_sha256"] = model_digest(hybrid)
            with self.assertRaisesRegex(ValueError, "wrong donor tensor"):
                check_saved_hybrid(row)


if __name__ == "__main__":
    unittest.main()
