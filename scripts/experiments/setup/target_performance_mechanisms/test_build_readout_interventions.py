import unittest

import torch

from .build_readout_interventions import READOUT_KEYS, MODES, hybrid_state, validate_inventory


class ReadoutInterventionTests(unittest.TestCase):
    def setUp(self):
        self.initial = {k: torch.full((2, 2) if k.endswith("weight") else (2,), .1) for k in READOUT_KEYS}
        self.initial["upstream.weight"] = torch.eye(2)
        self.initial["downstream.weight"] = torch.ones(2, 2)
        self.initial["normalization.num_batches_tracked"] = torch.tensor(0)
        self.terminal = {k: v + 1 for k, v in self.initial.items()}

    def test_exact_donor_all_untouched_tensors_both_directions(self):
        for mode in MODES:
            hybrid, changed = hybrid_state(self.initial, self.terminal, mode)
            self.assertEqual(set(changed), READOUT_KEYS)
            background, donor = (self.terminal, self.initial) if mode == MODES[0] else (self.initial, self.terminal)
            for key, value in hybrid.items():
                self.assertTrue(torch.equal(value, (donor if key in READOUT_KEYS else background)[key]))
                self.assertNotEqual(value.data_ptr(), (donor if key in READOUT_KEYS else background)[key].data_ptr())

    def test_noop_missing_key_nonfinite_and_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            hybrid_state(self.initial, self.initial, MODES[0])
        bad = self.terminal.copy()
        bad.pop(sorted(READOUT_KEYS)[0])
        with self.assertRaises(ValueError):
            hybrid_state(self.initial, bad, MODES[0])
        for value in (torch.ones(3, 3), torch.full((2, 2), float("nan"))):
            bad = self.terminal.copy()
            bad["upstream.weight"] = value
            with self.assertRaises(ValueError):
                hybrid_state(self.initial, bad, MODES[0])

    def test_smoke_or_incomplete_inventory_rejected(self):
        with self.assertRaises(ValueError):
            validate_inventory({"valid": True, "models": 24}, [])


if __name__ == "__main__":
    unittest.main()
