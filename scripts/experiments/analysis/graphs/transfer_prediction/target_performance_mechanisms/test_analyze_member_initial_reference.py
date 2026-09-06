import copy
import unittest

import pandas as pd

from .analyze_member_initial_reference import validate_manifest, validate_inputs, validate_grid, endpoint_changes
from .analyze_trajectories import DECODERS, PANEL


class InitialReferenceTests(unittest.TestCase):
    def setUp(self):
        self.arms = pd.DataFrame([dict(model_id=f"m{s}_{i}", seed=s, initial_sha256=f"h{s}",
                                      checkpoint=f"/run/m{s}_{i}/state_dict_2500.ckpt")
                                  for s in range(3) for i in range(8)])
        self.manifest = pd.DataFrame([dict(model_id=f"init{s}", seed=s, sources=[], weights_sha256=f"h{s}",
                                          checkpoint=f"/run/m{s}_0/state_dict_0.ckpt") for s in range(3)])

    def test_exact_shared_initialization_not_arbitrary_random(self):
        validate_manifest(self.manifest, self.arms)
        for field, value in (("weights_sha256", "new-random"), ("checkpoint", "/other/state_dict_0.ckpt")):
            bad = self.manifest.copy(deep=True)
            bad.loc[0, field] = value
            with self.assertRaises(ValueError):
                validate_manifest(bad, self.arms)
        bad = self.arms.copy()
        bad.loc[7, "initial_sha256"] = "other-source-init"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, bad)

    def test_both_streams_every_cached_batch(self):
        receipt = dict(all_cached_batches_identical=True, stream_target_cells=10,
                       cells=[dict(stream=s, target=t, episode_fingerprint=s+t,
                                   batch_sha256=[f"{s}/{t}/{i}" for i in range(32)])
                              for s in ("original", "fresh") for t in PANEL])
        validate_inputs(receipt, receipt)
        bad = copy.deepcopy(receipt)
        bad["cells"][0]["batch_sha256"][31] = "changed"
        with self.assertRaises(ValueError):
            validate_inputs(bad, receipt)
        bad = copy.deepcopy(receipt)
        bad["cells"][-1] = copy.deepcopy(bad["cells"][0])
        with self.assertRaises(ValueError):
            validate_inputs(bad, receipt)

    def test_complete_finite_fixed_input_grid(self):
        rows = [dict(model_id=m.model_id, dataset=t, decoder=d, sources=[], checkpoint=m.checkpoint,
                     weights_sha256=m.weights_sha256, variant="baseline", episodes=128, queries=3072,
                     episode_fingerprint=t, roc_auc=.6, accuracy=.5, f1=.4, nll=.8)
                for m in self.manifest.itertuples() for t in PANEL for d in sorted(DECODERS)]
        cells = pd.DataFrame(rows)
        reference = cells.copy(deep=True)
        self.assertEqual(len(validate_grid(cells, self.manifest, reference, "original")), 255)
        for column, value in (("weights_sha256", "wrong"), ("nll", float("nan")),
                              ("episode_fingerprint", "changed"), ("roc_auc", .7)):
            bad = cells.copy(deep=True)
            raw_index = bad.index[bad.decoder == "raw_center/ridge"][0]
            bad.loc[raw_index, column] = value
            with self.assertRaises(ValueError):
                validate_grid(bad, self.manifest, reference, "original")
        with self.assertRaises(ValueError):
            validate_grid(cells.iloc[1:], self.manifest, reference, "original")

    def test_changes_match_training_seed_and_stream(self):
        common = dict(stream="original", dataset="twibot20", seed=0, decoder="full_model", episode_fingerprint="episode")
        initial = pd.DataFrame([common | dict(model_id="init0", checkpoint="init", weights_sha256="h0",
                                              roc_auc=.6, accuracy=.5, f1=.4, nll=.8)])
        final = pd.DataFrame([common | dict(model_id="m0_0", checkpoint="final", weights_sha256="trained",
                                            roc_auc=.65, accuracy=.55, f1=.45, nll=.75)])
        result = endpoint_changes(final, initial, self.arms)
        self.assertAlmostEqual(result.change_0_to_2500_roc_auc.iloc[0], .05)
        self.assertAlmostEqual(result.change_0_to_2500_nll.iloc[0], -.05)
        for column, value in (("weights_sha256", "wrong-seed"), ("episode_fingerprint", "fresh-episode")):
            bad = initial.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                endpoint_changes(final, bad, self.arms)
        with self.assertRaises(ValueError):
            endpoint_changes(final, initial.iloc[:0], self.arms)


if __name__ == "__main__":
    unittest.main()
