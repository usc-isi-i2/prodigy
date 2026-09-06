from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from .analyze_support_identity_gradients import HERE, validate, derive, validate_prefix_receipts


class CompletedGradientTests(unittest.TestCase):
    def test_all_retained_sources_states_modes_and_matched_groups(self):
        c, g, b, groups, done = validate(HERE/"data/support_identity_gradients")
        contrasts, cancellation = derive(c, b, groups)
        self.assertEqual((len(c), len(g), len(b), len(contrasts), len(cancellation)), (432, 2592, 864, 144, 432))
        self.assertEqual(done["observed_deterministic_baseline_repeats"], 144)
        self.assertTrue((g[g.block=="label_input"].active_parameter_tensors == 0).all())
        prefix = validate_prefix_receipts(c, HERE/"data/support_identity_inputs")
        self.assertEqual(prefix["local_full_input_hash_links_checked"], 36)

    def test_missing_condition_and_changed_query_path_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"audit"
            shutil.copytree(HERE/"data/support_identity_gradients", root)
            original = pd.read_csv(root/"cells.csv")
            original.iloc[1:].to_csv(root/"cells.csv", index=False)
            with self.assertRaisesRegex(ValueError, "cell grid"):
                validate(root)
            changed = original.copy()
            index = changed.index[changed["mode"]=="meta_frozen"][0]
            changed.loc[index, "post_query_max_difference"] = .1
            changed.to_csv(root/"cells.csv", index=False)
            with self.assertRaisesRegex(ValueError, "query vectors changed"):
                validate(root)

    def test_missing_cancellation_evidence_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"audit"
            shutil.copytree(HERE/"data/support_identity_gradients", root)
            groups = pd.read_csv(root/"groups.csv")
            groups.iloc[1:].to_csv(root/"groups.csv", index=False)
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                validate(root)

    def test_inactive_parameter_alignment_is_not_zero_or_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"audit"
            shutil.copytree(HERE/"data/support_identity_gradients", root)
            gradients = pd.read_csv(root/"gradients.csv")
            index = gradients.index[gradients.block=="label_input"][0]
            gradients.loc[index, "cosine"] = 1.
            gradients.to_csv(root/"gradients.csv", index=False)
            with self.assertRaisesRegex(ValueError, "alignment must be undefined"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
