import unittest

import torch

from .label_interface import LABEL_VARIANTS, PERMUTATION_SEED, label_interface_mode


class TinyLabelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = {"ignore_label_embeddings": False}
        self.initial_label_mlp = torch.nn.Linear(3, 5)
        self.learned_label_embedding = torch.nn.Embedding(12, 5)

    def forward(self, x):
        y = self.initial_label_mlp(x)
        if self.params["ignore_label_embeddings"]:
            y = self.learned_label_embedding(torch.arange(len(x)))
        return y


class LabelInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28)
        self.model = TinyLabelModel()
        self.x = torch.randn(4, 3)
        self.original = self.model(self.x).detach().clone()
        self.state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def test_norm_changes_only_shared_magnitude_not_direction(self):
        with label_interface_mode(self.model, "train_norm_label_text", 6) as audit:
            actual = self.model(self.x)
            expected_norm = self.model.learned_label_embedding.weight[:6].norm(dim=1).mean()
            torch.testing.assert_close(actual.norm(dim=1).mean(), expected_norm)
            torch.testing.assert_close(torch.nn.functional.normalize(actual), torch.nn.functional.normalize(self.original))
            self.assertEqual(audit["forward_calls"], 1)
        torch.testing.assert_close(self.model(self.x), self.original, rtol=0, atol=0)

    def test_table_and_permuted_table_are_exact(self):
        for variant in ("train_label_table", "permuted_train_label_table"):
            index = torch.arange(4)
            if variant.startswith("permuted"):
                index = torch.randperm(6, generator=torch.Generator().manual_seed(PERMUTATION_SEED))[:4]
            with label_interface_mode(self.model, variant, 6):
                torch.testing.assert_close(self.model(self.x), self.model.learned_label_embedding.weight[index], rtol=0, atol=0)
        self.assertFalse(self.model.params["ignore_label_embeddings"])

    def test_zero_is_projected_zero_not_only_zero_raw_text(self):
        with label_interface_mode(self.model, "zero_projected_labels", 6):
            self.assertEqual(int(torch.count_nonzero(self.model(self.x))), 0)
        self.assertGreater(int(torch.count_nonzero(self.model(torch.zeros_like(self.x)))), 0)

    def test_context_restores_everything_even_on_exception(self):
        for variant in LABEL_VARIANTS:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with label_interface_mode(self.model, variant, 6):
                    self.model(self.x)
                    raise RuntimeError("intentional")
            torch.testing.assert_close(self.model(self.x), self.original, rtol=0, atol=0)
            self.assertFalse(self.model.params["ignore_label_embeddings"])
            self.assertEqual(len(self.model.initial_label_mlp._forward_hooks), 0)
            self.assertEqual(len(self.model.learned_label_embedding._forward_hooks), 0)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, self.state[key], rtol=0, atol=0)

    def test_baseline_and_rng_unchanged_and_invalid_range_rejected(self):
        rng = torch.get_rng_state().clone()
        with label_interface_mode(self.model, "baseline", 6) as audit:
            torch.testing.assert_close(self.model(self.x), self.original, rtol=0, atol=0)
            self.assertEqual(audit, {})
        with label_interface_mode(self.model, "permuted_train_label_table", 6):
            self.model(self.x)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            with label_interface_mode(self.model, "train_label_table", 20):
                pass


if __name__ == "__main__":
    unittest.main()
