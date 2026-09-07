import unittest
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from types import SimpleNamespace

from models.general_gnn import SingleLayerGeneralGNN
from .audit_update_numerics import check_reference_inputs, differences, index_select_decode
from .replay import batch_hash
from torch_geometric.data import Data


class NumericalComparisonTests(unittest.TestCase):
    def test_exact_and_changed_tensor_inventories(self):
        a = {"weight": torch.tensor([1., 2.]), "counter": torch.tensor(2)}
        b = {k: v.clone() for k, v in a.items()}
        self.assertTrue(differences(a, b)["bit_exact"])
        b["weight"][1] += .5
        d = differences(a, b)
        self.assertFalse(d["bit_exact"])
        self.assertEqual(d["changed_keys"], 1)
        self.assertEqual(d["maximum_absolute_difference"], .5)
        self.assertEqual(d["largest_differences"], {"weight": .5})

    def test_missing_tensor_is_not_an_exact_match(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            differences({"weight": torch.tensor(1.)}, {})

    def test_real_inputs_require_full_reference_hash_and_step(self):
        batches = [[Data(x=torch.tensor([[1., 2.]])), torch.tensor([1])]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'reference.jsonl.gz'
            with gzip.open(path, 'wt') as handle:
                handle.write(json.dumps({'step': 1, 'batch_sha256': batch_hash(batches[0])}) + '\n')
            self.assertTrue(check_reference_inputs(batches, path)['all_inputs_bit_exact'])
            batches[0][0].x[0, 1] += 1
            with self.assertRaisesRegex(ValueError, 'completed training input audit'):
                check_reference_inputs(batches, path)

    def test_index_select_preserves_both_decoder_forward_paths(self):
        rng = torch.Generator().manual_seed(424)
        model = SimpleNamespace(cos=torch.nn.CosineSimilarity(dim=1), logit_scale=torch.tensor(2.3))
        x, y = torch.randn(7, 16, generator=rng), torch.randn(3, 16, generator=rng)
        for bipartite in (False, True):
            edges = torch.tensor([[0, 0, 2, 2, 2, 6], [0, 1, 0, 0, 2, 1]])
            if not bipartite:
                edges[1] += len(x)
            a = SingleLayerGeneralGNN.decode(model, x, y, edges, bipartite)
            b = index_select_decode(model, x, y, edges, bipartite)
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
