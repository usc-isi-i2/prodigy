import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .support_identity_gradients import support_plan, label_intervention, cancellation, verify_interventions, gradient_probe, tensor_receipt
from .replay import batch_hash, clone_batch


class RandomWorkerDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return torch.tensor([float(i), float(torch.rand(()))])


def fixture():
    # Two episodes, three classes, three supports and two queries per class.
    ids = torch.arange(30)
    ids[5], ids[10] = ids[0].clone(), ids[0].clone()
    ids[21] = ids[16].clone()
    query = torch.tensor([0, 0, 0, 1, 1]*6).bool()
    labels = torch.arange(3).repeat_interleave(5).repeat(2)
    truth = F.one_hot(labels, 3).float()
    edges = torch.stack([torch.arange(30).repeat_interleave(3),
        torch.arange(3).repeat(30)+30+torch.arange(2).repeat_interleave(45)*3])
    attr = torch.stack([query.repeat_interleave(3).float(),
        ((truth*2-1)*(~query)[:, None]).reshape(-1)], dim=1)
    g = Data(x=torch.randn(30, 8), ptr=torch.arange(31), global_node_ids=ids,
        task_id_per_sample=torch.arange(2).repeat_interleave(15), edge_index=torch.empty((2, 0), dtype=torch.long))
    return [g, torch.randn(6, 8), truth, edges, attr, query.repeat_interleave(3),
            torch.zeros(1), torch.zeros(1), torch.zeros(1)]


class SupportIdentityTests(unittest.TestCase):
    def test_finite_prefix_preserves_spawned_worker_input_values(self):
        from .run_support_identity_gradients import prefix_loader
        original = torch.utils.data.DataLoader(RandomWorkerDataset(), batch_size=2, num_workers=2,
            multiprocessing_context="spawn")
        torch.manual_seed(7021)
        full = list(original)
        torch.manual_seed(7021)
        prefix = list(prefix_loader(original, 4))
        self.assertEqual(len(prefix), 4)
        for a, b in zip(full, prefix):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mass_and_magnitude_matched_support_only_interventions(self):
        b = fixture()
        before = batch_hash(b)
        plan = support_plan(b, 199)
        report = verify_interventions(b, plan)
        self.assertEqual(report["groups"], 2)
        self.assertEqual(report["affected_support_positions"], 5)
        self.assertEqual(report["squared_perturbation"], report["null_squared_perturbation"])
        self.assertEqual(batch_hash(b), before)
        a = label_intervention(b, plan, "identity_soft")[4].reshape(30, 3, 2)
        for g in plan["groups"]:
            torch.testing.assert_close(a[g], a[g[0]].expand_as(a[g]), rtol=0, atol=0)

    def test_query_truth_and_queries_do_not_select_soft_labels(self):
        b = fixture()
        original = support_plan(b, 9)
        q = original["query"]
        b[2][q] = b[2][q].roll(1, dims=1)
        b[0].global_node_ids[q] = 1000
        altered = support_plan(b, 9)
        self.assertEqual([v.tolist() for v in original["groups"]], [v.tolist() for v in altered["groups"]])
        self.assertEqual([v.tolist() for v in original["null_groups"]], [v.tolist() for v in altered["null_groups"]])

    def test_no_duplicates_is_a_true_noop(self):
        b = fixture()
        b[0].global_node_ids = torch.arange(30)
        plan = support_plan(b, 4)
        self.assertEqual(len(plan["groups"]), 0)
        for v in ("identity_soft", "permuted_soft"):
            self.assertEqual(batch_hash(b), batch_hash(label_intervention(b, plan, v)))

    def test_exact_opposition_and_agreement_gradients(self):
        g = torch.tensor([[1., 0.], [-1., 0.], [1., 0.]])
        r = cancellation(g, [torch.tensor([0, 1]), torch.tensor([0, 2])])
        self.assertEqual(r[0]["shared_gradient_ratio"], 0)
        self.assertEqual(r[0]["pair_cosine_mean"], -1)
        self.assertEqual(r[1]["shared_gradient_ratio"], 1)
        self.assertEqual(r[1]["pair_cosine_mean"], 1)

    def test_malformed_support_relations_rejected(self):
        b = fixture()
        b[4][0, 1] = .2
        with self.assertRaisesRegex(ValueError, "hard single-class"):
            support_plan(b, 1)

    def test_production_gradient_repeat_and_training_only_query_coupling(self):
        from .test_replay import fixture as model_fixture
        model, batch = model_fixture()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state()
        trainer = SimpleNamespace(model=model, parameter={"attr_regression_weight": 0},
            _restore_rng_state=torch.set_rng_state,
            get_loss_and_acc=lambda y, p: (F.cross_entropy(p, y), None), get_aux_loss=lambda g: 0.)
        changed = clone_batch(batch)
        changed[4][:, 1] = 0
        query = batch[5].reshape(-1, 2)[:, 0].bool()
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            for mode in ("training", "meta_frozen"):
                base = gradient_probe(trainer, state, batch, mode, rng)
                repeat = gradient_probe(trainer, state, batch, mode, rng)
                self.assertEqual(tensor_receipt(base), tensor_receipt(repeat))
                treated = gradient_probe(trainer, state, changed, mode, rng)
                torch.testing.assert_close(base["pre"], treated["pre"], rtol=0, atol=0)
                error = (base["post"][:len(query)][query]-treated["post"][:len(query)][query]).abs().max()
                self.assertEqual(float(error) == 0, mode == "meta_frozen")
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
