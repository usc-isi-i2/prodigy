import random
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .natural_support import frozen_contract, meta_inputs, meta_logits, support_cv_loss, support_plan, replace_graphs
from .replay import clone_batch, trace_stages
from .test_replay import fixture
from .role_context import query_mask
from .run_natural_support import score_draws


def support_fixture():
    model, original = fixture()
    graphs = original[0].to_data_list()
    expanded = []
    for i in range(16):
        g = graphs[i % 8].clone()
        g.global_node_ids = torch.tensor([i * 3, i * 3 + 1, i * 3 + 2, -1])
        expanded.append(g)
    g = Batch.from_data_list(expanded)
    g.task_id_per_sample = torch.arange(2).repeat_interleave(8)
    g.task_label_map = torch.tensor([[0, 1], [1, 0]])
    y = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1] * 2)
    q = torch.tensor([False] * 6 + [True] * 2).repeat(2)
    edges, attrs, mask = meta_inputs(y, q, g.task_id_per_sample, 4)
    return model, [g, original[1], F.one_hot(y, 2).float(), edges, attrs, mask, torch.empty(0), torch.empty(0), torch.empty(0)]


class NaturalSupportTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_pool_plan_ignores_query_labels_and_preserves_unique_id_contract(self):
        _, batch = support_fixture()
        rng_state = random.getstate()
        mappings, plan = support_plan([batch], 4, 13)
        altered = clone_batch(batch)
        q = query_mask(batch)
        altered[2][q] = altered[2][q].flip(1)
        other, receipt = support_plan([altered], 4, 13)
        self.assertEqual(plan, receipt)
        self.assertEqual(random.getstate(), rng_state)
        for a, b in zip(mappings, other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        centers = batch[0].global_node_ids[batch[0].ptr[:-1]]
        for row in plan["audits"]:
            chosen = row["selected"]
            self.assertEqual(len(set(r["center"] for r in chosen)), row["shots"])
            forbidden = set(centers[q & batch[0].task_id_per_sample.eq(row["episode"])].tolist())
            self.assertFalse(forbidden.intersection(r["center"] for r in chosen))
            self.assertTrue(all(r["episode"] != row["episode"] and r["global_label"] == row["global_class"] for r in chosen))

    def test_insufficient_unique_support_pool_rejected(self):
        _, b = fixture()
        b[0].task_id_per_sample.zero_()
        b[0].task_label_map = torch.tensor([[0, 1]])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            support_plan([b], 2, 0)

    def test_natural_whole_graph_matches_suffix_and_queries_stay_exact(self):
        model, batch = support_fixture()
        frozen_contract(model)
        mappings, _ = support_plan([batch], 2, 13)
        q, y, tasks = query_mask(batch), batch[2].argmax(1), batch[0].task_id_per_sample
        with torch.no_grad():
            with trace_stages(model, batch[0]) as trace:
                _, base, _ = model(*clone_batch(batch))
            pre = trace["U1_pre_meta"]
            x0, whole0 = meta_logits(model, pre, batch[1], y, q, tasks)
            torch.testing.assert_close(base, whole0[q], rtol=0, atol=1e-6)
            for mapping in mappings[0]:
                changed = replace_graphs(batch, batch[0].to_data_list(), mapping)
                _, direct, _ = model(*changed)
                x1, whole1 = meta_logits(model, pre[mapping], batch[1], y, q, tasks)
                torch.testing.assert_close(direct, whole1[q], rtol=0, atol=1e-6)
                torch.testing.assert_close(x0[q], x1[q], rtol=0, atol=0)

    def test_cross_validation_uses_no_query_inputs_or_query_labels(self):
        model, batch = support_fixture()
        q, y, task = query_mask(batch), batch[2].argmax(1), batch[0].task_id_per_sample
        pre = torch.randn(len(q), 8)
        with torch.no_grad():
            first = support_cv_loss(model, pre, batch[1], y, q, task)
            changed = pre.clone()
            changed[q] = 1000
            changed_y = y.clone()
            changed_y[q] = 1 - changed_y[q]
            second = support_cv_loss(model, changed, batch[1] , changed_y, q, task)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            held = q.clone()
            held[[0, 3, 8, 11]] = True
            _, a = meta_logits(model, pre, batch[1], y, held, task)
            tampered = y.clone()
            tampered[held] = 1 - tampered[held]
            _, b = meta_logits(model, pre, batch[1], tampered, held, task)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_vectorized_cv_matches_independent_folds(self):
        model, batch = support_fixture()
        q, y, task = query_mask(batch), batch[2].argmax(1), batch[0].task_id_per_sample
        pre = torch.randn(len(q), 8)
        s = (~q).nonzero().flatten()
        with torch.no_grad():
            batched = support_cv_loss(model, pre, batch[1], y, q, task)
            expected = torch.zeros(2)
            for fold in range(3):
                held = torch.zeros(len(s), dtype=torch.bool)
                held[[fold, 3 + fold, 6 + fold, 9 + fold]] = True
                _, logits = meta_logits(model, pre[s], batch[1], y[s], held, task[s])
                losses = F.cross_entropy(logits[held], y[s][held], reduction="none").reshape(2, 2).mean(1)
                expected += losses / 3
            torch.testing.assert_close(batched, expected, rtol=0, atol=1e-6)

    def test_selection_is_fixed_by_support_cv_not_query_outcomes(self):
        predictions = torch.randn(8, 4, 2)
        cv = torch.ones(8, 2)
        cv[3, 0], cv[6, 1] = 0, 0
        labels = dict(local_y=torch.tensor([0, 1, 0, 1]), mapping=torch.tensor([[0, 1]] * 4),
                      episode_ids=torch.tensor([0, 0, 1, 1]), use_global=True)
        _, episodes, cohorts, selected = score_draws(predictions, cv, labels, torch.tensor([0, 1, 0, 1]), {})
        self.assertEqual(selected.tolist(), [3, 6])
        altered = dict(labels, local_y=1-labels["local_y"])
        _, _, _, other = score_draws(-predictions, cv, altered, torch.tensor([0, 1, 0, 1]), {})
        torch.testing.assert_close(selected, other, rtol=0, atol=0)
        self.assertEqual(len(episodes), 16)
        self.assertEqual(sum(r["selected"] for r in episodes), 2)
        for row in cohorts:
            self.assertEqual(row["correctness_flips"] + row["always_correct"] + row["always_wrong"], row["queries"])


if __name__ == "__main__":
    unittest.main()
