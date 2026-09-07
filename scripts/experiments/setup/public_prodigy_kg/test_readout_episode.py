import unittest
import torch

from .readout_episode import ridge_tasks, run_episode
from .run_native import capture_forward_state


def fixture(ways=20, tasks=2):
    per_task = ways * 7
    n = per_task * tasks
    labels = torch.eye(ways).repeat_interleave(7, dim=0).repeat(tasks, 1)
    features = labels.clone()
    query = (torch.arange(n) % 7 >= 3).repeat_interleave(ways)
    destinations = torch.cat([torch.arange(n + task * ways, n + (task + 1) * ways).repeat(per_task)
                              for task in range(tasks)])
    edges = torch.stack([torch.arange(n).repeat_interleave(ways), destinations])
    return features, labels, edges, query


class ReadoutTests(unittest.TestCase):
    def test_real_layout_support_only_and_task_isolation(self):
        x, y, edges, query = fixture()
        scores, tasks = ridge_tasks(x, y, edges, query)
        self.assertEqual(tuple(scores.shape), (160, 20))
        self.assertEqual(len(tasks), 2)
        expected = y[query.reshape(len(x), -1)[:, 0]] * .75
        torch.testing.assert_close(scores, expected)
        changed = y.clone()
        changed[query.reshape(len(x), -1)[:, 0]] = float("nan")
        torch.testing.assert_close(ridge_tasks(x, changed, edges, query)[0], scores)
        changed_x = x.clone()
        changed_x[140:] = torch.roll(changed_x[140:], 1, dims=1)
        torch.testing.assert_close(ridge_tasks(changed_x, y, edges, query)[0][:80], scores[:80])

    def test_task_and_class_permutation(self):
        x, y, edges, query = fixture()
        original, _ = ridge_tasks(x, y, edges, query)
        columns = torch.arange(19, -1, -1)
        permuted_edges = edges.reshape(2, len(x), 20)[:, :, columns].reshape(2, -1)
        changed, _ = ridge_tasks(x, y[:, columns], permuted_edges, query)
        torch.testing.assert_close(changed, original[:, columns])

    def test_rejects_local_edges_and_overlap(self):
        x, y, edges, query = fixture()
        wrong = edges.clone()
        wrong[1] -= len(x)
        with self.assertRaises(ValueError):
            ridge_tasks(x, y, wrong, query)
        wrong = edges.clone().reshape(2, len(x), 20)
        wrong[1, 140:, 0] = len(x)
        with self.assertRaises(ValueError):
            ridge_tasks(x, y, wrong.reshape(2, -1), query)

    def test_hook_and_native_state_restoration(self):
        class Meta(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gnn_layers = torch.nn.ModuleList([])
                self.bn = torch.nn.BatchNorm1d(2)

            def forward(self, *, x, start_right):
                return self.bn(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_list = torch.nn.ModuleList([Meta()])

            def forward(self, x, unused, y, edges, attrs, query):
                z = self.layer_list[0](x=x, start_right=len(x))
                q = query.reshape(len(x), 2)[:, 0]
                return y[q], self.decode(z, unused, edges)[q], x

            def decode(self, input_x, label_x, edges, edgelist_bipartite=False):
                return input_x

        model = Model()
        x, y, edges, query = fixture(2, 1)
        args = (x, torch.zeros(1), y, edges, torch.zeros(1), query)
        state = capture_forward_state(model, torch)
        with torch.no_grad():
            truth, logits, _ = model(*args)
        artifacts = {"input": args, "state": state, "output": {"y_true": truth, "logits": logits}}
        before = capture_forward_state(model, torch)
        result = run_episode(model, artifacts, "cpu")
        self.assertEqual(result["native"]["max_abs_error"], 0)
        self.assertEqual(tuple(result["ridge_logits"].shape), (8, 2))
        after = capture_forward_state(model, torch)
        for name in before["buffers"]:
            self.assertTrue(torch.equal(before["buffers"][name], after["buffers"][name]))
        self.assertTrue(torch.equal(before["torch_rng"], after["torch_rng"]))
        self.assertEqual(before["training"], after["training"])
        self.assertFalse(model.layer_list[0]._forward_pre_hooks)
        result_post = run_episode(model, artifacts, "cpu", compare_post=True)
        expected_post, _ = ridge_tasks(result_post["geometry"]["post"], y, edges, query)
        torch.testing.assert_close(result_post["post_ridge_logits"], expected_post)
        torch.testing.assert_close(result_post["ridge_logits"], result["ridge_logits"])
        self.assertNotIn("decode", model.__dict__)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value.cpu(), before["buffers"][name]))


if __name__ == "__main__":
    unittest.main()
