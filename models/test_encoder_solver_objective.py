"""CPU controlled full-forward and AdamW boundary tests (no dataset required)."""
import copy
import unittest
import torch
from torch_geometric.data import Data, Batch
from models.general_gnn import SingleLayerGeneralGNN
from models.layer_classes import BackgroundGNNLayer, SupernodeAggrLayer, MetagraphLayer
from models.encoder_solver_objective import configure_objective, ridge_query_loss


class Encoder(torch.nn.Module, BackgroundGNNLayer):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
    def forward(self, original, x, *args):
        return self.bn(self.linear(x))


class Pool(torch.nn.Module, SupernodeAggrLayer):
    def forward(self, x, edge, idx, batch):
        return x[idx]


class Solver(torch.nn.Module, MetagraphLayer):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x, **kwargs):
        return self.linear(x)


def fixture(mode):
    params = dict(emb_dim=4, layers="S,U,M", task_name="neighbor_matching", n_way=2,
                  zero_shot=False, ignore_label_embeddings=False, zero_label_embeddings=False,
                  skip_path=False, encoder_solver_objective=mode)
    configure_objective(params)
    model = SingleLayerGeneralGNN(torch.nn.ModuleList([Encoder(), Pool(), Solver()]), params=params)
    n = 8
    graphs = Batch.from_data_list([Data(x=torch.randn(1, 4), edge_index=torch.empty(2, 0, dtype=torch.long),
                                       edge_index_supernode=torch.empty(2, 0, dtype=torch.long),
                                       supernode=torch.tensor([0])) for _ in range(n)])
    y = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1] * 2), 2).float()
    q = torch.tensor([False, True, False, True] * 2).repeat_interleave(2)
    targets = torch.arange(2).repeat(n) + torch.tensor([8] * 4 + [10] * 4).repeat_interleave(2)
    edge = torch.stack([torch.arange(n).repeat_interleave(2), targets])
    return model, (graphs, torch.randn(4, 4), y, edge, torch.stack([q.float(), torch.zeros(16)], 1), q)


class IsolationTest(unittest.TestCase):
    def test_losses_gradients_and_adamw(self):
        torch.manual_seed(42)
        original, batch = fixture("joint")
        models = {mode: copy.deepcopy(original) for mode in ("native", "joint", "isolated", "ridge_only")}
        outputs, native_losses, grads = {}, {}, {}
        for mode, model in models.items():
            model.params["encoder_solver_objective"] = mode
            model.encoder_solver_training = True
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            yt, yp, _ = model(*copy.deepcopy(batch))
            outputs[mode] = yp.detach()
            loss = torch.nn.functional.cross_entropy(yp, yt)
            native_losses[mode] = loss.detach()
            if mode == "isolated":
                native_grad = torch.autograd.grad(loss, model.layer_list[0].linear.weight, retain_graph=True, allow_unused=True)[0]
                self.assertIsNone(native_grad)
            if mode == "ridge_only":
                loss = model.encoder_solver_ridge_loss
            elif mode != "native":
                loss = loss + model.encoder_solver_ridge_loss
            loss.backward()
            grads[mode] = model.layer_list[0].linear.weight.grad.clone()
            opt.step()
        for mode in models:
            self.assertTrue(torch.equal(outputs["native"], outputs[mode]))
        self.assertTrue(torch.equal(grads["isolated"], grads["ridge_only"]))
        self.assertFalse(torch.equal(grads["joint"], grads["isolated"]))
        for name, value in models["isolated"].layer_list[0].state_dict().items():
            self.assertTrue(torch.equal(value, models["ridge_only"].layer_list[0].state_dict()[name]), name)
        self.assertTrue(torch.allclose(grads["joint"], grads["native"] + grads["isolated"], atol=1e-6))

    def test_eval_unchanged_and_no_new_parameters(self):
        model, batch = fixture("isolated")
        default = copy.deepcopy(model)
        default.params.pop("encoder_solver_objective")
        a = model(*copy.deepcopy(batch))[1]
        b = default(*copy.deepcopy(batch))[1]
        self.assertTrue(torch.equal(a, b))
        self.assertIsNone(model.encoder_solver_ridge_loss)
        self.assertEqual(list(model.state_dict()), list(default.state_dict()))

    def test_task_separation_and_query_label_not_used_for_fit(self):
        _, batch = fixture("joint")
        z = torch.randn(8, 4, requires_grad=True)
        y, edge, q = batch[2], batch[3], batch[5]
        total = ridge_query_loss(z, y, edge, q)
        first = ridge_query_loss(z[:4], y[:4], edge[:, :8], q[:8])
        second_edge = edge[:, 8:].clone()
        second_edge[0] -= 4
        second = ridge_query_loss(z[4:], y[4:], second_edge, q[8:])
        self.assertTrue(torch.allclose(total, (first + second) / 2))
        total.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_reject_incompatible(self):
        with self.assertRaises(ValueError):
            configure_objective(dict(encoder_solver_objective="isolated", layers="S,U,M2"))


if __name__ == "__main__":
    unittest.main()
