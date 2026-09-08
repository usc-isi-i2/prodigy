"""Small CPU tests for the export hot path without importing the training stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]


def export_method():
    tree = ast.parse((ROOT / "experiments/trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainerFS")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prediction_records_for_batch")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "export", "exec"), namespace)
    return namespace[method.name]


class ExportTest(unittest.TestCase):
    def test_support_order_limits_and_task_isolation(self):
        graph = SimpleNamespace(center_node_idx=torch.arange(12),
                                task_id_per_sample=torch.tensor([0]*6+[1]*6),
                                task_label_map=torch.tensor([[101, 102], [201, 202]]))
        labels = torch.eye(2)[torch.tensor([0,1,0,1,0,1]*2)]
        query = torch.tensor([False]*4+[True]*2+[False]*4+[True]*2)
        batch = [graph, None, labels, None, None, query]
        yt = labels[query]
        yp = torch.tensor([[0.,1.], [0.,1.], [0.,1.], [0.,1.]])
        fake = SimpleNamespace(parameter={"export_predictions":True, "prediction_support_per_label":1},
                               dataset_name="toy", _sampled_context_nodes=lambda *a: [])
        method = export_method()
        records = method(fake, batch, yt, yp, "test", 0, {})
        self.assertEqual([r["query_node_id"] for r in records], [4,5,10,11])
        self.assertEqual([[s["node_id"] for s in r["supports"]] for r in records], [[0,1],[1],[6,7],[7]])
        self.assertEqual([r["prediction"] for r in records], [102,102,202,202])
        fake.parameter["prediction_support_per_label"] = 0
        self.assertTrue(all(not r["supports"] for r in method(fake,batch,yt,yp,"test",0,{})))
        fake.parameter["prediction_support_per_label"] = 2
        rows = method(fake,batch,yt,yp,"test",0,{})
        self.assertEqual([s["node_id"] for s in rows[0]["supports"]], [0,1,2,3])


if __name__ == "__main__":
    unittest.main()
