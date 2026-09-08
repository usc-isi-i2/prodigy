"""Regression tests for batched episode IDs in private paired-query exports."""
import ast
from itertools import zip_longest
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


class PairingTest(unittest.TestCase):
    def test_tasks_in_same_batch_are_distinct_episodes(self):
        tree = ast.parse(Path(__file__).with_name("cluster_nm_bios.py").read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "read_pairs")
        env = {"pd":pd, "json":json, "zip_longest":zip_longest}
        exec(compile(ast.Module(body=[method],type_ignores=[]),"pairing","exec"),env)
        rows = [dict(dataset="toy",split="test",batch_index=0,task_id=tid,sample_index=tid*210,
                     query_node_id=99,gt=0,gt_local=0,prediction=0,pred_local=0,correct=True,
                     episode_label_map=list(range(30)), probabilities=[1.]+[0.]*29,
                     supports=[dict(node_id=i,local_label=0) for i in range(3)]) for tid in [0,1]]
        with tempfile.TemporaryDirectory() as temporary:
            paths = {key:Path(temporary)/(key+".jsonl") for key in ["ukr","hk"]}
            for path in paths.values():
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            result = env["read_pairs"](paths,"test")
            self.assertEqual(result.episode.nunique(),2)
            self.assertEqual(result.ukr_true_rank.tolist(),[1,1])
            rows[1]["query_node_id"] = 100
            paths["hk"].write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaises(AssertionError):
                env["read_pairs"](paths,"test")


if __name__ == "__main__":
    unittest.main()
