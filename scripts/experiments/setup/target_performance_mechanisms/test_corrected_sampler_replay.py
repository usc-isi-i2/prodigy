import json
from pathlib import Path
import tempfile
import unittest

from .run_corrected_sampler_replay import CONTRACT, FORWARD, SOURCES, STEPS, PANEL, DECODERS, validate_config, verify_replay


class CorrectedReuseTests(unittest.TestCase):
    def test_effective_source_budget_forward_and_sidecar_contract(self):
        params = {**CONTRACT, **FORWARD, "neighbor_sampling_source_subset": "covid", "neighbor_matching_edge_split": True,
                  "neighbor_sampling_episode_source": "graph_id", "neighbor_sampling_episode_source_weighting": "balanced",
                  "neighbor_sampling_batch_source_mode": "independent", "neighbor_sampling_cross_source_prob": 0.,
                  "use_edge_features": False, "eval_only": False, "ablate_features": "none", "ablate_edges": "none"}
        validate_config(params, "covid", CONTRACT)
        for field, value in (("neighbor_sampling_source_subset", "ukr_rus"), ("epochs", 2), ("emb_dim", 512), ("skip_path", True)):
            with self.assertRaises(ValueError):
                validate_config({**params, field: value}, "covid", CONTRACT)
        with self.assertRaisesRegex(ValueError, "saved training contract"):
            validate_config(params, "covid", {**CONTRACT, "workers": 0})

    def test_complete_replay_and_weight_cell_mutations(self):
        inventory = [{"model_id": f"{s}_{step}", "checkpoint": f"/{s}/{step}", "sources": [s], "weights_sha256": f"weight-{s}-{step}"}
                     for s in SOURCES for step in STEPS]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {}
            for stream in ("original", "fresh"):
                output = Path(tmp) / stream
                outputs[stream] = output
                output.mkdir()
                (output / "DONE").touch()
                for target in PANEL:
                    root = output / target
                    root.mkdir()
                    (root / "cache.json").write_text(json.dumps({"episode_fingerprint": stream + target}))
                    rows = [{**model, "decoder": d, "dataset": target, "variant": "baseline", "episodes": 128,
                             "episode_fingerprint": stream + target, "roc_auc": .7, "accuracy": .6, "f1": .6, "nll": .5}
                            for model in inventory for d in DECODERS]
                    (root / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            self.assertEqual(verify_replay(outputs, inventory), 6120)
            path = outputs["fresh"] / PANEL[-1] / "metrics.jsonl"
            original = path.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["weights_sha256"] = "wrong"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "weights/input identity"):
                verify_replay(outputs, inventory)
            path.write_text("\n".join(original.splitlines()[1:]))
            with self.assertRaisesRegex(ValueError, "complete corrected"):
                verify_replay(outputs, inventory)


if __name__ == "__main__":
    unittest.main()
