import copy
import json
from pathlib import Path
import tempfile
import unittest
from .finish_member_pipeline import GATES, TARGETS, require_substantive_receipt, compare_cached_inputs


class PipelineGateTests(unittest.TestCase):
    def test_requires_substantive_matched_cpu_run(self):
        receipt = dict.fromkeys(GATES, True)
        receipt.update(models=24, steps_per_model=2500, launch_revision="frozen")
        manifest = dict(mode="training", device="cpu", jobs=[{}] * 24, revision="frozen")
        require_substantive_receipt(receipt, manifest, "frozen")
        for key, value in (("research_result", False), ("steps_per_model", 20), ("launch_revision", "other")):
            changed = copy.deepcopy(receipt)
            changed[key] = value
            with self.assertRaises(ValueError):
                require_substantive_receipt(changed, manifest, "frozen")

    def test_cached_features_not_just_node_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {s: root / s for s in ("original", "fresh")}
            for stream, output in outputs.items():
                output.mkdir()
                (output / "DONE").touch()
                for target in TARGETS:
                    prior = "fresh_stage_cpu_20260906" if stream == "fresh" else (
                        "specialist_cpu_tail_20260906" if target in {"twibot20", "ukr_rus_suspended"} else "specialist_cpu_20260906")
                    value = dict(episode_fingerprint=f"{stream}/{target}", batch_sha256=["features"] * 32,
                                 episodes=128, graph_path=f"/{target}.pt")
                    for parent in (output / target, root / prior / target):
                        parent.mkdir(parents=True, exist_ok=True)
                        (parent / "cache.json").write_text(json.dumps(value))
            self.assertEqual(compare_cached_inputs(outputs, root)["stream_target_cells"], 10)
            path = outputs["fresh"] / TARGETS[0] / "cache.json"
            value = json.loads(path.read_text())
            value["batch_sha256"][0] = "changed feature with same node ids"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "batch_sha256"):
                compare_cached_inputs(outputs, root)


if __name__ == "__main__":
    unittest.main()
