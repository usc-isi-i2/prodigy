import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import run_cardinality_readout as cardinality
from . import run_native as native
from .test_run_native import FIXTURE_EVAL
from . import test_readout_replication as replication_tests


class CardinalityTests(unittest.TestCase):
    def test_command_changes_only_ways_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            upstream = base / "upstream"
            upstream.mkdir()
            (upstream / "kg_commands.py").write_text(
                "in_context_learning_eval_runs = [['ckpt', 'S2,UX,M2', 'InContext_eval_PRODIGY']]\n"
                "n_rels = {'FB15K-237': 200}\n"
                "def print_in_context_learning_evaluation_cmds():\n"
                "    cmd_template = " + repr(FIXTURE_EVAL) + "\n")
            kwargs = dict(upstream=upstream, root=base / "assets", output=base / "out",
                          checkpoint=base / native.NOMINATED_CHECKPOINT, gpu=3)
            baseline = native.build_plan(phase="eval", seed=0, **kwargs)
            for ways in (15, 20):
                plan = cardinality.build_plan(ways=ways, **kwargs)
                restored = plan["command"].copy()
                native.replace_option(restored, "--n_way", "20")
                native.replace_option(restored, "-test_cap", "500")
                self.assertEqual(restored, baseline["command"])
                self.assertEqual(plan["cardinality"]["effective_sampler_seed"], 300455)
                self.assertEqual(plan["cardinality"]["episodes"], 128)
                self.assertFalse((base / "out").exists())
            with self.assertRaises(ValueError):
                cardinality.build_plan(ways=10, **kwargs)
            argv = sum((["--" + name, str(value)] for name, value in kwargs.items()), [])
            with mock.patch.object(native, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as stream:
                cardinality.main(argv + ["--ways", "15"])
            execute.assert_not_called()
            self.assertTrue(json.loads(stream.getvalue())["dry_run"])

    def test_online_preserves_bn_rng_and_predictions(self):
        # Reuse the established two-forward state test but exercise this new hook.
        with mock.patch("scripts.experiments.setup.public_prodigy_kg.test_readout_replication.install_online",
                        side_effect=lambda *args: cardinality.install_online(*args, expected=2)):
            replication_tests.ReplicationTests("test_online_replay_preserves_native_stream").test_online_replay_preserves_native_stream()


if __name__ == "__main__":
    unittest.main()
