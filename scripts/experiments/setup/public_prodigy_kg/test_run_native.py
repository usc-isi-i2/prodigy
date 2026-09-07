"""Pure launcher/protocol tests: no torch, datasets, model forwards, or training."""
import contextlib
import ast
import io
import json
import pickle
from pathlib import Path
import random
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock

from . import run_native as native


FIXTURE_TRAIN = (
    "python3 experiments/run_single_experiment.py --root {root} --dataset Wiki "
    "--emb_dim 256 --device {device} --input_dim 768 --workers 7 --layers S2,UX,M2 "
    "-ds_cap 10010 -lr 1e-3 --prefix Wiki_PT_PRODIGY -esp 500 --eval_step 1000 "
    "--epochs 1 --dropout 0 --n_way 15 -bs 10 -qry 4 -shot 3 --task cls_nm "
    "--ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 "
    "-aug_test True -ckpt_step 1000 --all_test True -attr 1000 -meta_pos True"
)
FIXTURE_EVAL = (
    "python3 experiments/run_single_experiment.py --root {dataset_path} --dataset {ds} "
    "--emb_dim 256 -shot {nshot} --device {device} --input_dim 768 --layers {layers} "
    "-ds_cap 10 --prefix {prefix} -esp 500 --eval_step 1000 --epochs 1 --dropout 0 "
    "--n_way {nway} -bs 1 -qry 4 --ignore_label_embeddings True --task multiway_classification "
    "-test_cap 500 -val_cap 10 --workers 15 {pos_only_msg} --eval_only True {pt_suffix} "
    "--all_test True {suffix_lblsplit}"
)


class NativeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.upstream = self.base / "upstream"
        self.upstream.mkdir()
        (self.upstream / "kg_commands.py").write_text(
            "pretrain_cmds = " + repr({"PRODIGY": FIXTURE_TRAIN}) + "\n"
            "in_context_learning_eval_runs = [['<CKPT>', 'S2,UX,M2', 'InContext_eval_PRODIGY']]\n"
            "n_rels = {'FB15K-237': 200}\n"
            "def print_in_context_learning_evaluation_cmds():\n"
            "    cmd_template = " + repr(FIXTURE_EVAL) + "\n"
            "raise RuntimeError('must never execute upstream recipe file')\n")

    def plan(self, phase="train", **overrides):
        arguments = dict(upstream=self.upstream, root=self.base / "assets with spaces",
                         output=self.base / (phase + " output"), phase=phase, gpu=2)
        arguments.update(overrides)
        return native.build_plan(**arguments)

    @staticmethod
    def option(plan, flag):
        args = plan["command"]
        return args[args.index(flag) + 1]

    def test_train_is_literal_recipe_plus_declared_io_seed(self):
        plan = self.plan()
        expected = shlex.split(FIXTURE_TRAIN.format(
            root=shlex.quote(plan["root"]), device=0))[2:]
        self.assertEqual(plan["command"][2:2 + len(expected)], expected)
        self.assertEqual(self.option(plan, "-ds_cap"), "10010")
        self.assertEqual(self.option(plan, "-bs"), "10")
        self.assertEqual(self.option(plan, "--seed"), "0")
        self.assertEqual(self.option(plan, "--timestamp"), "native_train_seed0")
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(self.option(plan, "--device"), "0")
        self.assertEqual(plan["nomination"]["optimizer_updates"], 8001)
        self.assertFalse(Path(plan["output"]).exists())

    def test_smoke_only_changes_four_limits_and_timestamp(self):
        smoke, train = self.plan("smoke"), self.plan()
        for flag, value in (("-ds_cap", "3"), ("--workers", "0"),
                            ("-val_cap", "1"), ("-test_cap", "1")):
            self.assertEqual(self.option(smoke, flag), value)
        for flag in ("--layers", "--n_way", "-bs", "-qry", "-shot", "--task",
                     "-aug", "-aug_test", "-attr", "-meta_pos", "--dropout"):
            self.assertEqual(self.option(smoke, flag), self.option(train, flag))
        self.assertNotEqual(smoke["output"], train["output"])

    def test_eval_fixed_native_contract(self):
        checkpoint = self.base / "train" / native.NOMINATED_CHECKPOINT
        plan = self.plan("eval", checkpoint=checkpoint, gpu=3)
        for flag, value in (("--dataset", "FB15K-237"), ("--n_way", "20"), ("-shot", "3"),
                            ("-qry", "4"), ("-bs", "1"), ("-test_cap", "500"),
                            ("--workers", "15"), ("--ignore_label_embeddings", "True"),
                            ("--eval_only", "True"), ("--no_split_labels", "True")):
            self.assertEqual(self.option(plan, flag), value)
        args = plan["command"]
        start = args.index("--label_set") + 1
        self.assertEqual(args[start:start + 200], list(map(str, range(200))))
        self.assertNotIn("-train_cap", args)
        self.assertEqual(self.option(plan, "-pretrained"), str(checkpoint))
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "3")
        self.assertIn("train mode", plan["native_protocol_preserved"]["eval_mode"])
        self.assertIn("AddAggregation", plan["native_protocol_preserved"]["aggregation"])

    def test_invalid_phase_gpu_checkpoint_and_paths(self):
        cases = [dict(gpu=0), dict(phase="other"), dict(root=Path("relative")),
                 dict(output=self.upstream / "output"), dict(output=self.base),
                 dict(phase="eval"), dict(phase="eval", checkpoint=self.base / "state_dict_10000.ckpt"),
                 dict(checkpoint=self.base / native.NOMINATED_CHECKPOINT)]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, FileExistsError)):
                self.plan(**arguments)
        existing = self.base / "already_exists"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan(output=existing)

    def test_pin_and_clean_verification(self):
        with mock.patch.object(native.subprocess, "check_output",
                               side_effect=[native.UPSTREAM_COMMIT + "\n", ""]):
            self.assertTrue(native.verify_upstream(self.upstream)["clean"])
        with mock.patch.object(native.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaises(ValueError):
                native.verify_upstream(self.upstream)
        with mock.patch.object(native.subprocess, "check_output",
                               side_effect=[native.UPSTREAM_COMMIT, "?? changed.py"]):
            with self.assertRaises(ValueError):
                native.verify_upstream(self.upstream)

    def test_dry_run_never_executes_or_creates_output(self):
        args = ["--phase", "smoke", "--upstream", str(self.upstream), "--root", str(self.base / "assets"),
                "--output", str(self.base / "smoke"), "--gpu", "2"]
        with mock.patch.object(native, "verify_upstream", return_value={"clean": True}), \
                mock.patch.object(native, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as printed:
            native.main(args)
        execute.assert_not_called()
        self.assertTrue(json.loads(printed.getvalue())["dry_run"])
        self.assertFalse((self.base / "smoke").exists())

    def test_receipts_must_match_and_assets_must_exist(self):
        stage = Path(native.__file__).with_name("stage_assets.py")
        expected = native.literal_assignment(stage, "ASSETS")["Wiki"]
        extras = native.literal_assignment(stage, "EXTRA_REQUIRED")["Wiki"]
        feature = native.literal_assignment(stage, "FEATURE_FILE")
        root, dataset = self.base / "assets", "Wiki"
        target = root / dataset
        for relative in ("graph.pt", "Wiki_adj.pt", feature, *extras):
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture only, never unpickled")
        receipts = root / "receipts"
        receipts.mkdir()
        receipt = dict(dataset=dataset, status="complete", expected=expected,
                       target=str(target), url="https://snap.stanford.edu/prodigy/Wiki.zip", sha256="a" * 64)
        path = receipts / "Wiki.json"
        path.write_text(json.dumps(receipt))
        self.assertEqual(len(native.verify_receipts(root, [dataset])), 1)
        receipt["status"] = "extracting"
        path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            native.verify_receipts(root, [dataset])
        receipt["status"] = "complete"
        path.write_text(json.dumps(receipt))
        (target / "graph.pt").unlink()
        with self.assertRaises(ValueError):
            native.verify_receipts(root, [dataset])

    def test_receipt_preflight_failure_happens_before_runtime_or_outputs(self):
        plan = self.plan()
        with mock.patch.object(native, "verify_upstream", return_value={"clean": True}), \
                mock.patch.object(native, "runtime_versions") as versions:
            with self.assertRaises(FileNotFoundError):
                native.execute(plan)
        versions.assert_not_called()
        self.assertFalse(Path(plan["output"]).exists())

    def test_no_model_import_during_planning(self):
        before = set(sys.modules)
        self.plan()
        self.assertFalse({"torch", "torch_geometric", "models", "experiments.trainer"} & (set(sys.modules) - before))

    def test_legacy_set_sampling_matches_tuple_and_preserves_rng_population(self):
        population = {9, 1, 20, 4, 11}
        original = population.copy()
        actual, expected = random.Random(123), random.Random(123)
        proxy = native.LegacySetSampleRNG(actual)
        for _ in range(10):
            self.assertEqual(proxy.sample(population, 3), expected.sample(tuple(population), 3))
            self.assertEqual(proxy.sample(range(30), 4), expected.sample(range(30), 4))
            self.assertEqual(proxy.choices(range(4), k=3), expected.choices(range(4), k=3))
        self.assertEqual(actual.getstate(), expected.getstate())
        self.assertEqual(population, original)

    def test_scoped_compatibility_shim_only_applied_on_311(self):
        class Task:
            def sample(self, num_label, num_member, num_shot, num_query, rng):
                return rng.sample({1, 2, 5}, num_label)

        original = Task.sample
        self.assertFalse(native.install_sampler_compat(Task, (3, 10))["active"])
        self.assertIs(Task.sample, original)
        self.assertTrue(native.install_sampler_compat(Task, (3, 11))["active"])
        actual = Task().sample(2, 0, 0, 0, random.Random(8))
        self.assertEqual(actual, random.Random(8).sample(tuple({1, 2, 5}), 2))

    def test_namespace_imports_accept_native_reject_mixed(self):
        namespace = types.ModuleType("experiments")
        namespace.__path__ = [str(self.upstream / "experiments")]
        with mock.patch.dict(native.sys.modules, {"experiments": namespace}, clear=True):
            self.assertIn("namespace_paths", native.verify_imports(self.upstream)["experiments"])
            namespace.__path__.append(str(self.base / "foreign"))
            with self.assertRaises(RuntimeError):
                native.verify_imports(self.upstream)

    def test_actual_upstream_recipe_and_sampler_without_importing_models(self):
        # Optional local integration: AST only, never execute imports/construct datasets.
        pinned = Path("/private/tmp/gfm-public-prodigy.tmd74G/upstream")
        if not (pinned / "kg_commands.py").exists():
            self.skipTest("Pinned local audit clone is not present")
        self.assertEqual(native.verify_upstream(pinned)["head"], native.UPSTREAM_COMMIT)
        plan = self.plan(upstream=pinned)
        self.assertEqual(self.option(plan, "--layers"), "S2,UX,M2")
        source = ast.parse((pinned / "data/dataloader.py").read_text())
        klass = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MulticlassTask")
        sample = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
        namespace = {}
        exec(compile(ast.Module(body=[sample], type_ignores=[]), "<native MulticlassTask.sample AST>", "exec"), namespace)
        task = types.SimpleNamespace(linear_probe=False, label_set={0, 1}, train_label=None)
        # num_label=0 isolates the real failing sample(set) line without fake data arrays.
        if sys.version_info[:2] == (3, 11):
            with self.assertRaises(TypeError):
                namespace["sample"](task, 0, 0, 0, 0, random.Random(0))
        self.assertEqual(namespace["sample"](task, 0, 0, 0, 0,
                                             native.LegacySetSampleRNG(random.Random(0))), {})

    def test_episode_capture_preserves_actual_inputs_and_outputs(self):
        # Standard-library stand-ins exercise hooks/serialization, not a model.
        class HookSurface:
            def register_forward_pre_hook(self, hook):
                self.pre = hook

            def register_forward_hook(self, hook):
                self.post = hook

        class Value:
            def __init__(self, values):
                self.values = list(values)

            def clone(self):
                return Value(self.values)

            def cpu(self):
                return self

            def detach(self):
                return self

        saved = {}

        def save(value, path):
            saved[str(path)] = value
            # Disk payload only supports digest testing; saved holds clone identities.
            Path(path).write_bytes(pickle.dumps(str(path)))

        output = self.base / "capture_output"
        output.mkdir()
        surface = HookSurface()
        native.install_episode_capture(surface, output=output,
                                       torch_module=types.SimpleNamespace(save=save))
        arguments = (Value([1, 2]), Value([3]))
        with mock.patch.object(native, "capture_forward_state", return_value={"fixture": True}):
            self.assertIsNone(surface.pre(surface, arguments))
        arguments[0].values[0] = 999  # Stand-in for native in-place graph mutation.
        result = (Value([0, 1]), Value([0.25, 0.75]), arguments[0])
        self.assertIsNone(surface.post(surface, arguments, result))
        result[1].values[0] = 888
        index = json.loads((output / "episode_capture/index.json").read_text())
        self.assertEqual(index["captured_batches"], 1)
        record = index["records"][0]
        self.assertEqual(index["schema_version"], 2)
        self.assertEqual(record["state_sha256"], native.file_sha256(output / "episode_capture" / record["state_file"]))
        captured_inputs = saved[str(output / "episode_capture" / record["input_file"])]
        captured_outputs = saved[str(output / "episode_capture" / record["output_file"])]
        self.assertEqual(captured_inputs[0].values, [1, 2])
        self.assertEqual(captured_outputs["logits"].values, [0.25, 0.75])
        self.assertEqual(arguments[0].values, [999, 2])
        self.assertEqual(record["input_sha256"], native.file_sha256(output / "episode_capture" / record["input_file"]))
        self.assertEqual(record["output_sha256"], native.file_sha256(output / "episode_capture" / record["output_file"]))


if __name__ == "__main__":
    unittest.main()
