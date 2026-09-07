"""Provenance gates plus small actual-upstream CPU replay; no public data."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from . import evaluate_cached as runner


def common_params():
    return dict(layers="S2,UX,M2", emb_dim=256, input_dim=770, dropout=0,
                text_features_dropout=0, ignore_label_embeddings=True,
                zero_label_embeddings=False, meta_gnn_pos_only=True, has_final_back=False,
                no_bn_encoder=False, no_bn_metagraph=False, skip_path=False,
                zero_shot=False, not_freeze_learned_label_embedding=False,
                reset_after_layer=None, n_shots=3, n_query=4, seed=0,
                attention_mask_scheme="causal")


class CachedRunnerTests(unittest.TestCase):
    def test_accuracy_amendment_preserves_failed_logit_verdict(self):
        row = dict(passed=False, labels_equal=True, argmax_equal=True,
                   finite=True, shape_dtype_equal=True)
        replay = dict(passed=False, atol=runner.REPLAY_ATOL, rtol=runner.REPLAY_RTOL,
                      per_episode=[dict(row) for _ in range(500)])
        original = copy.deepcopy(replay)
        self.assertFalse(runner.replay_accepted(replay))
        self.assertTrue(runner.replay_accepted(replay, "accuracy-equivalence-20260907"))
        self.assertEqual(replay, original)
        for key in ("labels_equal", "argmax_equal", "finite", "shape_dtype_equal"):
            bad = copy.deepcopy(replay)
            bad["per_episode"][0][key] = False
            self.assertFalse(runner.replay_accepted(bad, "accuracy-equivalence-20260907"))
        self.assertFalse(runner.replay_accepted(dict(replay, per_episode=replay["per_episode"][:-1]),
                                               "accuracy-equivalence-20260907"))
        self.assertFalse(runner.replay_accepted(dict(replay, atol=1e-2), "accuracy-equivalence-20260907"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.eval_run = self.base / "native_eval"
        self.train = self.base / "train"
        self.eval_run.mkdir()
        self.train.mkdir()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_bundle(self):
        train_params = dict(common_params(), dataset="Wiki", n_way=15, batch_size=10,
                            dataset_len_cap=10010, epochs=1, task_name="cls_nm",
                            augmentation="ND0.5,NZ0.5", augment_test=True,
                            attr_regression_weight=1000, exp_name="native_train")
        params = dict(common_params(), dataset="FB15K-237", n_way=20, batch_size=1,
                      test_len_cap=500, train_cap=None, no_split_labels=True,
                      task_name="multiway_classification", eval_only=True,
                      augmentation="", augment_test=False, all_test=True,
                      label_set=list(map(str, range(200))))
        checkpoint = self.train / "state/native_train/checkpoint" / runner.native.NOMINATED_CHECKPOINT
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture, never load")
        (self.train / "initial_model.ckpt").write_bytes(b"initial fixture, never load")
        self.write(self.train / "effective_params.json", train_params)
        self.write(self.eval_run / "effective_params.json", params)
        self.write(self.train / "protocol.json", dict(phase="train", dry_run=False, dataset="Wiki",
                   seed=0, upstream_commit=runner.native.UPSTREAM_COMMIT))
        self.write(self.eval_run / "protocol.json", dict(phase="eval", dry_run=False, dataset="FB15K-237",
                   seed=0, upstream_commit=runner.native.UPSTREAM_COMMIT, checkpoint=str(checkpoint),
                   checkpoint_sha256=runner.native.file_sha256(checkpoint)))
        self.write(self.eval_run / "execution_status.json", {"status": "complete"})
        self.write(self.eval_run / "model_contract.json", {"training": True, "batch_norm": {"bn": {"training": True}}})
        self.write(self.eval_run / "eval_metrics.json", {"events": [{"split": "test", "batches": 500,
                   "model_training_before": True}]})
        capture = self.eval_run / "episode_capture"
        capture.mkdir()
        records = []
        for ordinal in range(500):
            record = {"ordinal": ordinal}
            for kind, prefix in (("input", "batch"), ("output", "output")):
                path = capture / f"{prefix}_{ordinal:05d}.pt"
                path.write_bytes(f"{kind}{ordinal}, not a pickle".encode())
                record[kind + "_file"] = path.name
                record[kind + "_sha256"] = runner.native.file_sha256(path)
            records.append(record)
        self.write(capture / "index.json", {"captured_batches": 500, "records": records})
        return checkpoint

    def test_complete_hashed_bundle_and_same_train_nomination(self):
        checkpoint = self.make_bundle()
        bundle = runner.validate_bundle(self.eval_run, self.train)
        self.assertEqual(len(bundle["entries"]), 500)
        self.assertEqual(bundle["checkpoint"], checkpoint)
        self.assertEqual(bundle["identity"]["checkpoint_sha256"], runner.native.file_sha256(checkpoint))

    def test_bad_hash_missing_episode_and_ordinal_rejected(self):
        self.make_bundle()
        path = self.eval_run / "episode_capture/batch_00002.pt"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            runner.validate_bundle(self.eval_run, self.train)
        index_path = self.eval_run / "episode_capture/index.json"
        index = runner.read_json(index_path)
        index["records"][0]["ordinal"] = 1
        self.write(index_path, index)
        with self.assertRaisesRegex(ValueError, "ordinal"):
            runner.validate_bundle(self.eval_run, self.train)
        index["records"].pop()
        self.write(index_path, index)
        with self.assertRaisesRegex(ValueError, "500"):
            runner.validate_bundle(self.eval_run, self.train)

    def test_train_smoke_mode_or_foreign_checkpoint_rejected(self):
        checkpoint = self.make_bundle()
        protocol_path = self.train / "protocol.json"
        protocol = runner.read_json(protocol_path)
        protocol["phase"] = "smoke"
        self.write(protocol_path, protocol)
        with self.assertRaisesRegex(ValueError, "phase"):
            runner.validate_bundle(self.eval_run, self.train)
        protocol["phase"] = "train"
        self.write(protocol_path, protocol)
        ep_path = self.eval_run / "protocol.json"
        ep = runner.read_json(ep_path)
        ep["checkpoint"] = str(checkpoint.with_name("state_dict_10000.ckpt"))
        self.write(ep_path, ep)
        with self.assertRaisesRegex(ValueError, "specified full training"):
            runner.validate_bundle(self.eval_run, self.train)

    def test_quality_gate_rejects_failed_missing_or_foreign_identity(self):
        from . import decision
        quality = self.base / "quality"
        quality.mkdir()
        self.write(quality / "execution_status.json", {"status": "complete"})
        native, raw, untrained = [0.8] * 500, [0.3] * 500, [0.2] * 500
        replay = dict(passed=True, atol=runner.REPLAY_ATOL, rtol=runner.REPLAY_RTOL,
                      per_episode=[dict(passed=True)] * 500)
        result = dict(phase="quality", identity={"id": "same"}, completed_episodes=500,
                      replay=replay, decision=decision.decide_quality(native, raw, untrained),
                      native=dict(episode_accuracy=native, replay=replay),
                      raw=dict(episode_accuracy=raw), untrained=dict(episode_accuracy=untrained))
        self.write(quality / "quality.json", result)
        self.assertTrue(runner.validate_quality_gate(quality, {"id": "same"})["decision"]["gate_passed"])
        amended = copy.deepcopy(result)
        amended["replay_policy"] = "accuracy-equivalence-20260907"
        amended["replay"]["passed"] = False
        amended["replay"]["per_episode"] = [dict(passed=False, labels_equal=True,
            argmax_equal=True, finite=True, shape_dtype_equal=True) for _ in range(500)]
        amended["native"]["replay"] = amended["replay"]
        self.write(quality / "quality.json", amended)
        with self.assertRaises(ValueError):
            runner.validate_quality_gate(quality, {"id": "same"})
        self.assertTrue(runner.validate_quality_gate(quality, {"id": "same"},
            "accuracy-equivalence-20260907")["decision"]["gate_passed"])
        amended["replay"]["per_episode"][17]["argmax_equal"] = False
        self.write(quality / "quality.json", amended)
        with self.assertRaises(ValueError):
            runner.validate_quality_gate(quality, {"id": "same"}, "accuracy-equivalence-20260907")
        for changed in (dict(result, replay={"passed": False}), dict(result, decision={"gate_passed": False}),
                        dict(result, completed_episodes=499), dict(result, identity={"id": "other"}),
                        dict(result, raw=dict(episode_accuracy=[0.9] * 500)),
                        dict(result, raw=dict(episode_accuracy=raw[:-1])),
                        dict(result, replay=dict(replay, per_episode=replay["per_episode"][:-1])),
                        dict(result, replay=dict(replay, atol=1e-2))):
            self.write(quality / "quality.json", changed)
            with self.assertRaises(ValueError):
                runner.validate_quality_gate(quality, {"id": "same"})

    def test_gpu_busy_rejected(self):
        with mock.patch.object(runner.subprocess, "check_output", return_value="4118917\n"):
            with self.assertRaisesRegex(RuntimeError, "already has compute"):
                runner.ensure_gpu_idle(2)
        with mock.patch.object(runner.subprocess, "check_output", return_value=""):
            runner.ensure_gpu_idle(3)

    def test_dry_run_defaults_gpu3_and_never_executes(self):
        args = ["--phase", "quality", "--upstream", str(self.base / "upstream"),
                "--native-eval", str(self.eval_run), "--train-run", str(self.train),
                "--output", str(self.base / "quality")]
        with mock.patch.object(runner.native, "verify_upstream", return_value={"clean": True}), \
                mock.patch.object(runner, "execute") as execute, \
                contextlib.redirect_stdout(io.StringIO()) as captured:
            runner.main(args)
        plan = json.loads(captured.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["gpu"], 3)
        self.assertEqual(plan["replay_gate"]["atol"], 1e-6)
        self.assertEqual(plan["replay_gate"]["rtol"], 1e-5)
        self.assertIsNone(plan["protocol"]["private_final_state_capture"])
        execute.assert_not_called()
        self.assertFalse((self.base / "quality").exists())

    def test_roles_dry_plan_declares_observation_only_states(self):
        args = ["--phase", "roles", "--upstream", str(self.base / "upstream"),
                "--native-eval", str(self.eval_run), "--train-run", str(self.train),
                "--quality-run", str(self.base / "quality"), "--output", str(self.base / "roles")]
        with mock.patch.object(runner.native, "verify_upstream", return_value={"clean": True}), \
                mock.patch.object(runner, "execute") as execute, \
                contextlib.redirect_stdout(io.StringIO()) as captured:
            runner.main(args)
        capture = json.loads(captured.getvalue())["protocol"]["private_final_state_capture"]
        self.assertEqual(capture["arms"], ["native", "support", "query"])
        self.assertEqual(capture["stages"], runner.STATE_STAGES)
        self.assertIn("no added forwards", capture["scope"])
        execute.assert_not_called()
        self.assertFalse((self.base / "roles").exists())

    def test_real_upstream_synthetic_constructor_replay_and_roles(self):
        upstream = Path(os.environ.get("PRODIGY_TEST_UPSTREAM",
                                       "/private/tmp/gfm-public-prodigy.tmd74G/upstream"))
        python = Path(os.environ.get("PRODIGY_TEST_PYTHON",
                                     "/Users/philipp/miniconda3/envs/prodigy/bin/python"))
        if not upstream.exists() or not python.exists():
            self.skipTest("Local pinned upstream and prodigy runtime unavailable")
        # Independent process prevents upstream namespace modules contaminating
        # other repo tests. This uses synthetic tensors only, on CPU.
        module_name = __name__
        script = (f"from {module_name} import synthetic_worker; "
                  f"synthetic_worker({str(upstream)!r})")
        completed = subprocess.run([str(python), "-B", "-c", script], text=True,
                                   capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


def synthetic_worker(upstream_path):
    """Actual unmodified native model, tiny synthetic two-class CPU streams."""
    import types
    import numpy as np
    upstream = Path(upstream_path)
    runner.native.verify_upstream(upstream)
    torch, layers, general = runner.import_runtime(upstream)
    torch.set_num_threads(1)
    from scripts.experiments.setup.public_prodigy_kg import cached_tensors as helpers
    from scripts.experiments.setup.public_prodigy_kg.test_cached_tensors import make_episode
    params = common_params()
    # Native construction reference uses exact upstream classes; its full state
    # is loaded strictly by our constructor, including unused/frozen branches.
    torch.manual_seed(9)
    label = torch.nn.Linear(768, 256)
    modules = layers.get_module_list("S2,UX,M2", 256, 768, 770, 0, None, "causal", False, True,
                                     batch_norm_encoder=True, batch_norm_metagraph=True, gnn_use_relu=True)
    reference_model = general.SingleLayerGeneralGNN(torch.nn.ModuleList(modules), initial_label_mlp=label,
                                                    params=params, text_dropout=torch.nn.Dropout(0))
    for parameter in reference_model.learned_label_embedding.parameters():
        parameter.requires_grad = False
    state = {name: value.detach().clone() for name, value in reference_model.state_dict().items()}
    model = runner.construct_model(params, state, "cpu")
    assert set(model.state_dict()) == set(state)
    malformed = dict(state)
    malformed.pop(next(iter(malformed)))
    try:
        runner.construct_model(params, malformed, "cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Strict model-state loading did not reject a missing key")
    arguments = [make_episode(), make_episode()]
    original_arguments = [runner.clone_arguments(args, "cpu") for args in arguments]
    rng = runner.rng_snapshot(torch, np)
    runner.restore_rng(rng, torch, np)
    references = []
    with torch.no_grad():
        reference_model.train(True)
        for args in arguments:
            y, logits, _ = reference_model(*runner.clone_arguments(args, "cpu"))
            references.append({"y_true": y.clone(), "logits": logits.clone()})
    small_helpers = types.SimpleNamespace(
        validate_episode=lambda args: helpers.validate_episode(args, expected_ways=2),
        suppress_background=helpers.suppress_background,
        cosine_prototype_logits=helpers.cosine_prototype_logits)
    run = lambda arm, **kwargs: runner.model_stream(model, state, zip(arguments, references),
              device="cpu", helpers=small_helpers, rng=rng, numpy=np, arm=arm, **kwargs)
    # Independent observation of the same existing UX forwards checks the saved
    # pre-M stage. There are no extra model forwards for state capture.
    observed_ux = []
    ux_handle = model.layer_list[1].proj.register_forward_hook(
        lambda _module, _args, output: observed_ux.append(output.detach().cpu().clone()))
    temporary_states = tempfile.TemporaryDirectory()
    state_output = Path(temporary_states.name)
    def capture_sink(arm):
        writer = runner.final_state_sink(state_output, arm, torch)
        def sink(ordinal, payload):
            assert torch.equal(payload["pre_meta_points"], observed_ux[-1])
            assert payload["stages"] == runner.STATE_STAGES
            for value in payload.values():
                if torch.is_tensor(value):
                    assert value.device.type == "cpu" and not value.requires_grad
            writer(ordinal, payload)
            # Hostile callback mutation must not touch model params, source
            # episode tensors, saved query-drift references, or the file.
            for value in payload.values():
                if torch.is_tensor(value):
                    value.zero_()
        return sink
    intact, queries = run("native", collect_queries=True, state_sink=capture_sink("native"))
    assert intact["replay"]["passed"] and intact["replay"]["max_abs_error"] == 0
    for count in intact["batch_norm_final_counts"].values():
        assert count == 2
    again, _ = run("native")
    assert again["episode_accuracy"] == intact["episode_accuracy"]
    assert again["batch_norm_final_counts"] == intact["batch_norm_final_counts"]
    untrained, _ = run("untrained")
    assert len(untrained["episode_accuracy"]) == 2 and untrained["weights_unchanged"]
    for arm in ("support", "query"):
        result, _ = run(arm, query_reference=queries, state_sink=capture_sink(arm))
        assert len(result["query_drift"]) == 2 and result["weights_unchanged"]
    ux_handle.remove()
    for arm in ("native", "support", "query"):
        directory = state_output / "final_states" / arm
        index = runner.read_json(directory / "index.json")
        assert index["saved_episodes"] == 2 and index["stages"] == runner.STATE_STAGES
        for ordinal, record in enumerate(index["records"]):
            path = directory / record["file"]
            assert record["sha256"] == runner.native.file_sha256(path)
            captured_state = torch.load(path, map_location="cpu")
            assert captured_state["ordinal"] == ordinal and captured_state["arm"] == arm
            assert captured_state["pre_meta_points"].shape == (14, 256)
            assert captured_state["final_points"].shape == (14, 256)
            assert captured_state["final_queries"].shape == (8, 256)
            assert captured_state["final_labels"].shape == (2, 256)
            assert torch.equal(captured_state["final_queries"],
                               captured_state["final_points"][captured_state["is_query"]])
            assert torch.equal(captured_state["query_point_indices"],
                               torch.where(captured_state["is_query"])[0])
            reconstructed = torch.nn.functional.cosine_similarity(
                captured_state["final_queries"][:, None, :],
                captured_state["final_labels"][None, :, :], dim=-1,
                eps=captured_state["cosine_eps"]) * captured_state["logit_scale"].exp()
            torch.testing.assert_close(reconstructed, captured_state["query_logits"], atol=1e-6, rtol=1e-5)
            assert torch.equal(reconstructed.argmax(-1), captured_state["query_logits"].argmax(-1))
            assert torch.equal(captured_state["query_y_true"], references[ordinal]["y_true"])
            if arm == "native":
                assert torch.equal(captured_state["query_logits"], references[ordinal]["logits"])
                assert torch.equal(captured_state["final_queries"], queries[ordinal])
    # Shape/order errors are rejected before a sink can label the state as valid.
    sample = torch.load(state_output / "final_states/native/episode_00000.pt", map_location="cpu")
    synthetic_capture = {"descriptor": sample["pre_meta_points"],
                         "meta": torch.cat((sample["final_points"], sample["final_labels"]))}
    info = helpers.validate_episode(arguments[0], expected_ways=2)
    for bad_capture, bad_info in (
        (dict(synthetic_capture, meta=synthetic_capture["meta"][:-1]), info),
        (synthetic_capture, dict(info, labels=1 - info["labels"])),
    ):
        try:
            runner.final_state_payload(model, bad_capture, arguments[0], bad_info,
                                       sample["query_y_true"], sample["query_logits"], ordinal=0, arm="native")
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed final-state shape/order was accepted")
    temporary_states.cleanup()
    for args, before in zip(arguments, original_arguments):
        assert args[0].x.shape[1] == 770
        for key in before[0].keys:
            assert torch.equal(args[0][key], before[0][key])
        for actual, expected in zip(args[1:], before[1:]):
            assert torch.equal(actual, expected)
    reference = references[0]
    perturbed = reference["logits"].clone()
    perturbed[0, 0] += 0.1
    assert not runner.compare_native(reference["y_true"], perturbed, reference)["passed"]


if __name__ == "__main__":
    unittest.main()
