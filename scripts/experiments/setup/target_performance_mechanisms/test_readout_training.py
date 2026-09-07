import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .build_readout_interventions import READOUT_KEYS
from .readout_training_constraint import CONDITIONS, SOURCES, configure_trainer
from .run_readout_training import make_plans
from .verify_readout_training import check_pairs, verify_checkpoint_states


def toy_model():
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.reset_mlp_c = torch.nn.Linear(2, 2)
    layer.reset_mlp_m = torch.nn.Linear(2, 2)
    layer.upstream = torch.nn.Linear(2, 2)
    model.layer_list = torch.nn.ModuleList([layer])
    return model


class ReadoutTrainingTests(unittest.TestCase):
    def test_constraint_preserves_initial_values_rng_and_other_trainability(self):
        for condition in CONDITIONS:
            model = toy_model()
            initial = copy.deepcopy(model.state_dict())
            rng = torch.random.get_rng_state().clone()
            with tempfile.TemporaryDirectory() as tmp:
                trainer = SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters()),
                    parameter={"readout_training_condition": condition}, resume_step=0, logging_dir=tmp)
                receipt = configure_trainer(trainer)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(all(torch.equal(initial[k], v) for k, v in model.state_dict().items()))
                self.assertEqual(receipt["steps_checked"], 0)
                actual = {k for k, p in model.named_parameters() if p.requires_grad}
                self.assertEqual(actual, set(initial) - READOUT_KEYS if condition == "frozen" else set(initial))
                with self.assertRaisesRegex(ValueError, "sequence"):
                    trainer.training_step_observer(1)
                receipt["inputs_hashed"] = 1
                trainer.training_step_observer(1)
                if condition == "frozen":
                    receipt["inputs_hashed"] = 2
                    with torch.no_grad():
                        next(iter(dict(model.named_parameters())[k] for k in READOUT_KEYS)).add_(1)
                    with self.assertRaisesRegex(ValueError, "changed"):
                        trainer.training_step_observer(2)

    def test_checkpoint_audit_catches_frozen_drift_and_idle_other_network(self):
        initial = toy_model().state_dict()
        valid = {k: v + (0 if k in READOUT_KEYS else 1) for k, v in initial.items()}
        self.assertEqual(verify_checkpoint_states(initial, {0: initial, 3: valid}, "frozen"), [0, 3])
        bad = copy.deepcopy(valid)
        bad[next(iter(READOUT_KEYS))].add_(1)
        with self.assertRaises(ValueError):
            verify_checkpoint_states(initial, {3: bad}, "frozen")
        with self.assertRaises(ValueError):
            verify_checkpoint_states(initial, {3: initial}, "frozen")
        with self.assertRaises(ValueError):
            verify_checkpoint_states(initial, {3: valid}, "free")
        initial["normalization.num_batches_tracked"] = torch.tensor(0)
        buffers_only = copy.deepcopy(initial)
        buffers_only["normalization.num_batches_tracked"] += 1
        with self.assertRaisesRegex(ValueError, "other network weights"):
            verify_checkpoint_states(initial, {3: buffers_only}, "frozen")

    def test_complete_pair_grid_and_full_inputs_required(self):
        rows = [{"source": source, "seed": seed, "condition": condition,
                 "initial_sha256": str(seed), "anchor_hashes": ["a"], "member_set_hashes": ["b"],
                 "member_order_hashes": ["c"], "input_hashes": ["d"], "final_walk_rng_sha256": "e",
                 "summary": {"episodes": 10000}}
                for source in SOURCES for seed in range(3) for condition in CONDITIONS]
        check_pairs(rows)
        with self.assertRaises(ValueError):
            check_pairs(rows[:-1])
        rows[1]["input_hashes"] = ["different features or topology"]
        with self.assertRaisesRegex(ValueError, "input_hashes"):
            check_pairs(rows)

    def test_plan_retains_original_recipe_and_only_varies_declared_constraint(self):
        plans = make_plans(Path("/private/tmp/readout-plan-only"))
        self.assertEqual(len(plans), 18)
        self.assertEqual({p["neighbor_sampling_source_subset"] for p in plans}, set(SOURCES))
        ignored = {"prefix", "exp_name", "timestamp", "readout_training_condition"}
        for free, frozen in zip(plans[::2], plans[1::2]):
            self.assertEqual({k: v for k, v in free.items() if k not in ignored},
                             {k: v for k, v in frozen.items() if k not in ignored})
            self.assertEqual(free["epochs"] * free["dataset_len_cap"], 2500)
            self.assertEqual(free["neighbor_matching_member_policy"], "lowest_sorted")
            self.assertEqual(str(free["device"]), "cpu")


if __name__ == "__main__":
    unittest.main()
