"""Freeze exactly the initialized readout; audit every consumed input and update."""
import gzip
import json
from pathlib import Path

import torch

from .build_readout_interventions import READOUT_KEYS
from .replay import batch_hash
from .verify_member_training import model_digest

CONDITIONS = ("free", "frozen")
SOURCES = ("ukr_rus", "cp_hk", "covid")


def configure_trainer(trainer):
    condition = trainer.parameter["readout_training_condition"]
    if condition not in CONDITIONS or trainer.resume_step or trainer.optimizer.state:
        raise ValueError("requires a declared condition and unused initial optimizer")
    parameters = dict(trainer.model.named_parameters())
    if not READOUT_KEYS <= parameters.keys() or any(not parameters[k].requires_grad for k in READOUT_KEYS):
        raise ValueError("expected four initially trainable readout tensors")
    before = {name for name, p in parameters.items() if p.requires_grad}
    initial = {k: parameters[k].detach().clone() for k in READOUT_KEYS}
    if condition == "frozen":
        frozen_ids = {id(parameters[k]) for k in READOUT_KEYS}
        for key in READOUT_KEYS:
            parameters[key].requires_grad_(False)
        for group in trainer.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in frozen_ids]
    wanted = before - READOUT_KEYS if condition == "frozen" else before
    by_id = {id(p): name for name, p in parameters.items()}

    def optimizer_names():
        return [[by_id[id(p)] for p in group["params"]] for group in trainer.optimizer.param_groups]

    names = optimizer_names()
    if set(sum(names, [])) != wanted or sum(map(len, names)) != len(wanted):
        raise ValueError("optimizer does not contain exactly the declared trainable parameters")
    receipt = {"condition": condition, "initial_model_sha256": model_digest(trainer.model.state_dict()),
               "initial_readout_sha256": model_digest(initial), "readout_keys": sorted(READOUT_KEYS),
               "optimizer_parameter_names": names, "other_trainable_parameters_unchanged": True,
               "steps_checked": 0, "inputs_hashed": 0}
    trainer.training_constraint_receipt = receipt
    input_path = Path(trainer.logging_dir) / "constraint_inputs.jsonl.gz"
    if input_path.exists():
        raise ValueError("refusing to append to an existing input audit")

    def capture_inputs(module, inputs):
        if not module.training:
            return
        receipt["inputs_hashed"] += 1
        row = {"step": receipt["inputs_hashed"], "batch_sha256": batch_hash(inputs)}
        with gzip.open(input_path, "at", compresslevel=1) as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def check_step(step):
        if step != receipt["steps_checked"] + 1 or receipt["inputs_hashed"] != step:
            raise ValueError("input/update audit sequence differs")
        if {k for k, p in parameters.items() if p.requires_grad} != wanted or optimizer_names() != names:
            raise ValueError("trainable parameter or optimizer inventory changed")
        if condition == "frozen" and any(
                parameters[k].grad is not None or not torch.equal(parameters[k], initial[k]) for k in READOUT_KEYS):
            raise ValueError("frozen readout changed or acquired gradients")
        receipt["steps_checked"] = step

    trainer.model.register_forward_pre_hook(capture_inputs)
    trainer.training_step_observer = check_step
    return receipt
