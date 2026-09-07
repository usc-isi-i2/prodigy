"""Independent torch re-computation from saved prediction tensors and actual inputs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from .replay import batch_hash


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    done = json.loads((a.run/"DONE.json").read_text())
    protocol = json.loads((a.run/"protocol.json").read_text())
    if not done["complete"] or done["smoke"]:
        raise ValueError("complete non-smoke result required")
    output = a.run/"independent_verification.json"
    if output.exists():
        raise ValueError("verification output exists")
    torch.set_num_threads(4)
    with (a.run/"groups.csv").open() as f:
        records = list(csv.DictReader(f))
    cells = defaultdict(list)
    for r in records:
        cells[(r["model_id"], int(r["input_step"]), int(r["checkpoint_step"]), r["mode"])].append(r)
    checked, maximum_error = 0, 0.
    for job in range(9):
        inputs = Path(protocol["inputs"])/f"job_{job:03d}"
        receipt = json.loads((inputs/"DONE.json").read_text())
        for step in range(1, 5):
            b = torch.load(inputs/f"batch_{step:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(b) != receipt["inputs"][step-1]["batch_sha256"]:
                raise ValueError("input changed")
            query = b[5].reshape(-1, 30)[:, 0].bool()
            ids = b[0].global_node_ids[b[0].ptr[:-1]][query]
            tasks = b[0].task_id_per_sample[query]
            truth = b[2][query].argmax(1)
            grouped = defaultdict(list)
            for j, key in enumerate(zip(tasks.tolist(), ids.tolist())):
                grouped[key].append(j)
            for checkpoint in (0, 2500):
                for mode in ("training", "meta_frozen"):
                    key = (receipt["arm"]["model_id"], step, checkpoint, mode)
                    rows = cells.pop(key)
                    lookup = {(int(r["episode"]), int(r["center_id"])): r for r in rows}
                    if len(rows) != len(grouped) or lookup.keys() != grouped.keys():
                        raise ValueError("actual group inventory differs")
                    raw = torch.load(Path(protocol["probes"])/f"job_{job:03d}"/
                        f"input{step}_ckpt{checkpoint}_{mode}.pt", map_location="cpu", weights_only=False)
                    z = raw["conditions"]["baseline"]["logits"].double()
                    for group_key, indices in grouped.items():
                        idx = torch.tensor(indices)
                        y, logits = truth[idx], z[idx]
                        if len(y.unique()) != len(idx):
                            raise ValueError("same-class repetition")
                        # Explicitly enumerate all k cyclic score assignments; no
                        # call to the producer's analytic averaging implementation.
                        lp = logits.log_softmax(1)
                        losses, correct = [], []
                        for shift in range(len(idx)):
                            moved = lp.roll(shift, dims=0)
                            losses.append(float(F.nll_loss(moved, y, reduction="sum")))
                            correct.append(float((moved.argmax(1) == y).sum()))
                        expected = {"size": len(idx), "original_nll_sum": losses[0],
                            "original_correct": correct[0], "symmetrized_nll_sum": sum(losses)/len(idx),
                            "symmetrized_correct": sum(correct)/len(idx)}
                        row = lookup[group_key]
                        for name, value in expected.items():
                            error = abs(float(row[name])-value)
                            maximum_error = max(maximum_error, error)
                            if error > 1e-10:
                                raise ValueError(f"cyclic tensor enumeration differs: {name}, {error}")
                        checked += 1
                    del raw
            del b
        print(f"Verified raw tensor cyclic enumeration: job {job}", flush=True)
    if cells or checked != done["identity_group_rows"]:
        raise ValueError("incomplete raw tensor verification")
    witnesses = torch.load(a.run/"whole_input_witnesses.pt", map_location="cpu", weights_only=False)
    if len(witnesses) != 36:
        raise ValueError("incomplete whole-input witnesses")
    for w in witnesses:
        query = torch.tensor([False]*3+[True]*4).repeat(120)
        positions = torch.where(query)[0]
        reverse = torch.full((840,), -1, dtype=torch.long)
        reverse[positions] = torch.arange(480)
        qp = reverse[w["permutation"][positions]]
        if (qp < 0).any():
            raise ValueError("witness moved support into query")
        expected = w["baseline_logits"][qp]
        for observed in (w["suffix_swapped_logits"], w["whole_swapped_logits"]):
            torch.testing.assert_close(observed, expected, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(w["whole_swapped_pre"], w["expected_pre"], atol=1e-4, rtol=1e-5)
    write_json(output, {"complete": True, "actual_input_batches_rehashed": 36,
        "baseline_tensor_cells": 144, "identity_groups_enumerated": checked,
        "every_cyclic_assignment_enumerated_from_raw_logits": True,
        "maximum_group_statistic_error": maximum_error, "whole_input_tensor_witnesses_rechecked": 36,
        "not_a_new_forward_for_every_cyclic_assignment": True})


if __name__ == "__main__":
    main()
