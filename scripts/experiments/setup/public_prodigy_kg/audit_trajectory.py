"""Print paired correctness transitions from a completed trajectory, read-only."""
import argparse
import json
from pathlib import Path

import torch

from .run_native import file_sha256


def transitions(old, new):
    if old.dtype != torch.bool or new.dtype != torch.bool or old.shape != new.shape:
        raise ValueError("Expected aligned boolean correctness vectors")
    return dict(correct_both=int((old & new).sum()),
                wrong_both=int((~old & ~new).sum()),
                corrected=int((~old & new).sum()), corrupted=int((old & ~new).sum()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.run.resolve()
    if json.loads((root / "execution_status.json").read_text())["status"] != "complete":
        raise ValueError("Trajectory must be complete")
    summary = json.loads((root / "summary.json").read_text())
    expected = {(step, f"{step}/episode_{i:05d}.pt") for step in (2000,4000,8000) for i in range(128)}
    receipts = summary["receipts"]
    if len(receipts) != 384 or {(r["step"],r["file"]) for r in receipts} != expected:
        raise ValueError("Unexpected trajectory inventory")
    for receipt in receipts:
        path = root / receipt["file"]
        if not path.resolve().is_relative_to(root) or file_sha256(path) != receipt["sha256"]:
            raise ValueError("Output receipt mismatch")
    before, after = [], []
    for i in range(128):
        old, new = [torch.load(root / str(step) / f"episode_{i:05d}.pt",
                              map_location="cpu", weights_only=False) for step in (2000,8000)]
        if old["source_sha256"] != new["source_sha256"] or not torch.equal(old["y_true_onehot"],new["y_true_onehot"]):
            raise ValueError("Queries are not paired")
        truth = old["y_true_onehot"].argmax(1)
        for saved, result in ((old,before),(new,after)):
            result.append(torch.stack([saved[key].argmax(1) == truth
                                       for key in ("native_logits","centered_logits")],1))
    old, new = torch.cat(before), torch.cat(after)
    regression = old[:,0] & ~new[:,0]
    recovery = ~old[:,0] & new[:,0]
    print(json.dumps(dict(query_occurrences=len(old),
        native=transitions(old[:,0],new[:,0]), u1=transitions(old[:,1],new[:,1]),
        native_regressions_u1_correct_both=int((regression & old[:,1] & new[:,1]).sum()),
        native_regressions_u1_corrupted=int((regression & old[:,1] & ~new[:,1]).sum()),
        native_recoveries_u1_correct_both=int((recovery & old[:,1] & new[:,1]).sum())),indent=2))


if __name__ == "__main__":
    main()
