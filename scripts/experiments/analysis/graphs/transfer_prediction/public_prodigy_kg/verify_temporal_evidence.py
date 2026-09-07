"""Verify exported initialization-specific summaries and print paper tables.

Prediction tensors remain on Tucker. Their hashes were checked before export;
this script checks local summary identity, inventory and aggregation, not the
underlying model execution or independent-sample statistical inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

RUNS = {
    "trajectory": ("publickg_trajectory_20260907",384,"7881b468f089cc2f2dbe3453dd44a8dc6d2ac4e771bc0300a958948954f5bf3f"),
    "crossover": ("publickg_crossover_20260907",512,"507e1169eb782be76223643ec3b81eb2ba8ca5a3faccf3d7cfb3805519f5d672"),
    "normalization": ("publickg_normalization_20260907",256,"7eb3cee78da8f38c47975a29108d3d0f3e68d1a6a6c295568e4401344285b552"),
}
SEED1_RUNS = {
    "trajectory": ("publickg_trajectory_seed1_20260907",384,"ed540a037da01d56828e4e5d850e8fe26a3ca516adb21f6240858459584d2aa0"),
    "crossover": ("publickg_crossover_seed1_20260907",512,"22c68738f422d2815f8661f5ef2039ce6004dbd79c3edb138673654d07ecfdb9"),
    "normalization": ("publickg_normalization_seed1_20260907",256,"3827e707ff2c6b27df2a6e536ef69d7228477644f77f09caace86930312dd394"),
}


def verify(root, runs=RUNS):
    result = {}
    source_hashes = set()
    for kind,(name,count,digest) in runs.items():
        directory = root/name
        payload = (directory/"summary.json").read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:
            raise ValueError(f"Changed exported summary: {name}")
        summary = json.loads(payload)
        protocol = json.loads((directory/"protocol.json").read_text())
        if json.loads((directory/"execution_status.json").read_text())["status"]!="complete":
            raise ValueError(f"Incomplete run: {name}")
        source_hashes.add(protocol["source_index_sha256"])
        if len(summary["rows"])!=count or len(summary["receipts"])!=count:
            raise ValueError(f"Incomplete inventory: {name}")
        if len({r["file"] for r in summary["receipts"]})!=count:
            raise ValueError(f"Duplicate artifact receipt: {name}")
        groups = {}
        for row in summary["rows"]:
            key = row["condition"] if kind=="crossover" else str(row["step"])
            groups.setdefault(key,[]).append(row)
        expected = {"E2000I2000","E2000I8000","E8000I2000","E8000I8000"} if kind=="crossover" else ({"2000","4000","8000"} if kind=="trajectory" else {"2000","8000"})
        if set(groups)!=expected or set(summary["means"])!=expected:
            raise ValueError(f"Unexpected conditions: {name}")
        for key,rows in groups.items():
            if len(rows)!=128 or {r["ordinal"] for r in rows}!=set(range(128)):
                raise ValueError(f"Unpaired episode inventory: {name}/{key}")
            means = summary["means"][key]
            entries = [(None,means)] if kind=="crossover" else list(means.items())
            for arm,metrics in entries:
                for metric,value in metrics.items():
                    values = [r["metrics"][metric] if arm is None else r["metrics"][arm][metric] for r in rows]
                    if not all(math.isfinite(v) for v in values) or not math.isclose(math.fsum(values)/128,value,rel_tol=0,abs_tol=1e-12):
                        raise ValueError(f"Aggregation differs: {name}/{key}/{arm}/{metric}")
        result[kind] = summary
    if len(source_hashes)!=1:
        raise ValueError("Runs use different saved episode streams")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0,1), default=0)
    args = parser.parse_args()
    evidence = verify(Path(__file__).parent/"data", RUNS if args.seed==0 else SEED1_RUNS)
    print("Verified: summary hashes, complete conditions, paired ordinals and every reported mean.")
    print("\n| Encoder | Inference | Accuracy | Macro-F1 | AUC |")
    print("|---|---|---:|---:|---:|")
    for e,i in ((2000,2000),(2000,8000),(8000,2000),(8000,8000)):
        m = evidence["crossover"]["means"][f"E{e}I{i}"]
        print(f"| {e+1:,} | {i+1:,} | {100*m['accuracy']:.2f} | {100*m['macro_f1']:.2f} | {100*m['ovr_auc']:.2f} |")
    print("\n| Checkpoint | Statistics | Native accuracy | U1 accuracy |")
    print("|---|---|---:|---:|")
    for step in (2000,8000):
        for mode in ("batch","running"):
            m = evidence["normalization"]["means"][str(step)]
            print(f"| {step+1:,} | {mode} | {100*m[mode+'/native']['accuracy']:.2f} | {100*m[mode+'/centered']['accuracy']:.2f} |")


if __name__=="__main__":
    main()
