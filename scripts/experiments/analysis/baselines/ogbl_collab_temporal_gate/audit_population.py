"""Describe cached positive populations; no training or test access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root):
    result = {}
    for year in (2017, 2018):
        path = root / f"year{year}.npz"
        with np.load(path) as data:
            x = data["pfeatures"]
            cold = x[:, 5] == 0
            panels = {}
            for label, mask in (("all", np.ones(len(x), dtype=bool)),
                                ("cold_endpoint", cold), ("both_warm", ~cold)):
                panels[label] = {
                    "count": int(mask.sum()),
                    "nonzero_aa_count": int((x[mask, 0] > 0).sum()),
                    "previous_pair_count": int((x[mask, 8] > 0).sum()),
                }
        result[str(year)] = {"cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "positive_populations": panels}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.cache_root), indent=2) + "\n")
