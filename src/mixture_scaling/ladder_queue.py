"""Cross-process rung claims, held until training completes or the worker exits."""
import fcntl
import json
import os
from pathlib import Path


def claimed_rows(root, rows):
    root = Path(root)
    claims = root / "_claims"
    claims.mkdir(parents=True, exist_ok=True)
    for run_id, sources in sorted(rows, key=lambda row: -len(row[1])):
        with (claims / f"{run_id}.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                if (root / "lp" / run_id / "summary.json").exists():
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps({"pid": os.getpid(), "run_id": run_id}))
                handle.flush()
                yield run_id, sources
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
