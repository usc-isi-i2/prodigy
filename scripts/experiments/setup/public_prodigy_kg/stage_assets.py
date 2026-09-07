"""Stage official PRODIGY KG archives; no training or pickle loading.

Dry run by default. Retains archives and records SHA256 (upstream publishes no
checksum). HTTPS provenance, pinned HTTP metadata, ZIP CRCs and path checks are
distinct checks, not a claim of a publisher-authenticated checksum.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ASSETS = {
    "FB15K-237": {"bytes": 601952672, "etag": '"2a60200-23e111a0-5fe8c1556157e"'},
    "Wiki": {"bytes": 14338377652, "etag": '"2a601ff-356a247b4-5fe8089b56ff0"'},
}
BASE_URL = "https://snap.stanford.edu/prodigy/"
FEATURE_FILE = "preproc_text_feats/text_feats_sentence-transformers_all-mpnet-base-v2_.pkl"
EXTRA_REQUIRED = {
    "Wiki": ("entity2id.json", "relation2id.json", "text_features_web_scraped.pb"),
    "FB15K-237": ("path_graph.json", "entity2id.json", "relation2id.json", "mid2name_dict.pkl"),
}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError(f"Refusing symlink receipt: {path}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_members(archive: zipfile.ZipFile, dataset: str) -> list[zipfile.ZipInfo]:
    """Validate every member before extraction, including ignored Mac metadata."""
    members, seen = [], set()
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or "\\" in item.filename or ":" in item.filename or "\x00" in item.filename):
            raise ValueError(f"Unsafe ZIP path: {item.filename!r}")
        if str(path) in seen:
            raise ValueError(f"Duplicate ZIP path: {item.filename!r}")
        seen.add(str(path))
        if item.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP member: {item.filename!r}")
        kind = stat.S_IFMT(item.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Non-regular ZIP member: {item.filename!r}")
        if path.parts[0] == "__MACOSX":
            continue
        if path.parts[0] != dataset:
            raise ValueError(f"Unexpected archive root: {item.filename!r}")
        members.append(item)
    names = {item.filename for item in members}
    required_paths = ("graph.pt", f"{dataset}_adj.pt", FEATURE_FILE, *EXTRA_REQUIRED[dataset])
    for required in (f"{dataset}/{path}" for path in required_paths):
        if required not in names:
            raise ValueError(f"Missing required asset: {required}")
    return members


def stage_one(root: Path, dataset: str) -> dict:
    metadata = ASSETS[dataset]
    url = BASE_URL + dataset + ".zip"
    receipt_path = root / "receipts" / f"{dataset}.json"
    target = root / dataset
    archive_path = root / "archives" / f"{dataset}.zip"
    part = archive_path.with_suffix(".zip.part")
    for path in (target, archive_path, part, receipt_path):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
    if target.exists():
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old.get("status") == "complete" and old.get("expected") == metadata:
                print(f"{dataset}: already staged; leaving existing files untouched", flush=True)
                return old
        raise FileExistsError(f"Existing dataset without matching complete receipt: {target}")
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        observed = {"bytes": int(response.headers["Content-Length"]),
                    "etag": response.headers["ETag"]}
        if observed != metadata or response.url != url:
            raise ValueError(f"Upstream metadata changed: {observed}; expected {metadata}")
    receipt = {"dataset": dataset, "url": url, "expected": metadata,
               "status": "downloading", "started_unix": time.time(),
               "publisher_checksum_available": False,
               "license": "Not stated in inspected official distribution; redistribution permission unresolved"}
    write_json(receipt_path, receipt)
    if not archive_path.exists():
        print(f"{dataset}: downloading/resuming {metadata['bytes']} bytes", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--proto", "=https",
                        "--proto-redir", "=https", "--connect-timeout", "30",
                        "--retry", "4", "--retry-delay", "3", "--continue-at", "-",
                        "--output", str(part), url], check=True)
        if part.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Incomplete or changed download: {part}")
        part.rename(archive_path)
    if archive_path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Incorrect archive byte count: {archive_path}")
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    receipt.update(status="checking_zip", sha256=digest.hexdigest())
    write_json(receipt_path, receipt)
    with zipfile.ZipFile(archive_path) as archive:
        members = checked_members(archive, dataset)
        expanded_bytes = sum(item.file_size for item in members)
        if expanded_bytes > 40_000_000_000:
            raise ValueError("Unexpectedly large archive expansion")
        if shutil.disk_usage(root).free < expanded_bytes + 5_000_000_000:
            raise OSError("Insufficient free storage for extraction plus 5 GB reserve")
        print(f"{dataset}: validating ZIP CRCs ({expanded_bytes} extracted bytes)", flush=True)
        failed = archive.testzip()
        if failed is not None:
            raise ValueError(f"ZIP CRC failed: {failed}")
        pending = Path(tempfile.mkdtemp(prefix=dataset + "-", dir=root / "pending"))
        receipt.update(status="extracting", expanded_bytes=expanded_bytes,
                       members=len(members), pending_directory=str(pending))
        write_json(receipt_path, receipt)
        archive.extractall(pending, members=members)
        if target.exists():
            raise FileExistsError(target)
        (pending / dataset).rename(target)
        pending.rmdir()  # only remove this newly created, now-empty staging directory
    receipt.update(status="complete", completed_unix=time.time(), target=str(target))
    write_json(receipt_path, receipt)
    print(f"{dataset}: complete, SHA256 {receipt['sha256']}", flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/dataMeR1/phil/data/prodigy_public_original"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(ASSETS), default=list(ASSETS))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"root": str(args.root), "datasets": args.datasets,
                      "download_bytes": sum(ASSETS[name]["bytes"] for name in args.datasets),
                      "execute": args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    if args.root.is_symlink() or not args.root.is_absolute() or len(args.root.parts) < 4:
        raise ValueError("Use a specific, absolute, non-symlink data directory")
    args.root.mkdir(parents=True, exist_ok=True)
    for name in ("archives", "receipts", "pending"):
        directory = args.root / name
        if directory.is_symlink():
            raise ValueError(f"Refusing symlink: {directory}")
        directory.mkdir(exist_ok=True)
    lock_path = args.root / "stage.lock"
    if lock_path.is_symlink():
        raise ValueError("Refusing symlink lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(args.root).free < 60_000_000_000:
            raise OSError("Reserve at least 60 GB before starting this two-archive staging")
        receipts = [stage_one(args.root, dataset) for dataset in args.datasets]
        write_json(args.root / "receipts" / "staging_complete.json", receipts)


if __name__ == "__main__":
    main()
