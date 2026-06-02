"""Command-line interface for the tweet embedding pipeline."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is pinned in requirements.
    yaml = None

from .constants import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVISION,
    OUTPUT_DIM,
    PREPROCESSING_VERSION,
)
from .io_utils import atomic_replace, atomic_write_json
from .logging_utils import setup_logger
from .manifest import write_run_manifest
from .source_index import (
    build_source_index,
    load_source_index,
    total_source_rows,
    write_source_index,
)
from .worker import worker_main


def parse_shard_list(value: str) -> list[int] | None:
    if not value.strip():
        return None
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed Ukraine-Russia tweet Parquet files into deterministic fp16 shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Optional YAML config to load before CLI overrides.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-files", default="", help="Optional precomputed source_files.parquet.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-folder", default="")
    parser.add_argument("--gpus", default="1,2,3,4", help="Recorded GPU list; also sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-rank", type=int, default=-1, help="Run only one worker rank.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=OUTPUT_DIM)
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=65_536)
    parser.add_argument("--source-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-source-index", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-shards", type=int, default=0, help="Limit to the first N selected shards.")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--shards", default="", help="Comma/range list such as 0,4,8-12.")
    parser.add_argument("--index-only", action="store_true", help="Build source_files/config/manifest only.")
    parser.add_argument("--write-empty-skipped", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = build_parser()
    config_only, _ = base_parser.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if config_only.config:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load --config")
        with Path(config_only.config).open("r", encoding="utf-8") as handle:
            config_defaults = yaml.safe_load(handle) or {}
        known_dests = {action.dest for action in base_parser._actions}
        config_defaults = {key: value for key, value in config_defaults.items() if key in known_dests}
        base_parser.set_defaults(**config_defaults)
    return base_parser.parse_args()


def args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    cfg["input_root"] = str(Path(cfg["input_root"]))
    cfg["output_root"] = str(Path(cfg["output_root"]))
    cfg["preprocessing_version"] = PREPROCESSING_VERSION
    return cfg


def write_config(output_root: Path, cfg: dict[str, Any]) -> None:
    if yaml is None:
        atomic_write_json(output_root / "config.json", cfg)
        return
    tmp_path = output_root / f"config.yaml.tmp.{socket.gethostname()}.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)
    atomic_replace(tmp_path, output_root / "config.yaml")


def prepare_source_files(cfg: dict[str, Any], logger: logging.Logger) -> list[dict[str, Any]]:
    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    output_source_path = output_root / "source_files.parquet"
    external_source_path = Path(cfg["source_files"]) if cfg.get("source_files") else None

    if external_source_path and external_source_path.exists() and not bool(cfg["rebuild_source_index"]):
        logger.info("loading source index from %s", external_source_path)
        rows = load_source_index(external_source_path)
        if external_source_path.resolve() != output_source_path.resolve():
            shutil.copy2(external_source_path, output_source_path)
        return rows

    if output_source_path.exists() and not bool(cfg["rebuild_source_index"]):
        logger.info("loading source index from %s", output_source_path)
        return load_source_index(output_source_path)

    logger.info("building source index under %s checksums=%s", input_root, cfg["source_checksums"])
    rows = build_source_index(input_root, checksum=bool(cfg["source_checksums"]))
    write_source_index(output_source_path, rows)
    return rows


def selected_shards(cfg: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[int]:
    total_rows = total_source_rows(source_rows)
    total_shards = math.ceil(total_rows / int(cfg["shard_size"]))
    explicit = parse_shard_list(str(cfg.get("shards") or ""))
    if explicit is not None:
        shards = [sid for sid in explicit if 0 <= sid < total_shards]
    else:
        shards = list(range(int(cfg["start_shard"]), total_shards))
    if int(cfg["smoke_shards"]) > 0:
        shards = shards[: int(cfg["smoke_shards"])]
    return shards


def main() -> int:
    args = parse_args()
    cfg = args_to_config(args)

    if cfg.get("gpus") and not bool(cfg["cpu"]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"])

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    for child in ["shards", "skipped_rows", "logs"]:
        (output_root / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root, "run")
    write_config(output_root, cfg)
    source_rows = prepare_source_files(cfg, logger)
    selected = selected_shards(cfg, source_rows)
    logger.info(
        "source_files=%s source_rows=%s selected_shards=%s",
        len(source_rows),
        total_source_rows(source_rows),
        len(selected),
    )
    write_run_manifest(output_root, cfg, source_rows, selected)

    if bool(cfg["index_only"]):
        logger.info("index-only complete")
        return 0
    if not selected:
        logger.info("no shards selected")
        return 0

    if int(cfg["worker_rank"]) >= 0:
        worker_main(int(cfg["worker_rank"]), cfg, source_rows, selected)
    elif int(cfg["num_workers"]) == 1:
        worker_main(0, cfg, source_rows, selected)
    else:
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=worker_main, args=(rank, cfg, source_rows, selected))
            for rank in range(int(cfg["num_workers"]))
        ]
        for process in processes:
            process.start()
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
                logger.error("worker pid=%s exitcode=%s", process.pid, process.exitcode)
        if failed:
            raise RuntimeError("one or more workers failed")

    write_run_manifest(output_root, cfg, source_rows, selected)
    logger.info("embedding run complete")
    return 0
