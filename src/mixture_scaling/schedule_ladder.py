"""Four-GPU shared queue, with optional checkpoint-boundary legacy draining."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def import_completed(legacy, root, rung):
    source = legacy / "lp" / f"ladder_r{rung}"
    summary_path = source / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or not summary.get("convergence") or not (source / "best.pt").is_file():
        raise ValueError(f"invalid legacy result: {source}")
    target = root / "lp" / source.name
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)
    elif target.resolve() != source.resolve():
        raise ValueError(f"legacy import collision: {target}")
    return True


def verify_legacy(pid, gpu, legacy):
    try:
        tokens = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    if ("mixture_scaling.node_mlp_ladder" not in tokens or str(legacy) not in tokens
            or "--device" not in tokens or tokens[tokens.index("--device")+1] != str(gpu)):
        raise RuntimeError(f"PID {pid} does not match the declared legacy ladder worker")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--log-root", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--legacy-state-root", type=Path)
    p.add_argument("--legacy-worker", action="append", default=[], help="GPU:PID:active_rung")
    p.add_argument("--legacy-worker-root", action="append", default=[], help="GPU:state_root override")
    args, extra = p.parse_known_args()
    worker_roots = {int(v.split(":", 1)[0]): Path(v.split(":", 1)[1]) for v in args.legacy_worker_root}
    def legacy_root(gpu):
        return worker_roots.get(gpu, args.legacy_state_root)
    gpus = [int(g) for g in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= {0,1,2,3}:
        p.error("select unique owned GPUs 0–3")
    legacy_workers = [tuple(map(int, value.split(":"))) for value in args.legacy_worker]
    if legacy_workers and not args.legacy_state_root:
        p.error("legacy workers need a legacy state root")
    for gpu, pid, rung in legacy_workers:
        if gpu not in gpus or not 1 <= rung <= 9:
            p.error("invalid legacy worker")
        verify_legacy(pid, gpu, legacy_root(gpu))
    args.log_root.mkdir(parents=True, exist_ok=False)
    (args.state_root / "lp").mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache = args.state_root / "_cache"
    if cache.is_symlink() or cache.exists():
        if cache.resolve() != args.cache_root.resolve():
            raise ValueError("cache root mismatch")
    else:
        cache.symlink_to(args.cache_root, target_is_directory=True)
    imported = set()
    if args.legacy_state_root:
        for rung in range(1,10):
            if import_completed(args.legacy_state_root, args.state_root, rung):
                imported.add(rung)
    reserved = imported | {rung for _,_,rung in legacy_workers}
    pending = sorted(set(range(1,10)) - reserved, reverse=True)
    common = ["--state-root", str(args.state_root), "--output-root", str(args.output_root),
              "--cache-root", str(args.cache_root), *extra]
    prefix = [sys.executable, "-u", "-m", "mixture_scaling.node_mlp_ladder"]
    (args.log_root / "revision.txt").write_text(subprocess.check_output(["git","rev-parse","HEAD"], text=True))
    (args.log_root / "STARTED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (args.log_root / "allocation.json").write_text(json.dumps({"gpus":gpus,"pending":pending,
        "imported":sorted(imported),"legacy_workers":legacy_workers,"arguments":extra},indent=2))
    workers, handles = {}, []

    def launch(gpu):
        if not pending:
            return
        memory = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits", "-i", str(gpu)],text=True).strip())
        if memory >= 2000:
            raise RuntimeError(f"GPU {gpu} occupied ({memory} MiB); refusing to collide")
        handle = (args.log_root / f"train_gpu{gpu}.log").open("w")
        handles.append(handle)
        workers[gpu] = subprocess.Popen(prefix + ["train", *common, "--queue", "--workers","1",
            "--device",str(gpu),"--rungs",",".join(map(str,pending))],stdout=handle,stderr=subprocess.STDOUT)
        print(json.dumps({"launched_gpu":gpu,"pid":workers[gpu].pid}),flush=True)

    for gpu in gpus:
        if gpu not in {g for g,_,_ in legacy_workers}:
            launch(gpu)
    draining = list(legacy_workers)
    while draining or any(proc.poll() is None for proc in workers.values()):
        for gpu,pid,rung in list(draining):
            if import_completed(legacy_root(gpu),args.state_root,rung):
                if verify_legacy(pid,gpu,legacy_root(gpu)):
                    # The imported summary is written after checkpoint and W&B finalization.
                    # Retire this worker before it trains further obsolete static-queue jobs.
                    os.kill(pid,signal.SIGTERM)
                    for _ in range(60):
                        if not Path(f"/proc/{pid}").exists():
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"legacy worker {pid} did not exit")
                draining.remove((gpu,pid,rung))
                print(json.dumps({"imported_rung":rung,"released_gpu":gpu}),flush=True)
                launch(gpu)
            elif not verify_legacy(pid,gpu,legacy_root(gpu)):
                raise RuntimeError(f"legacy worker {pid} exited before rung {rung} completed")
        for gpu,proc in workers.items():
            if proc.poll() not in (None,0):
                raise RuntimeError(f"queue worker on GPU {gpu} failed: {proc.returncode}")
        time.sleep(3)
    for handle in handles:
        handle.close()
    # All models are now complete. Reuse the canonical target-sharded evaluation.
    jobs=[]
    for index,gpu in enumerate(gpus):
        handle=(args.log_root/f"eval_gpu{gpu}.log").open("w")
        handles.append(handle)
        jobs.append(subprocess.Popen(prefix+["eval",*common,"--worker-index",str(index),
            "--workers",str(len(gpus)),"--device",str(gpu)],stdout=handle,stderr=subprocess.STDOUT))
    statuses = [proc.wait() for proc in jobs]
    if any(status != 0 for status in statuses):
        raise RuntimeError("evaluation failed; see per-GPU logs")
    subprocess.run(prefix+["aggregate",*common],check=True)
    for handle in handles:
        if not handle.closed:
            handle.close()
    (args.log_root / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))


if __name__ == "__main__":
    main()
