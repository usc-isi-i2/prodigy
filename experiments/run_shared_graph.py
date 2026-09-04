"""Train independent source-restricted NM models against one shared CPU graph.

See docs/fast_training.md. The supervisor never initializes CUDA. Each spawned
trainer owns its model, optimizer and RNG; CPU loaders also use spawn. Graph
features, topology, source pools, and all positive-view CSR indices are shared.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Fail closed: only these differences may reuse an identical loaded dataset.
# Everything else (including future newly added options) must match across jobs.
RUN_KEYS = set('''config prefix timestamp exp_name device seed workers detect_anomaly
loader_start_method epochs dataset_len_cap val_len_cap test_len_cap eval_step
checkpoint_step checkpoint_steps state_dir log_dir tags print_step n_way n_shots
n_query n_way_upper n_shots_upper n_query_upper batch_size learning_rate weight_decay
emb_dim layers gnn_type n_layer dropout neighbor_sampling_source_subset
neighbor_sampling_source_sequence neighbor_sampling_source_sequence_steps
neighbor_sampling_episode_source_weighting neighbor_sampling_cross_source_prob
neighbor_sampling_batch_source_mode pretrained_model_run resume_training_checkpoint
source_gradient_diagnostics_every source_gradient_diagnostics_max_sources'''.split())


def validate_configs(params):
    if not params:
        raise ValueError('At least one config is required')
    for p in params:
        if p['dataset'] != 'covid19_twitter' or p['task_name'] != 'neighbor_matching':
            raise ValueError('Shared launcher currently supports covid19_twitter-format NM graphs only')
        if not p['original_features'] or p.get('structural_features', 'none') != 'none':
            raise ValueError('Shared launcher requires original_features and structural_features=none')
        if p.get('neighbor_sampling_episode_source') != 'graph_id':
            raise ValueError('Use neighbor_sampling_episode_source=graph_id for source-restricted models')
        if p.get('midterm_label_downsample') or p.get('target_feature'):
            raise ValueError('Shared launcher does not support label downsampling or target-feature transforms')
        if p.get('resume_training_checkpoint') and p['workers'] != 0:
            raise ValueError('Exact resume requires --workers-per-model 0')
        if p.get('eval_only'):
            raise ValueError('This is a training launcher, not an eval sweep launcher')
    first = params[0]
    for index, p in enumerate(params[1:], 1):
        different = [key for key in first.keys() | p.keys()
                     if key not in RUN_KEYS and first.get(key) != p.get(key)]
        if different:
            raise ValueError(f'Config {index} changes shared-data/protocol settings: {sorted(different)}. Run separate groups.')


def validate_disjoint_sources(graph, chunk_size=1_000_000):
    """Source restrictions are valid only when context edges cannot cross sources."""
    for start in range(0, graph.edge_index.shape[1], chunk_size):
        edges = graph.edge_index[:, start:start + chunk_size]
        if not bool((graph.graph_id[edges[0]] == graph.graph_id[edges[1]]).all()):
            raise ValueError('Context graph contains cross-source edges; source-constrained training would leak')


def prepare_shared_dataset(dataset):
    import numpy as np
    import torch
    # Human-readable user IDs are unused by NM training and would be serialized
    # as millions of Python strings for every spawned process. Preserve node IDs
    # in the graph tensors; compact only this supervisor's shallow graph wrapper.
    validate_disjoint_sources(dataset.graph)
    dataset.graph = copy.copy(dataset.graph)
    dataset.graph.user_ids = []
    dataset.graph.share_memory_()
    for value in vars(dataset).values():
        if hasattr(value, 'whole_adj'):
            value.whole_adj.csr()
            value.whole_adj.share_memory_()
    ids = dataset.graph.graph_id.numpy()
    dataset.source_node_pools = {
        int(source): torch.from_numpy(np.flatnonzero(ids == source)).share_memory_()
        for source in np.unique(ids)
    }
    return dataset


def shared_storage_report(dataset):
    rowptr, col, values = dataset.neighbor_sampler.whole_adj.csr()
    return {key: tensor.is_shared() for key, tensor in {
        'features': dataset.graph.x, 'edges': dataset.graph.edge_index,
        'graph_id': dataset.graph.graph_id, 'csr_rowptr': rowptr,
        'csr_col': col, 'csr_values': values,
        **{f'source_{k}': v for k, v in dataset.source_node_pools.items()},
    }.items()}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def start_on_gpu(process, gpu):
    """Set visibility before the fresh interpreter imports/unpickles any library."""
    values = {'CUDA_VISIBLE_DEVICES': str(gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        process.start()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cuda_probe(gpu, queue, barrier):
    try:
        import torch
        value = torch.ones(1, device='cuda:0')
        torch.cuda.synchronize()
        queue.put(dict(ok=True, physical_gpu=gpu, visible=os.environ['CUDA_VISIBLE_DEVICES'],
                       device_count=torch.cuda.device_count()))
        barrier.wait(timeout=60)
        del value
    except BaseException:
        queue.put(dict(ok=False, physical_gpu=gpu, error=traceback.format_exc()))
        barrier.abort()
        raise


def cuda_preflight(slots):
    import torch
    context = torch.multiprocessing.get_context('spawn')
    queue = context.Queue()
    barrier = context.Barrier(len(slots))
    processes = []
    try:
        for gpu in slots:
            process = context.Process(target=_cuda_probe, args=(gpu, queue, barrier))
            start_on_gpu(process, gpu)
            processes.append(process)
        results = [queue.get(timeout=90) for _ in slots]
        for process in processes:
            process.join(timeout=10)
        if any(not r['ok'] or r.get('device_count') != 1 for r in results) or any(p.exitcode != 0 for p in processes):
            raise RuntimeError(f'Concurrent CUDA preflight failed: {results}')
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()


def train_one(dataset, params, job_dir, threads):
    os.setsid()  # only this trainer and its descendants are killed on cancellation
    job_dir = Path(job_dir)
    with (job_dir / 'console.log').open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        physical_gpu = params['device'].index
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu):
            raise RuntimeError('GPU visibility must be assigned before spawning the trainer')
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Spawned trainer unexpectedly initialized CUDA before device isolation')
        params = dict(params, device=torch.device('cuda:0'), physical_gpu=physical_gpu)
        write_json(job_dir / 'effective_config.json', params)
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(threads)
        from experiments.run_single_experiment import seed_everything
        from experiments.trainer import TrainerFS
        result = dict(status='initializing', pid=os.getpid(), exp_name=params['exp_name'],
                      device=str(params['device']), physical_gpu=physical_gpu, started=time.time(),
                      shared_storage=shared_storage_report(dataset))
        result_path = job_dir / 'result.json'
        write_json(result_path, result)
        try:
            if not all(result['shared_storage'].values()):
                raise RuntimeError('Trainer did not receive shared graph storage')
            seed_everything(params)
            torch.autograd.set_detect_anomaly(params['detect_anomaly'])
            trainer = TrainerFS(dataset, params)
            first_timed = None
            last_timed = None
            warmup = 10
            def observe(step):
                nonlocal first_timed, last_timed
                now = time.time()
                if first_timed is None and step >= trainer.resume_step + warmup:
                    first_timed = (step, now)
                last_timed = (step, now)
            trainer.training_step_observer = observe
            result.update(status='training', training_started=time.time())
            write_json(result_path, result)
            trainer.train()
            torch.cuda.synchronize(params['device'])
            result.update(status='complete', completed=time.time(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(params['device']),
                          checkpoint_dir=trainer.ckpt_dir)
            if first_timed is not None and last_timed[0] > first_timed[0]:
                steps = last_timed[0] - first_timed[0]
                elapsed = last_timed[1] - first_timed[1]
                result.update(steady_steps=steps, steady_seconds=elapsed,
                              steady_steps_per_second=steps/elapsed,
                              steady_started=first_timed[1], steady_finished=last_timed[1])
            write_json(result_path, result)
        except BaseException:
            result.update(status='failed', error=traceback.format_exc(), completed=time.time())
            write_json(result_path, result)
            traceback.print_exc()
            raise


def make_plan(args, overrides):
    import torch
    from experiments.params import get_params
    active = min(len(args.configs), len(args.gpus)*args.models_per_gpu)
    workers = args.workers_per_model
    if workers is None:
        workers = min(16, args.worker_budget // active)
    if workers < 0 or active * workers > args.worker_budget:
        raise ValueError('workers_per_model must be nonnegative and fit the total worker budget')
    if workers == 0 and args.workers_per_model is None:
        raise ValueError('Worker budget must allow at least one worker per active model')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    params = []
    for index, config in enumerate(args.configs):
        p = get_params(['--config', str(Path(config).resolve()), *overrides])
        if args.smoke_steps and p.get('neighbor_sampling_source_sequence'):
            raise ValueError('Smoke mode requires interleaved configs; do not truncate a blocked-source schedule')
        p.update(device=torch.device(f'cuda:{args.gpus[index % len(args.gpus)]}'),
                 workers=workers, loader_start_method='spawn',
                 state_dir=str(args.run_dir/'state'), log_dir=str(args.run_dir/'log'))
        if args.smoke_steps:
            p.update(epochs=1, dataset_len_cap=args.smoke_steps,
                     val_len_cap=2, test_len_cap=2, checkpoint_steps='',
                     checkpoint_step=args.smoke_steps, eval_step=args.smoke_steps+1,
                     eval_test_before_train=False, eval_val_before_train=False, eval_after_train=False)
        p['exp_name'] = f"{'smoke_' if args.smoke_steps else ''}{p['prefix']}_{stamp}_{index:03d}"
        params.append(p)
    validate_configs(params)
    return params, workers


def main():
    import torch
    from experiments.run_single_experiment import load_dataset
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    own, overrides = argv[:split], argv[split+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, choices=(2, 3), default=[2])
    parser.add_argument('--models-per-gpu', type=int, default=2)
    parser.add_argument('--worker-budget', type=int, default=32)
    parser.add_argument('--workers-per-model', type=int)
    parser.add_argument('--threads-per-model', type=int, default=4)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='Check concurrent CUDA contexts without loading a graph')
    args = parser.parse_args(own)
    if min(args.models_per_gpu, args.threads_per_model) < 1 or args.worker_budget < 0 or args.smoke_steps < 0:
        parser.error('Invalid concurrency, thread, worker, or smoke-step count')
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('GPUs must be unique')
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        parser.error('Unset CUDA_VISIBLE_DEVICES: select physical owned GPUs with --gpus 2 and/or 3')
    args.run_dir = args.run_dir.resolve()
    params, workers = make_plan(args, overrides)
    slots = [gpu for gpu in args.gpus for _ in range(args.models_per_gpu)]
    plan = dict(configs=[str(Path(c).resolve()) for c in args.configs], gpus=args.gpus,
                models_per_gpu=args.models_per_gpu, workers_per_model=workers,
                worker_budget=args.worker_budget, mode='smoke' if args.smoke_steps else 'training',
                revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                run_dir=str(args.run_dir), jobs=params)
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if args.dry_run:
        return
    plan['cuda_preflight'] = cuda_preflight(slots[:min(len(params), len(slots))])
    print('Concurrent CUDA preflight passed', flush=True)
    if args.preflight_only:
        return
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir/'manifest.json', plan)
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_DIR', str(args.run_dir))
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(False)
    start = time.time()
    dataset = prepare_shared_dataset(load_dataset(params[0]))
    if torch.cuda.is_initialized():
        raise RuntimeError('Supervisor must remain CPU-only')
    from data.covid19_twitter import resolve_source_subset
    for p in params:
        resolve_source_subset(p['neighbor_sampling_source_subset'], sorted(dataset.source_node_pools), dataset.graph.source_graph_names)
    plan.update(shared_graph_ready=time.time(), shared_graph_setup_seconds=time.time()-start,
                shared_storage=shared_storage_report(dataset))
    write_json(args.run_dir/'manifest.json', plan)
    context = torch.multiprocessing.get_context('spawn')
    active, finished = {}, []
    index = 0
    def stop_children():
        for process, _ in active.values():
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
        for process, _ in active.values():
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while index < len(params) or active:
            for slot, gpu in enumerate(slots):
                if slot in active or index >= len(params):
                    continue
                p = params[index]
                p['device'] = torch.device(f'cuda:{gpu}')
                job_dir = args.run_dir/f'job_{index:03d}'
                job_dir.mkdir()
                write_json(job_dir/'effective_config.json', p)
                process = context.Process(target=train_one, args=(dataset, p, str(job_dir), args.threads_per_model))
                start_on_gpu(process, gpu)
                active[slot] = (process, index)
                print(f"Started job {index} on GPU {gpu}; log {job_dir/'console.log'}", flush=True)
                index += 1
                write_json(args.run_dir/'manifest.json', plan)
            for slot, (process, job_index) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                finished.append(dict(job=job_index, exitcode=process.exitcode))
                del active[slot]
                if process.exitcode != 0:
                    raise RuntimeError(f'Job {job_index} failed with exit code {process.exitcode}; see its console.log')
            time.sleep(1)
        write_json(args.run_dir/'status.json', dict(status='complete', finished=finished, completed=time.time()))
        print('ALL SHARED-GRAPH JOBS COMPLETE', flush=True)
    except BaseException:
        stop_children()
        write_json(args.run_dir/'status.json', dict(status='failed_or_interrupted', finished=finished,
                                                   error=traceback.format_exc(), completed=time.time()))
        raise


if __name__ == '__main__':
    main()
