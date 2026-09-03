#!/usr/bin/env python3
"""Bounded profiling of the real NM ladder; no benchmark-result checkpoints.

Run on Tucker. Loads the graph once, measures isolated CPU stages, then loader
throughput, then synchronized GPU stages on fresh CPU batches. All outputs are
profiling diagnostics, not trained-model results. See README.md.
"""
import argparse
import cProfile
import functools
import gc
import json
import os
from pathlib import Path
import platform
import pstats
import random
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / 'scripts/experiments/setup/nm_ladder_nhop2/configs/train_ordA_r8.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', type=int, choices=(2, 3), default=2)
    parser.add_argument('--episodes-per-source', type=int, default=8)
    parser.add_argument('--loader-episodes', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--workers', default='0,2,4,8')
    parser.add_argument('--threads', default='4,1')
    parser.add_argument('--gpu-steps', type=int, default=16)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if min(args.episodes_per_source, args.loader_episodes, args.gpu_steps) < 1 or args.warmup < 0:
        parser.error('episode counts must be positive; warmup must be nonnegative')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_DIR', str(out))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from experiments.params import get_params
    from experiments.run_single_experiment import load_dataset
    from data.covid19_twitter import get_covid19_twitter_dataloader

    params = get_params(['--config', args.config, '--device', str(args.device)])
    if params['task_name'] != 'neighbor_matching' or params['batch_size'] != 1:
        raise ValueError('This profiler supports batch_size=1 neighbor_matching only')
    if params.get('neighbor_sampling_source_sequence') or params.get('neighbor_sampling_center_radii'):
        raise ValueError('Use the source-confined ladder protocol')
    torch.set_num_threads(4)  # same as run_single_experiment.py
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    rows = []
    metadata = dict(arguments=vars(args), params=params, hostname=platform.node(),
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    torch_version=torch.__version__, cpu_count=os.cpu_count(),
                    affinity=sorted(os.sched_getaffinity(0)), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    def save():
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
        (out / 'measurements.json').write_text(json.dumps(rows, indent=2) + '\n')
    def emit(row):
        rows.append(row)
        print('MEASUREMENT ' + json.dumps(row), flush=True)
        save()
    save()
    print('Loading full graph once', flush=True)
    start = time.perf_counter()
    dataset = load_dataset(params)
    emit(dict(phase='graph_load_and_csr', seconds=time.perf_counter()-start,
              nodes=dataset.graph.num_nodes, edges=dataset.graph.edge_index.shape[1],
              feature_dim=dataset.graph.x.shape[1]))
    # Use the production dataloader builder once, retaining its exact source strata,
    # episode semantics, collator, and augmentation. Avoid repeatedly scanning graph_id.
    kwargs = dict(params)
    kwargs.update(n_shot=params['n_shots'], bert=None, num_workers=0,
                  aug=params['augmentation'], aug_test=params['augment_test'],
                  split_labels=not params['no_split_labels'])
    kwargs.pop('dataset', None)
    start = time.perf_counter()
    loader = get_covid19_twitter_dataloader(dataset, split='train', node_split='',
                batch_count=args.loader_episodes+args.warmup, **kwargs)
    emit(dict(phase='dataloader_setup', seconds=time.perf_counter()-start))
    sampler = loader.batch_sampler
    task = sampler.task
    if not task.confine_to_single_stratum:
        raise ValueError('Expected source-confined episodes')
    strata = task.strata
    source_names = dataset.graph.source_graph_names
    names = [source_names[int(dataset.graph.graph_id[s[0]])] for s in strata]
    saved_batches = []

    # Instrument existing calls without changing sampling or feature construction.
    totals = {}
    originals = []
    def wrap(obj, name, label, count_rejected=False):
        original = getattr(obj, name)
        @functools.wraps(original)
        def timed(*a, **kw):
            start = time.perf_counter()
            value = original(*a, **kw)
            totals[label+'_seconds'] = totals.get(label+'_seconds', 0.) + time.perf_counter()-start
            totals[label+'_calls'] = totals.get(label+'_calls', 0) + 1
            if count_rejected and value is None:
                totals['rejected_centers'] = totals.get('rejected_centers', 0) + 1
            return value
        originals.append((obj, name, original))
        setattr(obj, name, timed)
    wrap(task, '_sample_center_members', 'center_members', True)
    wrap(dataset.neighbor_sampler, 'sample_node', 'neighborhood')
    wrap(dataset, 'get_subgraph', 'get_subgraph')
    wrap(dataset, 'add_pooling_supernode', 'pooling')
    try:
        for threads in [int(v) for v in args.threads.split(',')]:
            torch.set_num_threads(threads)
            for source_idx, name in enumerate(names):
                rng = random.Random(542 + source_idx)
                torch.manual_seed(542 + source_idx)
                batch_param = sampler.param_sampler(rng)
                profile = cProfile.Profile()
                for episode in range(args.warmup + args.episodes_per_source):
                    totals.clear()
                    if episode == args.warmup:
                        profile.enable()
                    t0 = time.perf_counter()
                    example = task._sample_from_stratum(batch_param.n_way, batch_param.n_member, rng, source_idx)
                    t1 = time.perf_counter()
                    fetched = dataset[([example], batch_param)]
                    t2 = time.perf_counter()
                    batch = loader.collate_fn(fetched)
                    t3 = time.perf_counter()
                    if episode == args.warmup:
                        profile.disable()
                        stats_path = out / f'cpu_{name}_threads{threads}.txt'
                        with stats_path.open('w') as f:
                            pstats.Stats(profile, stream=f).strip_dirs().sort_stats('cumulative').print_stats(45)
                    # Keep cProfile's overhead out of reported stage distributions.
                    elif episode > args.warmup:
                        emit(dict(phase='cpu_stages', source=name, threads=threads, episode=episode,
                                  select_seconds=t1-t0, fetch_seconds=t2-t1, collate_seconds=t3-t2,
                                  total_seconds=t3-t0, nodes=int(batch[0].num_nodes),
                                  **totals))
                    if threads == 4 and episode == args.warmup:
                        saved_batches.append(batch)
                del fetched, batch
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
    # Workers run before CUDA initialization. PyTorch worker tensor ops use one thread.
    torch.set_num_threads(4)
    for workers in [int(v) for v in args.workers.split(',')]:
        sampler.rng.seed(542)
        torch.manual_seed(0)
        bench_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                  collate_fn=loader.collate_fn)
        start = time.perf_counter()
        iterator = iter(bench_loader)
        for _ in range(args.warmup):
            next(iterator)
        startup = time.perf_counter()-start
        samples = []
        start = time.perf_counter()
        for _ in range(args.loader_episodes):
            t0 = time.perf_counter()
            b = next(iterator)
            samples.append(time.perf_counter()-t0)
        elapsed = time.perf_counter()-start
        emit(dict(phase='loader_throughput', workers=workers, main_threads=4,
                  episodes=args.loader_episodes, warmup_seconds=startup, seconds=elapsed,
                  episodes_per_second=args.loader_episodes/elapsed,
                  median_wait_seconds=statistics.median(samples), waits_seconds=samples))
        del iterator, bench_loader, b
        gc.collect()
    if args.cpu_only:
        return

    # Reuse production trainer/model/loss, but bypass duplicate train/val/test setup.
    # No train() call: these disposable optimizer steps write no model checkpoint.
    from experiments.trainer import TrainerFS
    class ProfileTrainer(TrainerFS):
        def _build_dataloaders(self, dataset, dataset_name):
            return loader, loader, loader, loader
    params.update(prefix='ladder_profile', exp_name='ladder_profile',
                  state_dir=str(out / 'state'), log_dir=str(out / 'log'))
    trainer = ProfileTrainer(dataset, params)
    torch.cuda.set_device(args.device)
    device = trainer.device
    metadata['gpu'] = torch.cuda.get_device_name(device)
    metadata['parameter_count'] = sum(p.numel() for p in trainer.model.parameters())
    for anomaly in (True, False):
        torch.autograd.set_detect_anomaly(anomaly)
        for step in range(args.warmup+args.gpu_steps):
            # PyG Data.to() mutates the object: preserve the saved CPU batches.
            cpu_batch = [v.clone() for v in saved_batches[step % len(saved_batches)]]
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            batch = [v.to(device) for v in cpu_batch]
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            trainer.model.train()
            trainer.optimizer.zero_grad()
            yt, yp, graph = trainer.model(*batch)
            loss, _ = trainer.get_loss_and_acc(yt, yp)
            total_loss = loss + trainer.parameter['attr_regression_weight'] * trainer.get_aux_loss(graph)
            torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            total_loss.backward()
            torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            trainer.optimizer.step()
            torch.cuda.synchronize(device)
            t4 = time.perf_counter()
            if step >= args.warmup:
                emit(dict(phase='gpu_stages', anomaly_detection=anomaly, step=step,
                          source=names[step % len(saved_batches)], transfer_seconds=t1-t0,
                          forward_loss_seconds=t2-t1, backward_seconds=t3-t2,
                          optimizer_seconds=t4-t3, total_seconds=t4-t0,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
            del batch, yt, yp, graph, loss, total_loss
    import wandb
    wandb.finish()
    metadata['completed'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
