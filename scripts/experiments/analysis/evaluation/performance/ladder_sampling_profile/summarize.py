#!/usr/bin/env python3
"""Summarize saved profiling measurements without training dependencies."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'cpu_stages':
            groups[row['source'], row['threads']].append(row)
    cpu = []
    for (source, threads), group in sorted(groups.items()):
        row = dict(source=source, threads=threads, episodes=len(group))
        for key in ['select_seconds', 'fetch_seconds', 'collate_seconds', 'total_seconds',
                    'neighborhood_seconds', 'get_subgraph_seconds', 'pooling_seconds',
                    'center_members_calls', 'rejected_centers', 'nodes']:
            row[key] = statistics.mean(r.get(key, 0) for r in group)
        row['gather_and_graph_seconds'] = row['get_subgraph_seconds'] - row['neighborhood_seconds']
        cpu.append(row)
    gpu = []
    for anomaly in [True, False]:
        group = [r for r in rows if r['phase'] == 'gpu_stages' and r['anomaly_detection'] == anomaly]
        if group:
            row = dict(anomaly_detection=anomaly, steps=len(group))
            for key in ['transfer_seconds', 'forward_loss_seconds', 'backward_seconds', 'optimizer_seconds', 'total_seconds']:
                row[key] = statistics.mean(r[key] for r in group)
            row['peak_allocated_gib'] = max(r['peak_allocated_bytes'] for r in group)/2**30
            row['peak_reserved_gib'] = max(r['peak_reserved_bytes'] for r in group)/2**30
            gpu.append(row)
    return dict(startup=[r for r in rows if r['phase'] in ('graph_load_and_csr', 'dataloader_setup')],
                cpu=cpu, loader=[r for r in rows if r['phase']=='loader_throughput'], gpu=gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'data/measurements.json')
    args = parser.parse_args()
    summary = summarize(json.loads(args.input.read_text()))
    path = args.input.with_name('summary.json')
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print('CPU stage means (milliseconds per episode)')
    print('| Source | Threads | Select | Fetch | Collate | Total | Rejected centers |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for r in summary['cpu']:
        print(f"| {r['source']} | {r['threads']} | {r['select_seconds']*1000:.1f} | {r['fetch_seconds']*1000:.1f} | {r['collate_seconds']*1000:.1f} | {r['total_seconds']*1000:.1f} | {r['rejected_centers']:.1f} |")
    print('\nLoader throughput')
    for r in summary['loader']:
        print(f"workers={r['workers']}: {r['episodes_per_second']:.2f} episodes/s")
    print('\nGPU stage means (milliseconds per step)')
    for r in summary['gpu']:
        print(f"anomaly={r['anomaly_detection']}: transfer={r['transfer_seconds']*1000:.1f}, forward={r['forward_loss_seconds']*1000:.1f}, backward={r['backward_seconds']*1000:.1f}, optimizer={r['optimizer_seconds']*1000:.1f}, total={r['total_seconds']*1000:.1f}, peak={r['peak_allocated_gib']:.2f} GiB")


if __name__ == '__main__':
    main()
