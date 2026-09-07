"""Verify and summarize downloaded shared-training smoke metadata."""
import argparse
import json
from pathlib import Path


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    status = json.loads((root/'status.json').read_text())
    checks = json.loads((root/'checkpoint_checks.json').read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(root.glob('job_*/result.json'))]
    configs = [json.loads(p.read_text()) for p in sorted(root.glob('job_*/effective_config.json'))]
    assert status['status'] == 'complete'
    assert len(jobs) == len(configs) == len(manifest['jobs']) == len(checks)
    assert manifest['mode'] == 'smoke'
    assert all(j['status'] == 'complete' and all(j['shared_storage'].values()) for j in jobs)
    assert all(j['physical_gpu'] == 2 for j in jobs)
    assert all(c['epochs'] == 1 and c['dataset_len_cap'] == 200 and c['workers'] == 4 for c in configs)
    assert all(c['exp_name'].startswith('smoke_') for c in configs)
    assert all(c['completed_steps'] == 200 and c['all_finite'] for c in checks)
    assert len({c['model_sha256'] for c in checks}) == len(checks)
    steady_start = min(j['steady_started'] for j in jobs)
    steady_end = max(j['steady_finished'] for j in jobs)
    return {
        'revision': manifest['revision'], 'models': len(jobs), 'physical_gpu': 2,
        'steps_per_model': 200, 'workers_per_model': 4,
        'shared_graph_setup_seconds': manifest['shared_graph_setup_seconds'],
        'all_shared_storage_checks_passed': True, 'all_terminal_checkpoints_finite_and_distinct': True,
        'all_model_steady_intervals_overlap_seconds': min(j['steady_finished'] for j in jobs)-max(j['steady_started'] for j in jobs),
        'aggregate_steady_steps': sum(j['steady_steps'] for j in jobs),
        'aggregate_steady_window_seconds': steady_end-steady_start,
        'aggregate_steady_steps_per_second': sum(j['steady_steps'] for j in jobs)/(steady_end-steady_start),
        'summed_per_model_steady_rates': sum(j['steady_steps_per_second'] for j in jobs),
        'training_phase_seconds_including_startup_and_terminal_saving': max(j['completed'] for j in jobs)-min(j['training_started'] for j in jobs),
        'max_per_model_peak_allocated_gib': max(j['peak_allocated_bytes'] for j in jobs)/2**30,
        'jobs': [dict(run=j['exp_name'], source_subset=c['neighbor_sampling_source_subset'],
                      steady_steps_per_second=j['steady_steps_per_second'],
                      peak_allocated_gib=j['peak_allocated_bytes']/2**30)
                 for j,c in zip(jobs, configs)],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    (args.input/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
