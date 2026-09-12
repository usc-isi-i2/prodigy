"""Compare two sequential versus two concurrent independent mapped readers."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--graph', type=Path, required=True)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
summary = {}
for mode in ('sequential', 'parallel'):
    start = time.perf_counter()
    jobs = []
    for i in range(2):
        output = a.output / f'{mode}-{i}.json'
        log = (a.output / f'{mode}-{i}.log').open('w')
        command = [sys.executable, str(Path(__file__).with_name('mmap_probe.py')),
                   '--graph', str(a.graph), '--cache', str(a.cache),
                   '--output', str(output), '--repeats', '1', '--modes', 'mmap']
        job = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((job, log, output))
        if mode == 'sequential':
            assert job.wait() == 0, f'Failed reader: {output}'
    for job, log, output in jobs:
        code = job.wait()
        log.close()
        assert code == 0, f'Failed reader: {output}'
    elapsed = time.perf_counter() - start
    reports = [json.loads(output.read_text())['runs'][0] for _, _, output in jobs]
    summary[mode] = dict(wall_s=elapsed, readers=reports)
    print(mode, elapsed, flush=True)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
signatures = {row['sample_sha256'] for report in summary.values() for row in report['readers']}
assert len(signatures) == 1, 'Concurrent readers changed sampled results'
