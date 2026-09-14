"""Evaluate added targets now, then complete the nine-target aggregate."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path('/dataMeR1/phil/gfm/mixture-scaling')
state = root / 'state/node_mlp_ladder_s0_convergence_recovery'
output = root / 'results/node_mlp_ladder_s0_convergence_recovery/raw'
common = ['--state-root', str(state), '--output-root', str(output), '--cache-root',
          '/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer/_cache',
          '--device', '1', '--workers', '1', '--convergence']
prefix = [sys.executable, '-u', '-m', 'mixture_scaling.node_mlp_ladder']
added = ['covid_political', 'election2020', 'ukr_rus_suspended']
finished = [str(k) for k in range(1, 10) if (state / 'lp' / f'ladder_r{k}' / 'summary.json').exists()]
subprocess.run(prefix + ['eval', *common, '--rungs', ','.join(finished), '--targets', *added], check=True)
print('Added targets evaluated for finished rungs', flush=True)
while not (root / 'log/node_mlp_ladder_s0_convergence_recovery/COMPLETE').exists():
    time.sleep(15)
subprocess.run(prefix + ['eval', *common, '--targets', *added], check=True)
subprocess.run(prefix + ['aggregate', *common], check=True)
result = json.loads((output / 'COMPLETE.json').read_text())
assert result['cells'] == 81
print(json.dumps(result), flush=True)
