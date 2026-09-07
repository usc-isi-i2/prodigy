"""Private JSON-only summary of the frozen natural-pair test; no forwards."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).parent
    data = root / 'data/kv_generality_20260907'
    read = lambda name: json.loads((data / name).read_text())
    rows, geometry, receipts = read('metrics.json'), read('geometry.json'), read('receipts.json')
    assert len(rows) == 10 and len(receipts) == 2
    assert read('protocol.json')['frozen_prediction']['practical_separation_auc'] == .01
    conditions = ['keys_only', 'values_only', 'joint_kv']
    colors = ['#7964A7', '#207F83', '#C56832']
    streams = ['original', 'fresh']
    summary = []
    for stream in streams:
        block = {r['condition']: r for r in rows if r['stream'] == stream}
        assert set(block) == {'intact', 'removed', *conditions}
        delta = {c: 100*(block[c]['within_episode_auc']-block['intact']['within_episode_auc']) for c in conditions}
        assert delta['values_only'] < 0 and delta['values_only'] <= delta['keys_only']-1
        receipt = next(r for r in receipts if r['stream'] == stream)
        assert all(receipt[k] for k in ['native_endpoints_bit_exact', 'joint_endpoint_bit_exact', 'label_attention_routes_bit_exact'])
        assert receipt['exact_query_checks'] == 160
        for c in conditions:
            group = [r for r in geometry if r['stream'] == stream and r['condition'] == c]
            assert len(group) == 128
            tv = np.mean([r['attention_tv_intact'] for r in group])
            if c == 'values_only': assert all(r['attention_tv_intact'] == 0 for r in group)
            summary.append(dict(stream=stream, condition=c, delta_auc_points=delta[c],
                delta_accuracy_points=100*(block[c]['accuracy']-block['intact']['accuracy']),
                attention_tv=float(tv)))
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8, 1.9), gridspec_kw={'width_ratios': [1.2, 1]})
    for i, (condition, color) in enumerate(zip(conditions, colors)):
        group = [r for r in summary if r['condition'] == condition]
        x = np.arange(2)+(i-1)*.23
        y = [r['delta_auc_points'] for r in group]
        axes[0].bar(x, y, width=.21, color=color)
        for xx, yy in zip(x, y): axes[0].text(xx, yy+(.2 if yy>=0 else -.35), f'{yy:+.2f}', ha='center', va='bottom' if yy>=0 else 'top', fontsize=7)
        axes[1].bar(x, [r['attention_tv'] for r in group], width=.21, color=color)
    for ax in axes:
        ax.set_xticks([0, 1], ['Original', 'Fresh'])
        ax.axhline(0, color='#64748b', lw=.6)
    axes[0].set_ylim(-8.5, 3.5)
    axes[0].set_ylabel('Within-episode AUC change (pp)', fontsize=7)
    axes[0].set_title('D  Ukraine to bots: useful values', loc='left', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, .1)
    axes[1].set_ylabel('Attention total variation', fontsize=7)
    axes[1].set_title('E  Keys engaged; value routes fixed', loc='left', fontsize=9, fontweight='bold')
    axes[1].text(.5, .091, 'Values-only: exactly zero in every episode', ha='center', fontsize=7, color=colors[1])
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.82, wspace=.38)
    fig.savefig(root/'figures/kv_generality.png', dpi=240)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
