"""Private figure from complete saved-prediction pair accounting."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).parent
    results = json.loads((root/'data/degree_pair_accounting_20260907.json').read_text())
    rows = results['rows']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = ['#386f9f', '#c0473b', '#3a8961', '#343a40']
    names = ['Equal degree', 'Degree cue correct', 'Degree cue wrong', 'Total']
    means = {}
    for stream in ('original', 'fresh'):
        selected = [r for r in rows if r['stream'] == stream and r['decoder'] == 'full_model']
        values = [100*np.mean([r['change_contribution'] for r in selected if r['group'] == g])
                  for g in ('equal_degree', 'cue_correct', 'cue_wrong')]
        means[stream] = values + [sum(values)]
    for i, stream in enumerate(('original', 'fresh')):
        bars = axes[0].bar(np.arange(4)+(i-.5)*.34, means[stream], width=.32,
                          color=colors, alpha=1 if i == 0 else .45,
                          label=stream.capitalize())
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x()+bar.get_width()/2, h + (.35 if h >= 0 else -.35),
                         f'{h:+.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
    axes[0].set_xticks(range(4), ['Equal\ndegree', 'Cue\ncorrect', 'Cue\nwrong', 'Total'])
    axes[0].set_ylabel('Contribution to full-model AUC change (points)')
    axes[0].set_ylim(-16.3, 5.6)
    axes[0].set_title('A  Loss on cue-aligned pairs outweighs gains', loc='left', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    for stream, marker in (('original', 'o'), ('fresh', '^')):
        selected = [r for r in rows if r['stream'] == stream and r['decoder'] == 'full_model']
        for source in sorted({r['source'] for r in selected}):
            a = next(r for r in selected if r['source'] == source and r['group'] == 'cue_correct')
            b = next(r for r in selected if r['source'] == source and r['group'] == 'cue_wrong')
            axes[1].scatter(100*(a['late_group_ranking']-a['early_group_ranking']),
                            100*(b['late_group_ranking']-b['early_group_ranking']),
                            marker=marker, c='#386f9f', alpha=.75, s=45)
    axes[1].axvline(0, color='#888', lw=.8)
    axes[1].set_xlabel('Ranking change where degree cue is correct (points)')
    axes[1].set_ylabel('Ranking change where degree cue is wrong (points)')
    axes[1].set_title('B  A shared tradeoff across nine sources', loc='left', fontsize=11)
    for ax in axes:
        ax.axhline(0, color='#888', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('TwiBot20: training shifts rankings away from an informative degree cue', fontsize=13)
    out = root/'figures/degree_pair_argument.png'
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == '__main__':
    main()
