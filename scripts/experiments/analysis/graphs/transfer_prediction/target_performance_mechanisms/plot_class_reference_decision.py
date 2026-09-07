"""Four-panel argument: scope, final computation, learned reversal, generality."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SOURCES = ['covid', 'covid_political', 'cp_hk', 'election2020', 'facebook_page_reference',
           'midterm', 'twibot20', 'ukr_rus', 'ukr_rus_suspended']
SOURCE_LABELS = ['covid', 'covid-political', 'hongkong', 'election2020', 'facebook-page',
                 'midterm', 'twibot20', 'ukraine', 'ukraine-suspended']
TARGETS = ['covid_political', 'election2020', 'facebook_page_reference', 'twibot20', 'ukr_rus_suspended']
THREE = ['covid_political', 'facebook_page_reference', 'twibot20']
COLORS = {'removed': '#C86A2C', 'orientation_only': '#6553A0', 'strength_only': '#24837B'}
POOLED = 'delta_pooled_margin_auc_points'
WITHIN = 'delta_within_episode_auc_points'


def contrast_panel(ax, frame, initial):
    frame = frame[frame.source.eq('cp_hk') & frame.step.eq(2500 if initial else 50000)]
    for condition, offset in [('removed', -.27), ('orientation_only', -.09), ('strength_only', .09)]:
        for index, target in enumerate(THREE):
            cell = frame[frame.target.eq(target) & frame.condition.eq(condition)]
            means = cell.groupby('seed')[POOLED].mean() if initial else cell.set_index('stream')[POOLED]
            x = index + offset
            ax.scatter(x, means.mean(), color=COLORS[condition], s=25, zorder=4)
            if initial:
                ax.errorbar(x, means.mean(), yerr=means.std(ddof=1), color=COLORS[condition], capsize=2, lw=.9)
            else:
                ax.scatter(x + np.array([-.025, .025]), means, color=COLORS[condition], s=9, alpha=.55)
        # Means summarize complete seed/stream cells, not selected episodes.
    for index, target in enumerate(THREE):
        cell = frame[frame.target.eq(target) & frame.condition.eq('removed')]
        means = cell.groupby('seed')[WITHIN].mean() if initial else cell.set_index('stream')[WITHIN]
        ax.scatter(index + .27, means.mean(), color='#192B3C', s=22, marker='D', zorder=4)
        if initial:
            ax.errorbar(index + .27, means.mean(), yerr=means.std(ddof=1), color='#192B3C', capsize=2, lw=.9)
        else:
            ax.scatter(index + .27 + np.array([-.025, .025]), means, color='#192B3C', s=8, alpha=.5, marker='D')
    ax.axhline(0, color='#89929A', lw=.6)
    ax.grid(axis='y', alpha=.15)
    ax.set_xticks(range(3), ['Political', 'Pages', 'Bots'])
    ax.set_xlim(-.55, 2.55)
    ax.set_ylabel('AUC change (points)')
    ax.set_ylim((-14, 12) if initial else (-23, 13))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root', type=Path, default=Path(__file__).parent / 'data')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.data_root / 'classref_decision_20260907'
    grid = pd.read_csv(args.data_root / 'message_scale_20260907/nine_source_map.csv')
    initial = pd.read_csv(root / 'initial_contrast_cells.csv')
    long = pd.read_csv(root / 'long_contrast_cells.csv')
    steps = pd.read_csv(root / 'training_ranking_cells.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'axes.titleweight': 'bold', 'xtick.labelsize': 7.5, 'ytick.labelsize': 7})
    fig = plt.figure(figsize=(8, 5.75), facecolor='white')
    heat = fig.add_axes([.17, .535, .30, .38])
    upper = fig.add_axes([.62, .535, .355, .38])
    training = fig.add_axes([.17, .095, .30, .275])
    lower = fig.add_axes([.62, .095, .355, .275])
    heat.set_title('A  Support effects depend on the source', loc='left', pad=16)
    means = grid.groupby(['source', 'target']).support_delta_points.mean().unstack().reindex(index=SOURCES, columns=TARGETS)
    heat.imshow(means, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-11, vcenter=0, vmax=11))
    for yi, source in enumerate(SOURCES):
        for xi, target in enumerate(TARGETS):
            values = grid[(grid.source == source) & (grid.target == target)].support_delta_points
            value = means.loc[source, target]
            mixed = values.min() < 0 < values.max()
            heat.text(xi, yi, f'{value:+.1f}' + ('*' if mixed else ''), ha='center', va='center',
                      fontsize=7, color='white' if abs(value) > 6 else '#182737')
            if source == target:
                heat.add_patch(Rectangle((xi - .48, yi - .48), .96, .96, fill=False, lw=.8, color='#152230'))
    heat.set_xticks(range(5), ['CP', 'E20', 'FB', 'BOT', 'SUSP'])
    heat.set_yticks(range(9), SOURCE_LABELS)
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)
    fig.text(.17, .437, 'Pooled AUC points: blue helps, red hurts; * signs differ.\n'
             'Box: source = target. Two-stream means; one seed/source.\n'
             'Historical checkpoints, distinct from panel B models.', fontsize=6.4, linespacing=1.25)
    contrast_panel(upper, initial, True)
    upper.set_title('B  Final contrast: orientation dominates', loc='left', pad=16)
    upper.text(.5, 1.015, 'Hong Kong, 2.5k; mean +/- SD of 3 seed means', transform=upper.transAxes,
               ha='center', fontsize=6.7, color='#53616E')
    handles = [Line2D([], [], marker='o', color=COLORS[c], ls='', ms=4, label=l) for c, l in
               [('removed', 'Full removal'), ('orientation_only', 'Orientation only'), ('strength_only', 'Strength only')]]
    handles.append(Line2D([], [], marker='D', color='#192B3C', ls='', ms=4, label='Full, within episode'))
    upper.legend(handles=handles, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.10),
                 frameon=False, fontsize=6.8, columnspacing=.8, handletextpad=.4, borderaxespad=0)
    training.set_title('C  Training reverses page support utility', loc='left', pad=16)
    page = steps[steps.source.eq('cp_hk') & steps.target.eq('facebook_page_reference') &
                 steps.condition.eq('removed') & steps.step.isin([100, 300, 900, 2500])]
    for metric, color, label, offset in [(POOLED, '#C86A2C', 'Pooled', -.06), (WITHIN, '#192B3C', 'Within episode', .06)]:
        seed = page.groupby(['step', 'seed'])[metric].mean().reset_index()
        means = seed.groupby('step')[metric].agg(['mean', 'std']).sort_index()
        training.errorbar(np.arange(4) + offset, means['mean'], yerr=means['std'], marker='o', ms=3,
                          color=color, capsize=2, lw=1.1, label=label)
    training.axhline(0, color='#89929A', lw=.6)
    training.set_xticks(range(4), ['100', '300', '900', '2,500'])
    training.set_xlabel('Training step (fixed target inputs)')
    training.set_ylabel('Removal AUC change (points)')
    training.grid(axis='y', alpha=.15)
    training.set_ylim(-6.5, 3)
    training.legend(frameon=False, fontsize=6.7, loc='lower right')
    training.text(.5, 1.015, 'Hong Kong; mean +/- SD of 3 seed means', transform=training.transAxes,
                  ha='center', fontsize=6.7, color='#53616E')
    contrast_panel(lower, long, False)
    lower.set_title('D  50k: a mixed generality result', loc='left', pad=16)
    lower.text(.5, 1.015, 'One nominated checkpoint; large: mean; small: 2 streams', transform=lower.transAxes,
               ha='center', fontsize=6.7, color='#53616E')
    fig.text(.08, .004, 'CP: political labels; E20: election; FB: page classes; BOT: twibot20; SUSP: suspension. '
             'Streams are not seeds. Error bars are not confidence intervals.', fontsize=6.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
