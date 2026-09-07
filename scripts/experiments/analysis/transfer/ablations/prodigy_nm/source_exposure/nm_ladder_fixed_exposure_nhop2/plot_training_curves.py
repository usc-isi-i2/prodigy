"""Export sampled W&B training histories and plot the completed fixed-exposure ladder.

Run with --fetch to refresh the read-only W&B export; otherwise plot saved CSVs.
W&B history sampling is approximate, not a complete per-step archive.
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT = 'eibl-usc/graph-clip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    data = ROOT / 'data'
    figures = ROOT / 'figures'
    data.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    if args.fetch:
        import wandb
        api = wandb.Api(timeout=60)
        runs = api.runs(PROJECT, filters={'displayName': {'$regex': '^nm_ladder_fx10k_h2m_ord'}, 'state': 'finished'}, per_page=50)
        histories, metadata = [], []
        for run in runs:
            match = re.search(r'ord([AC])_r(\d+)_', run.name)
            if not match:
                continue
            order, rung = match.group(1), int(match.group(2))
            frame = run.history(samples=args.samples, keys=['train_loss', 'train_acc'], pandas=True).sort_values('_step')
            frame = frame[['_step', 'train_loss', 'train_acc']].drop_duplicates('_step')
            frame['order'], frame['rung'], frame['run_id'] = order, rung, run.id
            frame['optimizer_updates'] = frame['_step'] + 1
            histories.append(frame)
            metadata.append(dict(order=order, rung=rung, run_id=run.id, run_name=run.name, url=run.url, state=run.state, git_commit=run.commit, last_history_step=run.lastHistoryStep, requested_samples=args.samples, exported_rows=len(frame)))
            print(f'{order}{rung}: {len(frame)} sampled rows, last history step {run.lastHistoryStep}', flush=True)
        meta = pd.DataFrame(metadata).sort_values(['order', 'rung'])
        assert set(zip(meta.order, meta.rung)) == {('A', r) for r in range(1, 9)} | {('C', r) for r in range(1, 8)}
        assert (meta.last_history_step == meta.rung * 10000 - 1).all()
        pd.concat(histories).sort_values(['order', 'rung', '_step']).to_csv(data / 'training_history_sampled.csv', index=False)
        meta.to_csv(data / 'training_runs.csv', index=False)
    history = pd.read_csv(data / 'training_history_sampled.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey='row')
    colors = plt.get_cmap('viridis')
    for col, order in enumerate(['A', 'C']):
        selected = history[history.order == order]
        if order == 'C':
            selected = pd.concat([selected, history[(history.order == 'A') & (history.rung == 8)]])
        for rung, frame in selected.groupby('rung'):
            frame = frame.sort_values('optimizer_updates')
            # Fixed update-width bins prevent irregular W&B sample density from
            # changing the horizontal scale of the smoothing window.
            frame = frame.assign(bin=(frame.optimizer_updates - 1) // 1000)
            mean = frame.groupby('bin')[['optimizer_updates', 'train_loss', 'train_acc']].mean()
            label = f'{rung} sources' + (' (shared)' if order == 'C' and rung == 8 else '')
            for row, metric in enumerate(['train_loss', 'train_acc']):
                axes[row, col].plot(mean.optimizer_updates / 1000, mean[metric], color=colors((rung - 1) / 7), label=label, lw=1.6)
        axes[0, col].set_title(f'Order {order}')
        axes[1, col].set_xlabel('Optimizer updates (thousands)')
        axes[0, col].legend(fontsize=8, ncol=2)
    axes[0, 0].set_ylabel('NM training loss')
    axes[1, 0].set_ylabel('NM training accuracy')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('PRODIGY fixed-exposure ladder: training histories', fontsize=15)
    fig.text(.5, .015, 'W&B sampled histories; means within 1,000-update bins. One training seed. A8 and C8 share one model.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for suffix in ['png', 'pdf']:
        fig.savefig(figures / f'training_curves.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
