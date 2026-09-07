"""Read-only matched comparison; no target selection, fitting, or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

TARGETS = ('covid_political', 'election2020', 'facebook_page_reference',
           'twibot20', 'ukr_rus_suspended')
METRICS = ('accuracy', 'macro_f1', 'roc_auc', 'nll')
OLD_HASH = 'b87f9ddfd0d1a8fa2c7025f8ab4953aa099f3c345113ba50e93990f661891202'


def compare(old, new):
    """Positive deltas mean higher values, including NLL (where higher is worse)."""
    old_index = {(r['stream'], r['schedule'], r['target'], r['mode'], r['readout']): r
                 for r in old['cells']}
    new_index = {(r['stream'], r['schedule'], r['target'], r['decoder']): r
                 for r in new['rows']}
    assert len(old_index) == len(old['cells']) == 400
    assert len(new_index) == len(new['rows']) == 80
    comparisons = []
    accounting = []
    for stream in ('original', 'fresh'):
        for schedule in ('blocked', 'interleaved'):
            for target in TARGETS:
                direct = new_index[stream, schedule, target, 'u1_centered']
                raw = new_index[stream, schedule, target, 'raw_centered']
                native = old_index[stream, schedule, target, 'native', 'full']
                native_centered = old_index[stream, schedule, target, 'native', 'u1_centered']
                terms = {
                    'readout_change': {m: native_centered[m]-native[m] for m in METRICS},
                    'training_change': {m: direct[m]-native_centered[m] for m in METRICS},
                    'total': {m: direct[m]-native[m] for m in METRICS}}
                for metric in METRICS:
                    assert abs(terms['total'][metric]-terms['readout_change'][metric]
                               -terms['training_change'][metric]) < 1e-12
                accounting.append(dict(stream=stream, schedule=schedule, target=target, **terms))
                for mode in ('native', 'joint', 'isolated', 'ridge_only'):
                    reference = old_index[stream, schedule, target, mode, 'u1_centered']
                    old_raw = old_index[stream, schedule, target, mode, 'raw_centered']
                    # Identical input/readout diagnostic, not a substitute for tensor receipts.
                    for metric in METRICS:
                        assert abs(raw[metric] - old_raw[metric]) < 1e-6, (stream, schedule, target, metric)
                    comparisons.append(dict(stream=stream, schedule=schedule, target=target,
                                            reference=mode,
                                            delta={m: direct[m]-reference[m] for m in METRICS},
                                            direct={m: direct[m] for m in METRICS},
                                            baseline={m: reference[m] for m in METRICS}))
    panels = []
    for stream in ('original', 'fresh'):
        for schedule in ('blocked', 'interleaved'):
            for mode in ('native', 'joint', 'isolated', 'ridge_only'):
                for panel in ('fixed4', 'all5'):
                    rows = [r for r in comparisons if r['stream'] == stream and
                            r['schedule'] == schedule and r['reference'] == mode and
                            (panel == 'all5' or r['target'] != 'ukr_rus_suspended')]
                    assert len(rows) == (4 if panel == 'fixed4' else 5)
                    panels.append(dict(stream=stream, schedule=schedule, reference=mode,
                                       panel=panel, delta={m: statistics.mean(r['delta'][m] for r in rows)
                                                          for m in METRICS}))
    accounting_panels = []
    for stream in ('original', 'fresh'):
        for schedule in ('blocked', 'interleaved'):
            for panel in ('fixed4', 'all5'):
                selected = [r for r in accounting if r['stream'] == stream and
                            r['schedule'] == schedule and
                            (panel == 'all5' or r['target'] != 'ukr_rus_suspended')]
                accounting_panels.append(dict(stream=stream, schedule=schedule, panel=panel,
                    **{term: {m: statistics.mean(r[term][m] for r in selected) for m in METRICS}
                       for term in ('readout_change', 'training_change', 'total')}))
    return dict(cells=comparisons, panels=panels, accounting=accounting,
                accounting_panels=accounting_panels,
                scope='Direct minus reference; same fixed centered readout. NLL higher is worse. '
                      'One training seed, reused streams; no uncertainty or universal-necessity claim.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--accounting-figure', type=Path)
    args = parser.parse_args()
    old_bytes = args.reference.read_bytes()
    assert hashlib.sha256(old_bytes).hexdigest() == OLD_HASH
    status = json.loads((args.evaluation.parent/'execution_status.json').read_text())
    assert status['status'] == 'complete'
    new_bytes = args.evaluation.read_bytes()
    result = compare(json.loads(old_bytes), json.loads(new_bytes))
    result['summary_sha256'] = dict(reference=OLD_HASH,
                                    evaluation=hashlib.sha256(new_bytes).hexdigest())
    result['source_summary_sha256'] = json.loads(new_bytes).get('source_summary_sha256')
    if args.accounting_figure:
        import matplotlib.pyplot as plt
        rows = [r for r in result['accounting_panels'] if r['panel'] == 'fixed4']
        readout = [100*r['readout_change']['macro_f1'] for r in rows]
        training = [100*r['training_change']['macro_f1'] for r in rows]
        fig, ax = plt.subplots(figsize=(8.3,4.8))
        ax.bar(range(4), readout, color='#177f89', label='Change readout; keep native-trained encoder')
        ax.bar(range(4), training, bottom=readout, color='#e5aa58', label='Then change training objective')
        for i, (a,b) in enumerate(zip(readout,training)):
            ax.text(i,a/2,f'+{a:.2f}',ha='center',va='center',color='white')
            ax.text(i,a+b/2,f'+{b:.2f}',ha='center',va='center')
        ax.set(xticks=range(4),xticklabels=['Original\nblocked','Original\ninterleaved',
               'Fresh\nblocked','Fresh\ninterleaved'],ylabel='Macro-F1 gain over native model (points)',ylim=(0,6.6))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9,loc='upper left')
        fig.suptitle('Most of the measured gain requires no retraining',weight='bold')
        fig.text(.5,.025,'Fixed four-target mean; one training seed and two reused streams.\n'
                 'Exact readout-first accounting, not a causal mediation fraction. Per-target effects can be negative.',
                 ha='center',fontsize=9)
        fig.subplots_adjust(left=.11,right=.98,top=.88,bottom=.23)
        args.accounting_figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.accounting_figure,dpi=170)
        plt.close(fig)
    if args.figure:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        labels = ['Political', 'Election', 'Facebook', 'TwiBot20', 'Suspension']
        for ax, mode in zip(axes, ('native', 'joint')):
            for schedule, offset, color in [('blocked', -.1, '#177f89'), ('interleaved', .1, '#b2543f')]:
                rows = [next(r for r in result['cells'] if r['stream'] == 'fresh' and
                             r['schedule'] == schedule and r['target'] == t and r['reference'] == mode)
                        for t in TARGETS]
                ax.scatter([100*r['delta']['macro_f1'] for r in rows],
                           [i+offset for i in range(5)], color=color, label=schedule, s=45)
            ax.axvline(0, color='gray', lw=.8)
            ax.set_title('Direct training − '+mode+' training')
            ax.set(yticks=range(5), yticklabels=labels, xlabel='Macro-F1 difference (points)', xlim=(-4,7))
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].invert_yaxis()
        axes[1].legend(frameon=False)
        fig.suptitle('A strengthened direct objective removes the claimed training advantage', weight='bold')
        fig.text(.5, .035, 'Same support-centered readout for every encoder · fresh stream · one training seed\n'
                 'All five targets shown; no uncertainty intervals. Centering and learned training scale changed together.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.12, right=.98, top=.82, bottom=.22, wspace=.22)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
