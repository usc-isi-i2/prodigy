"""Frozen seed-zero campaign; source names match immutable final-core artifact."""
from pathlib import Path
import argparse
import json
import yaml

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
HOLDOUT = 'twibot20'
ORDER = ('ukr_rus','covid','midterm','covid_political','election2020','ukr_rus_suspended','cp_hk','facebook_page_reference')
TARGETS = ('ukr_rus','covid','midterm','covid_political','election2020','ukr_rus_suspended','twibot20','cp_hk','facebook_page_reference')
ARMS = {
 'baseline': ('', 'Balanced source-confined NM, shared 256-dimensional encoder'),
 'exposure': ('proportional', 'Source probability proportional to source node count'),
 'schedule': ('blocked', '64 consecutive episodes per source, cycling over active sources'),
 'composition': ('cross_graph', 'Each class chooses an active source with balanced probability'),
 'centers': ('degree_balanced', 'Choose a positive-degree log2 band uniformly, then a center'),
 'eligibility': ('low_degree', 'Keep degree-2+ centers; repeat only within disjoint support/query partitions'),
 'positives': ('uniform_positive', 'Uniform unique one-hop positives instead of sorted random-walk discoveries'),
 'negatives': ('degree_hard', 'Competing centers share a log2 degree band; band probability proportional to size'),
 'context': ('one_hop', 'One-hop training context, fanout 100, same 101-node cap; evaluation stays two-hop'),
 'optimization': ('grad_normalized', 'Normalize each episode gradient to unit global norm before AdamW'),
 'alignment': ('feature_standardized', 'Per-node feature standardization without learned affine parameters'),
 'sharing': ('source_affine', 'Learn a source-specific input affine transform; absent sources remain identity'),
 'capacity': ('wide', '512-dimensional encoder instead of 256; equal episode cap, report runtime'),
 'objective': ('aux_reconstruction', 'NM plus 0.1 masked-feature reconstruction, masking 15% of sampled real nodes'),
 'region_adaptive': ('region_adaptive', '70% uniform-band / 30% bounded loss-adaptive degree-region sampling'),
 'coverage': ('coverage_cycle', 'Random-start cyclic traversal of eligible center pools'),
 'budget': ('per_source_budget', 'Cap grows by 1250 episodes per source, up to 10000 at rung eight'),
}


def generate(output=HERE/'configs', cap=10000, interval=2000, arms=None, rungs=None, combined_flags=None):
    base = yaml.safe_load((ROOT/'scripts/experiments/setup/final_core/training.yaml').read_text())
    base.update(batch_size=1, learning_rate=0.001, epochs=1, dataset_len_cap=cap,
        val_len_cap=1, test_len_cap=1, workers=4, seed=0,
        eval_test_before_train=False, eval_val_before_train=False, eval_after_train=False,
        eval_step=cap+1, checkpoint_step=cap+1, checkpoint_steps='',
        early_stopping_patience=2, campaign_protocol=True, campaign_eval_interval=interval,
        campaign_min_delta=0.001, campaign_val_per_source=16, campaign_holdout=HOLDOUT,
        attr_regression_weight=0.1, detect_anomaly=False,
        debug=False)
    # The auxiliary head exists for every arm to preserve initialization parity;
    # its loss is used only by the objective arm.
    base.pop('debug', None)
    definitions = dict(ARMS)
    if combined_flags is not None:
        definitions['combined'] = (','.join(sorted(combined_flags)), 'Training-validation-selected compatible combination')
    arms = arms or list(definitions)
    rungs = rungs or [8,1,2,3,4,5,6,7]
    rows=[]
    output.mkdir(parents=True, exist_ok=True)
    for rung in rungs:
        for arm in arms:
            flags, description = definitions[arm]
            p = {**base, 'campaign_flags':flags, 'prefix':f'nmi_{arm}_r{rung}_s0',
                 'neighbor_sampling_source_subset':','.join(ORDER[:rung])}
            if 'wide' in flags.split(','):
                p['emb_dim']=512
            if 'per_source_budget' in flags.split(','):
                p['dataset_len_cap']=min(cap,1250*rung)
            path=output/f'{arm}_r{rung}_s0.yaml'
            path.write_text(yaml.safe_dump(p,sort_keys=True))
            rows.append(dict(arm=arm,rung=rung,seed=0,flags=flags,description=description,
                sources=list(ORDER[:rung]),holdout=HOLDOUT,cap=p['dataset_len_cap'],config=str(path.relative_to(ROOT))))
    (output.parent/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=HERE/'configs')
    a.add_argument('--cap',type=int,default=10000);a.add_argument('--interval',type=int,default=2000)
    a.add_argument('--arms',nargs='+');a.add_argument('--rungs',nargs='+',type=int)
    p=a.parse_args(); rows=generate(p.output,p.cap,p.interval,p.arms,p.rungs)
    print(f'{len(rows)} models; held out {HOLDOUT}; {sum(r["cap"] for r in rows)} capped episodes')
