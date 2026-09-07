"""Read-only pair accounting conditional on BOTH relations' early rank credit.

This tests an opportunity-count explanation, not causal graph mediation.
Nine exhaustive early-state strata retain ties and equal episode weighting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

DECODERS = ('full_model', 'S0_pool/ridge', 'U1_pre_meta/ridge')
STATES = ((0., 'wrong'), (.5, 'tied'), (1., 'correct'))


def transition_accounting(y, retweet_early, retweet_late, follow_early, follow_late):
    arrays = [np.asarray(x) for x in (y, retweet_early, retweet_late, follow_early, follow_late)]
    y = arrays[0]
    if any(x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Aligned finite vectors required')
    if set(y) != {0, 1}:
        raise ValueError('Both binary classes required')
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    credits = []
    for score in arrays[1:]:
        d = score[pos, None] - score[neg][None, :]
        credits.append((d > 0).astype(float) + .5*(d == 0))
    re, rl, fe, fl = credits
    rows = []
    for rvalue, rname in STATES:
        for fvalue, fname in STATES:
            mask = (re == rvalue) & (fe == fvalue)
            row = dict(retweet_early_state=rname, follow_early_state=fname,
                       pairs=int(mask.sum()), mass=float(mask.mean()))
            values = dict(retweet_early=re, retweet_late=rl, follow_early=fe,
                          follow_late=fl, retweet_change=rl-re, follow_change=fl-fe,
                          difference_in_changes=(fl-fe)-(rl-re))
            for name, value in values.items():
                row[name+'_contribution'] = float(value[mask].sum()/mask.size)
            rows.append(row)
    totals = [float(x.mean()) for x in credits]
    if sum(r['pairs'] for r in rows) != re.size:
        raise ValueError('Incomplete pair partition')
    expected = totals[3]-totals[2]-totals[1]+totals[0]
    if abs(sum(r['difference_in_changes_contribution'] for r in rows)-expected) > 1e-12:
        raise ValueError('Interaction accounting failed')
    return rows, totals


def main():
    import torch
    from sklearn.metrics import roc_auc_score
    from .analyze_mixture_predictions import input_labels
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--follow-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    done = json.loads((args.follow_root/'DONE.json').read_text())
    cache_done = json.loads((args.cache_root/'DONE.json').read_text())
    if done['rows'] != 612 or len(done['receipts']) != 36:
        raise ValueError('Completed matched endpoint evaluation required')
    if len(cache_done['receipts']) != 2 or not all(
            r['exact_centers_features_labels_task_tensors'] for r in cache_done['receipts']):
        raise ValueError('Matched task receipt required')
    rows, receipts = [], []
    for stream in ('original', 'fresh'):
        ref = args.reference_root/stream/'twibot20'
        cache = args.cache_root/stream/'twibot20'
        follow = args.follow_root/stream/'twibot20'
        old, new = input_labels(ref, 'twibot20'), input_labels(cache, 'twibot20')
        for key in ('local_y', 'mapping', 'episode_ids'):
            if not torch.equal(old[key], new[key]):
                raise ValueError('Task labels or ordering changed')
        # Verify current center identities, not only a historical receipt.
        for bi in range(32):
            a = torch.load(ref/'batches'/f'batch_{bi:03d}.pt', map_location='cpu', weights_only=False)[0]
            b = torch.load(cache/'batches'/f'batch_{bi:03d}.pt', map_location='cpu', weights_only=False)[0]
            if not torch.equal(a.global_node_ids[a.ptr[:-1]], b.global_node_ids[b.ptr[:-1]]):
                raise ValueError('Center identities changed')
            if not torch.equal(a.x[a.ptr[:-1]], b.x[b.ptr[:-1]]):
                raise ValueError('Center features changed')
        metadata = [[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
                    for d in (ref, follow)]
        arms = [r for r in metadata[0] if r['decoder']=='full_model' and r['model_id'].endswith('_step100')]
        if len(arms) != 9 or len({r['sources'][0] for r in arms}) != 9:
            raise ValueError('Nine source checkpoints required')
        y, episodes = old['local_y'].numpy(), old['episode_ids'].numpy()
        for arm in arms:
            scores = {}
            for relation, directory, labels, meta in zip(('retweet','follow'), (ref,follow), (old,new), metadata):
                for step in (100,2500):
                    mid = arm['model_id'].rsplit('_step',1)[0]+f'_step{step}'
                    path = directory/(mid+'__baseline.pt')
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    if len(saved) != 32 or any(item['batch'] != bi or item['batch_sha256'] != labels['cache']['batch_sha256'][bi]
                                              for bi,item in enumerate(saved)):
                        raise ValueError('Prediction/cache identity mismatch')
                    for decoder in DECODERS:
                        logits = torch.cat([item['logits'][decoder] for item in saved])
                        if logits.shape != (len(y),2) or not torch.isfinite(logits).all():
                            raise ValueError('Invalid logits')
                        score = logits.softmax(1)[:,1].numpy()
                        scores[relation,step,decoder] = score
                        episode_auc = np.mean([roc_auc_score(y[episodes==ep], score[episodes==ep]) for ep in np.unique(episodes)])
                        records = [r for r in meta if r['model_id']==mid and r['decoder']==decoder]
                        if len(records)!=1:
                            raise ValueError('Missing metric reference')
                        # The old replay may not record episode AUC; validate pooled AUC too.
                        mapping = labels['mapping']
                        truth = mapping[torch.arange(len(y)), labels['local_y']].numpy()
                        prob = logits.softmax(1)
                        semantic = prob[torch.arange(len(y)), (mapping==1).long().argmax(1)].numpy()
                        if abs(roc_auc_score(truth,semantic)-records[0]['roc_auc']) > 1e-6:
                            raise ValueError('Pooled AUC not reproduced')
                        if 'mean_episode_auc' in records[0] and abs(episode_auc-records[0]['mean_episode_auc']) > 1e-12:
                            raise ValueError('Episode AUC not reproduced')
                    receipts.append(dict(stream=stream, relation=relation, model_id=mid,
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for decoder in DECODERS:
                parts, totals = [], []
                for ep in np.unique(episodes):
                    mask = episodes==ep
                    part,total = transition_accounting(y[mask], *[
                        scores[relation,step,decoder][mask] for relation,step in
                        (('retweet',100),('retweet',2500),('follow',100),('follow',2500))])
                    parts.append(part); totals.append(total)
                for index in range(9):
                    selected = [part[index] for part in parts]
                    row = dict(stream=stream, source=arm['sources'][0], decoder=decoder,
                               retweet_early_state=selected[0]['retweet_early_state'],
                               follow_early_state=selected[0]['follow_early_state'],
                               episodes=len(selected), episodes_present=sum(r['pairs']>0 for r in selected),
                               pairs=sum(r['pairs'] for r in selected),
                               endpoint_auc_means=np.mean(totals,axis=0).tolist())
                    for key in ('mass',)+tuple(k for k in selected[0] if k.endswith('_contribution')):
                        row[key] = float(np.mean([r[key] for r in selected]))
                    for key in tuple(k for k in row if k.endswith('_contribution')):
                        row[key.replace('_contribution','_conditional')] = row[key]/row['mass'] if row['mass'] else None
                    rows.append(row)
    if len(rows)!=486 or len(receipts)!=72:
        raise ValueError('Incomplete grid')
    args.output.mkdir(parents=True)
    result = dict(rows=rows, receipts=receipts, new_model_forwards=0, new_training=False,
                  weighting='equal episode; half-credit ties; conditional scores divide weighted credit by weighted mass',
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows), receipts=len(receipts), output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
