"""One fixed geometry residual, tested on untouched canonical cached NM inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from models.nm_geometry_readout import GeometryResidualReadout, mean_support_cosine
from scripts.experiments.setup.nm_hk_goal.run_mechanism import digest, bank_metadata


def summary(d):
    native = d.native_correct.astype(bool)
    out = dict(rows=len(d))
    if 'query' in d:
        out['distinct_queries'] = int(d['query'].nunique())
    for head in ['native', 'geometry', 'residual']:
        correct = d[head + '_correct'].astype(bool)
        out[head] = dict(correct=int(correct.sum()), accuracy=float(correct.mean()),
            recovered=int((~native & correct).sum()), lost=int((native & ~correct).sum()))
        if 'query' in d:
            out[head]['node_weighted_accuracy'] = float(d.groupby('query')[head + '_correct'].mean().mean())
    delta = d.residual_correct.astype(int) - native.astype(int)
    if 'episode' in d:
        ep = delta.groupby(d.episode).mean()
        out['paired_episode_deltas'] = dict(episodes=len(ep), improved=int((ep > 0).sum()),
            worsened=int((ep < 0).sum()), tied=int((ep == 0).sum()), mean=float(ep.mean()),
            sd=float(ep.std()), quantiles={str(k):float(v) for k,v in ep.quantile([0,.1,.5,.9,1]).items()},
            interpretation='Descriptive episode variability; repeated nodes and shared trained seed are not independent replications')
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_source_stage_20260908_v4'))
    p.add_argument('--original', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908'))
    p.add_argument('--extremes', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    assert 1 <= a.threads <= 2
    if a.dry_run:
        print(json.dumps(dict(device='cpu', threads=a.threads, targets=2, checkpoints=2,
            occurrences_per_cell=61440, fitting=False, weight_sweep=False, graph_loading=False,
            encoding=False, methods=['equal_standardized_native_mean_cosine']), indent=2))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    receipt = json.loads((a.cache / 'receipt.json').read_text())
    assert receipt['complete'] and receipt['model_states_unchanged']
    report = dict(complete=False, device='cpu', threads=a.threads,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=digest(Path(__file__).parent/'README.md'),
        source_receipt_sha256=digest(a.cache/'receipt.json'), checkpoints=receipt['checkpoints'],
        targets={}, fitting=False, graph_loading=False, encoding=False, new_sampling=False,
        rule='Equal mean of per-query standardized native and mean-support-cosine scores; population std floor 1e-8')
    head = GeometryResidualReadout().eval()
    slots = torch.tensor([i for i in range(210) if i % 7 >= 3])
    truth = slots // 7
    for target in ['cp_hk', 'ukr_rus']:
        info = receipt['targets'][target]
        refpath = a.cache / target / 'stage_predictions_private.csv'
        assert digest(refpath) == info['csv_sha256']
        ref = pd.read_csv(refpath).set_index(['model','episode','sample'])
        assert len(ref) == 122880 and ref.index.is_unique
        chunks, checked, numerical = [], [], []
        for bi,item in enumerate(info['batches']):
            path = a.cache / target / item['file']
            assert digest(path) == item['sha256']
            packed = torch.load(path, map_location='cpu', weights_only=False)
            checked.append(dict(file=item['file'],sha256=item['sha256']))
            for model, episodes in packed['models'].items():
                for ep,v in episodes.items():
                    pre, labels, edges, attrs, roles = v['inputs']
                    geometry = mean_support_cosine(pre,edges,attrs,roles,30)[slots]
                    error = float((geometry-v['mean_cosine']).abs().max())
                    assert error < 1e-6, (target,bi,ep,model,error)
                    native = v['native_logits']
                    scores = head(native,geometry)
                    index = pd.MultiIndex.from_tuples([(model,f'{bi}:{ep}',ep*210+int(q)) for q in slots])
                    base = ref.loc[index].reset_index()
                    assert torch.equal(native.argmax(1).eq(truth),torch.tensor(base.native_correct.values))
                    assert np.array_equal(native.argmax(1).numpy(),base.replay_native_prediction.values)
                    assert np.array_equal(v['mean_cosine'].argmax(1).numpy(),base.encoded_prediction.values)
                    d = base[['model','episode','sample','query','anchor','native_prediction','native_correct']].copy()
                    d['truth'] = truth.numpy()
                    d['geometry_prediction'] = geometry.argmax(1).numpy()
                    d['geometry_correct'] = geometry.argmax(1).eq(truth).numpy()
                    d['residual_prediction'] = scores.argmax(1).numpy()
                    d['residual_correct'] = scores.argmax(1).eq(truth).numpy()
                    d['residual_gap'] = (scores.topk(2,dim=1).values[:,0]-scores.topk(2,dim=1).values[:,1]).numpy()
                    numerical.append(dict(model=model,batch=bi,episode=ep,max_geometry_error=error,
                        geometry_argmax_differences=int((geometry.argmax(1)!=v['mean_cosine'].argmax(1)).sum()),
                        canonical_native_argmax_differences=int((native.argmax(1).numpy()!=base.native_prediction.values).sum()),
                        residual_ties_1e6=int((d.residual_gap<=1e-6).sum())))
                    chunks.append(d)
            print('BATCH',target,bi,'seconds',round(time.time()-started,2),flush=True)
        all_rows = pd.concat(chunks,ignore_index=True)
        assert len(all_rows)==122880 and not all_rows.duplicated(['model','episode','sample']).any()
        path = a.out / (target+'_predictions_private.csv')
        all_rows.to_csv(path,index=False)
        models = {}
        for model,s in all_rows.groupby('model'):
            assert len(s)==61440 and s.episode.nunique()==512
            assert s.groupby('episode').size().eq(120).all()
            freq = s.groupby('query')['query'].transform('size')
            models[model] = dict(all=summary(s),frequency_groups={label:summary(s[mask]) for label,mask in [
                ('1',freq.eq(1)),('2-4',freq.between(2,4)),('5-19',freq.between(5,19)),('20+',freq.ge(20))]})
        report['targets'][target] = dict(models=models,cache_files=checked,numerical=numerical,
            csv_sha256=digest(path),canonical_reference_sha256=info['csv_sha256'],
            complete_input_sha256=info['full_input_sha256'])
        (a.out/'receipt.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    # Secondary selected-case diagnostic: same fixed readout, no new selections.
    original_receipt = json.loads((a.original/'receipt.json').read_text())
    extreme_receipt = json.loads((a.extremes/'receipt.json').read_text())
    op = a.original/'embeddings_and_logits_private.pt'
    bp = a.extremes/'candidate_bank_private.pt'
    assert digest(op)==original_receipt['cache_sha256'] and digest(bp)==extreme_receipt['candidate_bank_sha256']
    original = torch.load(op,map_location='cpu',weights_only=False)
    bank = bank_metadata(bp)
    cases = {int(c['case']):c for c in original['cases']}
    tablepath = a.extremes/'selection_results_private.csv'
    assert digest(tablepath)==extreme_receipt['csv_sha256']
    selected = pd.read_csv(tablepath)
    persistent = selected.groupby('case').previously_persistent.first().to_dict()
    rows=[]
    for source,data in [('original',original),('bank',bank)]:
        names = data['head_order']
        scores = data['logits']
        native,geo = scores[:,names.index('native')],scores[:,names.index('mean_cosine')]
        combined = head(native,geo)
        for i,key in enumerate(data['logit_row_keys']):
            if source=='original' and key['condition']!='original':
                continue
            c = cases[int(key['case'])]
            r = dict(case=c['case'],cohort=c['cohort'],persistent=bool(persistent[c['case']]),
                condition='original' if source=='original' else key['method'],draw=int(key['draw']))
            r['original_native_correct'] = c['cohort']=='native_correct'
            for name,z in [('native',native),('geometry',geo),('residual',combined)]:
                r[name+'_prediction']=int(z[i].argmax())
                r[name+'_correct']=int(z[i].argmax())==c['truth']
            rows.append(r)
    d = pd.DataFrame(rows)
    assert len(d)==1800 and not d.duplicated(['case','condition','draw']).any()
    # Saved endpoints were checked during K/V; verify selected native predictions again.
    bankrows=d[d.condition!='original'].merge(selected,on=['case','draw'],suffixes=('','_ref'))
    bankrows=bankrows[bankrows.condition==bankrows.method]
    assert len(bankrows)==1600 and bankrows.native_prediction.eq(bankrows.native_prediction_ref).all()
    path = a.out/'selected_predictions_private.csv'
    d.to_csv(path,index=False)
    selected_summary={}
    for condition,s in d.groupby('condition'):
        selected_summary[condition]={}
        for cohort,frame in [('failed',s[s.cohort=='native_failed']),('correct',s[s.cohort=='native_correct']),('persistent',s[s.persistent])]:
            values=summary(frame)
            values['cases']=int(frame.case.nunique())
            values['comparison']='Recovery/loss uses native on the same original or modified inputs; accuracy remains selected-case/draw conditional'
            selected_summary[condition][cohort]=values
    report['selected']=dict(groups=selected_summary,csv_sha256=digest(path),
        original_cache_sha256=original_receipt['cache_sha256'],bank_sha256=extreme_receipt['candidate_bank_sha256'])
    report.update(complete=True,seconds=time.time()-started)
    (a.out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',report['seconds'],flush=True)


if __name__=='__main__':
    main()
