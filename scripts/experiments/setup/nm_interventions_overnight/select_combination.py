"""Read ONLY training-source validation histories to choose a combination."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import ARMS, generate
from scripts.experiments.setup.nm_interventions_overnight.evaluate import discover


def select(jobs):
    cells={}
    for job in jobs:
        p=job['params'];selection=job['selection']
        history=json.loads((Path(job['state'])/'validation_history.json').read_text())
        row=next(r for r in history if r['step']==selection['best_step'])
        # No test result paths or target scores enter this function.
        prefix=p['prefix'];arm,rung=prefix[4:].rsplit('_r',1);rung=int(rung.split('_')[0])
        cells[arm,rung]=row
    expected={(a,r) for a in ARMS for r in range(1,9)}
    missing=sorted(expected-set(cells))
    if missing:raise ValueError(f'Individual validation coverage incomplete: {missing}')
    scores={}
    for arm in ARMS:
        if arm=='baseline':continue
        differences=[];by_source={}
        for rung in range(1,9):
            row=cells[arm,rung];base=cells['baseline',rung]
            differences.append(row['macro_roc_auc']-base['macro_roc_auc'])
            for source in row['per_source']:
                delta=row['per_source'][source]['roc_auc']-base['per_source'][source]['roc_auc']
                by_source.setdefault(source,[]).append(delta)
        mean=statistics.mean(differences)
        worst=min(statistics.mean(v) for v in by_source.values())
        scores[arm]=dict(mean_delta=mean,positive_rungs=sum(v>0 for v in differences),
            worst_source_mean_delta=worst,by_rung=differences,
            eligible=mean>0.001 and sum(v>0 for v in differences)>=4 and worst>=-0.01)
    chosen=[a for a,s in scores.items() if s['eligible'] and a!='budget']
    # Alternative center policies: retain one, never let code precedence choose silently.
    exclusive={'centers','negatives','region_adaptive','coverage'}
    candidates=[a for a in chosen if a in exclusive]
    if 'region_adaptive' in candidates and scores['region_adaptive']['mean_delta']<=scores['centers']['mean_delta']:
        chosen.remove('region_adaptive');candidates.remove('region_adaptive')
    if candidates:
        winner=max(candidates,key=lambda a:scores[a]['mean_delta'])
        chosen=[a for a in chosen if a not in exclusive or a==winner]
    # Cross-source class mixing cannot be simultaneously a source-block schedule.
    if 'composition' in chosen and 'schedule' in chosen:
        chosen.remove(min(('composition','schedule'),key=lambda a:scores[a]['mean_delta']))
    # Low-degree eligibility and strict uniform-positive overrides are alternative
    # positive construction policies; do not silently ignore either component.
    if 'eligibility' in chosen and 'positives' in chosen:
        chosen.remove(min(('eligibility','positives'),key=lambda a:scores[a]['mean_delta']))
    flags=sorted({flag for a in chosen for flag in ARMS[a][0].split(',') if flag})
    return dict(protocol='training_validation_only_v1',scores=scores,chosen_arms=chosen,flags=flags,
        holdout_used_for_selection=False,source='training-source validation_history.json only')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-dirs',nargs='+',required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();result=select(discover(args.run_dirs))
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:raise ValueError('Refusing to alter a frozen combination')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['flags']:
        generate(output=args.output.parent/'combined_configs',arms=['combined'],combined_flags=result['flags'])
    print(json.dumps(result,indent=2))
