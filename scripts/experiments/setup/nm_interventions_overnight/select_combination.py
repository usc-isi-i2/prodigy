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


def compatible_arms(scores):
    """Retain higher validation gains when two implementations override each other."""
    center_policies={'centers','negatives','region_adaptive','coverage'}
    conflicts={frozenset((a,b)) for a in center_policies for b in center_policies if a!=b}
    conflicts.update(map(frozenset,[('composition','schedule'),('exposure','schedule'),
        ('composition','negatives'),('eligibility','positives')]))
    candidates=sorted((a for a,s in scores.items() if s['eligible'] and a!='budget'),
                      key=lambda a:(-scores[a]['mean_delta'],a))
    chosen=[];excluded={}
    for arm in candidates:
        if arm=='region_adaptive' and scores[arm]['mean_delta']<=scores['centers']['mean_delta']:
            excluded[arm]='Did not improve over the uniform degree-band centers arm'
            continue
        blockers=[a for a in chosen if frozenset((a,arm)) in conflicts]
        if blockers:
            excluded[arm]='Incompatible with higher-validation-gain arm(s): '+','.join(blockers)
        else:
            chosen.append(arm)
    return chosen,excluded


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
    chosen,excluded=compatible_arms(scores)
    flags=sorted({flag for a in chosen for flag in ARMS[a][0].split(',') if flag})
    return dict(protocol='training_validation_only_v2',scores=scores,chosen_arms=chosen,flags=flags,
        compatibility_exclusions=excluded,
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
