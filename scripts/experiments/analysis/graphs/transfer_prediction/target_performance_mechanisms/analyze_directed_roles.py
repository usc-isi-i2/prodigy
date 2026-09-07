"""Private fixed-gate summary of the directed-role experiment, only when complete."""
import argparse
import json
from pathlib import Path
import numpy as np

CONDITIONS=('intact','support','query','both')
DECODERS=('full_model','S0_pool/ridge','U1_pre_meta/ridge')


def summarize(rows):
    sources=sorted({r['source'] for r in rows})
    if len(sources)!=9 or len(rows)!=432:
        raise ValueError('All nine sources and 432 rows required')
    lookup={(r['stream'],r['source'],r['step'],r['condition'],r['decoder']):r for r in rows}
    expected={(stream,source,step,condition,decoder) for stream in ('original','fresh')
              for source in sources for step in (100,2500) for condition in CONDITIONS for decoder in DECODERS}
    if set(lookup)!=expected or len(lookup)!=len(rows):
        raise ValueError('Incomplete or duplicate experimental grid')
    cells=[]
    for stream in ('original','fresh'):
        for decoder in DECODERS:
            for source in sources:
                endpoint={step:{condition:lookup[stream,source,step,condition,decoder]['mean_episode_auc']
                                for condition in CONDITIONS} for step in (100,2500)}
                c={step:(a['intact']+a['both']-a['support']-a['query'])/2 for step,a in endpoint.items()}
                early=endpoint[100]
                cells.append(dict(stream=stream,decoder=decoder,source=source,endpoint=endpoint,
                    matching_contrast=c,contrast_change=c[2500]-c[100],
                    early_support_hurts=early['support']<early['intact'],
                    early_query_hurts=early['query']<early['intact'],
                    early_both_exceeds_each_single=early['both']>max(early['support'],early['query']),
                    early_half_advantage_margin=early['both']-.5-.5*(early['intact']-.5)))
    summaries=[]
    for stream in ('original','fresh'):
        for decoder in DECODERS:
            selected=[r for r in cells if r['stream']==stream and r['decoder']==decoder]
            endpoints={step:{condition:float(np.mean([r['endpoint'][step][condition] for r in selected]))
                             for condition in CONDITIONS} for step in (100,2500)}
            contrasts={step:float(np.mean([r['matching_contrast'][step] for r in selected])) for step in (100,2500)}
            e=endpoints[100]
            gates=dict(early_support_hurts=e['support']<e['intact'],early_query_hurts=e['query']<e['intact'],
                       early_both_exceeds_each_single=e['both']>max(e['support'],e['query']),
                       early_half_advantage_recovered=e['both']-.5>=.5*(e['intact']-.5),
                       contrast_weakens=contrasts[2500]<contrasts[100])
            summaries.append(dict(stream=stream,decoder=decoder,endpoint=endpoints,matching_contrast=contrasts,
                contrast_change=contrasts[2500]-contrasts[100],gates=gates,joint_gate=all(gates.values()),
                source_counts={key:sum(r[key] for r in selected) for key in
                    ('early_support_hurts','early_query_hurts','early_both_exceeds_each_single')},
                sources_with_weaker_contrast=sum(r['contrast_change']<0 for r in selected)))
    return dict(primary='full_model; other readouts cannot replace its criterion',source_cells=cells,summaries=summaries)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Do not overwrite a prior summary')
    result=summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in result['summaries']: print(json.dumps(row))


if __name__=='__main__': main()
