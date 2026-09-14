#!/usr/bin/env python3
"""Temporal, small residual correction to AA-DC. No test scoring."""
import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
STRENGTHS = (0., .25, .5, 1.)
FEATURES = ['log_aa', 'log_l3_aa_zero_only', 'external_ratio', 'lcc_min', 'lcc_max',
            'log_degree_min', 'log_degree_max', 'raw_cosine', 'log_prior_pair_events',
            'last_event_age_capped20', 'log_recent_activity_min', 'log_recent_activity_max', 'aa_zero']
UPSTREAM = 'b499c2046cfe76448545dfe08fad9effb58dd076'
FINGERPRINT = '07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def keys(edges, n):
    return np.minimum(edges[:, 0], edges[:, 1]) * n + np.maximum(edges[:, 0], edges[:, 1])


def negatives(positive, n, year, count=100000):
    rng = np.random.default_rng(year)
    forbidden = np.unique(keys(positive, n))
    retained, seen = [], set()
    while len(retained) < count:
        pairs = np.sort(rng.integers(0, n, size=(count, 2), dtype=np.int64), axis=1)
        k = keys(pairs, n)
        for key in k[~np.isin(k, forbidden)]:
            value = int(key)
            if value not in seen:
                retained.append(value)
                seen.add(value)
            if len(retained) == count:
                break
    k = np.asarray(retained, dtype=np.int64)
    return np.stack([k // n, k % n], axis=1)


def hits(pos, neg):
    assert np.isfinite(pos).all() and np.isfinite(neg).all()
    return float(np.mean(pos > np.partition(neg, -50)[-50]))


def calibrated_scores(aa, raw, edge, lcc, gate, scale):
    score, aval, _, ext, stats = raw
    score = np.zeros_like(score)
    for i in np.flatnonzero(aval > 0):
        u, v = edge[i]
        g = aa.compute_gate_value(float(ext[i]), 'progressive', .5, .5,
                                  (float(lcc[u]) + float(lcc[v])) / 2, gate)
        score[i] = float(float(aval[i]) * g)
    l3 = stats['l3_values']
    for i in np.flatnonzero(np.isfinite(l3)):
        score[i] = float(scale * float(l3[i]))
    return score


def calibrate(aa, rawp, rawn, pos, neg, lcc):
    pi, ni = np.flatnonzero(rawp[1] > 0), np.flatnonzero(rawn[1] > 0)
    assert len(pi) and len(ni)
    pi, ni = pi[np.argsort(rawp[0][pi])[-50:]], ni[np.argsort(rawn[0][ni])[-50:]]
    avg = lambda e: np.array([(lcc[u]+lcc[v])/2 for u,v in e], dtype=np.float32)
    gate = aa.calibrate_gate_thresholds(rawp[3][pi], avg(pos[pi]), rawn[3][ni], avg(neg[ni]),
                                      pos_protection_q=.95, margin=.1, neg_quantiles=(.33,.67))
    bp = calibrated_scores(aa, rawp, pos, lcc, gate, 0.)
    bn = calibrated_scores(aa, rawn, neg, lcc, gate, 0.)
    aa_neg = bn[rawn[1] > 0]
    cutoff = float(np.partition(aa_neg, -50)[-50]) if len(aa_neg) >= 50 else float(aa_neg.max())
    opt = aa.optimize_anchor_q(bp, bn, rawp[4]['l3_values'], rawn[4]['l3_values'], cutoff, K=50)
    return {'gate': gate, 'anchor_scale': opt['best_anchor_scale'], 'anchor_q': opt['best_q']}


def historical_inputs(graph, split, year):
    ty = split['train']['year'].flatten()
    mask = ty < year
    train, years = split['train']['edge'][mask], ty[mask]
    gy = graph['edge_year'].flatten()
    gm = gy < year
    # Filter years BEFORE lookup: no later event weight may reach an earlier fold.
    lookup = {(int(u),int(v)):float(w) for (u,v),w in
              zip(graph['edge_index'][:,gm].T, graph['edge_weight'].flatten()[gm])}
    weight = np.array([lookup.get((int(u),int(v)),lookup.get((int(v),int(u)),1.)) for u,v in train],dtype=np.float32)
    weight *= np.power(.95, year - years.astype(np.float32))
    del lookup
    assert years.max()<year and gy[gm].max()<year
    return train, years, weight, int(gy[gm].max())


def warm_positive_mask(pos, train, n):
    active = np.zeros(n, dtype=bool)
    active[train.flatten()] = True
    return active[pos].all(axis=1)


def rescue_eligible(features, edges):
    return ((features[:, 0] == 0) & (features[:, 8] == 0)
            & (edges[:, 0] != edges[:, 1]))


def residual_score(model, data, strength=1.):
    f, base, selfpair, eligible = data
    correction = strength * 12 * (model(f).flatten()/12).tanh()
    return (base + correction.masked_fill(~eligible, 0.)).masked_fill(selfpair, -1e9)


def make_year(aa, shared, graph, split, year, calibration, warm_only=False):
    started = time.monotonic()
    n = int(graph['num_nodes'])
    ty = split['train']['year'].flatten()
    train, years, weight, lookup_max = historical_inputs(graph,split,year)
    pos = split['valid']['edge'] if year == 2018 else split['train']['edge'][ty == year]
    neg = split['valid']['edge_neg'] if year == 2018 else negatives(pos, n, year)
    original_positive_count = len(pos)
    # Generate negatives from ALL target-year positives first, preserving pilot pairs.
    if warm_only and year < 2018:
        pos = pos[warm_positive_mask(pos, train, n)]
    assert len(pos) and years.max() < year
    assert not np.intersect1d(keys(pos,n),keys(neg,n)).size
    A = aa.build_weighted_adj(n, train, weight)
    inv = aa.precompute_inv_log_deg(A)
    adj = aa.build_adj(n, train)
    lcc = aa.compute_exact_lcc(A)
    common = dict(A=A,A_invlog=aa.precompute_aa_matrix(A,inv),inv_log_deg=inv,adj=adj,
                  use_gate=True,gate_mode='threshold',ext_threshold=.5,ext_penalty=.5,lcc=lcc,
                  use_l3=True,rescue_mode='anchor',anchor_scale=0.,collect_l3=True,show_progress=False)
    rp, rn = aa.score_edges(edges=pos,**common), aa.score_edges(edges=neg,**common)
    own_calibration = calibrate(aa,rp,rn,pos,neg,lcc) if year in (2015,2018) else None
    if calibration is None:
        calibration = own_calibration
    bp = calibrated_scores(aa,rp,pos,lcc,calibration['gate'],calibration['anchor_scale'])
    bn = calibrated_scores(aa,rn,neg,lcc,calibration['gate'],calibration['anchor_scale'])
    historical = keys(train,n)
    order = np.argsort(historical,kind='stable')
    unique, first, counts = np.unique(historical[order],return_index=True,return_counts=True)
    latest = np.maximum.reduceat(years[order],first)
    degree = np.array([len(a) for a in adj])
    recent = np.bincount(train[years == year-1].flatten(),minlength=n)
    x = np.asarray(graph['node_feat'],dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    def features(edge,raw):
        u,v = edge.T
        idx = np.searchsorted(unique,keys(edge,n))
        found = (idx<len(unique)) & (unique[np.minimum(idx,len(unique)-1)]==keys(edge,n))
        count, age = np.zeros(len(edge)), np.full(len(edge),20.)
        count[found] = counts[idx[found]]
        age[found] = np.minimum(20,year-latest[idx[found]])
        l3 = np.nan_to_num(raw[4]['l3_values'],nan=0.)
        f = np.stack([np.log1p(raw[1]),np.log1p(l3),np.nan_to_num(raw[3],nan=0.),
                      np.minimum(lcc[u],lcc[v]),np.maximum(lcc[u],lcc[v]),
                      np.log1p(np.minimum(degree[u],degree[v])),np.log1p(np.maximum(degree[u],degree[v])),
                      (x[u]*x[v]).sum(1),np.log1p(count),age,
                      np.log1p(np.minimum(recent[u],recent[v])),np.log1p(np.maximum(recent[u],recent[v])),
                      raw[1]==0],axis=1).astype(np.float32)
        assert f.shape==(len(edge),len(FEATURES)) and np.isfinite(f).all()
        return f
    output = dict(pos=pos,neg=neg,pfeatures=features(pos,rp),nfeatures=features(neg,rn),bp=bp,bn=bn)
    if year==2018:
        op=calibrated_scores(aa,rp,pos,lcc,own_calibration['gate'],own_calibration['anchor_scale'])
        on=calibrated_scores(aa,rn,neg,lcc,own_calibration['gate'],own_calibration['anchor_scale'])
        output.update(official_bp=op,official_bn=on)
        assert abs(hits(op,on)-.673557) < 5e-7, ('official AA replay mismatch',hits(op,on))
    meta = {'year':year,'graph_max_year':int(years.max()),'graph_lookup_max_year':lookup_max,
            'graph_events':len(train),'positives':len(pos),'original_positive_count':original_positive_count,
            'warm_only_applied':bool(warm_only and year<2018),
            'negative_fingerprint':shared.fingerprint(neg),
            'negatives':len(neg),'negative_same_year_collision_count':0,
            'negative_self_pairs':int((neg[:,0]==neg[:,1]).sum()),
            'pairs_fingerprint':shared.fingerprint(pos,neg),'features_fingerprint':shared.fingerprint(output['pfeatures'],output['nfeatures']),
            'graph_fingerprint':shared.fingerprint(train,years,weight),'elapsed_seconds':time.monotonic()-started,
            'frozen_aadc_hits_at_50':hits(bp,bn)}
    print(json.dumps({'event':'year_complete',**meta}),flush=True)
    return output,meta,calibration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/dataMeR1/phil/data/ogb'))
    parser.add_argument('--upstream',type=Path,default=Path('/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream'))
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--warm-only',action='store_true')
    parser.add_argument('--constrained-rescue',action='store_true')
    args=parser.parse_args()
    if args.constrained_rescue and not args.warm_only:
        parser.error('--constrained-rescue requires --warm-only')
    shared=load_module(Path(__file__).parents[1]/'ogbl_collab_mlp_lp/run.py','shared')
    config={'years':{'calibration':2015,'train':2016,'selection':2017,'assessment':2018},'seeds':SEEDS,
            'strengths':STRENGTHS,'steps':400,'hidden':32,'residual_bound':12.,'features':FEATURES,
            'negative_count':100000,'warm_only':args.warm_only,
            'constrained_rescue':args.constrained_rescue,
            'correction_eligibility':'zero AA and no prior pair event and nonself' if args.constrained_rescue else 'all nonself',
            'training_pool':'eligible positives and eligible negatives' if args.constrained_rescue else 'full warm panel',
            'negative_policy':'unchanged full-year-positive exclusion before positive filtering',
            'test_scored':False,'revision':shared.git_revision(),'threads':args.threads}
    print(json.dumps(config),flush=True)
    if args.dry_run:
        return
    import torch
    from torch import nn
    import wandb
    torch.set_num_threads(args.threads)
    rev=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    assert rev==UPSTREAM
    source=args.upstream/'aa_dc.py'
    assert source.read_bytes()==subprocess.check_output(['git','-C',str(args.upstream),'show','HEAD:aa_dc.py'])
    aa=load_module(source,'aa')
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'protocol.json',config)
    run=wandb.init(project='ogbl-collab-temporal-gate',group='constrained_v1' if args.constrained_rescue else ('warm_v1' if args.warm_only else 'pilot_v1'),name='temporal_gate_seeds012',
                   mode='offline',dir=str(args.out),config=config)
    begin=time.monotonic()
    graph,split,evaluator,version=shared.load_official_dataset(args.dataset_root)
    audit=shared.audit_dataset(graph,split,version)
    assert audit['split_fingerprint']==FINGERPRINT
    del split['test']
    years,metadata={},[]
    cal=None
    for year in (2015,2016,2017):
        years[year],meta,cal=make_year(aa,shared,graph,split,year,cal,args.warm_only)
        metadata.append(meta)
        np.savez_compressed(args.out/f'year{year}.npz',**years[year])
        run.log({'year':year,'feature_seconds':meta['elapsed_seconds'],'aadc_hits_at_50':meta['frozen_aadc_hits_at_50']})
    scale=float(np.partition(years[2015]['bn'],-50)[-50])
    assert scale>0
    fit=np.concatenate([years[2016]['pfeatures'],years[2016]['nfeatures']])
    mean,std=fit.mean(0),np.maximum(fit.std(0),1e-5)
    def tensors(d):
        return {side:(torch.from_numpy((d[side+'features']-mean)/std),
                       torch.from_numpy(np.log(np.maximum(d['b'+side],1e-6)/scale)),
                       torch.from_numpy(d['pos' if side=='p' else 'neg'][:,0]==d['pos' if side=='p' else 'neg'][:,1]),
                       torch.from_numpy(rescue_eligible(d[side+'features'],d['pos' if side=='p' else 'neg'])
                                        if args.constrained_rescue else np.ones(len(d['b'+side]),dtype=bool)))
                for side in ('p','n')}
    train,valid=tensors(years[2016]),tensors(years[2017])
    if args.constrained_rescue:
        train={side:tuple(t[d[3]] for t in d) for side,d in train.items()}
        assert len(train['p'][0])>0 and len(train['n'][0])>=2048
        assert all(bool(d[3].all()) and not bool(d[2].any()) for d in train.values())
    training_pool={'positives':len(train['p'][0]),'negatives':len(train['n'][0]),
                   'constrained_rescue':args.constrained_rescue}
    save(args.out/'training_pool.json',training_pool)
    def model_new():
        net=nn.Sequential(nn.Linear(len(FEATURES),32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        return net
    def score(model,data,strength=1.):
        return residual_score(model,data,strength)
    models,selections=[],[]
    for seed in SEEDS:
        started=time.monotonic(); torch.manual_seed(seed)
        model=model_new(); optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        best=hits(years[2017]['bp'],years[2017]['bn']); best_step=0; best_strength=0.
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        history=[]
        for step in range(1,401):
            if (step-1)%10==0:
                with torch.no_grad(): hard=torch.topk(score(model,train['n']),2048).indices
            pidx=torch.randint(len(train['p'][0]),(2048,))
            nidx=torch.cat([torch.randint(len(train['n'][0]),(1024,)),hard[torch.randint(2048,(1024,))]])
            sub=lambda d,i:tuple(t[i] for t in d)
            logits=torch.cat([score(model,sub(train['p'],pidx)),score(model,sub(train['n'],nidx))])
            labels=torch.cat([torch.ones(2048),torch.zeros(2048)])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,labels)
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            if step%20==0:
                with torch.no_grad():
                    cells=[]
                    for strength in STRENGTHS:
                        hp=score(model,valid['p'],strength).numpy(); hn=score(model,valid['n'],strength).numpy()
                        value=hits(hp,hn)
                        cells.append({'strength':strength,'hits_at_50':value})
                        if value>best:
                            best,best_step,best_strength=value,step,strength
                            state={k:v.detach().clone() for k,v in model.state_dict().items()}
                row={'step':step,'loss':float(loss),'selection':cells,'best_hits_at_50':best}
                history.append(row); run.log({'seed':seed,'step':step,'loss':float(loss),'selection_best':best})
        model.load_state_dict(state)
        path=args.out/f'gate_seed{seed}.pt'
        torch.save({'state_dict':state,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'scale':scale,
                    'strength':best_strength,'seed':seed,'step':best_step,'features':FEATURES,
                    'constrained_rescue':args.constrained_rescue},path)
        selection={'seed':seed,'step':best_step,'strength':best_strength,'selection_2017_hits_at_50':best,
                   'checkpoint':str(path),'sha256':shared.sha256_file(path),'parameter_count':sum(p.numel() for p in model.parameters()),
                   'elapsed_seconds':time.monotonic()-started,'history':history}
        selections.append(selection); models.append(model)
        save(args.out/f'selection_seed{seed}.json',selection)
        print(json.dumps({k:v for k,v in selection.items() if k!='history'}),flush=True)
    save(args.out/'selection_frozen.json',{'calibration':cal,'scale':scale,'selections':selections,'assessment_2018_scored':False})
    # No feature extraction or scoring on 2018 until all model selection is frozen.
    assessment,meta,_=make_year(aa,shared,graph,split,2018,cal)
    metadata.append(meta)
    np.savez_compressed(args.out/'year2018.npz',**assessment)
    final=tensors(assessment)
    base=hits(assessment['bp'],assessment['bn'])
    official=hits(assessment['official_bp'],assessment['official_bn'])
    rows=[]; arrays={}
    for model,selection in zip(models,selections):
        seed=selection['seed']
        with torch.no_grad():
            p=score(model,final['p'],selection['strength']).numpy()
            n=score(model,final['n'],selection['strength']).numpy()
        value=hits(p,n)
        if args.constrained_rescue:
            for side,values in [('p',p),('n',n)]:
                protected=(~final[side][3]) & (~final[side][2])
                assert np.array_equal(values[protected.numpy()],final[side][1][protected].numpy())
        evaluator.K=50
        assert evaluator.eval({'y_pred_pos':p,'y_pred_neg':n})['hits@50']==value
        row={'seed':seed,'selected_step':selection['step'],'selected_strength':selection['strength'],
             'hits_at_50':value,'delta_pp_vs_frozen_aadc':100*(value-base),'delta_pp_vs_official_aadc':100*(value-official)}
        rows.append(row); arrays[f'seed{seed}_positive']=p; arrays[f'seed{seed}_negative']=n
        print(json.dumps({'event':'assessment',**row}),flush=True)
    np.savez_compressed(args.out/'assessment_scores.npz',**arrays)
    result={'complete':True,'test_scored':False,'classification':'temporal_validation_pilot','revision':config['revision'],
            'constrained_rescue':args.constrained_rescue,'training_pool':training_pool,
            'frozen_2015_aadc_2018_hits_at_50':base,'official_2018_calibrated_aadc_hits_at_50':official,
            'seeds':rows,'mean_hits_at_50':float(np.mean([r['hits_at_50'] for r in rows])),
            'sample_std':float(np.std([r['hits_at_50'] for r in rows],ddof=1)),
            'metadata':metadata,'calibration':cal,'elapsed_seconds':time.monotonic()-begin}
    save(args.out/'results.json',result)
    receipt={'complete':True,'expected_seeds':list(SEEDS),'observed_seeds':[r['seed'] for r in rows],
             'revision':config['revision'],'dataset_fingerprint':FINGERPRINT,'upstream_revision':rev,
             'selection_frozen_sha256':shared.sha256_file(args.out/'selection_frozen.json'),
             'assessment_scores_sha256':shared.sha256_file(args.out/'assessment_scores.npz'),
             'all_temporal_graph_boundaries_verified':all(m['graph_max_year']<m['year'] and m['graph_lookup_max_year']<m['year'] for m in metadata),
             'all_negative_same_year_collisions_zero':True,'official_aadc_replay_passed':True,
             'official_metric_parity_passed':True,'test_scored':False,'runtime_root':str(args.out),
             'wandb_directory':run.dir,'checkpoints':[{k:v for k,v in s.items() if k!='history'} for s in selections],
             'qualification':'2018 validation was seen in earlier experiments; static OGB node features are not historically time-stamped.'}
    save(args.out/'validation_receipt.json',receipt)
    run.summary.update({'complete':True,'validation_mean_hits_at_50':result['mean_hits_at_50'],'elapsed_seconds':result['elapsed_seconds']})
    run.finish(); print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
