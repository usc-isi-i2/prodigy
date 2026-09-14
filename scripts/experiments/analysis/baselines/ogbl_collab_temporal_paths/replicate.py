"""Frozen temporal path replication; no 2019 arrays or model fitting."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ogbl_collab_compact_joint'))
from path_diagnosis import match,hits,top,sha,make_graph

def ages(u,v,adj,keys,first,year):
    if u==v:return []
    n=adj.shape[0]
    nb=lambda a:adj.indices[adj.indptr[a]:adj.indptr[a+1]]
    out=[]
    for a in nb(u):
        if a in (u,v):continue
        for b in np.intersect1d(nb(a),nb(v),assume_unique=True):
            if b in (u,v,a):continue
            es=np.array([[u,a],[a,b],[b,v]])
            k=es.min(1)*n+es.max(1); loc=np.searchsorted(keys,k)
            assert np.array_equal(keys[loc],k)
            out.append(int(year-first[loc].max()))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--raw',type=Path,default=Path('/dataMeR1/phil/data/ogb/ogbl_collab/raw'));a=p.parse_args()
    protocol=json.loads((Path(__file__).parent/'data/protocol.json').read_text());a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import wandb
    run=wandb.init(project='ogbl-collab-compact-joint',group='temporal-path-replication',name='h3-prerequisite',mode='offline',dir=str(a.out),config=protocol)
    start=time.monotonic();rt=a.root/'joint_v1';prepared=json.loads((rt/'prepared.json').read_text());stdpath=rt/'standardization.npz';assert sha(stdpath)==prepared['files']['standardization.npz'];std=np.load(stdpath)['std']
    edges=np.loadtxt(a.raw/'edge.csv.gz',delimiter=',',dtype=np.int64);years=np.loadtxt(a.raw/'edge_year.csv.gz',delimiter=',',dtype=np.int64).reshape(-1)
    assert int(years.max())==2017
    results={};private={}; hashchecks={}
    for year in (2017,2018):
        path=rt/'year2017.npz' if year==2017 else rt/'assessment/year2018.npz'
        expected=prepared['files']['year2017.npz'] if year==2017 else json.loads((rt/'assessment/results.json').read_text())['panel_sha256']
        assert sha(path)==expected;panel=np.load(path);scores=[];source=[]
        if year==2017:
            sr=a.root/'fusion_frozen_v1';sp=sr/'scores2017.npz';receipt=json.loads((sr/'selection.json').read_text());assert sha(sp)==receipt['scores2017_sha256'];z=np.load(sp)
            scores=[(z[f'joint_seed{s}_p'],z[f'joint_seed{s}_n']) for s in range(3)];source=[{'scores':str(sp),'sha256':sha(sp),'checkpoint_receipts':receipt['checkpoints']}]
        else:
            for s in range(3):
                sr=a.root/f'candidate_fresh_v1/select{s}';sp=sr/'best_scores.npz';receipt=json.loads((sr/'results.json').read_text());assert sha(sp)==receipt['best_score_sha256'];assert sha(sr/'best.pt')==receipt['best_checkpoint_sha256'];z=np.load(sp);scores.append((z['p'],z['n']));source.append({'scores':str(sp),'sha256':sha(sp),'checkpoint_sha256':receipt['best_checkpoint_sha256'],'selected_step':receipt['best']['step']})
        prefix='b' if year==2017 else 'official_b';base=hits(panel[prefix+'p'],panel[prefix+'n']);jh=np.stack([hits(x,y) for x,y in scores]);jt=np.stack([top(y) for x,y in scores]);cohorts={}
        for name,reducer in [('consistent',np.all),('any_seed',np.any)]:
            candidates=np.flatnonzero(~base & reducer(jh,axis=0));targets=np.flatnonzero(~top(panel[prefix+'n']) & reducer(jt,axis=0));cohorts[name]=match(panel['pfeatures'],panel['nfeatures'],candidates,targets,panel['psymmetric'],panel['nsymmetric'],std)
        mask=years<year;adj,keys,first,last=make_graph(edges[mask],years[mask],235868);assert np.array_equal(np.stack([keys//235868,keys%235868],1),panel['graph_edges']);assert first.max()<year
        cases={}
        for matches in cohorts.values():
            for m in matches:
                cases['n'+str(m['negative_index'])]=panel['neg'][m['negative_index']]
                for i in m['positives']:cases['p'+str(i)]=panel['pos'][i]
        features={};audit=[]
        for k,(u,v) in cases.items():
            values=ages(u,v,adj,keys,first,year);features[k]={'median_path_formation_age':float(np.median(values)) if values else None,'missing_path':int(not values),'paths':len(values)}
        # Independent set/dictionary enumeration with independently grouped first dates.
        dates={}
        for (u,v),y in zip(edges[mask],years[mask]):
            key=tuple(sorted((int(u),int(v))));dates[key]=min(int(y),dates.get(key,int(y)))
        nb=lambda u:set(adj.indices[adj.indptr[u]:adj.indptr[u+1]])
        for k in sorted(k for k in cases if k.startswith('n'))[:10]+sorted(k for k in cases if k.startswith('p'))[:10]:
            u,v=map(int,cases[k]); vv=[]
            if u!=v:
                for aa in nb(u)-{u,v}:
                    for bb in (nb(aa)&nb(v))-{u,v,aa}:
                        vv.append(year-max(dates[tuple(sorted(e))] for e in ((u,aa),(aa,bb),(bb,v))))
            assert len(vv)==features[k]['paths'];assert (float(np.median(vv)) if vv else None)==features[k]['median_path_formation_age'];audit.append(k)
        summ={}
        for name,ms in cohorts.items():
            groups=[];used=[];missing=[]
            for m in ms:
                neg=features['n'+str(m['negative_index'])];poss=[features['p'+str(i)] for i in m['positives']];used+=m['positives']
                if poss:missing.append(float(np.mean([v['missing_path'] for v in poss]))-neg['missing_path'])
                vals=[v['median_path_formation_age'] for v in poss if v['median_path_formation_age'] is not None]
                if neg['median_path_formation_age'] is not None and vals:
                    v=np.array(vals);n=neg['median_path_formation_age'];groups.append({'negative_index':m['negative_index'],'younger_preference':float(np.mean((v<n)+.5*(v==n))),'usable_positives':len(vals)})
            probs=np.array([g['younger_preference'] for g in groups]);trim=int(np.ceil(.2*len(probs)));ret=np.sort(probs)[:len(probs)-trim] if len(probs)>trim else np.array([])
            summ[name]={'target_groups':len(ms),'matched_groups':sum(bool(m['positives']) for m in ms),'usable_groups':len(groups),'positive_matches':len(used),'distinct_positive_matches':len(set(used)),'younger_preference':float(probs.mean()) if len(probs) else None,'delete_best_20percent_preference':float(ret.mean()) if len(ret) else None,'mean_missing_indicator_positive_minus_negative':float(np.mean(missing)) if missing else None,'groups':groups}
        results[str(year)]={'cohorts':summ,'panel_sha256':expected,'sources':source,'max_graph_year':int(years[mask].max()),'independent_audit_cases':audit,'graph_matches_panel':True,'n_cases':len(cases)};private[str(year)]={'matching':cohorts,'features':features}
    passes=[]
    for year,r in results.items():
        c=r['cohorts']['any_seed'];cc=r['cohorts']['consistent'];passed=c['usable_groups']>=30 and c['younger_preference']>.5 and c['delete_best_20percent_preference']>.5 and (cc['usable_groups']<10 or cc['younger_preference']>=.5);passes.append(passed);r['replication_gate_pass']=passed
    output={'complete':True,'gate_pass':all(passes),'years':results,'protocol':protocol,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':sha(Path(__file__)),'raw_edges_sha256':sha(a.raw/'edge.csv.gz'),'raw_years_sha256':sha(a.raw/'edge_year.csv.gz'),'std_sha256':sha(stdpath),'elapsed_seconds':time.monotonic()-start,'wandb_directory':run.dir,'test_read_or_scored':False}
    (a.out/'results.json').write_text(json.dumps(output,indent=2));(a.out/'private_cases.json').write_text(json.dumps(private,indent=2));run.summary.update({'gate_pass':all(passes),'elapsed_seconds':output['elapsed_seconds']});run.finish();print(json.dumps(output,indent=2))
if __name__=='__main__':main()
