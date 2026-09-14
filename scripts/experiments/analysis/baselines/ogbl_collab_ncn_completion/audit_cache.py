"""Independent sampled cache audit against raw graph neighborhoods, without model imports."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=dict(np.load(a.cache));panel=np.load(a.panel);g=panel['graph_edges'];pairs=np.concatenate([panel['pos'],panel['neg']])
np.testing.assert_array_equal(c['pairs'],pairs)
n=int(c['cn'].max());priority=np.random.default_rng(314159).permutation(n)
rng=np.random.default_rng(987654);ids=rng.choice(len(pairs),100,replace=False)
# Include nonempty and capped cases without using labels or scores.
for array in [c['scale'],c['candidate_scale'].max(1)]:
    choices=np.flatnonzero(array>1)
    ids=np.unique(np.concatenate([ids,choices[:20]]))
cache={}
def neighbors(node):
    if node not in cache:cache[node]=set(g[g[:,0]==node,1].tolist()+g[g[:,1]==node,0].tolist())
    return cache[node]
for i in ids:
    u,v=map(int,pairs[i]);nu,nv=neighbors(u),neighbors(v)
    common=nu&nv if u!=v else set();selected=sorted(common,key=lambda k:int(priority[k]))[:32]
    actual=c['cn'][i];assert actual[actual<n].tolist()==selected
    expected=len(common)/len(selected) if selected else 0
    assert np.isclose(c['scale'][i],expected)
    for side,(source,target) in enumerate(((u,v),(v,u))):
        possible=neighbors(source)-neighbors(target)-{u,v} if u!=v else set()
        selected=sorted(possible,key=lambda k:int(priority[k]))[:8]
        start=side*8;actual=c['candidates'][i,start:start+8]
        assert actual[actual<n].tolist()==selected
        for offset,k in enumerate(selected):
            slot=start+offset;assert np.isclose(c['candidate_scale'][i,slot],len(possible)/len(selected))
            child=int(c['child_index'][i,slot]);np.testing.assert_array_equal(c['child_pairs'][child],sorted([target,k]))
            assert k not in neighbors(target) and target!=k
            common=neighbors(target)&neighbors(k);wanted=sorted(common,key=lambda z:int(priority[z]))[:32]
            got=c['child_cn'][child];assert got[got<n].tolist()==wanted
            assert np.isclose(c['child_scale'][child],len(common)/len(wanted) if wanted else 0)
result=dict(complete=True,targets_audited=len(ids),all_target_pairs_exact=True,common_and_completion_membership_valid=True,child_neighborhoods_valid=True,sampling_and_mass_expansion_valid=True,no_target_label_used_for_neighborhood_construction=True)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
