"""Binary class-reference contrast diagnostics; no fitting or label-based choice."""
import numpy as np
from sklearn.metrics import roc_auc_score


def aligned_labels(local_y, mapping, use_global):
    y=np.asarray(local_y,dtype=np.int64)
    mapping=np.asarray(mapping,dtype=np.int64)
    if mapping.shape!=(len(y),2) or not np.isin(y,[0,1]).all():
        raise ValueError('binary local labels and one map per query required')
    if use_global:
        if not (np.sort(mapping,axis=1)==[0,1]).all():raise ValueError('not a global binary map')
        return mapping[np.arange(len(y)),y],np.where(mapping[:,1]==1,1.,-1.)
    return y,np.ones(len(y))


def ranking_metrics(local_margin,local_y,mapping,episode_ids,use_global):
    """Pooled and within-episode AUC; exact within/cross pair accounting.

    Compute in margin space to avoid artificial float32-softmax ties. Retain
    the original probability AUC separately in the caller. Episodes with only
    one observed class are undefined, never assigned AUC .5.
    """
    m=np.asarray(local_margin,dtype=np.float64)
    ids=np.asarray(episode_ids)
    y,sign=aligned_labels(local_y,mapping,use_global)
    if m.shape!=y.shape or ids.shape!=y.shape or not np.isfinite(m).all():raise ValueError('invalid scores')
    score=m*sign
    if len(np.unique(y))!=2:raise ValueError('pooled AUC requires both classes')
    pooled=float(roc_auc_score(y,score));rows=[];wins=0.;pairs=0
    for e in np.unique(ids):
        mask=ids==e;ye=y[mask];s=score[mask]
        if not (np.asarray(mapping)[mask]==np.asarray(mapping)[mask][0]).all():raise ValueError('mapping changes within episode')
        pos,neg=s[ye==1],s[ye==0];n=len(pos)*len(neg)
        auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/n) if n else None
        rows.append({'episode':int(e),'queries':int(mask.sum()),'positive_negative_pairs':n,'auc':auc})
        if n:wins+=auc*n;pairs+=n
    valid=[r['auc'] for r in rows if r['auc'] is not None]
    total=int((y==1).sum()*(y==0).sum());cross=total-pairs
    return {'pooled_margin_auc':pooled,'within_episode_auc':float(np.mean(valid)) if valid else None,
            'within_pair_weighted_auc':wins/pairs if pairs else None,
            'cross_episode_pair_auc':(pooled*total-wins)/cross if cross else None,
            'within_pair_fraction':pairs/total,'episodes':len(rows),'defined_episodes':len(valid),
            'undefined_episodes':len(rows)-len(valid),'queries':len(y)},rows


def contrast_parts(query,class_vectors,tau):
    """Final decoder inputs in local class order: q[N,D], labels[N,2,D]."""
    q=np.asarray(query,dtype=np.float64);l=np.asarray(class_vectors,dtype=np.float64)
    if q.ndim!=2 or l.shape!=(len(q),2,q.shape[1]) or not np.isfinite(tau) or tau<=0:
        raise ValueError('invalid final decoder shapes/temperature')
    qn=np.linalg.norm(q,axis=1);ln=np.linalg.norm(l,axis=2)
    if np.any(qn<=1e-8) or np.any(ln<=1e-8):raise ValueError('cosine epsilon regime requires explicit treatment')
    q=q/qn[:,None];l=l/ln[:,:,None]
    d=l[:,1]-l[:,0];r=np.linalg.norm(d,axis=1)
    if np.any(r<=1e-12):raise ValueError('undefined near-zero contrast direction; do not invent one')
    u=d/r[:,None];projection=np.sum(q*u,axis=1)
    return {'strength':r,'orientation':u,'projection':projection,'margin':tau*projection*r,
            'class_cosine':np.sum(l[:,0]*l[:,1],axis=1)}


def contrast_swaps(intact,removed,tau):
    """Final-score factorial: change orientation, strength, both or neither.

    Unlike support-vector norm swaps, these fix ||normalized_l1-normalized_l0||.
    They are diagnostic margins, not necessarily realizable latent label pairs.
    """
    return {'intact':intact['margin'],'removed':removed['margin'],
            'orientation_only':tau*removed['projection']*intact['strength'],
            'strength_only':tau*intact['projection']*removed['strength'],
            'unit_strength_intact':tau*intact['projection'],
            'unit_strength_removed':tau*removed['projection']}
