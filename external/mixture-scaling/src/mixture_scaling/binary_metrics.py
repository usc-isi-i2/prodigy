"""Binary metrics with explicit scoring, class balance, and validation-only decisions."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SCHEMA_VERSION = 2


def arrays(labels, scores):
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != s.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError('finite scores and nonempty matching binary labels required')
    return y, s


def classification(labels, predictions):
    y = np.asarray(labels, dtype=bool); p = np.asarray(predictions, dtype=bool)
    tp=int((y&p).sum()); tn=int((~y&~p).sum()); fp=int((~y&p).sum()); fn=int((y&~p).sum())
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    denominator=float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)
    return {'accuracy':(tp+tn)/len(y),'balanced_accuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None,
            'precision':precision,'recall':recall,'specificity':specificity,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'mcc':(tp*tn-fp*fn)/np.sqrt(denominator) if denominator else 0.,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def ranking_report(labels, scores):
    y,s=arrays(labels,scores); both=np.unique(y).size==2
    return {'roc_auc':float(roc_auc_score(y,s)) if both else None,
            'average_precision':float(average_precision_score(y,s)) if y.sum() else None}


def binary_report(labels, logits, *, include_ranking=True):
    y,s=arrays(labels,logits); p=expit(s)
    # Stable even for very large, confidently wrong logits.
    losses=np.logaddexp(0.,np.where(y==1,-s,s))
    bins=np.minimum((p*15).astype(int),14); calibration=[];ece=0.
    for i in range(15):
        mask=bins==i;count=int(mask.sum())
        if count:
            confidence=float(p[mask].mean());frequency=float(y[mask].mean());ece+=count/len(y)*abs(confidence-frequency)
            calibration.append({'bin':i,'n':count,'mean_probability':confidence,'positive_fraction':frequency})
    result={'n_pairs':len(y),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),'positive_fraction':float(y.mean()),
            'bce':float(losses.mean()),'positive_bce':float(losses[y==1].mean()) if y.sum() else None,
            'negative_bce':float(losses[y==0].mean()) if (1-y).sum() else None,'brier':float(np.square(p-y).mean()),
            'ece_15_equal_width':float(ece),'calibration_bins':calibration,
            'logit_quantiles':dict(zip(['min','p01','median','p99','max'],map(float,np.quantile(s,[0,.01,.5,.99,1])))),
            'at_probability_0_5':classification(y,s>=0)}
    if include_ranking:result.update(ranking_report(y,s))
    return result


def fit_decisions(labels, logits):
    y,s=arrays(labels,logits)
    if np.unique(y).size!=2:raise ValueError('decision fitting requires both classes in validation')
    fpr,tpr,thresholds=roc_curve(y,s,drop_intermediate=False)
    index=int(np.argmax((tpr+1-fpr)/2));threshold=float(thresholds[index]);constant_negative=not np.isfinite(threshold)
    center=float(s.mean());scale=max(float(s.std()),1e-12);x=(s-center)/scale;prior=float(y.mean())
    def objective(ab):
        z=ab[0]*x+ab[1];residual=expit(z)-y
        return float(np.logaddexp(0.,np.where(y==1,-z,z)).mean()),np.array([(residual*x).mean(),residual.mean()])
    fit=minimize(objective,[0.,np.log(prior/(1-prior))],jac=True,bounds=[(0.,None),(None,None)],method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError(f'calibration failed: {fit.message}')
    return {'threshold':None if constant_negative else threshold,'predict_all_negative':constant_negative,
            'threshold_policy':'maximize validation balanced accuracy; ties prefer highest threshold',
            'calibration':{'slope':float(fit.x[0]/scale),'intercept':float(fit.x[1]-fit.x[0]*center/scale),
                           'kind':'nonnegative affine logit calibration','fit_split':'validation','fit_success':True},
            'validation_prior':prior}


def evaluation_report(labels, logits, validation_mask, cosine_scores=None):
    y,s=arrays(labels,logits);mask=np.asarray(validation_mask,dtype=bool)
    if mask.shape!=y.shape or not mask.any() or mask.all():raise ValueError('nonempty disjoint validation/test subsets required')
    decisions=fit_decisions(y[mask],s[mask]);test=~mask;yt,st=y[test],s[test];cal=decisions['calibration']
    pred=np.zeros(len(yt),dtype=bool) if decisions['predict_all_negative'] else st>=decisions['threshold']
    result={'schema_version':SCHEMA_VERSION,'score_kind':'raw_dot_product','orientation':1,
            'validation':binary_report(y[mask],s[mask]),'test':binary_report(yt,st),'decisions':decisions,
            'test_at_validation_threshold':classification(yt,pred),'calibrated_test':binary_report(yt,cal['slope']*st+cal['intercept']),
            'baselines':{'constant_0_5':binary_report(yt,np.zeros(len(yt))),
                         'validation_prior':binary_report(yt,np.full(len(yt),np.log(decisions['validation_prior']/(1-decisions['validation_prior']))))},
            'mrr':{'status':'not_applicable','reason':'ungrouped pair classification; no per-query candidate ranking protocol'}}
    if cosine_scores is not None:
        _,cs=arrays(y,cosine_scores);result['cosine_test']=ranking_report(yt,cs[test]);result['cosine_test']['score_kind']='cosine'
    return result


def log_values(report, prefix):
    keys=['bce','positive_bce','negative_bce','brier','ece_15_equal_width','roc_auc','average_precision','n_pairs','n_positive','n_negative']
    out={f'{prefix}/{k}':report[k] for k in keys if report.get(k) is not None}
    out.update({f'{prefix}/{k}':v for k,v in report['at_probability_0_5'].items() if v is not None})
    return out


class PredictionWindow:
    """Bounded by the configured logging interval; no per-step sorting for AUC."""
    def __init__(self):self.parts={}
    def add(self,source,labels,logits):
        self.parts.setdefault(source,[]).append((labels.detach().cpu().numpy().copy(),logits.detach().cpu().numpy().copy()))
    def flush(self):
        out={};all_y=[];all_s=[]
        for source,parts in self.parts.items():
            y=np.concatenate([p[0] for p in parts]);s=np.concatenate([p[1] for p in parts]);all_y.append(y);all_s.append(s)
            out.update(log_values(binary_report(y,s,include_ranking=False),f'train/source/{source}'))
        if all_y:out.update(log_values(binary_report(np.concatenate(all_y),np.concatenate(all_s),include_ranking=False),'train'))
        self.parts.clear();return out
