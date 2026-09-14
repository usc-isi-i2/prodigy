import numpy as np
import pytest
from mixture_scaling.binary_metrics import binary_report, evaluation_report, fit_decisions


def test_constant_prior_and_neutral_baselines_are_not_confused():
    y=np.array([1]+[0]*5)
    neutral=binary_report(y,np.zeros(6));prior=binary_report(y,np.full(6,-np.log(5)))
    assert neutral['roc_auc']==.5
    assert neutral['average_precision']==pytest.approx(1/6)
    assert neutral['bce']==pytest.approx(np.log(2))
    assert prior['bce']==pytest.approx(.4505612088663046)
    assert prior['at_probability_0_5']['accuracy']==pytest.approx(5/6)
    assert prior['at_probability_0_5']['balanced_accuracy']==.5


def test_extreme_wrong_logits_remain_finite_and_class_losses_recombine():
    y=np.array([0,0,1]);s=np.array([10000,-10000,-20000])
    r=binary_report(y,s)
    assert r['bce']==pytest.approx(10000)
    assert r['bce']==pytest.approx((2*r['negative_bce']+r['positive_bce'])/3)
    assert np.isfinite(r['brier'])


def test_calibration_and_threshold_never_fit_test_labels():
    valy=np.tile([0,1,0,1],10);vals=np.tile([2.,4.,3.,2.5],10)
    mask=np.r_[np.ones(40,dtype=bool),np.zeros(4,dtype=bool)]
    scores=np.r_[vals,[-2.,1.,2.,4.]];y=np.r_[valy,[0,0,1,1]]
    first=evaluation_report(y,scores,mask);y[-4:]=1-y[-4:];second=evaluation_report(y,scores,mask)
    assert first['decisions']==second['decisions']
    assert first['test']['bce']!=second['test']['bce']
    assert first['decisions']['calibration']['slope']>=0


def test_raw_auc_is_not_silently_sign_flipped():
    y=np.tile([0,1],20);scores=np.tile([3.,-3.],20);mask=np.arange(40)<20
    r=evaluation_report(y,scores,mask)
    assert r['test']['roc_auc']==0
    assert r['orientation']==1
    assert r['calibrated_test']['bce']==pytest.approx(np.log(2),abs=1e-6)


def test_single_class_metrics_are_explicit_and_bad_inputs_rejected():
    assert binary_report([0,0],[0,1])['roc_auc'] is None
    assert binary_report([0,0],[0,1])['at_probability_0_5']['balanced_accuracy'] is None
    with pytest.raises(ValueError):binary_report([0,1],[0,float('nan')])
    with pytest.raises(ValueError):fit_decisions([0,0],[0,1])
