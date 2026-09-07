import unittest
import torch

from .test_role_topology import cycle_fixture
from .message_scale import forward_scaled, radial_swap, scale_control, summarize_trace
from .role_topology import encode, topology_batch
from .role_context import query_mask
from .replay import clone_batch, batch_hash
from .verify_member_training import model_digest


class MessageScaleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model,self.batch=cycle_fixture()
        self.model.layer_list[0].module_list[0].lin_edge_attr=None

    def test_scale_endpoints_and_frozen_queries(self):
        with torch.no_grad():
            original,_=encode(self.model,self.batch)
            baseline,b=forward_scaled(self.model,self.batch)
            torch.testing.assert_close(b['U1_pre_meta'],original,rtol=0,atol=0)
            removed,_=topology_batch(self.batch,'removed',seed=0,role='support')
            a,expected=encode(self.model,removed)
            zero,z=forward_scaled(self.model,self.batch,alpha=0.)
            torch.testing.assert_close(zero,expected,rtol=0,atol=0)
            torch.testing.assert_close(z['U1_pre_meta'],a,rtol=0,atol=0)
            q=query_mask(self.batch)
            for alpha in (1e-6,.0001,.01,.1,.3):
                _,t=forward_scaled(self.model,self.batch,alpha=alpha)
                for stage in ('U1_pre_meta','M2_post_meta'):
                    torch.testing.assert_close(t[stage][q],b[stage][q],rtol=0,atol=0)

    def test_tiny_scale_continuity_in_synthetic_forward(self):
        with torch.no_grad():
            zero,z=forward_scaled(self.model,self.batch,alpha=0.)
            tiny,t=forward_scaled(self.model,self.batch,alpha=1e-6)
            torch.testing.assert_close(tiny,zero,rtol=0,atol=1e-4)
            torch.testing.assert_close(t['post_relu'],z['post_relu'],rtol=0,atol=1e-5)

    def test_radial_swap_and_zero_direction_policy(self):
        a=torch.tensor([[3.,4.],[0.,0.],[2.,0.]])
        b=torch.tensor([[6.,8.],[1.,0.],[0.,0.]])
        result,audit=radial_swap(a,b)
        torch.testing.assert_close(result,torch.tensor([[6.,8.],[0.,0.],[0.,0.]]))
        self.assertEqual(audit['nonzero_norm_unassignable'],1)

    def test_norm_swaps_have_scoped_stage_and_restore(self):
        before=model_digest(self.model.state_dict());fingerprint=batch_hash(self.batch)
        with torch.no_grad():
            baseline,b=forward_scaled(self.model,self.batch)
            _,z=forward_scaled(self.model,self.batch,alpha=0.)
            q=query_mask(self.batch)
            for stage,key in (('post_relu','post_relu'),('pre_meta','U1_pre_meta')):
                replacement,_=radial_swap(b[key],z[key])
                _,t=forward_scaled(self.model,self.batch,replace_stage=stage,replacement=replacement)
                mask=~q if stage=='pre_meta' else ~q[self.batch[0].batch]
                torch.testing.assert_close(t[key][mask],replacement[mask],rtol=0,atol=0)
                torch.testing.assert_close(t['U1_pre_meta'][q],b['U1_pre_meta'][q],rtol=0,atol=0)
                after,_=forward_scaled(self.model,self.batch)
                torch.testing.assert_close(after,baseline,rtol=0,atol=0)
        self.assertEqual(before,model_digest(self.model.state_dict()))
        self.assertEqual(fingerprint,batch_hash(self.batch))

    def test_bias_and_bn_controls_are_query_scoped(self):
        with torch.no_grad():
            _,b=forward_scaled(self.model,self.batch)
            for opts in ({'bias_only':True},{'bias_only':True,'alpha':.01},{'bypass_bn':True},{'bypass_bn':True,'alpha':0.}):
                _,t=forward_scaled(self.model,self.batch,**opts)
                q=query_mask(self.batch)
                for stage in ('U1_pre_meta','M2_post_meta'):
                    torch.testing.assert_close(t[stage][q],b[stage][q],rtol=0,atol=0)

    def test_query_labels_do_not_change_predictions(self):
        other=clone_batch(self.batch);other[2][query_mask(other)]=other[2][query_mask(other)].flip(1)
        with torch.no_grad():
            a,_=forward_scaled(self.model,self.batch,alpha=.01)
            b,_=forward_scaled(self.model,other,alpha=.01)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_exception_and_invalid_control_leave_no_hooks(self):
        layer=self.model.layer_list[0].module_list[0]
        before=len(layer.lin_x._forward_hooks)
        with self.assertRaises(RuntimeError):
            with scale_control(self.model,self.batch,alpha=.01):raise RuntimeError('test')
        self.assertEqual(len(layer.lin_x._forward_hooks),before)
        for alpha in (float('nan'),-1.,2.):
            with self.assertRaises(ValueError):
                with scale_control(self.model,self.batch,alpha=alpha):pass
        self.assertEqual(len(layer.lin_x._forward_hooks),before)

    def test_geometry_denominators(self):
        with torch.no_grad():
            _,b=forward_scaled(self.model,self.batch);_,z=forward_scaled(self.model,self.batch,alpha=0.)
            rows=summarize_trace(self.batch,z,b,z)
        lookup={(r['stage'],r['cohort']):r for r in rows}
        self.assertEqual(lookup['post_relu','support_real']['count'],36)
        self.assertEqual(lookup['U1_pre_meta','support']['count'],4)
        self.assertEqual(lookup['U1_pre_meta','query']['max_abs_intact'],0.)


if __name__=='__main__':unittest.main()
