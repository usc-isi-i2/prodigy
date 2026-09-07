import unittest
import numpy as np
import torch
from .test_role_topology import cycle_fixture
from .run_class_reference_contrast import final_forward,summarize_contrasts
from .replay import batch_hash
from .role_context import query_mask
from .verify_member_training import model_digest


class FinalContrastTest(unittest.TestCase):
    def test_actual_decoder_and_no_query_or_state_changes(self):
        torch.set_num_threads(1);model,batch=cycle_fixture();model.layer_list[0].module_list[0].lin_edge_attr=None
        fingerprint=batch_hash(batch);digest=model_digest(model.state_dict())
        with torch.no_grad():
            a,ta,pa,ea=final_forward(model,batch,alpha=1.);b,tb,pb,eb=final_forward(model,batch,alpha=0.)
        self.assertLess(max(ea,eb),1e-4)
        q=query_mask(batch);ids=batch[0].task_id_per_sample[q]
        labels={'local_y':batch[2][q].argmax(1),'mapping':batch[0].task_label_map[ids],'episode_ids':ids,'use_global':True}
        scores,geometry,margins=summarize_contrasts(pa,pb,labels,float(model.logit_scale.exp()))
        self.assertEqual(len(scores),6);self.assertEqual(len(geometry),len(ids.unique()))
        torch.testing.assert_close(ta['final_input'][q],tb['final_input'][q],rtol=0,atol=0)
        self.assertEqual(batch_hash(batch),fingerprint);self.assertEqual(model_digest(model.state_dict()),digest)
        self.assertEqual(len(model.final_label_mlp._forward_hooks),0)


if __name__=='__main__':unittest.main()
