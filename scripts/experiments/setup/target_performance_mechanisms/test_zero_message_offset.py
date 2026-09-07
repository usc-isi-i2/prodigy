import unittest
from types import SimpleNamespace
import torch
from .run_zero_message_offset import subtract_default


class OffsetTest(unittest.TestCase):
    def fixture(self):
        layer=torch.nn.Module()
        layer.lin_x=torch.nn.Linear(3,2)
        layer.lin_edge_attr=None
        layer.mlp=torch.nn.Linear(2,2)
        layer.bn=torch.nn.Identity()
        bg=torch.nn.Module(); bg.module_list=torch.nn.ModuleList([layer])
        model=torch.nn.Module(); model.layer_list=torch.nn.ModuleList([bg]); model.eval()
        batch=[SimpleNamespace(batch=torch.tensor([0,0,1,1])),None,torch.zeros(2,2),None,None,
               torch.tensor([False,False,True,True])]
        return model,layer,batch

    def test_exact_support_only_subtraction_and_restoration(self):
        model,layer,batch=self.fixture(); x=torch.randn(4,2)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with subtract_default(model,batch) as audit:
            y=layer.bn(x)
        torch.testing.assert_close(y[:2],x[:2]-layer.mlp.bias,rtol=0,atol=0)
        torch.testing.assert_close(y[2:],x[2:],rtol=0,atol=0)
        torch.testing.assert_close(layer.bn(x),x,rtol=0,atol=0)
        self.assertEqual(audit['calls'],1)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_hook_restored_after_exception(self):
        model,layer,batch=self.fixture(); x=torch.randn(4,2)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with subtract_default(model,batch):
                raise RuntimeError('test')
        torch.testing.assert_close(layer.bn(x),x,rtol=0,atol=0)

    def test_train_mode_rejected(self):
        model,layer,batch=self.fixture(); model.train()
        with self.assertRaises(ValueError):
            with subtract_default(model,batch): pass


if __name__=='__main__': unittest.main()
