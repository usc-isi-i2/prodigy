import unittest
import torch
from .test_replay import fixture
from .replay import batch_hash, trace_stages
from .role_context import query_mask
from .directed_role_reversal import reverse_roles, CONDITIONS


class ReversalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model,self.batch=fixture()
        g=self.batch[0]
        mask=torch.arange(g.edge_index.shape[1])%2==0
        g.edge_index=g.edge_index[:,mask]; g.edge_attr=g.edge_attr[mask]

    def test_involution_scope_and_task_identity(self):
        original=batch_hash(self.batch)
        for condition in CONDITIONS:
            changed,audit=reverse_roles(self.batch,condition)
            restored,_=reverse_roles(changed,condition)
            self.assertEqual(batch_hash(restored),original)
            for key in ('x','global_node_ids','batch','ptr','edge_index_supernode','edge_attr'):
                torch.testing.assert_close(changed[0][key],self.batch[0][key],rtol=0,atol=0)
            for a,b in zip(changed[1:],self.batch[1:]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(audit['degree_exchange_exact'])
        self.assertEqual(batch_hash(self.batch),original)

    def test_support_reversal_preserves_final_queries(self):
        captured={}
        with torch.no_grad():
            for condition in CONDITIONS:
                b,_=reverse_roles(self.batch,condition)
                with trace_stages(self.model,b[0]) as trace:
                    self.model(*b)
                captured[condition]=trace['final_input'][query_mask(b)]
        torch.testing.assert_close(captured['intact'],captured['support'],rtol=0,atol=0)
        torch.testing.assert_close(captured['query'],captured['both'],rtol=0,atol=0)
        self.assertGreater(float((captured['intact']-captured['query']).abs().max()),0)

    def test_label_blind_and_invalid_edge_rejected(self):
        a,_=reverse_roles(self.batch,'support')
        self.batch[2]=self.batch[2].flip(1)
        b,_=reverse_roles(self.batch,'support')
        torch.testing.assert_close(a[0].edge_index,b[0].edge_index,rtol=0,atol=0)
        self.batch[0].edge_index[1,0]=int(self.batch[0].ptr[1])
        with self.assertRaises(ValueError): reverse_roles(self.batch,'both')


if __name__=='__main__': unittest.main()
