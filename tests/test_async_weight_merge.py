import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixture_scaling import async_weight_merge as m


class WeightMergeTests(unittest.TestCase):
    def test_blending_is_exact_at_endpoints_nonmutating_and_rejects_incompatible_states(self):
        kd = dict(weight=torch.tensor([1.e30, 1.e-30]), decoder_bias=torch.tensor(-1.),
                  counter=torch.tensor(7, dtype=torch.int64))
        ukraine = dict(weight=torch.tensor([-1.e30, -1.e-30]), decoder_bias=torch.tensor(2.),
                       counter=torch.tensor(7, dtype=torch.int64))
        original = copy.deepcopy((kd, ukraine))
        for alpha, endpoint in ((0., kd), (1., ukraine)):
            merged = m.blend_states(kd, ukraine, alpha)
            for key, value in endpoint.items():
                self.assertTrue(torch.equal(merged[key], value))
                self.assertNotEqual(merged[key].data_ptr(), value.data_ptr())
        middle = m.blend_states(kd, ukraine, .3)
        self.assertTrue(torch.equal(middle['weight'], .7 * kd['weight'] + .3 * ukraine['weight']))
        self.assertAlmostEqual(float(middle['decoder_bias']), -.1, places=6)
        self.assertEqual(middle['counter'].dtype, torch.int64)
        self.assertEqual(int(middle['counter']), 7)
        middle['weight'].zero_()
        middle['counter'].zero_()
        for state, saved in zip((kd, ukraine), original):
            for key in state:
                self.assertTrue(torch.equal(state[key], saved[key]))
        for case in ('keys', 'shape', 'dtype', 'integer', 'nonfinite'):
            with self.subTest(case=case):
                changed = copy.deepcopy(ukraine)
                if case == 'keys':
                    del changed['decoder_bias']
                elif case == 'shape':
                    changed['weight'] = changed['weight'].reshape(1, 2)
                elif case == 'dtype':
                    changed['weight'] = changed['weight'].double()
                elif case == 'integer':
                    changed['counter'] += 1
                else:
                    changed['weight'][0] = float('nan')
                with self.assertRaises(ValueError):
                    m.blend_states(kd, changed, .5)
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                m.blend_states(kd, ukraine, alpha)

    def test_selection_requires_both_floors_and_uses_declared_ties(self):
        def row(alpha, ukraine, facebook):
            return dict(alpha=alpha, validation=[dict(auc=ukraine), dict(auc=facebook)])
        floors = dict(zip(m.ac.SOURCES, (.9, .95)))
        invalid = [row(0., .89, 1.), row(1., 1., .94)]
        exact_floor = row(.4, .9, .95)
        rows = invalid + [exact_floor, row(.7, .95, .96), row(.2, .95, .97), row(.1, .95, .97)]
        selected, eligible = m.select_merge(rows, floors)
        self.assertEqual(selected['alpha'], .1)
        self.assertIn(.4, eligible)
        self.assertNotIn(0., eligible)
        self.assertNotIn(1., eligible)
        selected, _ = m.select_merge(rows + [row(.6, .96, .951)], floors)
        self.assertEqual(selected['alpha'], .6)
        self.assertEqual(m.select_merge(invalid, floors), (None, []))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.select_merge([row(.5, float('nan'), .99)], floors)

    def test_no_feasible_merge_skips_cuda_and_downstream_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'selection.json').write_text(json.dumps(dict(status='no_feasible_merge', selected_alpha=None)))
            args = SimpleNamespace(root=str(root), config='unused', device=0)
            with patch.object(torch.cuda, 'set_device') as cuda, \
                 patch.object(m.ac.base, 'evaluate') as downstream, \
                 patch.object(m.ac.base, 'preflight') as inputs, \
                 contextlib.redirect_stdout(io.StringIO()):
                m.evaluate(args)
            cuda.assert_not_called()
            downstream.assert_not_called()
            inputs.assert_not_called()
            self.assertFalse((root / 'results').exists())


if __name__ == '__main__':
    unittest.main()
