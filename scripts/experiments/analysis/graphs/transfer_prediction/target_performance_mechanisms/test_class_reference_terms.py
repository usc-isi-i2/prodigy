"""Pure saved-tensor examples: no model forwards or real experiment inputs."""
import copy
import unittest

import torch
import torch.nn.functional as F

from .class_reference_terms import COMPONENTS, PREFIX, decompose_class_reference


def capture_fixture():
    # Two data supports, one query, and two labels. Both supports connect to
    # each label with opposite signed relations; other nodes have self edges.
    x = torch.tensor([[1., 2.], [3., -1.], [2., 4.], [.5, -.2], [-.3, .7]])
    edges = torch.tensor([[0, 1, 0, 1, 0, 1, 2, 3, 4],
                          [3, 3, 4, 4, 0, 1, 2, 3, 4]])
    attrs = torch.tensor([[0., 1.], [0., -1.], [0., -1.], [0., 1.]] + [[0., 0.]] * 5)
    attention = torch.tensor([.6, .3, .2, .7, 1., 1., 1., .1, .1]).reshape(-1, 1, 1)
    kqv = torch.cat((x * .2, x * .7, x * 1.3), dim=1)
    weighted = (attention * kqv[edges[0], 4:].reshape(-1, 1, 2)).reshape(-1, 2)
    state = {PREFIX + key: value for key, value in {
        'out_proj.weight': torch.tensor([[1.3, -.2], [.7, .9]]),
        'out_proj.bias': torch.tensor([.13, -.31]),
        'bn.weight': torch.tensor([1.2, .8]),
        'bn.bias': torch.tensor([.7, -.4]),
        'bn.running_mean': torch.tensor([.1, .3]),
        'bn.running_var': torch.tensor([.9, 1.7]),
    }.items()}
    messages = F.linear(weighted, state[PREFIX + 'out_proj.weight'], state[PREFIX + 'out_proj.bias'])
    summed = torch.zeros_like(x).index_add_(0, edges[1], messages) + x
    output = F.batch_norm(summed, state[PREFIX + 'bn.running_mean'], state[PREFIX + 'bn.running_var'],
        state[PREFIX + 'bn.weight'], state[PREFIX + 'bn.bias'], training=False, momentum=0., eps=1e-5)
    audit = {'meta_input': x, 'final_inputs': output[:3], 'final_labels': output[3:],
        'weighted_values': weighted, 'edge_index': edges, 'edge_attr': attrs,
        'support_mask': torch.tensor([True, True, False, False, False]),
        'query_mask': torch.tensor([False, False, True]), 'attention': attention,
        'kqv_used': kqv, 'capture_calls': {'meta': 1, 'kqv': 1, 'attention': 1, 'weighted_values': 1, 'labels': 1}}
    return audit, state, torch.tensor([[0, 1]])


class ClassReferenceTermsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_replay_and_float64_additive_accounting(self):
        audit, state, pairs = capture_fixture()
        result = decompose_class_reference(audit, state, label_pairs=pairs)
        self.assertEqual(tuple(result['raw_components']), COMPONENTS)
        self.assertTrue(result['discrepancies']['native_all_nodes_bit_exact'])
        raw_sum = torch.stack(list(result['raw_components'].values())).sum(0)
        torch.testing.assert_close(raw_sum, audit['final_labels'].double(), rtol=1e-6, atol=1e-6)
        self.assertEqual(result['label_degrees'].tolist(), [3, 3])
        # Confirm actual per-edge output bias, not the more usual once/node bias.
        scale = state[PREFIX + 'bn.weight'].double() / (state[PREFIX + 'bn.running_var'].double() + 1e-5).sqrt()
        expected = (3 * state[PREFIX + 'out_proj.bias'].double() * scale).expand(2, -1)
        torch.testing.assert_close(result['raw_components']['output_bias'], expected)

    def test_common_bn_offset_cancels_raw_but_not_normalized_contrast(self):
        audit, state, pairs = capture_fixture()
        result = decompose_class_reference(audit, state, label_pairs=pairs)
        bn = result['raw_components']['bn_offset']
        torch.testing.assert_close(bn[1] - bn[0], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertNotEqual(float(result['label_norms'][0]), float(result['label_norms'][1]))
        expected = bn[0] * (1 / result['label_norms'][1] - 1 / result['label_norms'][0])
        torch.testing.assert_close(result['contrast_components']['bn_offset'][0], expected)
        self.assertGreater(float(expected.norm()), 0.)

    def test_normalization_is_additive_only_with_original_denominators(self):
        audit, state, pairs = capture_fixture()
        result = decompose_class_reference(audit, state, label_pairs=pairs)
        terms = result['contrast_components']
        torch.testing.assert_close(torch.stack(list(terms.values())).sum(0),
            result['normalized_contrast_observed'], rtol=1e-6, atol=1e-6)
        residual = result['raw_components']['label_residual']
        independently_normalized = F.normalize(residual, dim=1)
        wrong = independently_normalized[1] - independently_normalized[0]
        self.assertGreater(float((wrong - terms['label_residual'][0]).norm()), .01)

    def test_explicit_pair_order_is_respected(self):
        audit, state, pairs = capture_fixture()
        forward = decompose_class_reference(audit, state, label_pairs=pairs)
        reverse = decompose_class_reference(audit, state, label_pairs=pairs.flip(1))
        for key in COMPONENTS:
            torch.testing.assert_close(forward['contrast_components'][key], -reverse['contrast_components'][key])
            torch.testing.assert_close(forward['raw_components'][key], reverse['raw_components'][key])

    def test_interleaved_labels_in_two_episodes_require_explicit_pairing(self):
        audit, state, _ = capture_fixture()
        # Duplicate the pure saved capture but interleave the label rows:
        # episode A uses local labels 0/2, episode B uses 1/3.
        maps = [torch.tensor([0, 1, 2, 6, 8]), torch.tensor([3, 4, 5, 7, 9])]
        x = torch.zeros(10, 2)
        kqv = torch.zeros(10, 6)
        output = torch.zeros(10, 2)
        original_output = torch.cat((audit['final_inputs'], audit['final_labels']))
        for mapping in maps:
            x[mapping] = audit['meta_input']
            kqv[mapping] = audit['kqv_used']
            output[mapping] = original_output
        other = {**audit, 'meta_input': x, 'kqv_used': kqv,
            'final_inputs': output[:6], 'final_labels': output[6:],
            'edge_index': torch.cat([mapping[audit['edge_index']] for mapping in maps], dim=1),
            'edge_attr': audit['edge_attr'].repeat(2, 1),
            'attention': audit['attention'].repeat(2, 1, 1),
            'weighted_values': audit['weighted_values'].repeat(2, 1),
            'query_mask': audit['query_mask'].repeat(2),
            'support_mask': torch.tensor([True, True, False, True, True, False, False, False, False, False])}
        correct = decompose_class_reference(other, state, label_pairs=torch.tensor([[0, 2], [1, 3]]))
        torch.testing.assert_close(correct['normalized_contrast_observed'][0],
                                   correct['normalized_contrast_observed'][1], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            decompose_class_reference(other, state, label_pairs=torch.tensor([[0, 1], [2, 3]]))

    def test_inputs_and_parameters_are_not_modified(self):
        audit, state, pairs = capture_fixture()
        original_audit, original_state, original_pairs = copy.deepcopy(audit), copy.deepcopy(state), pairs.clone()
        decompose_class_reference(audit, state, label_pairs=pairs)
        for key in audit:
            if isinstance(audit[key], torch.Tensor):
                torch.testing.assert_close(audit[key], original_audit[key], rtol=0, atol=0)
            else:
                self.assertEqual(audit[key], original_audit[key])
        for key in state:
            torch.testing.assert_close(state[key], original_state[key], rtol=0, atol=0)
        torch.testing.assert_close(pairs, original_pairs, rtol=0, atol=0)

    def test_missing_or_unsupported_state_and_capture_rejected(self):
        for case in ('missing_capture', 'missing_state', 'second_layer', 'final_projection',
                     'unequal_degrees', 'pair_mismatch', 'bad_endpoint', 'epsilon'):
            audit, state, pairs = capture_fixture()
            options = {}
            if case == 'missing_capture':
                del audit['weighted_values']
            elif case == 'missing_state':
                del state[PREFIX + 'bn.bias']
            elif case == 'second_layer':
                state['layer_list.2.gnn_layers.1.out_proj.bias'] = torch.zeros(2)
            elif case == 'final_projection':
                state['final_label_mlp.weight'] = torch.eye(2)
            elif case == 'unequal_degrees':
                audit['edge_index'][1, 0] = 4
            elif case == 'pair_mismatch':
                pairs[0, 1] = 0
            elif case == 'bad_endpoint':
                audit['final_labels'] = audit['final_labels'] + .1
            else:
                options['bn_eps'] = .01
            with self.subTest(case=case), self.assertRaises((ValueError, AssertionError)):
                decompose_class_reference(audit, state, label_pairs=pairs, **options)


if __name__ == '__main__':
    unittest.main()
