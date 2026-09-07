import unittest
import torch
from .analyze import metrics, summarize, TARGETS, STREAMS, DECODERS, METRICS


class AnalysisTests(unittest.TestCase):
    def test_global_mapping_and_distinct_f1(self):
        # Locally every prediction is class 0. Swapped episodes must be mapped
        # before pooling semantic-class metrics, not after computing local F1.
        labels = dict(local_y=torch.tensor([0, 0, 0, 1]),
                      mapping=torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]]), use_global=True)
        logits = torch.tensor([[2., 0.]] * 4)
        result = metrics(logits, labels)
        self.assertAlmostEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['f1'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1'], (2 / 3 + .8) / 2)
        local = metrics(logits, dict(labels, use_global=False))
        self.assertEqual(local['f1'], 0.)
        self.assertAlmostEqual(local['macro_f1'], 3 / 7)
        self.assertNotEqual(result['roc_auc'], local['roc_auc'])

    def test_equal_target_means_and_paired_deltas(self):
        cells = []
        for stream in STREAMS:
            for target in TARGETS:
                for schedule in ('blocked', 'interleaved'):
                    for mode in ('native', 'joint', 'isolated', 'ridge_only'):
                        for decoder in DECODERS:
                            value = float(target == 'ukr_rus_suspended') + .1 * (mode == 'isolated') + .05 * (decoder == 'U1_pre_meta/ridge')
                            cells.append(dict(stream=stream, target=target, schedule=schedule, mode=mode, decoder=decoder,
                                              **{k: value for k in METRICS}))
        means, deltas = summarize(cells)
        row = next(r for r in means if r['panel'] == 'all5' and r['mode'] == 'native' and r['decoder'] == 'full_model')
        self.assertAlmostEqual(row['macro_f1'], .2)
        self.assertTrue(all(abs(r['macro_f1'] - (.1 if r['comparison'] != 'isolated_full_minus_own_U1' else -.05)) < 1e-8 for r in deltas))
        with self.assertRaises(ValueError):
            summarize(cells[:-1])


if __name__ == '__main__':
    unittest.main()
