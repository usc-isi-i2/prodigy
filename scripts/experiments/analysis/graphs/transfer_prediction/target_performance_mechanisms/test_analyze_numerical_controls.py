import unittest

import pandas as pd

from .analyze_numerical_controls import INITIAL, MODES, repeat_differences, validate_logits, validate_training
from .analyze_trajectories import DECODERS, PANEL


class NumericalControlAnalysisTests(unittest.TestCase):
    def test_full_control_contract_and_failed_prediction_are_distinct(self):
        arms = pd.DataFrame([dict(model_id=f'{m}_{r}', mode=m, repeat=r, seed=2, source='cp_hk',
                                  initial_sha256=INITIAL, final_sha256='det' if m == 'deterministic' else f'def{r}',
                                  inputs_matched=2500) for m in MODES for r in (0, 1)])
        receipt = dict(valid=True, exact_historical_inputs=True, same_initialization=True, smoke=False,
                       models=4, steps_per_model=2500, independent_seeds=1, deterministic_states_bit_exact=True)
        states = [dict(mode=m, step=s, rng_bit_exact=True, train_batch_sampler_bit_exact=True,
                       model_bit_exact=m == 'deterministic' or s == 0,
                       optimizer_bit_exact=m == 'deterministic' or s == 0)
                  for m in MODES for s in (0, 100, 300, 900, 2500)]
        self.assertTrue(validate_training(receipt, arms, states))
        states[-1]['model_bit_exact'] = False
        receipt['deterministic_states_bit_exact'] = False
        arms.loc[arms.model_id == 'deterministic_1', 'final_sha256'] = 'different'
        self.assertFalse(validate_training(receipt, arms, states))
        receipt['smoke'] = True
        with self.assertRaisesRegex(ValueError, 'full-length'):
            validate_training(receipt, arms, states)

    def test_prediction_grid_and_raw_controls(self):
        rows = []
        for stream in ('original', 'fresh'):
            for target in PANEL:
                for mode in MODES:
                    values = {d: .1 if mode == 'default' and d == 'full_model' else 0 for d in DECODERS}
                    rows.append(dict(stream=stream, target=target, mode=mode, comparisons=544,
                                     all_logits_bit_exact=mode == 'deterministic', maximum_absolute_difference_by_decoder=values))
        receipt = dict(complete=True, same_target_inputs=True, models=4, targets=5, streams=2,
                       deterministic_logits_bit_exact=True)
        self.assertTrue(validate_logits(receipt, rows))
        rows[0]['maximum_absolute_difference_by_decoder']['raw_center/ridge'] = .01
        with self.assertRaisesRegex(ValueError, 'raw-input'):
            validate_logits(receipt, rows)

    def test_repeat_metric_differences_require_all_targets(self):
        cells = pd.DataFrame([dict(stream=s, dataset=t, decoder=d, mode=m, repeat=r,
                                   episode_fingerprint=s+t, **{metric: .6 + (.01 * r if m == 'default' and not d.startswith('raw_') else 0)
                                                             for metric in ('roc_auc', 'accuracy', 'f1', 'nll')})
                              for s in ('original', 'fresh') for t in PANEL for d in DECODERS for m in MODES for r in (0, 1)])
        differences = repeat_differences(cells)
        self.assertEqual(len(differences), 340)
        self.assertAlmostEqual(differences.loc[(differences['mode'] == 'default') & (differences.decoder == 'full_model'), 'delta_roc_auc'].mean(), .01)
        with self.assertRaisesRegex(ValueError, 'complete'):
            repeat_differences(cells.iloc[1:])


if __name__ == '__main__':
    unittest.main()
