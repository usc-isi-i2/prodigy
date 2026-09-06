import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.params import get_params
from .run_numerical_controls import make_plans, reference_params, REFERENCE_ID


class NumericalControlsTests(unittest.TestCase):
    def setUp(self):
        self.reference = get_params(['--config', str(Path(__file__).parent / 'member_configs/00_memberctl_ukr_rus_lowest_sorted_s0.yaml'),
                                     '--device', '123', '--seed', '2', '--neighbor_sampling_source_subset', 'cp_hk',
                                     '--neighbor_matching_member_seed', '280102'])
        self.reference.update(readout_training_condition='free', prefix=REFERENCE_ID, device='cpu',
                              loader_start_method='spawn', exp_name='reference', state_dir='/private/reference/state',
                              log_dir='/private/reference/log')

    def test_grid_has_one_seed_and_only_numerical_conditions(self):
        original = copy.deepcopy(self.reference)
        plans = make_plans(self.reference, Path('/private/new-controls'), 0)
        self.assertEqual(self.reference, original)
        self.assertEqual(len(plans), 4)
        self.assertEqual({(p['numerical_mode'], p['numerical_repeat']) for p in plans},
                         {('default', 0), ('default', 1), ('deterministic', 0), ('deterministic', 1)})
        self.assertEqual({p['seed'] for p in plans}, {2})
        self.assertEqual({p['dataset_len_cap'] for p in plans}, {2500})
        self.assertEqual({p['neighbor_sampling_source_subset'] for p in plans}, {'cp_hk'})

    def test_smoke_preserves_initialization_sampling_and_source(self):
        plans = make_plans(self.reference, Path('/private/new-smoke'), 8)
        for p in plans:
            self.assertEqual(p['checkpoint_steps'], '0,8')
            self.assertEqual(p['dataset_len_cap'], 8)
            self.assertEqual(p['neighbor_matching_member_seed'], 280102)
            self.assertEqual(p['workers'], 2)

    def test_reference_is_unique_and_prespecified(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text(json.dumps({'jobs': [self.reference]}))
            self.assertEqual(reference_params(Path(folder))['prefix'], REFERENCE_ID)
            path.write_text(json.dumps({'jobs': [self.reference, self.reference]}))
            with self.assertRaisesRegex(ValueError, 'unique'):
                reference_params(Path(folder))
            self.reference['seed'] = 0
            path.write_text(json.dumps({'jobs': [self.reference]}))
            with self.assertRaisesRegex(ValueError, 'recipe'):
                reference_params(Path(folder))


if __name__ == '__main__':
    unittest.main()
