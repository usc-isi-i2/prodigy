import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixture_scaling import async_seed2_control as m
from tests.test_async_extension import make_parent


def fixture(root):
    """Synthetic long-horizon snapshots around a real tiny-model Adam state."""
    parent = make_parent(root / 'parent', seed=2)
    start = copy.deepcopy(parent['runtime']['counts'])

    def counts(ukraine, facebook=0):
        out = copy.deepcopy(start)
        for index, updates, prefix in ((0, ukraine, 'supervised'), (1, facebook, 'kd')):
            out[index][prefix + '_updates'] += updates
            for key, amount in (('positive_examples', 1024 * updates), ('negative_examples', 5120 * updates),
                                (prefix + '_positive_examples', 1024 * updates), (prefix + '_negative_examples', 5120 * updates)):
                out[index][key] += amount
        return out

    def row(arm, step):
        ua = step if arm == 'control' else step // 2
        fb = 0 if arm == 'control' else step // 2
        ck = copy.deepcopy(parent)
        ck['runtime']['counts'] = counts(ua, fb)
        ck['runtime']['logical_step'] += step
        ck['step'] += step
        for state in ck['optimizer']['state'].values():
            state['step'] += step
        # Equal source exposure has equal sampler state; Facebook remains frozen
        # in the control. No 60k-step toy training is required to test this audit.
        for index, added in enumerate((ua, fb)):
            if added:
                ck['samplers'][index]['offset'] = added % 1024
                ck['samplers'][index]['generator'] = torch.Generator().manual_seed(1000 + added).get_state()
                ck['samplers'][index]['order_generator'] = torch.Generator().manual_seed(2000 + added).get_state()
        path = root / arm / f'{step}.pt'
        m.ac.save(path, ck)
        auc, fb_auc = {0: (.6, .96), 28000: (.8, .96), 30000: (.9, .94),
                       56000: (.99, .99), 60000: (1., .99)}.get(step, (.7, .96))
        return dict(additional_step=step, checkpoint=str(path), counts=ck['runtime']['counts'],
                    validation=[dict(auc=auc, bce=.2), dict(auc=fb_auc, bce=.2)])

    rows = [row('control', step) for step in (0, 28000, 30000, 56000, 60000)]
    kd_rows = [row('kd', step) for step in (0, 56000, 60000)]
    physical = [m.ac.empty_counts(), m.ac.empty_counts()]
    physical[0].update(supervised_updates=60000, positive_examples=61440000, negative_examples=307200000,
                       supervised_positive_examples=61440000, supervised_negative_examples=307200000)
    summary = dict(status='complete', seed=2, data_seed=0, probe_seed=0, budget_steps=60000,
                   start_sha256='same_parent', start_counts=start, probe_sha256={}, graph_split_receipts={},
                   teacher_forward_batches=0, teacher_forward_pairs=0, physical_counts=physical,
                   protocol=dict(learning_rate=.0005, weight_decay=1e-5, optimizer='restored AdamW; no reset',
                                 validation_interval=2000, schedule='ukraine_only', kd_weight=0.))
    kd_summary = copy.deepcopy(summary)
    kd_summary['protocol'].update(schedule='alternating_1_to_1', kd_weight=.1)
    return summary, rows, kd_summary, kd_rows


class Seed2ControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_audit_accepts_shared_start_and_exact_source_exposure_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, rows, kd_summary, kd_rows = fixture(Path(temporary))
            result = m.audit_control(summary, rows, kd_summary, kd_rows)
            self.assertTrue(result['start_model_adam_sampler_rng_identical'])
            self.assertTrue(result['zero_added_facebook_training'])
            self.assertEqual(result['ukraine_exposure_and_sampler_match_at'], [56000, 60000])

    def test_audit_rejects_changed_start_facebook_work_and_exposure(self):
        for case in ('model', 'optimizer', 'rng', 'samplers', 'facebook_work', 'facebook_sampler', 'exposure'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                summary, rows, kd_summary, kd_rows = fixture(Path(temporary))
                if case in ('model', 'optimizer', 'rng', 'samplers'):
                    path = rows[0]['checkpoint']
                    ck = torch.load(path, weights_only=False)
                    if case == 'model':
                        key = next(iter(ck['model']))
                        ck['model'][key] += 1
                    elif case == 'optimizer':
                        state = next(iter(ck['optimizer']['state'].values()))
                        state['exp_avg'] += 1
                    elif case == 'rng':
                        ck['rng']['cpu'][0] ^= 1
                    else:
                        ck['samplers'][0]['offset'] += 1
                    m.ac.save(path, ck)
                elif case == 'facebook_work':
                    summary['physical_counts'][1]['supervised_updates'] = 1
                elif case == 'facebook_sampler':
                    ck = torch.load(rows[-1]['checkpoint'], weights_only=False)
                    ck['samplers'][1]['offset'] += 1
                    m.ac.save(rows[-1]['checkpoint'], ck)
                else:
                    rows[1]['counts'][0]['positive_examples'] += 1
                with self.assertRaises(ValueError):
                    m.audit_control(summary, rows, kd_summary, kd_rows)

    def test_prepare_selection_obeys_floor_grid_and_reuses_frozen_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, rows, kd_summary, kd_rows = fixture(root / 'snapshots')
            output, reference = root / 'output', root / 'reference'
            run = output / 'node_neighbors/lp/ukraine_only'
            run.mkdir(parents=True)
            (run / 'summary.json').write_text(json.dumps(summary))
            (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            models = {}
            for alias in m.REUSED:
                row = kd_rows[-1] if alias == 'kd_w010_fixed' else kd_rows[1]
                cp = reference / 'evaluation/node_neighbors/lp' / alias / 'best.pt'
                cp.parent.mkdir(parents=True)
                shutil.copyfile(row['checkpoint'], cp)
                models[alias] = dict(run_id=alias, original_checkpoint=row['checkpoint'],
                                     additional_step=row['additional_step'], validation=row['validation'])
            old_manifest = dict(singleton_source_auc={m.ac.SOURCES[1]: .9531494824886322})
            args = SimpleNamespace(root=str(output), config='unused')
            original_read = m.read

            def source_only_read(path):
                if '/results/' in str(path):
                    raise AssertionError('checkpoint selection must not read downstream reports')
                return original_read(path)

            with patch.object(m, 'reference', return_value=(reference, {}, old_manifest, models, kd_summary, kd_rows)), \
                 patch.object(m, 'read', side_effect=source_only_read), \
                 patch.object(m.ac, 'load_config', return_value={}), \
                 patch.object(m.ac.base, 'preflight', return_value={}), \
                 patch.object(m.ac.base, 'evaluate', side_effect=AssertionError('must not evaluate reused models')), \
                 contextlib.redirect_stdout(io.StringIO()):
                m.prepare_eval(args)
            manifest = m.read(output / 'evaluation_manifest.json')
            selected = next(row for row in manifest['models'] if row['run_id'] == 'ukraine_only_selected')
            self.assertEqual(selected['additional_step'], 28000)
            self.assertFalse(selected['start_fallback'])
            self.assertEqual(len([row for row in manifest['models'] if row['provenance']['kind'] == 'new']), 5)
            reused = [row for row in manifest['models'] if row['provenance']['kind'] == 'reused']
            self.assertEqual({row['run_id'] for row in reused}, set(m.REUSED))
            self.assertTrue(all(row['provenance']['evaluation_root'] == str(reference / 'evaluation') for row in reused))
            self.assertFalse(manifest['unavailable'])


if __name__ == '__main__':
    unittest.main()
