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
import torch.nn.functional as F

from mixture_scaling import async_extension as m
from tests.test_async_extension import TinyModel, graph, make_parent, tracking


class KDWeightTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_invalid_weights_rejected_before_loading_parent(self):
        for weight in (0., -.1, float('nan'), float('inf')):
            with self.subTest(weight=weight), patch.object(m, 'load_start') as load:
                args = SimpleNamespace(root='unused', arm='kd_extended', kd_weight=weight)
                with self.assertRaisesRegex(ValueError, 'finite and positive'):
                    m.train(args, torch.device('cpu'))
                load.assert_not_called()

    def test_weight_scales_only_soft_target_gradient(self):
        labels = torch.tensor([1., 0., 0.])
        teacher = torch.tensor([.7, -.4, .1], requires_grad=True)
        for weight in (.1, .3, .5, 1.):
            with self.subTest(weight=weight):
                logits = torch.tensor([.2, -.1, .4], requires_grad=True)
                loss = m.weighted_task_loss(logits, labels, teacher, weight)
                expected = weight * F.binary_cross_entropy_with_logits(logits, teacher.detach().sigmoid())
                self.assertTrue(torch.allclose(loss, expected))
                loss.backward()
                expected_gradient = weight * (logits.detach().sigmoid() - teacher.detach().sigmoid()) / 3
                self.assertTrue(torch.allclose(logits.grad, expected_gradient))
                self.assertIsNone(teacher.grad)
                # Hard labels must not re-enter Facebook's objective.
                changed_labels = m.weighted_task_loss(logits.detach(), 1 - labels, teacher, weight)
                self.assertTrue(torch.equal(loss.detach(), changed_labels))

                ukraine_logits = torch.tensor([.2, -.1, .4], requires_grad=True)
                supervised = m.weighted_task_loss(ukraine_logits, labels, None, weight)
                self.assertTrue(torch.equal(supervised, F.binary_cross_entropy_with_logits(ukraine_logits, labels)))
                supervised.backward()
                self.assertTrue(torch.allclose(ukraine_logits.grad, (ukraine_logits.detach().sigmoid() - labels) / 3))

    def test_positive_weight_changes_model_without_changing_sampling_or_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / 'parent'
            output = Path(temporary) / 'weights'
            original = make_parent(parent)
            batches, summaries, checkpoints, teachers = {}, {}, {}, []
            next_batch = m.ac.next_batch
            frozen_teacher = m.ac.frozen_teacher
            for weight, run_id in ((1., 'weight_1'), (.3, 'weight_03')):
                batches[weight] = []

                def capture_batch(g):
                    pairs, labels, npositive = next_batch(g)
                    batches[weight].append((g['source'], pairs.clone(), labels.clone(), npositive))
                    return pairs, labels, npositive

                def capture_teacher(path, device):
                    teacher = frozen_teacher(path, device)
                    teachers.append((teacher, copy.deepcopy(teacher.state_dict())))
                    return teacher

                args = SimpleNamespace(root=str(output), parent_root=str(parent), arm='kd_extended',
                                       run_id=run_id, kd_weight=weight, config='unused', seed=0,
                                       additional_steps=8, validation_interval=2, log_interval=2,
                                       selection_grid=2)
                with patch.object(m.ac.base, 'BiasMLP', TinyModel), \
                     patch.object(m.ac, 'load_config', return_value={}), \
                     patch.object(m.ac.base, 'preflight'), \
                     patch.object(m.ac.base, 'load_graph', side_effect=lambda source, *_: graph(source)), \
                     patch.object(m.ac, 'tracked_run', side_effect=tracking), \
                     patch.object(m.ac, 'next_batch', side_effect=capture_batch), \
                     patch.object(m.ac, 'frozen_teacher', side_effect=capture_teacher), \
                     contextlib.redirect_stdout(io.StringIO()):
                    m.train(args, torch.device('cpu'))
                run = output / 'node_neighbors/lp' / run_id
                summaries[weight] = json.loads((run / 'summary.json').read_text())
                checkpoints[weight] = torch.load(run / 'checkpoints/additional_000008.pt', weights_only=False)
                initial = torch.load(run / 'checkpoints/additional_000000.pt', weights_only=False)
                self.assertEqual(initial['metadata']['run_id'], run_id)
                self.assertEqual(initial['metadata']['protocol']['kd_weight'], weight)
                self.assertEqual(initial['runtime']['converged'], [False, True])
                for name, tensor in original['model'].items():
                    self.assertTrue(torch.equal(tensor, initial['model'][name]))
                for parameter, state in original['optimizer']['state'].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, initial['optimizer']['state'][parameter][field]))

            full, scaled = summaries[1.], summaries[.3]
            for field in ('surviving_counts', 'physical_counts', 'teacher_forward_batches',
                          'teacher_forward_pairs', 'start_sha256', 'teacher_sha256', 'probe_sha256'):
                self.assertEqual(full[field], scaled[field], field)
            self.assertEqual(full['teacher_forward_batches'], 4)
            self.assertEqual(len(full['replay_checks']), 1)
            self.assertTrue(full['replay_checks'][0]['matches'])
            self.assertEqual(scaled['replay_checks'], [])
            self.assertEqual(len(batches[1.]), 8)
            self.assertEqual(len(batches[.3]), 8)
            for left, right in zip(batches[1.], batches[.3]):
                self.assertEqual(left[0], right[0])
                self.assertEqual(left[3], right[3])
                self.assertTrue(torch.equal(left[1], right[1]))
                self.assertTrue(torch.equal(left[2], right[2]))
            self.assertTrue(any(row[3] < 1024 for row in batches[.3]))
            for left, right in zip(checkpoints[1.]['samplers'], checkpoints[.3]['samplers']):
                self.assertEqual(left['offset'], right['offset'])
                for field in ('order', 'generator', 'order_generator'):
                    self.assertTrue(torch.equal(left[field], right[field]))
            self.assertTrue(any(not torch.equal(tensor, checkpoints[.3]['model'][name])
                                for name, tensor in checkpoints[1.]['model'].items()))
            for counts in scaled['physical_counts']:
                self.assertEqual(counts['supervised_updates'] + counts['kd_updates'], 4)
            self.assertEqual(scaled['physical_counts'][0]['kd_updates'], 0)
            self.assertEqual(scaled['physical_counts'][1]['supervised_updates'], 0)
            for teacher, before in teachers:
                self.assertFalse(teacher.training)
                for name, tensor in teacher.state_dict().items():
                    self.assertTrue(torch.equal(tensor, before[name]))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
