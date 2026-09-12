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

from mixture_scaling import async_extension as m


class TinyModel(torch.nn.Module):
    def __init__(self, view=None):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.decoder_bias = torch.nn.Parameter(torch.tensor(-1.))

    def forward(self, x):
        return self.encoder(x)


def graph(source):
    npositive = 2051 if source == m.ac.SOURCES[0] else 1030
    return dict(source=source,
                x=torch.randn(7, 4, generator=torch.Generator().manual_seed(42)),
                positive=torch.tensor([[0, 1, 2], [1, 2, 3]]).repeat(1, (npositive + 2) // 3)[:, :npositive],
                validation=torch.tensor([[0, 1], [1, 2]]),
                sampler=m.ac.base.lp.ExactNonedges(7, torch.tensor([1, 9, 17])),
                receipt={'source': source})


@contextlib.contextmanager
def tracking(*args):
    yield SimpleNamespace(summary={}), lambda *args: None


def make_parent(root):
    """A terminal full state with real Adam moments and partial sampler offsets."""
    run = root / 'node_neighbors/lp/async_kd'
    run.mkdir(parents=True)
    graphs = [graph(source) for source in m.ac.SOURCES]
    m.ac.initialize_sampling(graphs, 0, torch.device('cpu'))
    m.ac.seed_everything(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-5)
    runtime = m.ac.empty_runtime()
    teacher_path = run / 'facebook_teacher.pt'
    teacher = None
    for step in range(4):
        index = step % 2
        pairs, labels, npositive = m.ac.next_batch(graphs[index])
        teacher_logits = None
        if index == 1 and teacher is not None:
            with torch.no_grad():
                teacher_logits = m.ac.base.score(teacher, graphs[index]['x'], pairs)
        optimizer.zero_grad(set_to_none=True)
        loss = m.ac.task_loss(m.ac.base.score(model, graphs[index]['x'], pairs), labels, teacher_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        runtime['logical_step'] += 1
        m.increment(runtime['counts'], index, npositive, teacher_logits is not None)
        if step == 1:
            m.ac.save(teacher_path, m.ac.snapshot(model, optimizer, graphs, runtime, {}))
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
    start_path = run / 'selected.pt'
    runtime['converged'] = [True, True]
    runtime['teacher_paths'] = [str(start_path), str(teacher_path)]
    validation = [m.ac.diagnostic.validation_pairs(g) for g in graphs]
    probes = [m.ac.fixed_probe(g, 813719 + 1009 * i) for i, g in enumerate(graphs)]
    for source, probe in zip(m.ac.SOURCES, probes):
        m.ac.save(run / f'probe_{source}.pt', m.ac.cpu_clone(probe))
    report = m.ac.measure(model, graphs, validation, probes)
    checkpoint = m.ac.snapshot(model, optimizer, graphs, runtime, {})
    m.ac.save(start_path, checkpoint)
    m.ac.save(run / 'endpoint.pt', checkpoint)
    summary = dict(seed=0, sources=list(m.ac.SOURCES), surviving_counts=copy.deepcopy(runtime['counts']), logical_step=4,
                   physical_steps=6, graph_split_receipts={s: g['receipt'] for s, g in zip(m.ac.SOURCES, graphs)},
                   events=[dict(teacher_checkpoint=str(start_path))], endpoint=report)
    (run / 'summary.json').write_text(json.dumps(summary))
    rows = [dict(logical_step=4, checkpoint=str(start_path), **report)]
    # Create one discarded parent patience-tail checkpoint for exact replay checking.
    for index in (0, 1):
        pairs, labels, npositive = m.ac.next_batch(graphs[index])
        with torch.no_grad():
            target = m.ac.base.score(teacher, graphs[index]['x'], pairs) if index == 1 else None
        optimizer.zero_grad(set_to_none=True)
        loss = m.ac.task_loss(m.ac.base.score(model, graphs[index]['x'], pairs), labels, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        runtime['logical_step'] += 1
        m.increment(runtime['counts'], index, npositive, index == 1)
    prior_path = run / 'discarded_tail.pt'
    m.ac.save(prior_path, m.ac.snapshot(model, optimizer, graphs, runtime, {}))
    rows.append(dict(logical_step=6, checkpoint=str(prior_path), **m.ac.measure(model, graphs, validation, probes)))
    (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return checkpoint


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_sampler_equal(self, left, right):
        self.assertEqual(left['offset'], right['offset'])
        for field in ('order', 'generator', 'order_generator'):
            self.assertTrue(torch.equal(left[field], right[field]), field)

    def test_full_state_continuation_reactivates_ukraine_and_matches_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / 'parent'
            output = Path(temporary) / 'extension'
            original = make_parent(parent)
            self.assertEqual(original['runtime']['converged'], [True, True])
            self.assertEqual(original['samplers'][0]['offset'], 2048)
            self.assertTrue(original['optimizer']['state'])
            batches, losses, teachers = {}, {}, []
            original_next = m.ac.next_batch
            original_loss = m.ac.task_loss
            original_teacher = m.ac.frozen_teacher
            for arm in m.ARMS:
                batches[arm], losses[arm] = [], []

                def capture_batch(g):
                    result = original_next(g)
                    batches[arm].append((g['source'], result[0].clone(), result[2]))
                    return result

                def capture_loss(logits, labels, teacher_logits=None):
                    losses[arm].append(teacher_logits is not None)
                    return original_loss(logits, labels, teacher_logits)

                def capture_teacher(path, device):
                    teacher = original_teacher(path, device)
                    teachers.append((teacher, copy.deepcopy(teacher.state_dict())))
                    return teacher

                args = SimpleNamespace(root=str(output), parent_root=str(parent), arm=arm,
                                       config='unused', seed=0, additional_steps=8,
                                       validation_interval=2, log_interval=2, selection_grid=2)
                with patch.object(m.ac.base, 'BiasMLP', TinyModel), \
                     patch.object(m.ac, 'load_config', return_value={}), \
                     patch.object(m.ac.base, 'preflight'), \
                     patch.object(m.ac.base, 'load_graph', side_effect=lambda source, *_: graph(source)), \
                     patch.object(m.ac, 'tracked_run', side_effect=tracking), \
                     patch.object(m.ac, 'next_batch', side_effect=capture_batch), \
                     patch.object(m.ac, 'task_loss', side_effect=capture_loss), \
                     patch.object(m.ac, 'frozen_teacher', side_effect=capture_teacher), \
                     contextlib.redirect_stdout(io.StringIO()):
                    m.train(args, torch.device('cpu'))
                initial = torch.load(output / 'node_neighbors/lp' / arm / 'checkpoints/additional_000000.pt', weights_only=False)
                for parameter, state in original['optimizer']['state'].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, initial['optimizer']['state'][parameter][field]))
                self.assertEqual(initial['runtime']['converged'], [False, arm == 'kd_extended'])
                self.assertIsNone(initial['runtime']['teacher_paths'][0])

            kd = torch.load(output / 'node_neighbors/lp/kd_extended/checkpoints/additional_000008.pt', weights_only=False)
            only_mid = torch.load(output / 'node_neighbors/lp/ukraine_only/checkpoints/additional_000004.pt', weights_only=False)
            only_end = torch.load(output / 'node_neighbors/lp/ukraine_only/checkpoints/additional_000008.pt', weights_only=False)
            self.assertEqual(kd['runtime']['counts'][0], only_mid['runtime']['counts'][0])
            self.assert_sampler_equal(kd['samplers'][0], only_mid['samplers'][0])
            self.assert_sampler_equal(original['samplers'][1], only_end['samplers'][1])
            self.assertEqual(original['runtime']['counts'][1], only_end['runtime']['counts'][1])
            for field in ('supervised_updates', 'supervised_positive_examples', 'supervised_negative_examples'):
                self.assertEqual(kd['runtime']['counts'][1][field], original['runtime']['counts'][1][field])
            ukraine_batches = [batch for source, batch, _ in batches['kd_extended'] if source == m.ac.SOURCES[0]]
            for expected, (_, actual, _) in zip(ukraine_batches, batches['ukraine_only']):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(any(npositive < 1024 for _, _, npositive in batches['kd_extended']))
            self.assertEqual(losses['kd_extended'], [False, True] * 4)
            self.assertEqual(losses['ukraine_only'], [False] * 8)
            for teacher, saved in teachers:
                self.assertFalse(teacher.training)
                for name, value in teacher.state_dict().items():
                    self.assertTrue(torch.equal(value, saved[name]))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            for state in kd['optimizer']['state'].values():
                self.assertEqual(int(state['step']), 12)
            summary = json.loads((output / 'node_neighbors/lp/kd_extended/summary.json').read_text())
            self.assertEqual(summary['teacher_forward_batches'], 4)
            self.assertEqual(summary['teacher_forward_pairs'], 6 * summary['physical_counts'][1]['kd_positive_examples'])
            self.assertEqual(len(summary['replay_checks']), 1)
            self.assertTrue(summary['replay_checks'][0]['matches'])
            with patch.object(m, 'reference_metrics', return_value=({}, [.5, 0.])), \
                 contextlib.redirect_stdout(io.StringIO()):
                m.prepare_eval(args)
            manifest = json.loads((output / 'evaluation_manifest.json').read_text())
            self.assertEqual(len(manifest['models']), 5)
            self.assertTrue(manifest['replay_check']['ukraine_exposure_and_sampler_match'])
            self.assertFalse(manifest['unavailable'])
            selected = [row for row in manifest['models'] if row['run_id'].endswith('_selected')]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row['start_fallback'] and row['additional_step'] == 0 for row in selected))

    def test_selection_grid_threshold_ties_and_absence(self):
        def row(step, updates, ukraine, facebook):
            return dict(additional_step=step, counts=[dict(supervised_updates=updates)],
                        validation=[dict(auc=ukraine), dict(auc=facebook)])
        start = row(0, 20, .8, .9)
        valid = row(4, 22, .85, .9)
        tie_later = row(6, 24, .85, .91)
        rows = [row(2, 21, .99, .95), row(3, 22, .98, .8999),
                row(8, 26, .99, .99), tie_later, valid, start]
        self.assertIs(m.select_preserving(rows, .9, 20, 4, 2), valid)
        self.assertIs(m.select_preserving([start, row(4, 22, .79, .99)], .9, 20, 4, 2), start)
        self.assertIsNone(m.select_preserving(rows, 1., 20, 4, 2))


if __name__ == '__main__':
    unittest.main()
