"""Frozen support-only boundary repair test on a new native CP episode stream."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from sklearn.metrics import f1_score

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import evaluate_logits
from .contrast_metrics import ranking_metrics
from .message_content import aggregation_audit
from .replay import batch_hash, clone_batch
from .role_context import query_mask
from .run_class_reference_long import native_dataset, recorded_model
from .support_boundary_episode import evaluate_episode
from .verify_member_training import model_digest
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    classification_targets, build_classification_loader, reset_episode_rng,
    iter_episodes, new_fingerprint, update_episode_fingerprint)

TARGET = 'covid_political'
OFFSET = 200009
CONDITIONS = ('native', 'value', 'u1_ridge', 'native_calibrated',
              'value_calibrated', 'u1_ridge_calibrated')


def build_protocol(args):
    source_path = args.long_root / 'protocol.json'
    source = json.loads(source_path.read_text())
    for key in ('checkpoint', 'weights_sha256', 'recorded_training_params'):
        if key not in source:
            raise ValueError('missing long protocol field: ' + key)
    return {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'long_protocol': str(source_path),
        'long_protocol_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'checkpoint': source['checkpoint'], 'weights_sha256': source['weights_sha256'],
        'recorded_training_params': source['recorded_training_params'],
        'source': 'cp_hk', 'training_step': 50000, 'target': TARGET,
        'stream': 'support_boundary_frozen', 'eval_episode_seed_offset': OFFSET,
        'previous_stream_offsets': [0, 100003], 'episodes': args.episodes,
        'smoke_only': args.episodes != 128, 'device': 'cpu', 'threads': 4,
        'data_root': str(args.data_root), 'catalog': args.catalog,
        'conditions': list(CONDITIONS), 'new_training': False,
        'sampling': {'hops': 1, 'fanout': 100, 'node_limit': 2000,
                     'n_way': 2, 'n_shot': 3, 'batch_size': 1, 'workers': 0},
        'calibration': 'Three balanced folds hide one support per class; each support is predicted once with its label hidden. Fixed support-standardized logistic calibration with slope and intercept L2 penalty lambda=1. Unconstrained slope. No query labels used for fitting or selection.',
        'u1_readout': 'L2-normalized U1 embeddings; support ridge lambda=1, no intercept, one-hot targets; identical calibration folds.',
        'metrics': 'Pooled probability AUC, accuracy, global positive-class F1, global macro-F1, NLL; separate margin AUC and within-episode ranking. Global labels use each episode task_label_map. Query occurrences are not independent accounts.',
        'decision': 'Compare value_calibrated against both native_calibrated and u1_ridge_calibrated and all uncalibrated arms. Report every arm and metric; do not tune on this stream.',
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--long-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, choices=(2, 128), default=128)
    parser.add_argument('--data-root', type=Path, default=Path('/dataMeR1/phil/data'))
    parser.add_argument('--catalog', default='docs/graph_catalog.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must not exist; preserve all previous runs')
    protocol = build_protocol(args)
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    if torch.cuda.is_available():
        raise ValueError('hide GPUs with CUDA_VISIBLE_DEVICES=""; this is CPU-only')
    torch.set_num_threads(4)
    params = protocol['recorded_training_params']
    expected = {'layers': 'S,U,M', 'emb_dim': 256, 'input_dim': 768,
                'n_hop': 1, 'n_shots': 3, 'gnn_type': 'sage',
                'ignore_label_embeddings': True, 'has_final_back': False,
                'no_bn_encoder': False, 'no_bn_metagraph': False,
                'dropout': 0, 'text_features_dropout': 0}
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError('nominated model recipe changed')
    state = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=True)['model']
    if model_digest(state) != protocol['weights_sha256']:
        raise ValueError('checkpoint identity changed')
    eval_params = {**params, 'task_name': 'classification', 'ignore_label_embeddings': False,
                   'zero_shot': False, 'n_way': 2, 'n_shots': 3, 'batch_size': 1}
    model = recorded_model(eval_params, state)
    if any(module.training for module in model.modules()):
        raise ValueError('all modules must be in eval mode')
    aggregation = aggregation_audit(model)
    if aggregation[0]['aggregation_module'] != 'SumAggregation':
        raise ValueError('native aggregation changed')
    targets = classification_targets(args.catalog, include_facebook=True)
    dataset, loader, path = native_dataset(TARGET, targets[TARGET], args.data_root)
    protocol['graph_path'] = str(path)
    protocol['catalog_sha256'] = hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.json', protocol)
    private = args.output / 'private_episodes'
    private.mkdir()
    batches = build_classification_loader(
        dataset_name=TARGET, data_root=args.data_root, target=targets[TARGET],
        dataset=dataset, get_dataloader=loader, graph_path=path, workers=0,
        eval_episode_seed_offset=OFFSET, batch_count=args.episodes,
        batch_size=1, n_way=2, n_shot=3)
    reset_episode_rng()
    fingerprint = new_fingerprint()
    counts, ys, maps, ids, hashes, files, fits = [], [], [], [], [], [], []
    pieces = {name: [] for name in CONDITIONS}
    start = time.monotonic()
    with torch.no_grad():
        for index, original in enumerate(batches):
            batch = clone_batch(original)
            before = batch_hash(batch)
            q = query_mask(batch)
            graph = batch[0]
            if graph.task_id_per_sample.unique().numel() != 1:
                raise ValueError('native batch1 violated')
            for episode in iter_episodes(batch, n_way=2, n_shot=3,
                                         n_query=targets[TARGET]['n_query'],
                                         equal_query_counts=not targets[TARGET]['eval_random_query']):
                update_episode_fingerprint(fingerprint, episode)
            result = evaluate_episode(model, batch)
            if set(result['logits']) != set(CONDITIONS) or result['query_labels_used_for_fit'] is not False:
                raise ValueError('incomplete or invalid episode result')
            if batch_hash(batch) != before or model_digest(model.state_dict()) != protocol['weights_sha256']:
                raise ValueError('input, weights, or buffers mutated')
            for name in CONDITIONS:
                logits = result['logits'][name].detach().cpu()
                if logits.shape != (int(q.sum()), 2) or not torch.isfinite(logits).all():
                    raise ValueError('invalid query logits')
                pieces[name].append(logits)
            ys.append(batch[2][q].argmax(1))
            maps.append(graph.task_label_map[graph.task_id_per_sample[q]])
            ids.append(torch.full((int(q.sum()),), index))
            counts.append(int(q.sum()))
            hashes.append(before)
            file = private / f'episode_{index:03d}.pt'
            torch.save({'batch': batch, 'result': result, 'batch_sha256': before}, file)
            files.append({'episode': index, 'path': str(file.relative_to(args.output)),
                          'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
            fits.append({'episode': index, 'calibration': result['calibration']})
            write_json(args.output / 'progress.json', {'episodes': index + 1, 'expected': args.episodes,
                       'elapsed_seconds': time.monotonic() - start})
            print(json.dumps({'episodes': index + 1, 'expected': args.episodes,
                              'elapsed_seconds': time.monotonic() - start}), flush=True)
    if len(counts) != args.episodes:
        raise ValueError('incomplete frozen stream')
    labels = {'local_y': torch.cat(ys), 'mapping': torch.cat(maps),
              'episode_ids': torch.cat(ids), 'use_global': True, 'batch_counts': counts}
    saved, rows = {}, []
    for name in CONDITIONS:
        logits = torch.cat(pieces[name])
        saved[name] = logits
        metrics = evaluate_logits(logits, labels)
        row_ids = torch.arange(len(logits))
        truth = labels['mapping'][row_ids, labels['local_y']].numpy()
        prediction = labels['mapping'][row_ids, logits.argmax(1)].numpy()
        metrics['macro_f1'] = float(f1_score(truth, prediction, average='macro', labels=[0, 1], zero_division=0))
        ranks, _ = ranking_metrics((logits[:, 1].double() - logits[:, 0].double()).numpy(),
            labels['local_y'].numpy(), labels['mapping'].numpy(), labels['episode_ids'].numpy(), True)
        rows.append({'target': TARGET, 'stream': protocol['stream'], 'condition': name, **metrics, **ranks})
    torch.save({'logits': saved, 'labels': labels}, args.output / 'private_predictions.pt')
    write_json(args.output / 'metrics.json', rows)
    write_json(args.output / 'calibration.json', fits)
    write_json(args.output / 'receipt.json', {'episode_fingerprint': fingerprint.hexdigest(),
        'batch_sha256': hashes, 'private_files': files, 'weights_sha256': protocol['weights_sha256'],
        'aggregation': aggregation, 'all_weights_unchanged': True, 'all_inputs_unchanged': True})
    write_json(args.output / 'DONE.json', {'episodes': len(counts), 'conditions': len(rows),
        'smoke_only': args.episodes != 128, 'elapsed_seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
