"""Fixed long-checkpoint label-initialization by support-context diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch

from .class_reference_kv import kv_forward
from .contrast_metrics import ranking_metrics
from .label_interface import label_interface_mode
from .replay import batch_hash
from .run_class_reference_kv import load_cell
from .verify_member_training import model_digest


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--long-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    if torch.cuda.is_available():
        raise ValueError('hide GPUs: CPU-only diagnostic')
    torch.set_num_threads(4)
    args.phase = 'long_test'
    protocol = json.loads((args.long_root / 'protocol.json').read_text())
    if protocol['recorded_training_params']['ignore_label_embeddings'] is not True:
        raise ValueError('training must use label table')
    args.output.mkdir(parents=True)
    write(args.output / 'protocol.json', {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'parent': str(args.long_root), 'parent_sha256': hashlib.sha256(
            (args.long_root / 'protocol.json').read_bytes()).hexdigest(),
        'streams': ['original', 'fresh'], 'episodes_per_stream': 128,
        'factors': ['current/train_label_table', 'intact/suppressed'],
        'decision': 'Compare within-episode suppression AUC changes separately in both streams; no tuning.',
        'scope': 'Label initialization only, not restoration of full training protocol.'})
    rows = []
    try:
        for stream in ('original', 'fresh'):
            model, labels, paths, expected, references, rec = load_cell(args, stream)
            if len(paths) != 128:
                raise ValueError('exactly 128 saved episodes required')
            values = {f'{mode}_{arm}': [] for mode in ('current', 'table')
                      for arm in ('intact', 'suppressed')}
            audits = []
            offset = 0
            for bi, path in enumerate(paths):
                batch = torch.load(path, map_location='cpu', weights_only=False)
                if batch_hash(batch) != expected[bi]:
                    raise ValueError('input digest differs')
                count = labels['batch_counts'][bi]
                for mode, variant in (('current', 'baseline'), ('table', 'train_label_table')):
                    with label_interface_mode(model, variant, training_label_count=3) as audit:
                        intact = kv_forward(model, batch, alpha=1.)
                        removed = kv_forward(model, batch, alpha=0.)
                        q = intact[4]['query_mask']
                        torch.testing.assert_close(intact[4]['final_inputs'][q],
                            removed[4]['final_inputs'][q], rtol=0, atol=0)
                        if mode == 'table':
                            for result in (intact, removed):
                                torch.testing.assert_close(result[4]['meta_input'][len(q):],
                                    model.learned_label_embedding.weight[:2], rtol=0, atol=0)
                        for j, (arm, result) in enumerate((('intact', intact), ('suppressed', removed))):
                            logits = result[0].detach().cpu()
                            if logits.shape != (count, 2) or not torch.isfinite(logits).all():
                                raise ValueError('invalid predictions')
                            if mode == 'current':
                                torch.testing.assert_close(logits, references[j][offset:offset+count], rtol=0, atol=0)
                            values[f'{mode}_{arm}'].append(logits)
                        audits.append({'batch': bi, 'mode': mode, **audit})
                    if model.params['ignore_label_embeddings'] is not False:
                        raise ValueError('flag not restored')
                if batch_hash(batch) != expected[bi] or model_digest(model.state_dict()) != rec['weights_sha256']:
                    raise ValueError('inputs or weights changed')
                offset += count
                write(args.output / 'progress.json', {'stream': stream, 'episodes': bi+1})
            saved = {k: torch.cat(v) for k, v in values.items()}
            torch.save({'logits': saved, 'labels': labels}, args.output / f'{stream}.pt')
            write(args.output / f'{stream}_audit.json', audits)
            y, mapping, ids = (labels[k].numpy() for k in ('local_y', 'mapping', 'episode_ids'))
            for condition, logits in saved.items():
                ranks, _ = ranking_metrics((logits[:, 1].double()-logits[:, 0].double()).numpy(), y, mapping, ids, True)
                pred = logits.argmax(1).numpy()
                global_pred = mapping[range(len(pred)), pred]
                rows.append({'stream': stream, 'condition': condition, **ranks,
                    'accuracy': float((pred == y).mean()),
                    'global_positive_prediction_rate': float((global_pred == 1).mean())})
            write(args.output / 'metrics.json', rows)
        write(args.output / 'DONE.json', {'streams': 2, 'episodes': 256, 'cells': len(rows)})
    except Exception as exc:
        write(args.output / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
