"""Frozen two-source stage accounting on every saved canonical NM test input."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import zipfile

os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/experiments/setup/final_core')]
from scripts.experiments.setup.nm_hk_support_extremes.run import MetadataOnly
from scripts.experiments.setup.nm_hk_mechanism.run import build_model, meta_args
from scripts.experiments.setup.nm_support_resampling.run import encode, capture, meta, digest_state
from scripts.experiments.setup.nm_complete_input_audit.run import update_hash
from scripts.experiments.setup.nm_support_geometry.run import digest
from evaluate_fixed_grid import git_commit


def map_features(path):
    """Map only x from an existing trusted, uncompressed torch-save archive."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        with z.open(name) as f:
            ref = MetadataOnly(f).load()['x']
        st = ref['storage']
        assert st['type'] == 'FloatStorage' and len(ref['size']) == 2
        entry = z.getinfo(name[:-len('data.pkl')]+'data/'+st['key'])
        assert entry.compress_type == zipfile.ZIP_STORED
        with path.open('rb') as f:
            f.seek(entry.header_offset+26)
            fn, extra = struct.unpack('<HH', f.read(4))
        start = entry.header_offset+30+fn+extra
        storage = np.memmap(path, dtype='<f4', mode='r', offset=start, shape=(st['count'],))
        x = np.ndarray(shape=ref['size'], dtype='<f4', buffer=storage,
                       offset=ref['offset']*4, strides=tuple(s*4 for s in ref['stride']))
    return x


def matrix_metrics(z, truth, prefix):
    z = z.float()
    assert torch.isfinite(z).all()
    ix = torch.arange(len(z), device=z.device)
    true = z[ix, truth]
    other = z.clone()
    other[ix, truth] = -torch.inf
    logp = z.log_softmax(1)
    return {
        prefix+'_correct': (z.argmax(1) == truth).cpu().numpy(),
        prefix+'_prediction': z.argmax(1).cpu().numpy(),
        prefix+'_rank': (1+(z > true[:, None]).sum(1)).cpu().numpy(),
        prefix+'_margin': (true-other.max(1).values).cpu().numpy(),
        prefix+'_probability': logp[ix, truth].exp().cpu().numpy(),
        prefix+'_nll': (-logp[ix, truth]).cpu().numpy(),
    }


def self_test():
    """Exercise strided storage/offset recovery and competing-class ranking."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'features.pt'
        base = torch.arange(240, dtype=torch.float32).reshape(20, 12)
        expected = base[2:18:2, 1:11:2]
        torch.save({'x': expected, 'irrelevant': torch.ones(3)}, path)
        assert np.array_equal(map_features(path), expected.numpy())
    z = torch.tensor([[.2, .8, .1], [.4, .4, .1], [.7, .6, .2]])
    t = torch.tensor([1, 1, 2])
    r = matrix_metrics(z, t, 'test')
    assert r['test_correct'].tolist() == [True, False, False]
    assert r['test_rank'].tolist() == [1, 1, 3]
    assert np.allclose(r['test_margin'], [.6, 0, -.5])
    print('SELF_TEST_OK', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2'))
    p.add_argument('--audit', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908'))
    p.add_argument('--bios', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908'))
    p.add_argument('--hk-cache', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908'))
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    assert args.out is not None
    source = json.loads((args.inputs/'receipt.json').read_text())
    protocol = json.loads((args.audit/'protocol_summary.json').read_text())
    params = json.loads((args.audit/'effective_config.json').read_text())
    if args.dry_run:
        print(json.dumps(dict(targets=['ukr_rus', 'cp_hk'], models=['ukr', 'hk'],
            episodes_per_target=512, query_occurrences_per_cell=61440,
            reuse_hk_cached_episodes=True, train=False, new_sampling=False,
            feature_access='read-only memory map of original backing x',
            fixed_tolerances=dict(pre=1e-5, logits=1e-4, probability=1e-5),
            device=args.device, revision=git_commit()), indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.time()
    assert source['complete'] and protocol['member_policy'] == 'lowest_sorted'
    features = map_features(Path(source['feature_artifact']))
    assert tuple(features.shape) == tuple(source['feature_shape'])
    cached_receipt = json.loads((args.hk_cache/'receipt.json').read_text())
    cached_path = args.hk_cache/'embeddings_and_logits_private.pt'
    assert digest(cached_path) == cached_receipt['cache_sha256']
    cached = torch.load(cached_path, map_location='cpu', weights_only=False)
    models = {}
    hashes = {}
    checkpoints = {}
    params['device'] = args.device
    for short, key in [('ukr', 'ukr_rus'), ('hk', 'cp_hk')]:
        ckpt = protocol['targets']['cp_hk']['splits']['test']['models'][key]['checkpoint']
        checkpoints[short] = dict(path=ckpt, sha256=digest(Path(ckpt)))
        models[short] = build_model(params, ckpt, args.device)
        hashes[short] = digest_state(models[short])
    assert hashes['hk'] == cached['model_state_sha256']
    report = dict(protocol='canonical_nm_source_stage_v1', revision=git_commit(),
        checkpoints=checkpoints, input_receipt_sha256=digest(args.inputs/'receipt.json'),
        feature_artifact=source['feature_artifact'], feature_shape=list(features.shape),
        cached_hk_sha256=cached_receipt['cache_sha256'], targets={},
        tolerances=dict(pre=1e-5, logits=1e-4, probability=1e-5),
        train=False, new_sampling=False, complete=False)
    query_slots = torch.tensor([j for j in range(210) if j % 7 >= 3], device=args.device)
    truth = query_slots//7
    for target, data_name in [('ukr_rus', 'ukr_rus_twitter'), ('cp_hk', 'cp_hk_twitter')]:
        dest = args.out/target
        dest.mkdir()
        info = source['targets'][target]
        offset = info['source_node_offset']
        refpath = args.bios/data_name/'paired_cluster_queries_private.tsv'
        refs = pd.read_csv(refpath, sep='\t')
        refs = refs[refs.split == 'test'].set_index(['episode', 'sample'])
        assert len(refs) == 61440 and refs.index.is_unique
        tables = []
        input_hash = hashlib.sha256()
        target_report = dict(reference_sha256=digest(refpath), batches=[],
            reencoded_subgraphs={m: 0 for m in models}, reused_episodes=0,
            max_probability_error={m: 0. for m in models},
            prediction_mismatches={m: 0 for m in models},
            correctness_mismatches={m: 0 for m in models}, witnesses=[])
        for bi, item in enumerate(info['files']):
            path = args.inputs/target/item['file']
            assert digest(path) == item['sha256']
            batch = torch.load(path, map_location='cpu', weights_only=False)['batch']
            g = batch[0]
            ids = g.global_node_ids.numpy()
            assert ((ids >= -1) & (ids < len(features))).all()
            g.x = torch.from_numpy(np.array(features[np.maximum(ids, 0)], copy=True))
            g.x[g.global_node_ids < 0] = 0
            update_hash(input_hash, batch)
            packed = dict(model_states=hashes, models={}, input_file_sha256=item['sha256'])
            with torch.inference_mode():
                for model_name, model in models.items():
                    reuse = cached['episodes'] if target == 'cp_hk' and model_name == 'hk' else {}
                    missing = [ep for ep in range(32) if (bi, ep) not in reuse]
                    graphs = [g.get_example(ep*210+j) for ep in missing for j in range(210)]
                    pieces = []
                    for k in range(0, len(graphs), 128):
                        pieces.append(encode(model, Batch.from_data_list(graphs[k:k+128]), args.device).cpu())
                    all_pre = torch.cat(pieces).reshape(len(missing), 210, -1) if pieces else []
                    fresh = {ep: all_pre[i] for i, ep in enumerate(missing)}
                    target_report['reencoded_subgraphs'][model_name] += len(graphs)
                    if model_name == 'hk':
                        target_report['reused_episodes'] += 32-len(missing)
                    witness = capture(model, batch, args.device) if bi == 0 else None
                    episodes = {}
                    for ep in range(32):
                        pre = reuse[bi, ep][0] if (bi, ep) in reuse else fresh[ep]
                        a = meta_args(batch, ep, pre.to(args.device), model)
                        if (bi, ep) in reuse:
                            for old, new in zip(reuse[bi, ep][1:], a[1:]):
                                assert torch.equal(old, new.cpu())
                        native = meta(model, *a)[query_slots]
                        support = a[0].reshape(30, 7, -1)[:, :3]
                        query = F.normalize(a[0][query_slots], dim=1)
                        mean_cosine = query @ F.normalize(support, dim=2).mean(dim=1).T
                        prototype = query @ F.normalize(support.mean(dim=1), dim=1).T
                        if witness is not None:
                            saved, full = witness
                            pre_err = float((a[0]-saved['pre'][ep*210:(ep+1)*210]).abs().max())
                            logit_err = float((native-full[ep*120:(ep+1)*120]).abs().max())
                            assert pre_err < 1e-5 and logit_err < 1e-4, (pre_err, logit_err)
                            target_report['witnesses'].append(dict(model=model_name, episode=ep,
                                max_pre_error=pre_err, max_logit_error=logit_err))
                        slots = query_slots.cpu().numpy()+ep*210
                        index = pd.MultiIndex.from_arrays([[f'{bi}:{ep}']*120, slots])
                        ref = refs.loc[index]
                        assert np.array_equal(g.center_node_idx[slots].numpy()-offset, ref['query'].to_numpy())
                        anchors = g.task_label_map[ep].cpu().numpy()-offset
                        assert np.array_equal(anchors[truth.cpu().numpy()], ref.anchor.to_numpy())
                        rows = dict(episode=[f'{bi}:{ep}']*120, sample=slots,
                            query=ref['query'].to_numpy(), anchor=ref.anchor.to_numpy(),
                            model=[model_name]*120)
                        for prefix, scores in [('native', native), ('encoded', mean_cosine), ('prototype', prototype)]:
                            rows.update(matrix_metrics(scores, truth, prefix))
                        prob_err = float(np.max(np.abs(rows['native_probability']-ref[model_name+'_true_probability'].to_numpy())))
                        wrong_dec = int(np.sum(rows['native_correct'] != ref[model_name+'_correct'].to_numpy()))
                        wrong_pred = int(np.sum(anchors[rows['native_prediction']] != ref[model_name+'_pred'].to_numpy()))
                        target_report['max_probability_error'][model_name] = max(target_report['max_probability_error'][model_name], prob_err)
                        target_report['correctness_mismatches'][model_name] += wrong_dec
                        target_report['prediction_mismatches'][model_name] += wrong_pred
                        assert prob_err < 1e-5 and wrong_dec == 0 and wrong_pred == 0, (target, bi, ep, model_name, prob_err, wrong_dec, wrong_pred)
                        tables.append(pd.DataFrame(rows))
                        episodes[ep] = dict(inputs=tuple(v.cpu() for v in a),
                            native_logits=native.cpu(), mean_cosine=mean_cosine.cpu(),
                            prototype_cosine=prototype.cpu())
                    packed['models'][model_name] = episodes
                    del graphs, pieces, all_pre, fresh, witness
                cachepath = dest/f'batch_{bi:03d}_embeddings_private.pt'
                torch.save(packed, cachepath)
                target_report['batches'].append(dict(file=cachepath.name, sha256=digest(cachepath), bytes=cachepath.stat().st_size))
            print('BATCH_OK', target, bi, 'elapsed', round(time.time()-started, 2), flush=True)
            del batch, g, packed
            gc.collect()
        assert input_hash.hexdigest() == info['expected_hash']
        d = pd.concat(tables, ignore_index=True)
        assert len(d) == 122880 and not d.duplicated(['episode', 'sample', 'model']).any()
        csv = dest/'stage_predictions_private.csv'
        d.to_csv(csv, index=False)
        target_report.update(full_input_sha256=input_hash.hexdigest(), rows=len(d),
            csv_sha256=digest(csv), native_accuracies=d.groupby('model').native_correct.mean().to_dict())
        report['targets'][target] = target_report
        (args.out/'receipt.partial.json').write_text(json.dumps(report, indent=2))
    for key, model in models.items():
        assert digest_state(model) == hashes[key]
    report.update(complete=True, model_states_unchanged=True, seconds=time.time()-started)
    (args.out/'receipt.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
