"""Train unchanged joint model on2018; explicitly select on2019 test fusion."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run as joint
from fusion import cutoff, hits, maximum, transform
from official_test import ALPHAS, BASES


def grid(panel, scores):
    rows = []
    for base in BASES:
        prefix = 'official_b' if base == 'validation2018' else 'b'
        b = {s:panel[prefix+s] for s in ('p','n')}
        z = transform({'bp':b['p'], 'bn':b['n']}, scores, cutoff(b['n']), cutoff(scores['n']))
        for alpha in ALPHAS:
            rows.append(dict(base=base, alpha=alpha, hits=hits(**maximum(z, alpha))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--test-runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[0,1,2], required=True)
    p.add_argument('--device', choices=['cuda:0','cuda:1','cuda:2','cuda:3'], required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = dict(name='train2018_v1', seed=args.seed, device=args.device, steps=2000,
        selection_interval=50, training_year=2018, training_graph_max_year=2017,
        assessment_year=2019, assessment_graph_max_year=2018,
        classification='test_tuned_development', test_used_for_selection=True,
        unchanged='architecture,2015 AA feature calibration,2016 standardization,balanced BCE,Adam.001/wd.0001,clip5,sampling2048pos/1024uniform/1024hard,hardpool2048 refreshed10steps',
        changed='training pairs and graph advance from2016 to2018; checkpoint selection now maximizes2019 fusion',
        initialization='fresh model, same optimization seeds0/1/2; not fine-tuning',
        bases=BASES, alphas=ALPHAS, expected_checkpoints=40,
        selection='max test fusion; ties earliest step, then declared base/alpha order',
        normalization='test-negative50 cutoffs; explicitly test-dependent',
        total_inference_scalars=496448, ensemble=False,
        stop='exactly2000 updates each seed; no automatic extension or retry',
        old_comparator='69.5590 best individual,69.4288 common mean; old checkpoint selection differs')
    if args.dry_run:
        print(json.dumps(cfg, indent=2)); return
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    out = args.out/f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    cfg['revision'] = joint.revision()
    joint.save(out/'protocol.json', cfg)
    import wandb
    tracking = wandb.init(project='ogbl-collab-compact-joint',group='train2018_v1',
        name=f'seed{args.seed}',mode='offline',dir=str(out),config=cfg)
    prepared = joint.read(args.runtime/'prepared.json')
    old = joint.read(args.runtime/'assessment/results.json')
    testreceipt = joint.read(args.test_runtime/'results.json')
    files = {str(args.runtime/'node_features.npy'):prepared['files']['node_features.npy'],
             str(args.runtime/'standardization.npz'):prepared['files']['standardization.npz'],
             str(args.runtime/'assessment/year2018.npz'):old['panel_sha256'],
             str(args.test_runtime/'year2019.npz'):testreceipt['panel_sha256']}
    for name, sha in files.items():
        assert joint.shared.sha256_file(Path(name)) == sha
    assert testreceipt['total_inference_scalars'] == cfg['total_inference_scalars'] < 1000000
    joint.save(out/'sources.json',dict(files=files,model_source_revision=prepared['revision'],
        test_panel_producer=joint.read(args.test_runtime/'protocol.json')['revision']))
    tr, te = np.load(args.runtime/'assessment/year2018.npz'), np.load(args.test_runtime/'year2019.npz')
    ms = np.load(args.runtime/'standardization.npz')
    x = torch.as_tensor(np.load(args.runtime/'node_features.npy'), device=args.device)
    train = joint.panel_tensors(tr,ms['mean'],ms['std'],args.device)
    test = joint.panel_tensors(te,ms['mean'],ms['std'],args.device)
    train_adj = joint.adjacency(tr['graph_edges'],len(x),args.device)
    test_adj = joint.adjacency(te['graph_edges'],len(x),args.device)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = joint.Model('joint').to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    from ogb.linkproppred import Evaluator
    evaluator = Evaluator(name='ogbl-collab'); evaluator.K = 50
    history, best, stream = [], None, hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(1,2001):
        if (step-1)%10 == 0:
            with torch.no_grad():
                model.eval()
                z = model.encode(x,train_adj)
                hard = joint.scores(model,z,train['n']).topk(2048).indices.cpu()
                del z
        pi, ni, stream_bytes = joint.sampled_indices(generator,len(train['p'][0]),len(train['n'][0]),hard)
        stream.update(stream_bytes)
        model.train(); optimizer.zero_grad(set_to_none=True)
        z = model.encode(x,train_adj)
        values = []
        for side,index in [('p',pi),('n',ni)]:
            index = index.to(args.device)
            edges, features = train[side]
            values.append(model.score(z,edges[index],features[index]))
        logits = torch.cat(values)
        labels = torch.cat([torch.ones_like(values[0]),torch.zeros_like(values[1])])
        loss = nn.functional.binary_cross_entropy_with_logits(logits,labels)
        assert torch.isfinite(loss)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(grad)
        optimizer.step()
        del z,values,logits
        if step%10 == 0:
            tracking.log({'step':step,'train/loss':float(loss),'train/gradient_norm':float(grad)})
        if step%50 == 0:
            scores = joint.evaluate(model,x,test_adj,test)
            assert hits(**scores) == evaluator.eval({'y_pred_pos':scores['p'],'y_pred_neg':scores['n']})['hits@50']
            rows = grid(te,scores)
            winner = max(rows,key=lambda row:row['hits'])
            checkpoint = out/f'step{step}.pt'
            torch.save(dict(model=model.state_dict(),step=step,seed=args.seed,revision=cfg['revision']),checkpoint)
            scorefile = out/f'scores{step}.npz'
            np.savez_compressed(scorefile,**scores)
            row = dict(step=step,train_loss=float(loss),joint_hits=hits(**scores),grid=rows,
                best=winner,elapsed_seconds=time.monotonic()-started,
                checkpoint_sha256=joint.shared.sha256_file(checkpoint),score_sha256=joint.shared.sha256_file(scorefile))
            history.append(row)
            if best is None or winner['hits'] > best['hits']:
                best = dict(step=step,**winner)
            joint.save(out/'history.json',history)
            tracking.log({'step':step,'test/joint_hits':row['joint_hits'],
                          'test/fusion_hits':winner['hits'],'test/best_fusion_hits':best['hits']})
            print(json.dumps(dict(seed=args.seed,step=step,loss=float(loss),best=best)),flush=True)
    state = torch.load(out/f"step{best['step']}.pt",map_location=args.device,weights_only=False)
    model.load_state_dict(state['model'])
    replay = joint.evaluate(model,x,test_adj,test)
    selected = np.load(out/f"scores{best['step']}.npz")
    for s in ('p','n'):
        np.testing.assert_array_equal(replay[s],selected[s])
    receipt = dict(complete=True,seed=args.seed,revision=cfg['revision'],best=best,steps_run=2000,
        test_scored=True,test_used_for_selection=True,total_inference_scalars=cfg['total_inference_scalars'],
        exact_selected_checkpoint_replay=True,elapsed_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),uniform_stream_sha256=stream.hexdigest(),
        history_sha256=joint.shared.sha256_file(out/'history.json'),wandb_directory=tracking.dir)
    joint.save(out/'results.json',receipt)
    tracking.finish()
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
