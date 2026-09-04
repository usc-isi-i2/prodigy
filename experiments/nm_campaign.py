"""Isolated, source-held-out NM intervention protocol.

Training, stopping, and combination selection never consume held-out-source metrics.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from data.dataloader import NeighborTask


def flags_of(value):
    return {x for x in str(value).split(',') if x}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class CampaignNeighborTask(NeighborTask):
    """All candidate draws stay inside the active source pools, including mixed episodes."""
    def __init__(self, *args, flags='', adaptive_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = flags_of(flags)
        self.adaptive_weights = adaptive_weights
        self.counter = 0
        self.bins = {}
        self.cursors = {}
        if not self.strata:
            raise ValueError('Campaign requires explicit source pools')
        if 'proportional' in self.flags:
            sizes = [len(x) for x in self.strata]
            self.stratum_weights = np.asarray(sizes) / sum(sizes)

    def _groups(self, candidates, key):
        if key not in self.bins:
            rowptr, _, _ = self.neighbor_sampler.whole_adj.csr()
            c = torch.as_tensor(candidates)
            degree = (rowptr[c+1] - rowptr[c]).numpy()
            bucket = np.minimum(np.floor(np.log2(np.maximum(degree, 1))).astype(int), 15)
            self.bins[key] = [(int(b), candidates[bucket == b]) for b in np.unique(bucket)]
        return self.bins[key]

    def _sample_center_members(self, center, num_member, rng):
        if 'uniform_positive' not in self.flags:
            return super()._sample_center_members(center, num_member, rng)
        rowptr, col, _ = self.neighbor_sampler.whole_adj.csr()
        neighbors = torch.unique(col[int(rowptr[center]):int(rowptr[center+1])]).tolist()
        if len(neighbors) < num_member:
            return None
        return rng.sample(neighbors, num_member)

    def _sample_from_stratum(self, num_label, num_member, rng, stratum_idx):
        changed = self.flags & {'degree_balanced', 'degree_hard', 'region_adaptive', 'coverage_cycle'}
        if not changed:
            return super()._sample_from_stratum(num_label, num_member, rng, stratum_idx)
        candidates = self._eligible_candidates(self.strata[stratum_idx], num_member, ('stratum', stratum_idx))
        self._require_candidates(candidates, num_label, 'campaign source')
        groups = self._groups(candidates, stratum_idx)
        pool = candidates
        if 'degree_hard' in self.flags:
            usable = [(b, c) for b, c in groups if len(c) >= num_label]
            # Size-weighted band choice preserves center marginals before rejection;
            # all competing centers now have comparable positive-view degree.
            if not usable:
                raise RuntimeError('No degree band contains enough classes')
            _, pool = rng.choices(usable, weights=[len(c) for _, c in usable], k=1)[0]
        task = {}
        for _ in range(max(30000, num_label*1000)):
            if 'coverage_cycle' in self.flags:
                # A random-start cyclic traversal gives no repeated candidates until
                # the source pool is exhausted. Degree filtering remains unchanged.
                if stratum_idx not in self.cursors:
                    self.cursors[stratum_idx] = rng.randrange(len(candidates))
                idx = self.cursors[stratum_idx] % len(candidates)
                self.cursors[stratum_idx] += 1
                center = int(candidates[idx])
            elif self.flags & {'degree_balanced', 'region_adaptive'}:
                weights = [1.0]*len(groups)
                if 'region_adaptive' in self.flags:
                    raw = [float(self.adaptive_weights[b]) for b, _ in groups]
                    mean = sum(raw)/len(raw)
                    weights = [0.7 + 0.3*min(2.0, max(0.5, v/max(mean, 1e-8))) for v in raw]
                _, band = rng.choices(groups, weights=weights, k=1)[0]
                center = int(rng.choice(band))
            else:
                center = int(rng.choice(pool))
            if center in task:
                continue
            members = self._sample_center_members(center, num_member, rng)
            if members is not None:
                task[center] = members
            if len(task) == num_label:
                return task
        raise RuntimeError('Campaign center sampler exhausted attempts')

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        if 'cross_graph' in self.flags:
            # Balanced source choice per class; unlike the historical global fallback,
            # this can never draw a held-out source or change to size-proportional exposure.
            task = {}
            for _ in range(30000):
                idx = rng.choices(range(len(self.strata)), weights=self.stratum_weights, k=1)[0]
                part = self._sample_from_stratum(1, num_member, rng, idx)
                task.update(part)
                if len(task) == num_label:
                    return task
            raise RuntimeError('Mixed-source sampler exhausted attempts')
        if 'blocked' in self.flags:
            idx = (self.counter // 64) % len(self.strata)
            self.counter += 1
            return self._sample_from_stratum(num_label, num_member, rng, idx)
        return super().sample(num_label, num_member, num_shot, num_query, rng)


def prepare_model_dataset(dataset, params):
    from data.covid19_twitter import resolve_source_subset
    allowed = resolve_source_subset(params['neighbor_sampling_source_subset'], sorted(dataset.source_node_pools), dataset.graph.source_graph_names)
    holdout_id = list(dataset.graph.source_graph_names).index(params['campaign_holdout'])
    if holdout_id in allowed:
        raise ValueError('Held-out source was included in training')
    result = copy.copy(dataset)
    result.campaign_region_weights = torch.ones(16).share_memory_()
    return result


@contextmanager
def fixed_rng(seed):
    python = random.getstate()
    numpy = np.random.get_state()
    with torch.random.fork_rng():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python)
            np.random.set_state(numpy)


def make_loader(dataset, params, source, split, episodes, seed=12345):
    from data.covid19_twitter import get_covid19_twitter_dataloader
    # Fixed two-hop context, independent of the training intervention.
    return get_covid19_twitter_dataloader(
        dataset, split=split, node_split='', batch_size=1, n_way=30, n_shot=3,
        n_query=4, batch_count=episodes, root=params['root'], bert=None,
        num_workers=0, aug='', aug_test=False, split_labels=False,
        train_cap=None, linear_probe=False, task_name='neighbor_matching',
        neighbor_sampling_episode_source='graph_id',
        neighbor_sampling_episode_source_weighting='balanced',
        neighbor_sampling_source_subset=source, neighbor_matching_edge_split=True,
        campaign_protocol=True, campaign_flags='', eval_episode_seed_offset=seed)


def materialize(dataset, params, source, split, episodes):
    seed = 70000 + sum(ord(c) for c in source) + (0 if split == 'val' else 10000)
    with fixed_rng(seed):
        batches = list(make_loader(dataset, params, source, split, episodes))
    digest = hashlib.sha256()
    for b in batches:
        for value in (b[0].global_node_ids, b[0].edge_index, b[2], b[5]):
            digest.update(value.cpu().contiguous().numpy().tobytes())
    return batches, digest.hexdigest()


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    ys, ps = [], []
    losses = []
    start = time.time()
    for batch in batches:
        cloned = [x.clone().to(device) for x in batch]
        yt, yp, _ = model(*cloned)
        if not torch.isfinite(yp).all():
            raise ValueError('Non-finite eval predictions')
        losses.append(float(F.cross_entropy(yp, yt.float())))
        ys.append(yt.cpu())
        ps.append(torch.softmax(yp, dim=1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return dict(roc_auc=float(roc_auc_score(y, p, average='macro')),
                accuracy=float((y.argmax(1) == p.argmax(1)).mean()),
                loss=float(np.mean(losses)), episodes=len(batches), seconds=time.time()-start)


def train(trainer):
    params = trainer.parameter
    flags = flags_of(params['campaign_flags'])
    dataset = trainer.train_dataloader.dataset
    # A one-hop TRAINING arm must still validate with the common two-hop dataset.
    eval_dataset = getattr(dataset, 'campaign_eval_dataset', dataset)
    source_names = list(eval_dataset.graph.source_graph_names)
    from data.covid19_twitter import resolve_source_subset
    source_ids = sorted(resolve_source_subset(params['neighbor_sampling_source_subset'], sorted(eval_dataset.source_node_pools), source_names))
    sources = [source_names[i] for i in source_ids]
    holdout_id = source_names.index(params['campaign_holdout'])
    assert holdout_id not in source_ids
    out = Path(trainer.state_dir)
    interval = int(params['campaign_eval_interval'])
    patience = int(params['early_stopping_patience'])
    min_delta = float(params['campaign_min_delta'])
    caches = {}
    print('Materializing fixed training-source validation panel', flush=True)
    for source in sources:
        caches[source] = materialize(eval_dataset, params, source, 'val', params['campaign_val_per_source'])
    atomic_json(out/'validation_protocol.json', dict(sources=sources, excluded=params['campaign_holdout'],
                fingerprints={s: c[1] for s, c in caches.items()}, episodes_per_source=params['campaign_val_per_source'],
                selection='macro source NM ROC-AUC', interval=interval, patience=patience, min_delta=min_delta))
    best_score, meaningful_best = -math.inf, -math.inf
    best_step = 0
    stale = 0
    history = []
    started = time.time()
    rolling_loss = []
    exposure = {s: 0 for s in source_names}
    params_trainable = [p for group in trainer.optimizer.param_groups for p in group['params']]
    train_iter = iter(trainer.train_dataloader)
    curve = (out/'training_curve.jsonl').open('a', buffering=1)
    stopped = 'cap'
    try:
        for step in range(1, trainer.steps + 1):
            trainer.model.train()
            cpu = next(train_iter)
            graph = cpu[0]
            real_ids = graph.graph_id[graph.global_node_ids >= 0].unique().tolist()
            if holdout_id in real_ids or not set(real_ids).issubset(source_ids):
                raise RuntimeError(f'Source leakage: observed {real_ids}, allowed {source_ids}')
            observed_source = int(graph.source_id_per_task[0])
            if observed_source >= 0:
                exposure[source_names[observed_source]] += 1
            batch = [x.to(trainer.device) for x in cpu]
            trainer.optimizer.zero_grad()
            target_x = None
            mask = None
            if 'aux_reconstruction' in flags:
                target_x = batch[0].x.detach().clone()
                real = batch[0].global_node_ids >= 0
                mask = (torch.rand(len(real), device=real.device) < 0.15) & real
                batch[0].x = batch[0].x.clone()
                batch[0].x[mask] = 0
            yt, yp, graph = trainer.model(*batch)
            loss, acc = trainer.get_loss_and_acc(yt, yp)
            total = loss
            if mask is not None and bool(mask.any()):
                total = total + 0.1*F.mse_loss(trainer.aux_header(graph.x[mask]), target_x[mask])
            if not torch.isfinite(total):
                raise RuntimeError(f'Non-finite loss at {step}')
            total.backward()
            if 'grad_normalized' in flags:
                grads = [p.grad for p in params_trainable if p.grad is not None]
                norm = torch.sqrt(sum(g.detach().square().sum() for g in grads)).clamp_min(1e-8)
                for grad in grads:
                    grad.div_(norm)
            trainer.optimizer.step()
            scalar = float(loss.detach())
            rolling_loss.append(scalar)
            if 'region_adaptive' in flags:
                # Source-independent degree bands of sampled data-point centers;
                # update using their episode's loss, without any held-out feedback.
                centers = graph.center_node_idx.detach().cpu().long()
                rowptr, _, _ = eval_dataset.neighbor_sampler.whole_adj.csr()
                degree = (rowptr[centers+1]-rowptr[centers]).clamp_min(1)
                bands = torch.floor(torch.log2(degree.float())).long().clamp_max(15).unique()
                weights = dataset.campaign_region_weights
                weights[bands] = 0.98*weights[bands] + 0.02*scalar
            if step % 100 == 0 or step == 1:
                row = dict(step=step, episodes=step*params['batch_size'], loss=float(np.mean(rolling_loss)),
                           seconds=time.time()-started, exposure=exposure.copy())
                curve.write(json.dumps(row)+'\n')
                wandb.log({'train_loss':row['loss'], 'episodes':row['episodes']}, step=step)
                print(f"TRAIN step={step} loss={row['loss']:.5f} seconds={row['seconds']:.1f}", flush=True)
                rolling_loss.clear()
            observer = getattr(trainer, 'training_step_observer', None)
            if observer:
                observer(step)
            if step % interval == 0 or step == trainer.steps:
                trainer.save_checkpoint(step)
                rows = {s: {**evaluate(trainer.model, cache, trainer.device), 'fingerprint':fp}
                        for s, (cache, fp) in caches.items()}
                score = float(np.mean([v['roc_auc'] for v in rows.values()]))
                history.append(dict(step=step, macro_roc_auc=score, per_source=rows))
                if score > best_score:
                    best_score, best_step = score, step
                if score > meaningful_best + min_delta:
                    meaningful_best, stale = score, 0
                else:
                    stale += 1
                atomic_json(out/'validation_history.json', history)
                atomic_json(out/'selection.json', dict(status='running', best_step=best_step, best_val=best_score,
                            checkpoint=str(Path(trainer.ckpt_dir)/f'state_dict_{best_step}.ckpt'), sources=sources,
                            flags=sorted(flags), training_steps=step, stale_checks=stale))
                print(f'VALIDATION step={step} macro_auc={score:.6f} stale={stale}/{patience}', flush=True)
                if stale >= patience:
                    stopped = 'validation_plateau'
                    break
        atomic_json(out/'selection.json', dict(status='complete', best_step=best_step, best_val=best_score,
                    checkpoint=str(Path(trainer.ckpt_dir)/f'state_dict_{best_step}.ckpt'), sources=sources,
                    flags=sorted(flags), training_steps=step, stop_reason=stopped,
                    seconds=time.time()-started, exposure=exposure))
        trainer.save_best_state_dict(best_step)
        wandb.run.summary['best_step'] = best_step
        wandb.run.summary['validation_macro_roc_auc'] = best_score
    finally:
        curve.close()
        wandb.finish()
