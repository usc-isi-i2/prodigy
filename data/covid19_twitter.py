import os
from functools import partial
from typing import Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler, sampler_kwargs_from_config
from .augment import get_aug
from .dataloader import (
    ParamSampler,
    BatchSampler,
    Collator,
    ContrastiveTask,
    NeighborTask,
    RegressionTask,
    MultiTaskSplitBatch,
)
from .dataset import SubgraphDataset
from .neighbor_matching_split import (
    configure_edge_split,
    ensure_static_views,
    positive_sampler_for_split,
)
from .midterm import (
    BinaryFutureLinkTask,
    StaticLinkTask,
    _normalize_view_name,
    _load_named_tensor,
    _load_edge_feature_names,
    _select_target_from_feature,
    _apply_feature_subset,
    _apply_edge_feature_subset,
    _build_stratified_node_splits,
    _build_regression_node_splits,
    _mask_labels_to_node_split,
    _apply_label_downsample,
    midterm_task,
    _deterministic_label_embeddings,
    _label_embedding_texts,
)


def _parse_center_radii(value):
    if value is None or str(value).strip() == "":
        return None
    radii = []
    for token in str(value).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"global", "none", "unbounded"}:
            radii.append(None)
        else:
            radius = int(token)
            if radius <= 0:
                raise ValueError(
                    f"neighbor_sampling_center_radii values must be positive or global, got {token!r}."
                )
            radii.append(radius)
    if not radii:
        raise ValueError("neighbor_sampling_center_radii did not contain a radius.")
    return radii


def _parse_center_radius_weights(value, count):
    if count is None or value is None or str(value).strip() == "":
        return None
    weights = [
        float(token.strip())
        for token in str(value).split(",")
        if token.strip()
    ]
    if len(weights) != count:
        raise ValueError(
            "neighbor_sampling_center_radius_weights must have one value per radius: "
            f"expected {count}, got {len(weights)}."
        )
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"center radius weights must be positive, got {weights}.")
    return weights


def _build_covid19_twitter_graph(raw: dict, **kwargs):
    edge_view = _normalize_view_name(kwargs.get("edge_view", kwargs.get("midterm_edge_view", "default")))
    edge_index, resolved_edge_view = _load_named_tensor(
        raw,
        edge_view,
        default_key="edge_index",
        views_key="edge_index_views",
        legacy_prefix="edge_index",
    )
    if edge_index is None:
        raise ValueError(
            f"No edge_index found for view '{resolved_edge_view}'. "
            "The graph file must contain at least 'edge_index'."
        )

    x = raw["x"]
    y = raw.get("y")
    if y is None:
        y = torch.full((x.shape[0],), -1, dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0])
    try:
        edge_attr, _ = _load_named_tensor(
            raw,
            resolved_edge_view,
            default_key="edge_attr",
            views_key="edge_attr_views",
            legacy_prefix="edge_attr",
        )
    except ValueError:
        edge_attr = None
    edge_feature_names = _load_edge_feature_names(raw, resolved_edge_view)
    if edge_attr is not None:
        graph.edge_attr = edge_attr
    graph.edge_attr_feature_names = edge_feature_names
    graph.feature_names = raw.get("feature_names", [])
    graph.label_names = raw.get("label_names", [])
    graph.label_type = raw.get("label_type", "classification")
    graph.user_ids = raw.get("user_ids", [])
    if "graph_id" in raw:
        graph.graph_id = raw["graph_id"]
    graph.source_graph_names = raw.get("source_graph_names", [])
    graph.node_targets = raw.get("node_targets", None)
    graph = _select_target_from_feature(
        graph,
        kwargs.get("target_feature", ""),
        keep_in_x=bool(kwargs.get("target_feature_keep_in_x", False)),
        transform=kwargs.get("target_transform", "none"),
    )
    if graph.label_type != "regression":
        graph.y = _apply_label_downsample(
            graph.y,
            graph.label_names,
            kwargs.get("midterm_label_downsample", ""),
            seed=int(kwargs.get("seed", 0) or 0),
        )
    graph = _apply_feature_subset(graph, kwargs.get("feature_subset", kwargs.get("midterm_feature_subset", "all")))
    graph = _apply_edge_feature_subset(
        graph,
        kwargs.get("edge_feature_subset", kwargs.get("midterm_edge_feature_subset", "all")),
        feature_names=edge_feature_names,
    )
    return graph, resolved_edge_view


def get_covid19_twitter_dataset(root: str, n_hop: int = 1, graph_filename: str = "retweet_graph.pt", **kwargs) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading covid19_twitter graph from {graph_path}...")
    raw = torch.load(graph_path, map_location="cpu")
    ensure_static_views(raw, kwargs)

    task_name = kwargs.get("task_name", "")
    if task_name == "temporal_link_prediction" and not kwargs.get("edge_view") and not kwargs.get("midterm_edge_view"):
        kwargs = {**kwargs, "edge_view": "temporal_history"}
        print("temporal_link_prediction: defaulting to 'temporal_history' edge view for subgraph construction.")
    if task_name == "static_link_prediction" and not kwargs.get("edge_view") and not kwargs.get("midterm_edge_view"):
        kwargs = {**kwargs, "edge_view": "static_background"}
        print("static_link_prediction: defaulting to 'static_background' edge view for subgraph construction.")

    graph, resolved_edge_view = _build_covid19_twitter_graph(raw, **kwargs)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, {graph.x.shape[1]} node features")
    print("Building neighbor sampler (CSR preprocessing)...", flush=True)
    sampler_kwargs = sampler_kwargs_from_config(kwargs, n_hop)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop, **sampler_kwargs)
    print("Neighbor sampler ready.", flush=True)
    dataset = SubgraphDataset(graph, neighbor_sampler, bidirectional=False)
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print(f"Edge features: {graph.edge_attr.shape[1]} dims from edge view '{resolved_edge_view}'")

    if task_name in ("temporal_link_prediction", "static_link_prediction"):
        default_target = "static_holdout" if task_name == "static_link_prediction" else "future"
        target_view = _normalize_view_name(
            kwargs.get("target_edge_view", kwargs.get("midterm_target_edge_view", default_target)),
            default=default_target,
        )
        future_edge_index, resolved_target_view = _load_named_tensor(
            raw,
            target_view,
            default_key="future_edge_index",
            views_key="target_edge_index_views",
            legacy_prefix="future_edge_index",
        )
        if future_edge_index is not None:
            print("Building target-edge neighbor sampler...", flush=True)
            future_graph = Data(edge_index=future_edge_index, num_nodes=graph.num_nodes)
            dataset.future_neighbor_sampler = NeighborSampler(
                future_graph, num_hops=n_hop, **sampler_kwargs
            )
            dataset.future_edge_view = resolved_target_view
            print("Target-edge neighbor sampler ready.", flush=True)
        else:
            dataset.future_edge_view = None
    elif task_name == "classification":
        dataset.future_edge_view = None
    else:
        dataset.future_edge_view = None
    configure_edge_split(
        raw, dataset, kwargs, n_hop=n_hop, sampler_kwargs=sampler_kwargs
    )
    return dataset


def resolve_source_subset(subset_spec, stratum_ids, source_names):
    """Resolve a --neighbor_sampling_source_subset spec to a set of graph_ids.

    `subset_spec` is a comma-separated list of source names (as stored in
    `graph.source_graph_names`) and/or integer graph_ids. Returns None when the spec is
    empty (meaning "all sources"), otherwise the selected graph_ids as a set.

    Raises ValueError on anything ambiguous -- an unknown name, an id absent from the
    graph, or an empty selection. A mis-specified subset must fail loudly: silently
    falling back to all sources would produce a full-merge run wearing a single-rung
    label, which is indistinguishable from a real rung in the results table.
    """
    tokens = [tok.strip() for tok in str(subset_spec or "").split(",")]
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return None

    name_to_id = {name: idx for idx, name in enumerate(source_names)}
    available = ", ".join(f"{i}:{source_names[i]}" if i < len(source_names) else str(i)
                          for i in stratum_ids)
    selected = set()
    for tok in tokens:
        if tok in name_to_id:
            graph_id = name_to_id[tok]
        else:
            try:
                graph_id = int(tok)
            except ValueError:
                raise ValueError(
                    f"neighbor_sampling_source_subset: unknown source {tok!r}. "
                    f"Available sources: {available}."
                )
        if graph_id not in stratum_ids:
            raise ValueError(
                f"neighbor_sampling_source_subset: graph_id {graph_id} (from {tok!r}) is not "
                f"present in this graph. Available sources: {available}."
            )
        selected.add(graph_id)
    if not selected:
        raise ValueError("neighbor_sampling_source_subset resolved to an empty set of sources.")
    return selected


def resolve_source_sequence(sequence_spec, stratum_ids, source_names):
    """Resolve an ordered source list, rejecting duplicates and omissions."""
    tokens = [tok.strip() for tok in str(sequence_spec or "").split(",")]
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return None

    name_to_id = {name: idx for idx, name in enumerate(source_names)}
    available = ", ".join(
        f"{i}:{source_names[i]}" if i < len(source_names) else str(i)
        for i in stratum_ids
    )
    sequence = []
    for tok in tokens:
        if tok in name_to_id:
            graph_id = name_to_id[tok]
        else:
            try:
                graph_id = int(tok)
            except ValueError:
                raise ValueError(
                    f"neighbor_sampling_source_sequence: unknown source {tok!r}. "
                    f"Available sources: {available}."
                )
        if graph_id not in stratum_ids:
            raise ValueError(
                f"neighbor_sampling_source_sequence: graph_id {graph_id} (from {tok!r}) "
                f"is not active. Available sources: {available}."
            )
        if graph_id in sequence:
            raise ValueError(
                f"neighbor_sampling_source_sequence contains duplicate source {tok!r}."
            )
        sequence.append(graph_id)
    if set(sequence) != set(stratum_ids):
        missing = [graph_id for graph_id in stratum_ids if graph_id not in sequence]
        raise ValueError(
            "neighbor_sampling_source_sequence must list every active source exactly once; "
            f"missing graph_ids {missing}."
        )
    return sequence


def parse_source_sequence_steps(steps_spec, expected_count, expected_total):
    """Parse and validate contiguous source-block episode counts."""
    try:
        steps = [
            int(tok.strip())
            for tok in str(steps_spec or "").split(",")
            if tok.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "neighbor_sampling_source_sequence_steps must be comma-separated integers."
        ) from exc
    if len(steps) != expected_count:
        raise ValueError(
            "neighbor_sampling_source_sequence_steps must have one count per source: "
            f"got {steps}, expected {expected_count} counts."
        )
    if any(step <= 0 for step in steps):
        raise ValueError(
            f"neighbor_sampling_source_sequence_steps must be positive, got {steps}."
        )
    if sum(steps) != expected_total:
        raise ValueError(
            "neighbor_sampling_source_sequence_steps must sum to the full training budget: "
            f"sum={sum(steps)}, expected epochs * dataset_len_cap={expected_total}."
        )
    return steps


def configure_cpu_loader_worker(worker_id, *, detect_anomaly=False):
    """Set debugging explicitly in spawned CPU workers, without initializing CUDA."""
    torch.autograd.set_detect_anomaly(detect_anomaly)


def get_covid19_twitter_dataloader(
        dataset: SubgraphDataset,
        split: str,
        node_split: str,
        batch_size: Union[int, range],
        n_way: Union[int, range],
        n_shot: Union[int, range],
        n_query: Union[int, range],
        batch_count: int,
        root: str,
        bert,
        num_workers: int,
        aug: str,
        aug_test: bool,
        split_labels: bool,
        train_cap: Optional[int],
        linear_probe: bool,
        label_set: Optional[Set[int]] = None,
        **kwargs
) -> DataLoader:
    del root
    seed = sum(ord(c) for c in split) + int(
        kwargs.get("eval_episode_seed_offset", 0) or 0
    )
    graph = dataset.graph
    task_name = kwargs.get("task_name", "neighbor_matching")
    aug_by_task = None  # set only by the nm_fp_cl rotation branch (per-episode aug map)
    if task_name == "neighbor_matching":
        positive_sampler = positive_sampler_for_split(dataset, split, kwargs)
        center_radii = _parse_center_radii(
            kwargs.get("neighbor_sampling_center_radii", "")
        )
        center_radius_weights = _parse_center_radius_weights(
            kwargs.get("neighbor_sampling_center_radius_weights", ""),
            None if center_radii is None else len(center_radii),
        )
        strata = None
        confine_to_single_stratum = False
        # Two graph_id-based modes (mutually exclusive in practice):
        #   neighbor_sampling_strata="graph_id"          -> balance sources WITHIN each episode
        #   neighbor_sampling_episode_source="graph_id"  -> confine each episode to ONE source
        strata_mode = kwargs.get("neighbor_sampling_strata", "")
        episode_source = kwargs.get("neighbor_sampling_episode_source", "")
        batch_source_mode = kwargs.get(
            "neighbor_sampling_batch_source_mode", "independent"
        )
        subset_spec = kwargs.get("neighbor_sampling_source_subset", "")
        sequence_spec = kwargs.get("neighbor_sampling_source_sequence", "")
        sequence_steps_spec = kwargs.get(
            "neighbor_sampling_source_sequence_steps", ""
        )
        sequence_steps = None
        if strata_mode == "graph_id" or episode_source == "graph_id":
            if not hasattr(graph, "graph_id"):
                raise ValueError("graph_id neighbor sampling requires graph.graph_id metadata.")
            graph_ids = graph.graph_id.detach().cpu().numpy()
            shared_pools = getattr(dataset, "source_node_pools", None)
            stratum_ids = sorted(shared_pools) if shared_pools is not None else sorted(set(graph_ids.tolist()))
            source_count = len(stratum_ids)
            source_names = list(getattr(graph, "source_graph_names", []))
            # Optionally restrict episodes to a subset of the merged sources. The merge is a
            # disjoint block-concat, so a neighborhood sampled around a center never leaves its
            # source graph; dropping the other strata is therefore equivalent to training on the
            # merge of just these sources (see nm_ladder_order_robustness-jul_23).
            subset = resolve_source_subset(subset_spec, stratum_ids, source_names)
            if subset is not None:
                stratum_ids = [graph_id for graph_id in stratum_ids if graph_id in subset]
            if str(sequence_spec or "").strip():
                is_training_stream = split == "train" and node_split != "val"
                if not is_training_stream:
                    sequence = None
                else:
                    if episode_source != "graph_id":
                        raise ValueError(
                            "neighbor_sampling_source_sequence requires "
                            "neighbor_sampling_episode_source='graph_id'."
                        )
                    if batch_size != 1:
                        raise ValueError(
                            "neighbor_sampling_source_sequence requires batch_size=1 so one "
                            "scheduled episode equals one optimizer step."
                        )
                    sequence = resolve_source_sequence(
                        sequence_spec, stratum_ids, source_names
                    )
                    expected_total = batch_count * int(kwargs.get("epochs", 1))
                    sequence_steps = parse_source_sequence_steps(
                        sequence_steps_spec, len(sequence), expected_total
                    )
                    stratum_ids = sequence
            elif str(sequence_steps_spec or "").strip():
                raise ValueError(
                    "neighbor_sampling_source_sequence_steps was set without "
                    "neighbor_sampling_source_sequence."
                )
            strata = ([shared_pools[graph_id].numpy() for graph_id in stratum_ids]
                      if shared_pools is not None else
                      [np.where(graph_ids == graph_id)[0].tolist() for graph_id in stratum_ids])
            confine_to_single_stratum = episode_source == "graph_id"
            summary = ", ".join(
                f"{source_names[graph_id] if graph_id < len(source_names) else graph_id}:{len(stratum)}"
                for graph_id, stratum in zip(stratum_ids, strata)
            )
            mode = "confine-to-one-source" if confine_to_single_stratum else "balance-within-episode"
            scope = f" [subset {len(stratum_ids)}/{source_count} sources]" if subset else ""
            print(f"Neighbor sampling graph_id strata ({mode}){scope}: {summary}", flush=True)
            if batch_source_mode == "complete":
                if episode_source != "graph_id":
                    raise ValueError(
                        "neighbor_sampling_batch_source_mode='complete' requires "
                        "neighbor_sampling_episode_source='graph_id'."
                    )
                if not isinstance(batch_size, int) or batch_size != len(strata):
                    raise ValueError(
                        "neighbor_sampling_batch_source_mode='complete' requires batch_size "
                        f"to equal the number of active sources ({len(strata)}), got "
                        f"{batch_size!r}."
                    )
                print(
                    "Neighbor sampling batches: complete source coverage "
                    f"({len(strata)} within-source episodes per batch)",
                    flush=True,
                )
            if sequence_steps is not None:
                schedule = ", ".join(
                    f"{source_names[graph_id] if graph_id < len(source_names) else graph_id}:"
                    f"{steps}"
                    for graph_id, steps in zip(stratum_ids, sequence_steps)
                )
                print(f"Blocked source schedule: {schedule}", flush=True)
        elif any(
            str(spec or "").strip()
            for spec in (subset_spec, sequence_spec, sequence_steps_spec)
        ):
            raise ValueError(
                "neighbor_sampling_source_subset/source_sequence requires "
                "neighbor_sampling_strata='graph_id' or "
                "neighbor_sampling_episode_source='graph_id'; got neither, so the requested "
                "source restriction would be silently ignored."
            )
        sampler = BatchSampler(
            batch_count,
            NeighborTask(
                positive_sampler,
                graph.num_nodes,
                "inout",
                kwargs.get("neighbor_sampling_strategy", "strict"),
                strata=strata,
                confine_to_single_stratum=confine_to_single_stratum,
                stratum_weighting=kwargs.get("neighbor_sampling_episode_source_weighting", "proportional"),
                cross_source_prob=float(kwargs.get("neighbor_sampling_cross_source_prob", 0.0)),
                stratum_schedule_steps=sequence_steps,
                filter_min_degree=bool(kwargs.get("neighbor_matching_edge_split", False)),
                batch_source_mode=batch_source_mode,
                center_radii=center_radii,
                center_radius_weights=center_radius_weights,
                center_region_fanout=int(
                    kwargs.get("neighbor_sampling_center_region_fanout", 64)
                ),
                center_region_node_limit=int(
                    kwargs.get("neighbor_sampling_center_region_node_limit", 4096)
                ),
                center_region_candidate_limit=int(
                    kwargs.get("neighbor_sampling_center_region_candidate_limit", 512)
                ),
                # Define graph distance on the leakage-free background adjacency even
                # when validation/test positives come from their disjoint holdout views.
                center_region_sampler=dataset.neighbor_sampler,
                center_max_attempts=int(
                    kwargs.get("neighbor_sampling_center_max_attempts", 200)
                ),
            ),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name == "nm_fp_cl":
        # Multi-task SSL rotation: every EPISODE is one of {nm, cl, fp}, drawn round-robin
        # over a count-weighted schedule (MultiTaskSplitBatch). All three share the metric
        # episode structure (nm=NeighborTask; cl/fp=ContrastiveTask), so the metagraph is
        # built identically; they differ only in (a) augmentation, set per task here via
        # aug_by_task, and (b) — for fp — the reconstruction loss, dispatched in the trainer
        # off graph.mix_is_fp (tagged by the Collator). Val/test fall back to pure NM so the
        # pretrain monitor metric stays coherent (mixing reconstruction and metric scores in
        # one accumulated eval is meaningless); the real comparison is the frozen-encoder
        # downstream sweep, run as separate eval-only jobs.
        if split == "train":
            counts_raw = str(kwargs.get("mix_task_counts", "1,1,1")).split(",")
            try:
                counts = [int(c) for c in counts_raw]
                assert len(counts) == 3 and all(c >= 0 for c in counts) and sum(counts) > 0
            except (ValueError, AssertionError):
                raise ValueError(
                    f"mix_task_counts must be three non-negative ints 'nm,cl,fp', got {counts_raw!r}."
                )
            task_base = MultiTaskSplitBatch(
                [
                    NeighborTask(
                        dataset.neighbor_sampler, graph.num_nodes, "inout",
                        kwargs.get("neighbor_sampling_strategy", "strict"),
                    ),
                    ContrastiveTask(graph.num_nodes),  # cl
                    ContrastiveTask(graph.num_nodes),  # fp
                ],
                ["nm", "cl", "fp"],
                counts,
            )
            sampler = BatchSampler(
                batch_count, task_base,
                ParamSampler(batch_size, n_way, n_shot, n_query, 1),
                seed=seed,
            )
            mask_ratio = float(kwargs.get("fp_mask_ratio", 0.3))
            mask_strategy = kwargs.get("fp_mask_strategy", "zero")
            if mask_strategy == "zero":
                fp_aug_spec = f"NZ{mask_ratio}"
            elif mask_strategy == "random":
                fp_aug_spec = f"NR{mask_ratio}"
            else:
                raise ValueError(f"Unknown fp_mask_strategy={mask_strategy!r}; use 'zero' or 'random'.")
            cl_aug_spec = kwargs.get("mix_cl_aug", "NZ0.2")
            aug_by_task = {
                "nm": get_aug("", graph.x),
                "cl": get_aug(cl_aug_spec, graph.x),
                "fp": get_aug(fp_aug_spec, graph.x),
            }
            label_embeddings = {
                "nm": torch.zeros(1, 768).expand(graph.num_nodes, -1),
                "cl": torch.zeros(1, 768).expand(graph.num_nodes, -1),
                "fp": torch.zeros(1, 768).expand(graph.num_nodes, -1),
            }
            print(
                f"nm_fp_cl rotation (per-episode): nm:cl:fp counts = "
                f"{counts[0]}:{counts[1]}:{counts[2]}, cl_aug={cl_aug_spec}, fp_aug={fp_aug_spec}",
                flush=True,
            )
        else:
            # coherent monitor: pure NM for val/test
            sampler = BatchSampler(
                batch_count,
                NeighborTask(
                    dataset.neighbor_sampler, graph.num_nodes, "inout",
                    kwargs.get("neighbor_sampling_strategy", "strict"),
                ),
                ParamSampler(batch_size, n_way, n_shot, n_query, 1),
                seed=seed,
            )
            label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name == "e4_multi":
        # E4 multi-task objective: masked-feature-recon ⊕ directed-LP ⊕ structural-property,
        # on E2's encoder. Every episode is a whole-node-masked ContrastiveTask episode (same
        # structure as masked_feature_prediction); the three heads read it in the trainer.
        #   * simultaneous -> mask every episode, trainer sums MFR+LP+structural
        #   * rotation     -> MultiTaskSplitBatch tags one head per episode (mfr/lp/struct),
        #                     Collator sets graph.e4_task; trainer dispatches that head only.
        # Masked node's bio block = MFR target, its structural (degree) block = structural
        # target (own degree input zeroed -> no passthrough leakage). Val/test = masked
        # episodes -> the trainer's simultaneous total is a coherent monitor.
        combine = kwargs.get("e4_combine", "simultaneous")
        if split == "train" and combine == "rotation":
            counts_raw = str(kwargs.get("e4_task_counts", "1,1,1")).split(",")
            try:
                counts = [int(c) for c in counts_raw]
                assert len(counts) == 3 and all(c >= 0 for c in counts) and sum(counts) > 0
            except (ValueError, AssertionError):
                raise ValueError(
                    f"e4_task_counts must be three non-negative ints 'mfr,lp,struct', got {counts_raw!r}."
                )
            task_base = MultiTaskSplitBatch(
                [
                    ContrastiveTask(graph.num_nodes),  # mfr
                    ContrastiveTask(graph.num_nodes),  # lp
                    ContrastiveTask(graph.num_nodes),  # struct
                ],
                ["mfr", "lp", "struct"],
                counts,
            )
            sampler = BatchSampler(
                batch_count, task_base,
                ParamSampler(batch_size, n_way, n_shot, n_query, 1),
                seed=seed,
            )
            mask_ratio = float(kwargs.get("fp_mask_ratio", 0.3))
            mask_strategy = kwargs.get("fp_mask_strategy", "zero")
            if mask_strategy == "zero":
                fp_aug_spec = f"NZ{mask_ratio}"
            elif mask_strategy == "random":
                fp_aug_spec = f"NR{mask_ratio}"
            else:
                raise ValueError(f"Unknown fp_mask_strategy={mask_strategy!r}; use 'zero' or 'random'.")
            # All episodes are masked (lp ignores the mask); the trainer picks the head.
            aug_by_task = {
                "mfr": get_aug(fp_aug_spec, graph.x),
                "lp": get_aug(fp_aug_spec, graph.x),
                "struct": get_aug(fp_aug_spec, graph.x),
            }
            label_embeddings = {
                "mfr": torch.zeros(1, 768).expand(graph.num_nodes, -1),
                "lp": torch.zeros(1, 768).expand(graph.num_nodes, -1),
                "struct": torch.zeros(1, 768).expand(graph.num_nodes, -1),
            }
            print(
                f"e4_multi rotation (per-episode): mfr:lp:struct counts = "
                f"{counts[0]}:{counts[1]}:{counts[2]}",
                flush=True,
            )
        else:
            sampler = BatchSampler(
                batch_count,
                ContrastiveTask(graph.num_nodes),
                ParamSampler(batch_size, n_way, n_shot, n_query, 1),
                seed=seed,
            )
            label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name in {"contrastive", "masked_feature_prediction"}:
        sampler = BatchSampler(
            batch_count,
            ContrastiveTask(graph.num_nodes),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name == "temporal_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError("temporal_link_prediction requires target edges, but no future edge view was found.")
        neg_ratio = int(kwargs.get("midterm_lp_neg_ratio", 1))
        if isinstance(n_way, int):
            invalid_n_way = n_way != 1
        else:
            invalid_n_way = any(v != 1 for v in n_way)
        if invalid_n_way:
            raise ValueError(
                "temporal_link_prediction now only supports binary LP episodes. "
                f"Use --n_way 1, got n_way={n_way}."
            )
        task = BinaryFutureLinkTask(dataset.future_neighbor_sampler, graph.num_nodes, neg_ratio=neg_ratio)
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = False
    elif task_name == "static_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError(
                "static_link_prediction requires a 'static_holdout' target edge view; "
                "enrich the graph with benchmark targets (enrich_graph_targets.py)."
            )
        neg_ratio = int(kwargs.get("midterm_lp_neg_ratio", 1))
        hard_negatives = bool(kwargs.get("hard_negatives", True))
        invalid_n_way = (n_way != 1) if isinstance(n_way, int) else any(v != 1 for v in n_way)
        if invalid_n_way:
            raise ValueError(f"static_link_prediction only supports n_way=1, got n_way={n_way}.")
        task = StaticLinkTask(
            dataset.future_neighbor_sampler,
            dataset.neighbor_sampler,
            graph.num_nodes,
            neg_ratio=neg_ratio,
            hard_negatives=hard_negatives,
        )
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = False
    elif task_name == "classification":
        if getattr(graph, "label_type", "classification") == "regression":
            raise ValueError(
                "covid19_twitter graph stores regression labels; use --task_name regression, not classification."
            )
        label_names = list(getattr(graph, "label_names", []))
        num_classes = len(label_names)
        if num_classes == 0:
            raise ValueError("covid19_twitter classification requires graph.label_names to be populated.")

        if bert is not None:
            label_texts = _label_embedding_texts(label_names, kwargs.get("label_emb_texts", ""))
            label_embeddings = bert.get_sentence_embeddings(label_texts)
        else:
            label_embeddings = _deterministic_label_embeddings(label_names, dim=768)

        labels = graph.y.numpy()
        if not hasattr(dataset, "_classification_node_splits"):
            dataset._classification_node_splits = _build_stratified_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._classification_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown covid19_twitter classification node split '{split_key}'. "
                f"Available: {sorted(node_splits.keys())}"
            )
        labels = _mask_labels_to_node_split(labels, node_splits[split_key])
        task = midterm_task(
            labels=labels,
            num_classes=num_classes,
            split=split,
            label_set=label_set,
            split_labels=False,
            train_cap=train_cap,
            linear_probe=linear_probe,
            random_query=kwargs.get("eval_random_query", False),
        )
        task.original_graph_labels = graph.y.numpy().copy()
        task.split_masked_labels = labels.copy()
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        is_multiway = True
    elif task_name == "regression":
        if getattr(graph, "label_type", "classification") != "regression":
            raise ValueError(
                f"covid19_twitter regression requires a graph with regression labels; "
                f"got label_type={getattr(graph, 'label_type', None)!r}."
            )
        if n_way != 1:
            raise ValueError(f"covid19_twitter regression only supports n_way=1, got {n_way}.")

        labels = graph.y.detach().cpu().numpy().astype(np.float32)
        if not hasattr(dataset, "_regression_node_splits"):
            dataset._regression_node_splits = _build_regression_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._regression_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown covid19_twitter regression node split '{split_key}'. "
                f"Available: {sorted(node_splits.keys())}"
            )
        split_idx = node_splits[split_key]
        if split_idx.size == 0:
            raise ValueError(
                f"covid19_twitter regression split '{split_key}' has no labeled nodes."
            )

        task = RegressionTask(labels=labels, node_indices=split_idx)
        task.original_graph_labels = labels.copy()
        task.split_masked_labels = labels.copy()
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, 1, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros((1, 768), dtype=torch.float)
        is_multiway = False
    else:
        raise ValueError(f"Unknown task for covid19_twitter: {task_name}")

    if task_name == "nm_fp_cl" or (
        task_name == "e4_multi"
        and split == "train"
        and kwargs.get("e4_combine", "simultaneous") == "rotation"
    ):
        # Per-episode aug is supplied via aug_by_task (train) or none (val/test fallback);
        # the global collator aug is identity so it never double-augments.
        aug_fn = get_aug("", graph.x)
    elif task_name in {"masked_feature_prediction", "e4_multi"}:
        # e4_multi (simultaneous, or any val/test) masks every episode like fp.
        mask_ratio = float(kwargs.get("fp_mask_ratio", 0.3))
        mask_strategy = kwargs.get("fp_mask_strategy", "zero")
        if mask_strategy == "zero":
            fp_aug = f"NZ{mask_ratio}"
        elif mask_strategy == "random":
            fp_aug = f"NR{mask_ratio}"
        else:
            raise ValueError(
                f"Unknown fp_mask_strategy={mask_strategy!r}; use 'zero' or 'random'."
            )
        aug_spec = (aug if task_name == "masked_feature_prediction" else "") or fp_aug
        aug_fn = get_aug(aug_spec, graph.x)
    else:
        aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=(kwargs.get("loader_start_method") or None) if num_workers else None,
        worker_init_fn=partial(configure_cpu_loader_worker,
                              detect_anomaly=bool(kwargs.get("detect_anomaly", False))),
        collate_fn=Collator(
            label_embeddings, aug=aug_fn, is_multiway=is_multiway,
            task_name=task_name, aug_by_task=aug_by_task,
        ),
    )
