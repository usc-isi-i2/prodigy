import torch
import argparse

import sys
import os
from datetime import datetime
sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

try:
    import yaml
except ImportError:  # pragma: no cover - depends on environment.
    yaml = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def _load_yaml_config(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config.")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    if "params" in config:
        config = config["params"]
        if not isinstance(config, dict):
            raise TypeError(f"{path}: 'params' must be a YAML mapping.")
    return config


def _coerce_config_value(action, value):
    if value is None:
        return None
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        return str2bool(value) if isinstance(value, str) else bool(value)
    if action.nargs in {"+", "*"}:
        values = value
        if isinstance(values, str):
            values = values.split()
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"Config value for '{action.dest}' must be a list.")
        if action.type is None:
            return list(values)
        return [action.type(item) for item in values]
    if action.type is None:
        return value
    if action.type is str2bool:
        return str2bool(value)
    if isinstance(value, action.type):
        return value
    return action.type(value)


def _apply_config_defaults(parser, config):
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }
    unknown = sorted(set(config) - set(action_by_dest))
    if unknown:
        raise ValueError(
            "Unknown config option(s): "
            f"{unknown}. Use argparse destination names, e.g. 'dataset_len_cap'."
        )
    for key, value in config.items():
        action = action_by_dest[key]
        coerced = _coerce_config_value(action, value)
        if action.choices is not None:
            values = coerced if isinstance(coerced, list) else [coerced]
            invalid = [item for item in values if item not in action.choices]
            if invalid:
                raise ValueError(
                    f"Invalid config value for '{key}': {invalid}. "
                    f"Allowed values: {list(action.choices)}"
                )
        action.default = coerced


def get_params():
    args = argparse.ArgumentParser()

    args.add_argument("--config", default="", type=str, help="Optional YAML config. CLI args override config values.")
    args.add_argument("-root", "--root", default="./FSdatasets", type=str)
    args.add_argument("-dataset", "--dataset", default="arxiv", type=str)
    args.add_argument("--csv_filename", default="twitter_data.csv", type=str)
    args.add_argument("--facebook_pkl_filename", default="facebook_2022-06-24_2022-06-25_part1.pkl", type=str)
    args.add_argument("--facebook_edges_filename", default="facebook_2022-06-24_2022-06-25_part1_edges.csv", type=str)
    args.add_argument("--facebook_node_features_filename", default="facebook_2022-06-24_2022-06-25_part1_node_features.csv", type=str)
    args.add_argument("--facebook_data_source", default="csv", type=str)
    args.add_argument("--facebook_use_edge_features", default=False, type=str2bool)
    args.add_argument("--facebook_edge_feature_columns", default="", type=str)
    args.add_argument("--facebook_source_pkl_path", default="", type=str)
    args.add_argument("--facebook_embeddings_path", default="", type=str)
    args.add_argument("--facebook_embedding_ids_path", default="", type=str)
    args.add_argument("--facebook_text_emb_model", default="", type=str)
    args.add_argument("--facebook_target_dim", default=768, type=int)
    args.add_argument("--facebook_filter_to_uk_ru", default=True, type=str2bool)
    args.add_argument("--facebook_max_posts", default=None, type=int)
    args.add_argument("--label_type", default="verified", type=str)
    args.add_argument("--max_users", default=None, type=int)
    args.add_argument("-invalidate_cache", "--invalidate_cache", default=False, type=str2bool)
    # if true, it will regenerate preprocessed cache
    args.add_argument("-ds_cap", "--dataset_len_cap", default=10000, type=int)
    args.add_argument("-val_cap", "--val_len_cap", default=None, type=int)
    args.add_argument("-test_cap", "--test_len_cap", default=None, type=int)
    args.add_argument("-shuffle_index", "--shuffle_index", default=False, type=str2bool) # For KG datasets, shuffle index to get a different task each time if using ds_cap 1
    # will cap length of training and testing datasets (use this for debugging or when using smaller GPU)
    args.add_argument("-force_cache", "--force_cache", default=False, type=str2bool)  # will use preprocessed cache
    args.add_argument("-cl_only", "--classification_only", default=False, type=str2bool) # only set this to true when using the very basic arxiv dataset!!! (this is basic node classification where labels are the same in train and test)
    args.add_argument("-esp", "--early_stopping_patience", default=20, type=int) # early stopping patience (in validation epochs, so with default eval_epoch argument 20 * 10 = 200 epochs)
    args.add_argument("--reset_after_layer", default=None, nargs='+', type=int)
    args.add_argument("-original_features", "--original_features", default=False, type=str2bool)
    args.add_argument("-override_log", "--override_log", default=False, type=str2bool)


    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument(
        "--eval_episode_seed_offset",
        default=0,
        type=int,
        help=(
            "Optional offset added to the deterministic split-derived eval episode "
            "seed. Default 0 preserves the fixed historical episode sets."
        ),
    )

    args.add_argument("-metric", "--metric", default="Acc", choices=["Acc"])

    # Training-specific params
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-epochs", "--epochs", default=12, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=10, type=int)  # deprecated
    args.add_argument("-eval_epo", "--eval_epoch", default=10, type=int)  # deprecated
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=10, type=int)  # deprecated
    args.add_argument("-prt_step", "--print_step", default=2000, type=int)
    args.add_argument("-eval_step", "--eval_step", default=2000, type=int)
    args.add_argument("-ckpt_step", "--checkpoint_step", default=2000, type=int)
    args.add_argument(
        "-ckpt_steps", "--checkpoint_steps", default="", type=str,
        help=(
            "Explicit checkpoint schedule as comma-separated step counts, e.g. "
            "'0,10,100,250,500,1000,2000'. Non-empty OVERRIDES the --checkpoint_step "
            "modulo cadence. A step here is a count of COMPLETED optimizer steps: 0 is "
            "the random-init state written before the first step, N is the state after "
            "N steps. Steps above the budget (epochs x dataset_len_cap) are never "
            "reached; the terminal checkpoint at the true step count is written either "
            "way. Use this for dense early-training trajectories (saturation curves), "
            "where a uniform modulo cadence cannot express log-spaced steps."
        ),
    )
    args.add_argument("-bs", "--batch_size", default=1, type=int) 
    args.add_argument("-weight_decay", "--weight_decay", default=0.001, type=float)
    args.add_argument("-dropout", "--dropout", default=0, type=float)
    args.add_argument("-txt_dropout", "--text_features_dropout", default=0, type=float)  # additionally drop out text features
    args.add_argument("-rel_sample_seed", "--rel_sample_random_seed", default=None, type=float)  # seed for sampling relations

    args.add_argument("-split_train_nodes", "--split_train_nodes", default=False, type=str2bool) # Split train nodes into 'train' and 'val'

    args.add_argument("-verbose", "--verbose", default=False, type=str2bool)

    args.add_argument("-workers", "--workers", default=10, type=int)  # Number of workers per dataloader
    args.add_argument("-gpu", "--device", default=123, type=int)  # device 123 means CPU


    # GNN- and model-specific parameters
    args.add_argument("-input_dim", "--input_dim", default=768, type=int)  # this is bert dim etc.
    args.add_argument("-emb_dim", "--emb_dim", default=256, type=int)
    args.add_argument("-gnn_type", "--gnn_type", default="sage", type=str)  # support "gin", "no_msg_passing", "sage"
    args.add_argument("-n_layer", "--n_layer", default=1, type=int)
    args.add_argument("-meta_n_layer", "--meta_n_layer", default=1, type=int)
    args.add_argument("-gnn2", "--second_gnn", default="Atten", type=str)  # "vanilla" or "gat"
    args.add_argument("--attention_mask_scheme", default="causal", type=str)  # the name of the pretraining task
    args.add_argument("-skip", "--skip_path", default=False, type=str2bool)
    args.add_argument("-has_final_back", "--has_final_back", default=False, type=str2bool)

    args.add_argument("-layers", "--layers", default="S,U,M", type=str)  # default: GraphSAGE->Up->Metagraph (see experiments/layers.py for more info)
    args.add_argument("-ignore_label_embs", "--ignore_label_embeddings", default=True, type=str2bool)
    args.add_argument("-zero_lbl", "--zero_label_embeddings", default=False, type=str2bool)
    args.add_argument("-not_freeze_learned_label_embedding", "--not_freeze_learned_label_embedding", default=False, type=str2bool)
    args.add_argument("-linear_probe", "--linear_probe", default=False, type=str2bool)
    args.add_argument("-fdf", "--fix_datasets_first", default=False,
                      type=str2bool)  # Whether to convert datasets to list first (no sampling involved later).
    # This should generally not be used as the resulting files would be way too large

    args.add_argument("-no_bn_metagraph", "--no_bn_metagraph", default=False,  # no batch norm metagraph
                      type=str2bool)
    args.add_argument("-no_bn_encoder", "--no_bn_encoder", default=False,  # no batch norm on S layers etc.
                      type=str2bool)

    args.add_argument("-calc_ranks", "--calc_ranks", default=False, type=str2bool)  # Whether to calc MRR and HITS ranks.
    args.add_argument(
        "--save_roc_curve",
        default=False,
        type=str2bool,
        help="If True, save ROC curve plot/points for binary evaluation splits.",
    )
    args.add_argument("-eval_only", "--eval_only", default=False, type=str2bool)  # Eval. only mode (no training, only one pass of testing ds at the beginning and then quit)
    args.add_argument(
        "--eval_test_before_train",
        default=False,
        type=str2bool,
        help="If True, run a test-set evaluation before training starts.",
    )
    args.add_argument(
        "--eval_val_before_train",
        default=False,
        type=str2bool,
        help="If True, run a validation-set evaluation before training starts.",
    )
    args.add_argument(
        "--eval_after_train",
        default=False,
        type=str2bool,
        help="If True, run validation/test evaluation once after the final training step.",
    )
    args.add_argument("-meta_pos", "--meta_gnn_pos_only", default=False, type=str2bool)  # Whether to use only positive edges for meta graph

    ###  Few-shot task parameters  ###
    args.add_argument("-task", "--task_name", default=None, type=str)  # the name of the pretraining task
    args.add_argument("-zeroshot", "--zero_shot", default=False, type=str2bool) # if True, messages will NOT be passed along the metagraph edges.
    args.add_argument("-no_split_labels", "--no_split_labels", default=True, type=str2bool) # split train/val/test with original dataset split
    args.add_argument("-all_test", "--all_test", default=False,
                      type=str2bool)  # Set train/test/val labels to the same label set (for testing purposes)
    args.add_argument("-train_cap", "--train_cap", default=None, type=int) # split train/val/test with original dataset split
    args.add_argument('--label_set', type=str, nargs='+')
    args.add_argument("-csr_split", "--csr_split", default=False, type=str2bool)  # Whether to use CSR split...

    args.add_argument("-way", "--n_way", default=3, type=int) # how many labels do we want in each few-shot task
    args.add_argument("-shot", "--n_shots", default=3, type=int) # if not zeroshot, how many shots do we want in the training dataset?
    args.add_argument("-qry", "--n_query", default=24, type=int)
    args.add_argument(
        "--neighbor_sampling_strategy",
        default="strict",
        choices=["strict", "replacement"],
        help=(
            "How NeighborTask handles centers with too few unique sampled neighbors. "
            "'strict' rejects the center; 'replacement' reuses sampled neighbors with replacement."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_strata",
        default="",
        choices=["", "graph_id"],
        help=(
            "Optional strata used for neighbor-matching center-node sampling. "
            "'graph_id' balances centers across disjoint merged source graphs."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_episode_source",
        default="",
        choices=["", "graph_id"],
        help=(
            "If 'graph_id', confine every neighbor-matching episode to a SINGLE merged "
            "source graph. Removes cross-source negatives so the model cannot exploit a "
            "source-discrimination shortcut. Source-selection weighting is controlled by "
            "--neighbor_sampling_episode_source_weighting."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_source_subset",
        default="",
        type=str,
        help=(
            "Comma-separated source graphs (names from graph.source_graph_names, or integer "
            "graph_ids) that neighbor-matching episodes may be drawn from. Empty -> all sources. "
            "Because merges are disjoint block-concats (no cross-source edges) and episodes are "
            "already confined to one source, restricting the set is equivalent to training on the "
            "merge of just those sources -- this lets one merged graph stand in for every rung of "
            "a curriculum ladder without rebuilding merges. Requires --neighbor_sampling_strata or "
            "--neighbor_sampling_episode_source to be 'graph_id'; errors otherwise rather than "
            "silently training on every source."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_source_sequence",
        default="",
        type=str,
        help=(
            "Optional comma-separated source order for BLOCKED sequential neighbor-matching "
            "training. Names/ids use the same vocabulary as --neighbor_sampling_source_subset. "
            "Every listed source is used for one contiguous block, in order, with block lengths "
            "from --neighbor_sampling_source_sequence_steps. Requires batch_size=1, "
            "--neighbor_sampling_episode_source=graph_id, and a zero cross-source probability. "
            "Empty preserves the historical interleaved sampler."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_source_sequence_steps",
        default="",
        type=str,
        help=(
            "Comma-separated positive episode counts for the blocked source sequence. The counts "
            "must match the sequence length and sum to epochs * dataset_len_cap."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_episode_source_weighting",
        default="proportional",
        choices=["proportional", "balanced"],
        help=(
            "How the per-episode source is chosen when confining episodes "
            "(--neighbor_sampling_episode_source=graph_id). 'proportional' -> P(source) "
            "proportional to node count (per-node marginal matches naive sampling); "
            "'balanced' -> uniform over sources (gives small sources equal episode share)."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_batch_source_mode",
        default="independent",
        choices=["independent", "complete"],
        help=(
            "How sources are assigned across neighbor-matching episodes in one batch. "
            "'independent' preserves historical per-episode draws. 'complete' emits "
            "exactly one internally within-source episode from every active graph_id "
            "per batch; it requires batch_size to equal the active source count, "
            "neighbor_sampling_episode_source=graph_id, and zero cross-source probability."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_cross_source_prob",
        default=0.0,
        type=float,
        help=(
            "Partial cross-source episodes (remedy #4). Requires "
            "--neighbor_sampling_episode_source=graph_id. Probability that a given "
            "neighbor-matching episode is a naive MIXED-source episode instead of being "
            "confined to one source. 0.0 -> pure within-source (default), 1.0 -> pure "
            "naive; values in between interpolate, trading the cross-source shortcut "
            "against source-discrimination signal."
        ),
    )
    args.add_argument(
        "--neighbor_matching_edge_split",
        default=False,
        type=str2bool,
        help=(
            "Leakage-free NM protocol. Requires edge_view=static_background and "
            "target_edge_view=static_holdout. Training positives and all message "
            "passing use the background view; val/test positives use only holdout edges."
        ),
    )

    args.add_argument("-way_u", "--n_way_upper", default=-1, type=int) # If defined, will set the upper bound for n_way
    args.add_argument("-shot_u", "--n_shots_upper", default=-1, type=int) # If defined, will set the upper bound for n_shots
    args.add_argument("-qry_u", "--n_query_upper", default=-1, type=int) # If defined, will set the upper bound for n_query
    args.add_argument("-max_length", default=-1, type=int) 

    ### data augmentation parameters ###
    args.add_argument("-aug", "--augmentation", default="", type=str)
    args.add_argument("-aug_test", "--augment_test", default=False, type=str2bool)  # if set, the valid set and the test set are also augmented
    args.add_argument(
        "-ablate_feat", "--ablate_features", default="none",
        choices=["none", "zero", "permute", "noise"],
        help=(
            "Controlled eval-time feature ablation for measuring feature vs. topology "
            "reliance. 'zero' replaces all node features with zeros; 'permute' shuffles "
            "feature rows across nodes within each subgraph (preserves the subgraph's "
            "feature multiset, breaks the node<->feature binding); 'noise' resamples every "
            "node's features from the full graph's feature distribution (destroys the local "
            "neighborhood content while keeping nodes distinct and in-distribution). "
            "'permute' vs 'noise' distinguishes features-as-distinguishers from "
            "features-as-content. The corresponding aug token (FZ/FP/NR1.0) is appended to "
            "--augmentation and --augment_test is forced on. For -eval_only True runs."
        ),
    )
    args.add_argument(
        "-ablate_edges", "--ablate_edges", default="none",
        choices=["none", "rewire"],
        help=(
            "Controlled eval-time EDGE ablation for measuring topology reliance "
            "(the 'rewired edge' half of the topology_feature_ssl 2x2). 'rewire' "
            "replaces each subgraph's real edges with a random directed edge set of "
            "the same size over the same node support (destroys real adjacency, keeps "
            "the node bag / features / edge count / supernode scaffolding). Composable "
            "with --ablate_features (the token ER is appended to --augmentation and "
            "--augment_test is forced on). For -eval_only True runs."
        ),
    )
    args.add_argument(
        "--structural_features", default="none", choices=["none", "directed3", "directed3_log", "directed6"],
        help=(
            "E1/E2: inject directed per-node structural features as INPUT features "
            "(topology becomes representable). 'directed3' = [in_deg, out_deg, "
            "log_deg] (exact, O(E), scales to the 34M-node merged graph); 'directed6' "
            "adds [k_core, pagerank, clustering] (networkx, small graphs only). "
            "Z-scored per graph and concatenated to graph.x, bumping input_dim by "
            "3/6. Must be set identically at "
            "pretrain AND eval (it defines the encoder's input space). Features are "
            "cached next to the graph file. Subject to the leakage control "
            "(leakage_baseline.py) — E1/E2 'learned structure' only if the frozen "
            "rep beats the raw-structural-feature probe on structure-linked targets."
        ),
    )
    args.add_argument("-attr", "--attr_regression_weight", default=0.0, type=float)
    args.add_argument(
        "--fp_mask_ratio",
        default=0.3,
        type=float,
        help="Node feature mask ratio for task_name=masked_feature_prediction.",
    )
    args.add_argument(
        "--fp_mask_strategy",
        default="zero",
        choices=["zero", "random"],
        help="How masked node features are corrupted for task_name=masked_feature_prediction.",
    )
    args.add_argument(
        "--mix_task_counts",
        default="1,1,1",
        type=str,
        help="Round-robin weights for the nm_fp_cl SSL rotation, as 'nm,cl,fp' integer "
        "counts (e.g. '1,1,1' = equal per-episode share). Only used when task_name=nm_fp_cl.",
    )
    args.add_argument(
        "--mix_cl_aug",
        default="NZ0.2",
        type=str,
        help="Feature augmentation applied to CONTRASTIVE (cl) episodes inside the nm_fp_cl "
        "rotation (two-view positives). NM episodes get no aug; FP episodes use "
        "fp_mask_ratio/fp_mask_strategy. Only used when task_name=nm_fp_cl.",
    )
    # --- E4: multi-task objective (masked-feature-recon ⊕ directed-LP ⊕ structural) ---
    # Built on E2's encoder (sage_multi + directed3). MFR + structural share the
    # whole-node masking (fp_mask_ratio/strategy): a masked node's bio block (cols
    # :768) is the MFR target and its structural block (cols 768:) the structural
    # target — the node's own degree input is zeroed, so predicting it from context
    # is non-trivial (no leakage). Directed-LP scores the episode's directed edges
    # against sampled negatives on the encoder embeddings.
    args.add_argument(
        "--e4_combine",
        default="simultaneous",
        choices=["simultaneous", "rotation"],
        help="How task_name=e4_multi combines its heads: 'simultaneous' (weighted sum of "
        "MFR+LP+structural every step) or 'rotation' (one head per episode, round-robin "
        "over e4_task_counts; requires batch_size=1).",
    )
    args.add_argument(
        "--e4_weights",
        default="1,1,1",
        type=str,
        help="Simultaneous-mode loss weights as 'mfr,lp,struct' floats. Only used when "
        "task_name=e4_multi and e4_combine=simultaneous.",
    )
    args.add_argument(
        "--e4_task_counts",
        default="1,1,1",
        type=str,
        help="Rotation-mode round-robin counts as 'mfr,lp,struct' ints. Only used when "
        "task_name=e4_multi and e4_combine=rotation.",
    )
    args.add_argument(
        "--e4_lp_neg_k",
        default=1,
        type=int,
        help="Negative edges sampled per positive directed edge for the e4 link-prediction head.",
    )


    args.add_argument("-prefix", "--prefix", default="exp1", type=str) # prefix for the experiment name for wandb
    args.add_argument("-timestamp", "--timestamp", default=None, type=str)
    args.add_argument("--tags", default=[], nargs="+", type=str, help="Tags to attach to the wandb run.")

    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)

    args.add_argument("-bert", "--bert_emb_model", default="multi-qa-distilbert-cos-v1")
    args.add_argument("-bert_cache", "--bert_cache", default="multi-qa-distilbert-cos-v1")
    args.add_argument(
        "--label_emb_model",
        default="",
        type=str,
        help=(
            "Optional sentence-transformer model used only for class label embeddings. "
            "This is independent of --original_features, so graph node features can stay "
            "precomputed while zero-shot classification labels use a matching text encoder."
        ),
    )
    args.add_argument(
        "--label_emb_texts",
        default="",
        type=str,
        help=(
            "Optional semicolon-separated label text overrides for embeddings, "
            "e.g. 'not_conservative=not conservative political user;"
            "conservative=conservative political user'. Graph label names and "
            "class ids are unchanged."
        ),
    )
    args.add_argument(
        "--label_emb_revision",
        default="",
        type=str,
        help="Optional model revision for --label_emb_model.",
    )
    args.add_argument(
        "--label_emb_trust_remote_code",
        default=True,
        type=str2bool,
        help="Passed to SentenceTransformer when --label_emb_model is set.",
    )
    args.add_argument(
        "--label_emb_normalize",
        default=True,
        type=str2bool,
        help=(
            "If True, L2-normalize --label_emb_model outputs. Keep True for GTE "
            "graph features generated with normalized embeddings."
        ),
    )

    args.add_argument("-kg_emb", "--kg_emb_model", default="", type=str)   #  "TransE", "ComplEx", etc.
    args.add_argument("-pretrained", "--pretrained_model_run", default="", type=str)
    #  Name of WanDB run to pull the best model from.

    args.add_argument("-n_hop", "--n_hop", default=1, type=int)  # number of hops for subgraph extraction
    args.add_argument(
        "--neighbor_sampling_hop_sizes",
        default="",
        type=str,
        help=(
            "Optional comma-separated fanout per extracted hop. Empty preserves the "
            "historical fanout of 100 at every hop. Example: '9,9' gives a two-hop "
            "subgraph of at most about 100 nodes before the hard node limit."
        ),
    )
    args.add_argument(
        "--neighbor_sampling_node_limit",
        default=2000,
        type=int,
        help="Hard maximum nodes per sampled subgraph (historical default: 2000).",
    )
    args.add_argument(
        "--neighbor_matching_walk_hops",
        default=0,
        type=int,
        help=(
            "Positive-pair random-walk length for neighbor matching. 0 follows --n_hop "
            "for historical behavior; set 1 to hold the one-hop NM objective fixed while "
            "extracting two-hop context."
        ),
    )
    args.add_argument("--graph_filename", default="graph_data.pt", type=str)  # graph file to load from root
    args.add_argument(
        "--target_feature",
        default="",
        type=str,
        help=(
            "Optional node feature name to use as the regression target at load time. "
            "The selected feature column is removed from graph.x to avoid leakage."
        ),
    )
    args.add_argument(
        "--target_feature_keep_in_x",
        default=False,
        type=str2bool,
        help=(
            "If True, keep --target_feature inside graph.x instead of removing it. "
            "Useful only as a leakage/sanity test."
        ),
    )
    args.add_argument(
        "--target_transform",
        default="none",
        type=str,
        help=(
            "Optional transform applied to the regression target: none | log1p. "
            "Use log1p for heavy-tailed profile counts (follower count, etc.)."
        ),
    )
    args.add_argument(
        "--feature_subset",
        "--midterm_feature_subset",
        dest="feature_subset",
        default="all",
        type=str,
        help=(
            "Feature subset spec for graph node features. "
            "Supported: all | constant1 | stats_only | emb_only | emb_only_plus_label | label_only | "
            "keep:<f1,f2,...> | drop:<f1,f2,...>"
        ),
    )
    args.add_argument(
        "--midterm_label_downsample",
        default="",
        type=str,
        help=(
            "Optional class-balance downsampling for midterm labels before train/val/test node splits. "
            "Examples: '50:50', '20:80'. Ratios follow graph.label_names order."
        ),
    )
    args.add_argument(
        "--eval_random_query",
        default=False,
        type=str2bool,
        help=(
            "If True, allocate query slots in val/test episodes proportionally to class frequency "
            "instead of equal n_query per class. Produces accumulated metrics that reflect the "
            "real class distribution rather than a balanced one."
        ),
    )
    args.add_argument(
        "--edge_view",
        "--midterm_edge_view",
        dest="edge_view",
        default="default",
        type=str,
        help="Named edge view for the background graph (default uses 'edge_index').",
    )
    args.add_argument(
        "--target_edge_view",
        "--midterm_target_edge_view",
        dest="target_edge_view",
        default="future",
        type=str,
        help="Named edge target view for temporal link prediction (default uses 'future_edge_index').",
    )
    args.add_argument(
        "--edge_feature_subset",
        "--midterm_edge_feature_subset",
        dest="edge_feature_subset",
        default="all",
        type=str,
        help="Edge feature subset: all | none | keep:<f1,f2,...> | drop:<f1,f2,...>",
    )
    args.add_argument(
        "--midterm_lp_neg_ratio",
        default=1,
        type=int,
        help="Negative:positive ratio for temporal/static link-prediction binary sampling (e.g., 5 means 5 negatives per positive).",
    )
    args.add_argument(
        "--hard_negatives",
        default=True,
        type=str2bool,
        help=(
            "static_link_prediction only: draw negatives from the center's 2-hop "
            "(but not 1-hop) neighborhood in the background graph. False = random "
            "negatives (task becomes much easier / AUC saturates)."
        ),
    )
    args.add_argument(
        "--use_edge_features",
        "--midterm_use_edge_features",
        dest="use_edge_features",
        default=False,
        type=str2bool,
        help="If True, configure the background GNN to consume graph edge_attr features.",
    )
    args.add_argument(
        "--midterm_debug_print_episodes",
        default=0,
        type=int,
        help="If >0, print full first-eval episode details for this many LP episodes.",
    )
    args.add_argument(
        "--debug_print_predictions",
        default=0,
        type=int,
        help=(
            "If >0, print this many eval query predictions from the first val/test batch. "
            "For regression, prints input features, prediction, target, and error."
        ),
    )

    args.add_argument("-smalldataset", "--small_dataset", default=False,
                      type=str2bool)  # use for debugging  - very small dataset

    args.add_argument("-exptype", "--experiment_type", default="metagraph")


    config_args, _ = args.parse_known_args()
    if config_args.config:
        _apply_config_defaults(args, _load_yaml_config(config_args.config))

    args = args.parse_args()

    params = {}
    for k, v in vars(args).items():
        params[k] = v
    if params["task_name"] is None:
        if params["dataset"] == "midterm":
            params["task_name"] = "neighbor_matching"
        else:
            params["task_name"] = "classification"
    task_aliases = {
        "nm": "neighbor_matching",
        "cl": "contrastive",
        "same_graph": "contrastive",
        "fp": "masked_feature_prediction",
        "mfp": "masked_feature_prediction",
        "slp": "static_link_prediction",
        "reg": "regression",
        "mix": "nm_fp_cl",
        "e4": "e4_multi",
    }
    params["task_name"] = task_aliases.get(params["task_name"], params["task_name"])

    # Feature-ablation intervention: compose the ablation aug into the eval path.
    # Meant for -eval_only True runs; if used during training it would also ablate
    # the training augmentations, so we warn when that combination is requested.
    ablation_tokens = []
    if params["ablate_features"] != "none":
        ablation_tokens.append(
            {"zero": "FZ", "permute": "FP", "noise": "NR1.0"}[params["ablate_features"]]
        )
    if params["ablate_edges"] != "none":
        ablation_tokens.append({"rewire": "ER"}[params["ablate_edges"]])
    if ablation_tokens:
        existing = params["augmentation"]
        token = ",".join(ablation_tokens)
        params["augmentation"] = f"{existing},{token}" if existing else token
        params["augment_test"] = True
        if not params.get("eval_only", False):
            active = [
                f"--ablate_{k}={params['ablate_' + k]}"
                for k in ("features", "edges")
                if params["ablate_" + k] != "none"
            ]
            print(
                f"[params] WARNING: {' '.join(active)} set without -eval_only True; "
                "the ablation aug will also be applied during training.",
                flush=True,
            )

    if args.device == 123:
        params["device"] = torch.device('cpu')
    else:
        params['device'] = torch.device('cuda:' + str(args.device))
    if params["timestamp"] is None:
        params["timestamp"] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    params["exp_name"] = params["prefix"] + "_" + params["timestamp"]

    return params
