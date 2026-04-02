import torch
import numpy as np
import sys
import os
import json
import wandb
import torch.optim as optim
import time
from tqdm import tqdm, trange
import shutil
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

from models.get_model import print_num_trainable_params
from models.model_eval_utils import accuracy
from models.general_gnn import SingleLayerGeneralGNN
from models.sentence_embedding import SentenceEmb
from experiments.layers import get_module_list

def _to_float(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return float(v.detach().cpu().mean().item())
    return float(v)

def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TrainerFS():
    def __init__(self, dataset, parameter):
        wandb.init(project="graph-clip", name=parameter["exp_name"])
        #wandb.run.log_code(".")
        wandb.run.summary["wandb_url"] = wandb.run.url
        _log("Initializing trainer")
        print("---------- Parameters ----------", flush=True)
        for k, v in parameter.items():
            print(f"  {k}: {v}", flush=True)
        print("--------------------------------", flush=True)
        wandb.config.trainer_fs = True

        self.parameter = parameter

        self.ignore_label_embeddings = parameter['ignore_label_embeddings']
        self.is_zero_shot = parameter['zero_shot']

        # parameters
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.dataset_len_cap = parameter['dataset_len_cap']
        self.invalidate_cache = parameter['invalidate_cache']
        self.early_stopping_patience = parameter['early_stopping_patience']

        # step
        self.steps = parameter["epochs"] * parameter['dataset_len_cap']
        self.print_step = parameter['print_step']
        self.eval_step = parameter['eval_step']
        self.checkpoint_step = parameter['checkpoint_step']

        self.dataset_name = parameter['dataset']
        self.classification_only = self.parameter["classification_only"]

        self.shots = parameter['n_shots']  # k shots!
        self.ways = parameter['n_way']  # n way classification!

        self.device = parameter['device']

        if parameter["task_name"] == "temporal_link_prediction" and self.ways != 1:
            raise ValueError(
                "temporal_link_prediction now only supports binary LP episodes. "
                f"Use --n_way 1, got n_way={self.ways}."
            )

        if self.ways > 1:
            self.loss = torch.nn.CrossEntropyLoss()
            self.is_multiway = True
        elif self.ways == 1:
            self.loss = torch.nn.BCEWithLogitsLoss()  # binary classification (positives/negatives)
            self.is_multiway = False
        else:
            raise Exception("Invalid number of ways:", self.ways)

        self.calc_ranks = parameter['calc_ranks']
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._printed_eval_example = False
        self._printed_train_example = False

        bert_dim = 768

        self.emb_dim = parameter["emb_dim"]
        self.gnn_type = parameter["gnn_type"]
        self.original_features = parameter["original_features"]

        self.fix_datasets = self.parameter['fix_datasets_first']


        initial_label_mlp = torch.nn.Linear(bert_dim, self.emb_dim)
                                              
        edge_attr_dim = None
        if self.dataset_name in ["NELL", "ConceptNet", "FB15K-237", "Wiki", "WikiKG90M"]:
            edge_attr_dim = bert_dim
            self.parameter["input_dim"] = bert_dim + 2  # add 2 to flag head and tail nodes
            if self.parameter["task_name"] == "neighbor_matching":
                edge_attr_dim = bert_dim
            if self.parameter["task_name"] == "sn_neighbor_matching":
                edge_attr_dim = bert_dim
                self.parameter["input_dim"] = bert_dim
            if self.parameter["kg_emb_model"]:
                # if KG embedding model is set, we ignore the input_dim kwarg
                kg_embedding_dim = 100
                edge_attr_dim = kg_embedding_dim
                self.parameter["input_dim"] = kg_embedding_dim + 2  # add 2 to flag head and tail nodes
        if self.dataset_name in ["CSG"]:
            edge_attr_dim = 128
        if self.dataset_name == "midterm" and self.parameter.get("midterm_use_edge_features", False):
            midterm_edge_attr = getattr(dataset.graph, "edge_attr", None)
            if midterm_edge_attr is None:
                raise ValueError(
                    "midterm_use_edge_features=True but the loaded midterm graph has no edge_attr. "
                    "Check --midterm_edge_view / --midterm_edge_feature_subset and graph_data.pt contents."
                )
            edge_attr_dim = midterm_edge_attr.shape[1] if midterm_edge_attr.dim() > 1 else 1
            _log(f"Using midterm edge features with edge_attr_dim={edge_attr_dim}")

        self.txt_dropout = torch.nn.Dropout(self.parameter["text_features_dropout"])
        self.msg_pos_only = "meta_gnn_pos_only" in self.parameter and self.parameter["meta_gnn_pos_only"]
        if self.parameter["layers"] != "SimpleDotProduct":
            batch_norm_encoder = not self.parameter["no_bn_encoder"]
            batch_norm_metagraph = not self.parameter["no_bn_metagraph"]
            layer_list = get_module_list(self.parameter["layers"], self.emb_dim, edge_attr_dim=edge_attr_dim,
                                         input_dim=self.parameter["input_dim"], dropout=self.parameter["dropout"],
                                         reset_after_layer = self.parameter["reset_after_layer"],
                                         attention_mask_scheme = self.parameter["attention_mask_scheme"],
                                         has_final_back = self.parameter["has_final_back"],
                                         msg_pos_only=self.msg_pos_only,
                                         batch_norm_metagraph=batch_norm_metagraph,
                                         batch_norm_encoder=batch_norm_encoder,
                                         gnn_use_relu = self.dataset_name in ["NELL", "ConceptNet", "FB15K-237", "Wiki", "WikiKG90M"])

            layer_list = torch.nn.ModuleList(layer_list)
            self.model = SingleLayerGeneralGNN(layer_list=layer_list, initial_label_mlp=initial_label_mlp,  # initial_input_mlp = initial_input_mlp,
                                                 params=self.parameter, text_dropout=self.txt_dropout)
        else:
            from models.simple_dot_product import SimpleDotProdModel
            self.model = SimpleDotProdModel(layer_list=None, initial_label_mlp=initial_label_mlp,
                                            params=self.parameter, text_dropout=self.txt_dropout)
        print(self.model)
        self.model.to(self.device)
        num_params = print_num_trainable_params(self.model)
        # Add logging of # params to summary.json
        wandb.run.summary["num_params"] = num_params

        # create a header to predict masked node attribute
        if self.parameter["attr_regression_weight"]:
            embed_dim = self.emb_dim
            output_dim = self.parameter["input_dim"]
            self.aux_header = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, output_dim),
            )
            self.aux_header.to(self.device)
            self.aux_loss = torch.nn.MSELoss()
            self.aux_loss.to(self.device)

        bert_model_name = self.parameter["bert_emb_model"]
        # Twitter/midterm + numerical features does not need sentence embeddings and can
        # run with random label embeddings in the dataloader.
        if self.dataset_name in {"twitter", "midterm", "instagram_mention"} and self.original_features:
            self.Bert = None
        else:
            self.Bert = SentenceEmb(
                bert_model_name,
                device=self.device,
                cache_folder=os.path.join(self.parameter["root"], "sbert"),
            )

        params = list(self.model.parameters())
        if hasattr(self, "aux_header"):
            params += list(self.aux_header.parameters())
        if not self.parameter["not_freeze_learned_label_embedding"]:
            for param in self.model.learned_label_embedding.parameters():
                param.requires_grad = False

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params),
                                     lr=self.learning_rate, weight_decay=self.parameter["weight_decay"])

        # self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 0, self.steps)

        wandb.config.params = parameter
        wandb.watch(self.model, log_freq=100)

        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['exp_name'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        # Symlink to latest checkpoint
        self.wandb_fdir = os.path.join(self.state_dir, 'files')
        if not os.path.isdir(self.wandb_fdir):
            os.symlink(wandb.run.dir, self.wandb_fdir)

        self.ckpt_dir = os.path.join(self.state_dir, 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        self.logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['exp_name'], 'data')
        self.cache_dir = os.path.join(self.parameter['log_dir'], "cache")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if not os.path.isdir(self.logging_dir):
            os.makedirs(self.logging_dir)
        else:
            if self.parameter["override_log"]:
                print(f"Overwriting {self.logging_dir} logging dir!")
                shutil.rmtree(self.logging_dir)
                os.makedirs(self.logging_dir)
            else:
                raise Exception(f"{self.logging_dir} logging dir already exists!!!")

        self.all_saveable_modules = {
            "model": self.model
        }
        self.pretrained_model_run = self.parameter["pretrained_model_run"]
        if self.pretrained_model_run != "":
            _log(f"Reloading state dict from {self.pretrained_model_run}")
            self.load_checkpoint(self.pretrained_model_run)

        # Data loader creation.
        self.train_dataloader, self.train_val_dataloader, self.val_dataloader, self.test_dataloader = self._build_dataloaders(dataset, self.dataset_name)

    def _build_dataloaders(self, dataset, dataset_name):
        kwargs = {}
        kwargs["root"] = os.path.join(self.parameter["root"], dataset_name)
        kwargs["num_workers"] = self.parameter["workers"]
        kwargs["batch_size"] = self.parameter["batch_size"]
        kwargs["n_way"] = self.parameter["n_way"]
        kwargs["n_shot"] = self.parameter["n_shots"]
        kwargs["n_query"] = self.parameter["n_query"]
        kwargs["bert"] = self.Bert
        kwargs["task_name"] = self.parameter["task_name"]
        kwargs["aug"] = self.parameter["augmentation"]
        kwargs["aug_test"] = self.parameter["augment_test"]
        kwargs["split_labels"] = not self.parameter["no_split_labels"]
        kwargs["train_cap"] = self.parameter["train_cap"]
        kwargs['linear_probe'] = self.parameter['linear_probe']
        kwargs["csv_filename"] = self.parameter["csv_filename"]
        kwargs["label_type"] = self.parameter["label_type"]
        kwargs["max_users"] = self.parameter["max_users"]
        kwargs["pkl_filename"] = self.parameter["facebook_pkl_filename"]
        kwargs["facebook_edges_filename"] = self.parameter["facebook_edges_filename"]
        kwargs["facebook_node_features_filename"] = self.parameter["facebook_node_features_filename"]
        kwargs["facebook_data_source"] = self.parameter["facebook_data_source"]
        kwargs["facebook_use_edge_features"] = self.parameter["facebook_use_edge_features"]
        kwargs["facebook_edge_feature_columns"] = self.parameter["facebook_edge_feature_columns"]
        kwargs["source_pkl_path"] = self.parameter["facebook_source_pkl_path"]
        kwargs["facebook_embeddings_path"] = self.parameter["facebook_embeddings_path"]
        kwargs["facebook_embedding_ids_path"] = self.parameter["facebook_embedding_ids_path"]
        kwargs["facebook_text_emb_model"] = self.parameter["facebook_text_emb_model"]
        kwargs["facebook_target_dim"] = self.parameter["facebook_target_dim"]
        kwargs["facebook_filter_to_uk_ru"] = self.parameter["facebook_filter_to_uk_ru"]
        kwargs["max_posts"] = self.parameter["facebook_max_posts"]
        kwargs["midterm_feature_subset"] = self.parameter["midterm_feature_subset"]
        kwargs["midterm_edge_view"] = self.parameter["midterm_edge_view"]
        kwargs["midterm_target_edge_view"] = self.parameter["midterm_target_edge_view"]
        kwargs["midterm_edge_feature_subset"] = self.parameter["midterm_edge_feature_subset"]
        kwargs["neighbor_sampling_strategy"] = self.parameter["neighbor_sampling_strategy"]
        kwargs["midterm_lp_neg_ratio"] = self.parameter.get("midterm_lp_neg_ratio", 1)
        if self.parameter["all_test"]:
            kwargs["all_test"] = True
        if self.parameter["label_set"]:
            kwargs["label_set"] = set([int(v) for v in self.parameter["label_set"]])
            print("Label set:", kwargs["label_set"])
        if self.parameter["csr_split"]:
            kwargs["csr_split"] = self.parameter["csr_split"]
        if dataset_name == "arxiv":
            from data.arxiv import get_arxiv_dataloader
            get_dataloader = get_arxiv_dataloader
        elif dataset_name == "mag240m":
            from data.mag240m import get_mag240m_dataloader
            get_dataloader = get_mag240m_dataloader
        elif dataset_name in ["Wiki", "WikiKG90M"]: # "NELL", "FB15K-237", "ConceptNet",  by default still use legacy for them for now
            from data.kg import get_kg_dataloader
            get_dataloader = get_kg_dataloader
        elif dataset_name in [ "NELL", "FB15K-237", "ConceptNet"]: 
            assert self.parameter["task_name"] != "classification"
            from data.kg import get_kg_dataloader
            get_dataloader = get_kg_dataloader
        elif dataset_name == "twitter":
            from data.twitter_csv import get_twitter_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_twitter_dataloader
        elif dataset_name in {"facebook-uk_ru", "facebook_uk_ru"}:
            from data.facebook_uk_ru import get_facebook_uk_ru_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_facebook_uk_ru_dataloader
        elif dataset_name == "midterm":
            from data.midterm import get_midterm_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_midterm_dataloader
        elif dataset_name == "ukr_rus_twitter":
            from data.ukr_rus_twitter import get_ukr_rus_twitter_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_ukr_rus_twitter_dataloader
        elif dataset_name == "covid19_twitter":
            from data.covid19_twitter import get_covid19_twitter_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_covid19_twitter_dataloader
        elif dataset_name == "instagram_mention":
            from data.instagram_mention import get_instagram_mention_dataloader
            kwargs["root"] = self.parameter["root"]
            get_dataloader = get_instagram_mention_dataloader
        else:
            raise NotImplementedError

        val_batch_count = self.parameter["val_len_cap"] if self.parameter["val_len_cap"] is not None else self.parameter["dataset_len_cap"]
        test_batch_count = self.parameter["test_len_cap"] if self.parameter["test_len_cap"] is not None else self.parameter["dataset_len_cap"]

        val_dataloader = get_dataloader(dataset, split="val", node_split="", batch_count=val_batch_count, **kwargs)
        test_dataloader = get_dataloader(dataset, split="test", node_split="", batch_count=test_batch_count, **kwargs)

        train_val_dataloader = None
        train_node_split = ""
        if self.parameter["split_train_nodes"]:
            train_val_dataloader = get_dataloader(dataset, split="train", node_split="val", batch_count=val_batch_count, **kwargs)
            train_node_split = "train"

        # Update the n_way, n_shot, n_query parameters with range objects for the dataset
        # This is only done for train
        if self.parameter["n_way_upper"] > 0:
            kwargs["n_way"] = range(kwargs["n_way"], self.parameter["n_way_upper"] + 1)
        if self.parameter["n_shots_upper"] > 0:
            kwargs["n_shot"] = range(kwargs["n_shot"], self.parameter["n_shots_upper"] + 1)
        if self.parameter["n_query_upper"] > 0:
            kwargs["n_query"] = range(kwargs["n_query"], self.parameter["n_query_upper"] + 1)
        train_dataloader = get_dataloader(dataset, split="train", node_split=train_node_split, batch_count=self.parameter["dataset_len_cap"], **kwargs)
        return train_dataloader, train_val_dataloader, val_dataloader, test_dataloader


    def move_to_device(self, bt_response):
        return tuple([x.to(self.device) for x in bt_response])
        

    def get_loss_and_acc(self, y_true_matrix, y_pred_matrix):
        loss = self.loss(y_pred_matrix, y_true_matrix.float())
        if not self.is_multiway:
            p_score = y_pred_matrix[y_true_matrix == 1]
            n_score = y_pred_matrix[y_true_matrix == 0]
            if (
                self.parameter.get("task_name") != "temporal_link_prediction"
                and len(p_score) == len(n_score)
            ):
                y = torch.Tensor([1]).to(y_true_matrix.device)
                loss = torch.nn.MarginRankingLoss(0.5)(p_score, n_score, y)
            else:
                pass  # keep BCE for temporal LP or when pos/neg counts differ

        return loss, accuracy(y_true_matrix, y_pred_matrix, calc_roc=not self.is_multiway)[2]
    
    def get_hits(self, y_true_matrix, y_pred_matrix, task_mask):
        # get HITS@10, HITS@5, HITS@1, MRR scores
        tasks = task_mask.unique()
        n_tasks = len(tasks)
        yt, yp = y_true_matrix.cpu().numpy().flatten(), y_pred_matrix.cpu().numpy().flatten()
        data = {"Hits@10": 0, "Hits@5": 0, "Hits@1": 0, "MRR": 0}
        for i in range(n_tasks):
            where = torch.where(task_mask == tasks[i])[0].cpu()
            x = torch.tensor(yp[where])
            query_idx = np.where(yt[where] == 1)[0]
            _, idx = torch.sort(x, descending=True)
            rank = list(idx.cpu().numpy()).index(query_idx) + 1
            if rank <= 10:
                data['Hits@10'] += 1
            if rank <= 5:
                data['Hits@5'] += 1
            if rank == 1:
                data['Hits@1'] += 1
            data['MRR'] += 1.0 / rank
        for key in data:
            data[key] = data[key] / n_tasks
        return data

    def get_aux_loss(self, graph):
        if hasattr(graph, "node_attr_mask") and self.parameter["attr_regression_weight"]:
            mask = ~graph.node_attr_mask
            if hasattr(graph, "node_mask"):
                mask = mask.logical_and(graph.node_mask)
            target = graph.x_orig[mask]
            input = graph.x[mask]
            output = self.aux_header(input)
            loss = self.aux_loss(output, target)
            return loss
        return torch.zeros(1, device=self.device)

    def save_checkpoint(self, step):
        state_dict = {key: value.state_dict() for key, value in self.all_saveable_modules.items()}
        torch.save(state_dict, os.path.join(self.ckpt_dir, 'state_dict_' + str(step) + '.ckpt'))

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        for key, module in self.all_saveable_modules.items():
            module.load_state_dict(state_dict[key], strict=False)

    def _maybe_save_roc_curve(self, y_true_matrix, y_pred_matrix, split_name, step=None):
        if not self.parameter.get("save_roc_curve", False):
            return
        if self.is_multiway:
            return
        y_true = y_true_matrix.detach().cpu().reshape(-1).numpy()
        y_score = y_pred_matrix.detach().cpu().reshape(-1).numpy()
        if y_true.size == 0:
            return
        if len(np.unique(y_true)) < 2:
            return

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        try:
            import matplotlib.pyplot as plt

            suffix = split_name if step is None else f"{split_name}_step{step}"
            png_path = os.path.join(self.logging_dir, f"roc_{suffix}.png")
            csv_path = os.path.join(self.logging_dir, f"roc_{suffix}.csv")

            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"{split_name} ROC-AUC")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve ({split_name})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(png_path, dpi=160)
            plt.close(fig)

            np.savetxt(
                csv_path,
                np.column_stack([fpr, tpr, thresholds]),
                delimiter=",",
                header="fpr,tpr,threshold",
                comments="",
            )
        except Exception as ex:
            _log(f"Failed to save ROC curve for {split_name}: {ex}")

    def _maybe_print_debug_example(self, batch, yt, yp, graph, split_name, printed_attr, require_flag=False, raw_graph=None):
        if split_name == "train" and getattr(self, printed_attr):
            return
        max_eps = int(self.parameter.get("midterm_debug_print_episodes", 0) or 0)
        if require_flag and max_eps <= 0:
            return

        ytrue = yt.detach().cpu()
        ypred = yp.detach().cpu()
        center_nodes = None
        center_graph = raw_graph if raw_graph is not None else graph
        if hasattr(center_graph, "center_node_idx"):
            try:
                center_nodes = center_graph.center_node_idx.detach().cpu().flatten().tolist()
            except Exception:
                center_nodes = None

        if ypred.ndim > 1 and ypred.shape[-1] > 1:
            pred_idx = int(torch.argmax(ypred[0]).item())
            if ytrue.ndim > 1 and ytrue.shape[-1] > 1:
                true_idx = int(torch.argmax(ytrue[0]).item())
            else:
                true_idx = int(ytrue[0].item())
            print(
                f"[debug-example] split={split_name} sample=0 pred={pred_idx} gt={true_idx} "
                f"logits={ypred[0].tolist()}"
            )
            feat_str = self._format_debug_node_features(raw_graph if raw_graph is not None else graph, sample_idx=0)
            if feat_str is not None:
                center0 = int(center_nodes[0]) if center_nodes is not None and len(center_nodes) > 0 else "na"
                print(f"[debug-features] split={split_name} sample=0 center_node={center0} {feat_str}")

            if self.parameter.get("task_name", "") == "classification":
                try:
                    labels_onehot = batch[2].detach().cpu()
                    num_labels = int(labels_onehot.shape[1])
                    gt_label_idx = torch.argmax(labels_onehot, dim=1).long()
                    meta_mask = batch[5].detach().cpu().view(-1, num_labels)
                    query_mask = meta_mask[:, 0].bool()

                    total_items = int(gt_label_idx.numel())
                    if isinstance(self.batch_size, int) and self.batch_size > 0 and total_items % self.batch_size == 0:
                        task_len = total_items // self.batch_size
                    else:
                        task_len = total_items

                    gt_t = gt_label_idx[:task_len]
                    q_t = query_mask[:task_len]
                    pred_t = torch.argmax(ypred[:task_len], dim=1).long().cpu()
                    prob_t = torch.softmax(ypred[:task_len], dim=1).cpu()
                    label_emb = batch[1].detach().cpu()[:num_labels]

                    print(f"[debug-classification] first {split_name} task")
                    if hasattr(graph, "task_label_map"):
                        task_label_map = graph.task_label_map.detach().cpu()
                        if task_label_map.ndim == 2 and task_label_map.shape[0] > 0:
                            print(f"  local->global label map: {task_label_map[0].tolist()}")
                    print(f"  label embedding shape: {tuple(label_emb.shape)}")
                    for n in range(num_labels):
                        emb_preview = ", ".join(f"{v:.4f}" for v in label_emb[n][: min(8, label_emb.shape[1])].tolist())
                        print(f"  label N{n + 1} emb[:8]=[{emb_preview}]")

                    task_obj = getattr(getattr(self, f"{split_name}_dataloader", None), "batch_sampler", None)
                    if task_obj is not None:
                        task_obj = getattr(task_obj, "task", None)
                    orig_labels = getattr(task_obj, "original_graph_labels", None)
                    split_labels = getattr(task_obj, "split_masked_labels", None)
                    raw_ptr = raw_graph.get("ptr") if isinstance(raw_graph, dict) else None

                    def center_row_for_sample(sample_idx: int):
                        if raw_ptr is not None and sample_idx < int(raw_ptr.numel()) - 1:
                            return int(raw_ptr[sample_idx].item())
                        return sample_idx

                    s_count = 0
                    q_count = 0
                    for n in range(num_labels):
                        s_idx = torch.where((gt_t == n) & (~q_t))[0][:5]
                        q_idx = torch.where((gt_t == n) & q_t)[0][:5]
                        for i in s_idx.tolist():
                            s_count += 1
                            center_row = center_row_for_sample(i)
                            center_i = int(center_nodes[i]) if center_nodes is not None and i < len(center_nodes) else "na"
                            feat_i = self._format_debug_node_features(raw_graph if raw_graph is not None else graph, sample_idx=center_row, emb_preview=4)
                            raw_y_i = orig_labels[center_i] if isinstance(center_i, int) and orig_labels is not None else "na"
                            split_y_i = split_labels[center_i] if isinstance(center_i, int) and split_labels is not None else "na"
                            print(
                                f"  S{s_count}: idx={i} center_row={center_row} center={center_i} raw_y={raw_y_i} split_y={split_y_i} "
                                f"local_gt=N{n + 1} pred=N{int(pred_t[i].item()) + 1}"
                            )
                            if feat_i is not None:
                                print(f"    features: {feat_i}")
                        for i in q_idx.tolist():
                            q_count += 1
                            center_row = center_row_for_sample(i)
                            center_i = int(center_nodes[i]) if center_nodes is not None and i < len(center_nodes) else "na"
                            logits_i = [float(v) for v in ypred[i].tolist()]
                            probs_i = [float(v) for v in prob_t[i].tolist()]
                            feat_i = self._format_debug_node_features(raw_graph if raw_graph is not None else graph, sample_idx=center_row, emb_preview=4)
                            raw_y_i = orig_labels[center_i] if isinstance(center_i, int) and orig_labels is not None else "na"
                            split_y_i = split_labels[center_i] if isinstance(center_i, int) and split_labels is not None else "na"
                            print(
                                f"  Q{q_count}: idx={i} center_row={center_row} center={center_i} raw_y={raw_y_i} split_y={split_y_i} "
                                f"pred=N{int(pred_t[i].item()) + 1} -> gt=N{n + 1} "
                                f"logits={logits_i} probs={probs_i}"
                            )
                            if feat_i is not None:
                                print(f"    features: {feat_i}")
                except Exception as ex:
                    print(f"[debug-classification] failed to decode episode: {ex}")

            if self.parameter.get("task_name", "") == "neighbor_matching" and center_nodes is not None:
                try:
                    labels_onehot = batch[2].detach().cpu()
                    num_labels = int(labels_onehot.shape[1])
                    gt_label_idx = torch.argmax(labels_onehot, dim=1).long()
                    meta_mask = batch[5].detach().cpu().view(-1, num_labels)
                    query_mask = meta_mask[:, 0].bool()

                    total_items = int(gt_label_idx.numel())
                    if isinstance(self.batch_size, int) and self.batch_size > 0 and total_items % self.batch_size == 0:
                        task_len = total_items // self.batch_size
                    else:
                        task_len = total_items

                    centers_t = torch.tensor(center_nodes, dtype=torch.long)[:task_len]
                    gt_t = gt_label_idx[:task_len]
                    q_t = query_mask[:task_len]
                    pred_t = torch.argmax(ypred[:task_len], dim=1).long().cpu()

                    print(f"[debug-episode] first {split_name} task")
                    for n in range(num_labels):
                        print(f"N{n + 1}: {n}")
                    s_count = 0
                    q_count = 0
                    for n in range(num_labels):
                        s_idx = torch.where((gt_t == n) & (~q_t))[0][:5]
                        q_idx = torch.where((gt_t == n) & q_t)[0][:5]
                        for i in s_idx.tolist():
                            s_count += 1
                            print(f"S{s_count}: {int(centers_t[i].item())} (N{n + 1})")
                        for i in q_idx.tolist():
                            q_count += 1
                            pred_n = int(pred_t[i].item()) + 1
                            print(f"Q{q_count}: {int(centers_t[i].item())} (pred N{pred_n} -> gt N{n + 1})")
                except Exception as ex:
                    print(f"[debug-episode] failed to decode episode: {ex}")
        else:
            pred_val = float(ypred.flatten()[0].item())
            true_val = float(ytrue.flatten()[0].item())
            print(f"[debug-example] split={split_name} sample=0 pred={pred_val:.4f} gt={true_val:.4f}")
            feat_str = self._format_debug_node_features(raw_graph if raw_graph is not None else graph, sample_idx=0)
            if feat_str is not None:
                center0 = int(center_nodes[0]) if center_nodes is not None and len(center_nodes) > 0 else "na"
                print(f"[debug-features] split={split_name} sample=0 center_node={center0} {feat_str}")
            if self.parameter.get("task_name", "") == "temporal_link_prediction":
                try:
                    if (
                        center_nodes is not None
                        and hasattr(graph, "task_id_per_sample")
                        and hasattr(graph, "lp_task_center_ids")
                    ):
                        task_ids = graph.task_id_per_sample.detach().cpu().flatten().long()
                        task_centers = graph.lp_task_center_ids.detach().cpu().flatten().long()
                        top_k = min(len(center_nodes), int(task_ids.numel()), int(ytrue.shape[0]))
                        probs = torch.sigmoid(ypred[:top_k].flatten()).detach().cpu().tolist()
                        print(f"[debug-lp] {split_name} examples by episode:")
                        current_tid = None
                        for i in range(top_k):
                            tid = int(task_ids[i].item())
                            fcenter = int(task_centers[tid].item())
                            if tid != current_tid:
                                current_tid = tid
                                print(f"  [episode {tid}] future_center={fcenter}")
                            cand = int(center_nodes[i])
                            gt_i = float(ytrue[i].item()) if ytrue.ndim == 1 else float(ytrue[i].flatten()[0].item())
                            logit_i = float(ypred[i].flatten()[0].item())
                            prob_i = float(probs[i])
                            print(
                                f"    i={i} pair=({cand}->{fcenter}) gt={int(round(gt_i))} "
                                f"logit={logit_i:.4f} prob={prob_i:.4f}"
                            )
                except Exception as ex:
                    print(f"[debug-lp] failed to decode LP example: {ex}")
            if self.parameter.get("task_name", "") == "temporal_link_prediction" and max_eps > 0:
                try:
                    labels_all = batch[2].detach().cpu().reshape(-1)
                    query_mask_all = batch[5].detach().cpu().reshape(-1).bool()
                    if (
                        center_nodes is not None
                        and hasattr(graph, "task_id_per_sample")
                        and hasattr(graph, "lp_task_center_ids")
                    ):
                        task_ids = graph.task_id_per_sample.detach().cpu().reshape(-1).long()
                        task_centers = graph.lp_task_center_ids.detach().cpu().reshape(-1).long()
                        n_eps = min(max_eps, int(task_centers.numel()))

                        query_indices = torch.where(query_mask_all)[0].tolist()
                        qpos_to_pred = {int(idx): k for k, idx in enumerate(query_indices)}

                        print(f"[debug-lp-full] printing first {n_eps} {split_name} episode(s)")
                        for ep in range(n_eps):
                            ep_idx = torch.where(task_ids == ep)[0].tolist()
                            fut_center = int(task_centers[ep].item())
                            print(f"[debug-lp-full][episode {ep}] future_center={fut_center}")

                            support_idx = [i for i in ep_idx if not bool(query_mask_all[i].item())]
                            query_idx = [i for i in ep_idx if bool(query_mask_all[i].item())]

                            print("  supports:")
                            for i in support_idx:
                                cand = int(center_nodes[i])
                                gt_i = int(round(float(labels_all[i].item())))
                                print(f"    cand={cand} pair=({cand}->{fut_center}) gt={gt_i}")

                            print("  queries:")
                            for i in query_idx:
                                cand = int(center_nodes[i])
                                gt_i = int(round(float(labels_all[i].item())))
                                if i in qpos_to_pred:
                                    k = qpos_to_pred[i]
                                    logit_i = float(ypred[k].flatten()[0].item())
                                    prob_i = float(torch.sigmoid(ypred[k].flatten()[0]).item())
                                    print(
                                        f"    cand={cand} pair=({cand}->{fut_center}) gt={gt_i} "
                                        f"logit={logit_i:.4f} prob={prob_i:.4f}"
                                    )
                                else:
                                    print(f"    cand={cand} pair=({cand}->{fut_center}) gt={gt_i}")
                except Exception as ex:
                    print(f"[debug-lp-full] failed to print full episodes: {ex}")

        if split_name == "train":
            setattr(self, printed_attr, True)

    def _maybe_log_eval_diagnostics(self, y_true_matrix, y_pred_matrix, split_name, step=None):
        # Focus diagnostics on 2-way classification, where sudden accuracy jumps can
        # come from a smooth logit-margin shift crossing the argmax boundary.
        if split_name == "train":
            return
        if y_pred_matrix is None or y_pred_matrix.numel() == 0:
            return
        if y_pred_matrix.ndim != 2 or y_pred_matrix.shape[1] != 2:
            return
        if self.parameter.get("task_name") != "classification":
            return

        y_pred_cpu = y_pred_matrix.detach().cpu()
        if y_true_matrix.ndim > 1 and y_true_matrix.shape[-1] > 1:
            y_true_cpu = torch.argmax(y_true_matrix.detach().cpu(), dim=1).long()
        else:
            y_true_cpu = y_true_matrix.detach().cpu().reshape(-1).long()

        pred_cpu = torch.argmax(y_pred_cpu, dim=1).long()
        probs_cpu = torch.softmax(y_pred_cpu, dim=1)
        p1_cpu = probs_cpu[:, 1]
        conf_cpu = probs_cpu.max(dim=1).values
        logit_margin_cpu = y_pred_cpu[:, 1] - y_pred_cpu[:, 0]
        true_margin_cpu = torch.where(y_true_cpu == 1, logit_margin_cpu, -logit_margin_cpu)

        gt0 = y_true_cpu == 0
        gt1 = y_true_cpu == 1
        pred0 = pred_cpu == 0
        pred1 = pred_cpu == 1

        def _mean_or_none(x):
            if x.numel() == 0:
                return None
            return float(x.float().mean().item())

        tn = int((gt0 & pred0).sum().item())
        fp = int((gt0 & pred1).sum().item())
        fn = int((gt1 & pred0).sum().item())
        tp = int((gt1 & pred1).sum().item())

        diag = {
            f"{split_name}_pred_class1_rate": float(pred_cpu.float().mean().item()),
            f"{split_name}_gt_class1_rate": float(y_true_cpu.float().mean().item()),
            f"{split_name}_mean_confidence": float(conf_cpu.mean().item()),
            f"{split_name}_mean_p_class1": float(p1_cpu.mean().item()),
            f"{split_name}_mean_true_margin": float(true_margin_cpu.mean().item()),
            f"{split_name}_mean_logit_margin": float(logit_margin_cpu.mean().item()),
            f"{split_name}_tn": tn,
            f"{split_name}_fp": fp,
            f"{split_name}_fn": fn,
            f"{split_name}_tp": tp,
        }

        class0_acc = _mean_or_none(pred_cpu[gt0] == 0)
        class1_acc = _mean_or_none(pred_cpu[gt1] == 1)
        p1_gt0 = _mean_or_none(p1_cpu[gt0])
        p1_gt1 = _mean_or_none(p1_cpu[gt1])
        if class0_acc is not None:
            diag[f"{split_name}_class0_acc"] = class0_acc
        if class1_acc is not None:
            diag[f"{split_name}_class1_acc"] = class1_acc
        if p1_gt0 is not None:
            diag[f"{split_name}_mean_p_class1_gt0"] = p1_gt0
        if p1_gt1 is not None:
            diag[f"{split_name}_mean_p_class1_gt1"] = p1_gt1

        print(
            f"[diag-{split_name}] n={int(y_true_cpu.numel())} "
            f"cm=[[{tn},{fp}],[{fn},{tp}]] "
            f"pred1_rate={diag[f'{split_name}_pred_class1_rate']:.4f} "
            f"gt1_rate={diag[f'{split_name}_gt_class1_rate']:.4f} "
            f"class0_acc={class0_acc:.4f} "
            f"class1_acc={class1_acc:.4f} "
            f"mean_true_margin={diag[f'{split_name}_mean_true_margin']:.4f} "
            f"mean_conf={diag[f'{split_name}_mean_confidence']:.4f} "
            f"p1(gt0)={p1_gt0:.4f} "
            f"p1(gt1)={p1_gt1:.4f}",
            flush=True,
        )

        wandb.log(diag, step=0 if step is None else step)

    def _compute_eval_metrics(self, y_true_matrix, y_pred_matrix):
        metrics = {}
        if y_true_matrix is None or y_pred_matrix is None:
            return metrics

        yt = y_true_matrix.detach().cpu()
        yp = y_pred_matrix.detach().cpu()

        try:
            if yp.ndim == 1 or (yp.ndim == 2 and yp.shape[1] == 1):
                # Binary single-logit case (e.g. temporal_link_prediction).
                y_true = yt.reshape(-1).numpy().astype(int)
                y_score = torch.sigmoid(yp.reshape(-1)).numpy()
                y_pred = (y_score >= 0.5).astype(int)
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
                if len(np.unique(y_true)) >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
                return metrics

            # Multi-logit classification case.
            if yt.ndim > 1 and yt.shape[1] > 1:
                y_true = torch.argmax(yt, dim=1).numpy().astype(int)
            else:
                y_true = yt.reshape(-1).numpy().astype(int)

            y_pred = torch.argmax(yp, dim=1).numpy().astype(int)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

            avg = "binary" if yp.shape[1] == 2 else "macro"
            metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

            probs = torch.softmax(yp, dim=1).numpy()
            n_classes = int(probs.shape[1])
            if n_classes == 2:
                if len(np.unique(y_true)) >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            elif len(np.unique(y_true)) >= 2:
                true_1hot = np.eye(n_classes, dtype=np.float32)[y_true]
                metrics["roc_auc"] = float(
                    roc_auc_score(true_1hot, probs, multi_class="ovr", average="macro")
                )
        except Exception as ex:
            _log(f"Failed to compute eval metrics: {ex}")

        return metrics

    def _log_eval_metrics(self, metrics, split_name, step=None):
        if not metrics:
            return
        prefix_metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}
        wandb.log(prefix_metrics, step=0 if step is None else step)

        suffix = split_name if step is None else f"{split_name}_step{step}"
        out_path = os.path.join(self.logging_dir, f"metrics_{suffix}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(prefix_metrics, f, indent=2, sort_keys=True)
        except Exception as ex:
            _log(f"Failed to save metrics file {out_path}: {ex}")

        msg = " ".join(f"{k}={v:.4f}" for k, v in prefix_metrics.items())
        _log(f"[metrics] {msg}")

    def _format_debug_node_features(self, graph, sample_idx: int = 0, emb_preview: int = 8):
        try:
            if isinstance(graph, dict):
                x = graph.get("x")
                feature_names = graph.get("feature_names", None)
            else:
                x = graph.x if hasattr(graph, "x") else None
                feature_names = getattr(graph, "feature_names", None)

            if x is None or x.ndim != 2:
                return None
            if sample_idx < 0 or sample_idx >= int(x.shape[0]):
                return None

            x_row = x[sample_idx].detach().cpu().flatten()
            if not feature_names or len(feature_names) != int(x_row.numel()):
                feature_names = [f"f{i}" for i in range(int(x_row.numel()))]

            stat_pairs = []
            emb_pairs = []
            for name, val in zip(feature_names, x_row.tolist()):
                pair = f"{name}={val:.4f}"
                if str(name).startswith("emb_"):
                    emb_pairs.append(pair)
                else:
                    stat_pairs.append(pair)

            parts = []
            if stat_pairs:
                parts.append("stats=[" + ", ".join(stat_pairs) + "]")
            if emb_pairs:
                if len(emb_pairs) <= emb_preview * 2:
                    emb_str = ", ".join(emb_pairs)
                else:
                    emb_str = ", ".join(emb_pairs[:emb_preview]) + ", ..., " + ", ".join(emb_pairs[-emb_preview:])
                parts.append(f"emb[{len(emb_pairs)}]=[" + emb_str + "]")
            return " ".join(parts)
        except Exception as ex:
            return f"<failed to format node features: {ex}>"

    def _extract_raw_debug_graph(self, batch):
        try:
            return batch[0]
        except Exception:
            return None

    def _snapshot_debug_graph(self, batch):
        raw_graph = self._extract_raw_debug_graph(batch)
        if raw_graph is None:
            return None
        snap = {}
        try:
            if hasattr(raw_graph, "x") and raw_graph.x is not None:
                snap["x"] = raw_graph.x.detach().cpu().clone()
        except Exception:
            pass
        try:
            if hasattr(raw_graph, "center_node_idx") and raw_graph.center_node_idx is not None:
                snap["center_node_idx"] = raw_graph.center_node_idx.detach().cpu().clone()
        except Exception:
            pass
        try:
            if hasattr(raw_graph, "ptr") and raw_graph.ptr is not None:
                snap["ptr"] = raw_graph.ptr.detach().cpu().clone()
        except Exception:
            pass
        try:
            feature_names = getattr(raw_graph, "feature_names", None)
            if feature_names is not None:
                snap["feature_names"] = list(feature_names)
        except Exception:
            pass
        return snap


    def save_best_state_dict(self, best_step):
        best_step = os.path.join(self.ckpt_dir, 'state_dict_' + str(best_step) + '.ckpt')
        best_ckpt = os.path.join(self.state_dir, 'state_dict')
        if os.path.exists(best_step):
            shutil.copy(best_step, best_ckpt)
        else:
            print('No such best checkpoint to copy: {}. Saving current model state instead.'.format(best_step))
            state_dict = {key: value.state_dict() for key, value in self.all_saveable_modules.items()}
            torch.save(state_dict, best_ckpt)
        print("Saved best model to {}".format(best_ckpt))
        self.best_state_dict_path = best_ckpt

    def train(self):

        # initialization
        best_step = 0
        best_val = 0
        test_acc_on_best_val = 0
        best_test_acc = 0
        other_metrics_on_best = {}
        bad_counts = 0

        # training by step
        t_load, t_one_step = 0, 0
        train_dataloader_itr = iter(self.train_dataloader)

        bad_counts = 0

        def prefix_dict(d, prefix):
            return {prefix + key: value for key, value in d.items()}

        run_test_before_train = bool(self.parameter.get("eval_test_before_train", False))
        run_val_before_train = bool(self.parameter.get("eval_val_before_train", False))
        eval_only = bool(self.parameter.get("eval_only", False))

        if run_test_before_train or eval_only:
            with torch.no_grad():
                _log("Pre-training eval on test set...")
                test_loss, test_acc, test_acc_std, test_aux_loss, ranks = self.do_eval(self.test_dataloader, split_name="test", step=0)
                _log(f"  [pre-train test]  acc={_to_float(test_acc):.4f} ± {_to_float(test_acc_std):.4f}  loss={_to_float(test_loss):.4f}")
                start_log_dict = {"start_test_acc": test_acc, "start_test_acc_std": test_acc_std}
                if ranks is not None:
                    for key in ranks:
                        start_log_dict["start_test_" + key] = ranks[key]
                wandb.log(start_log_dict, step=0)

        if eval_only:
            _log("Evaluation only — done.")
            wandb.finish()
            return

        if run_val_before_train:
            with torch.no_grad():
                _log("Pre-training eval on val set...")
                val_loss, val_acc, val_acc_std, val_aux_loss, ranks = self.do_eval(self.val_dataloader, split_name="val", step=0)
                _log(f"  [pre-train val]   acc={_to_float(val_acc):.4f} ± {_to_float(val_acc_std):.4f}  loss={_to_float(val_loss):.4f}")
                start_log_dict = {"start_val_acc": val_acc, "start_val_acc_std": val_acc_std}
                if ranks is not None:
                    for key in ranks:
                        start_log_dict["start_val_" + key] = ranks[key]
                wandb.log(start_log_dict, step=0)

        pbar = trange(self.steps)
        for e in pbar:
            self.model.train()

            self.optimizer.zero_grad()

            t1 = time.time()
            try:
                batch = next(train_dataloader_itr)
            except StopIteration:
                train_dataloader_itr = iter(self.train_dataloader)
                batch = next(train_dataloader_itr)
            t2 = time.time()
            batch = [i.to(self.device) for i in batch]
            raw_debug_graph = self._snapshot_debug_graph(batch)
            yt, yp, graph = self.model(*batch) # apply the model
            self._maybe_print_debug_example(
                batch,
                yt,
                yp,
                graph,
                split_name="train",
                printed_attr="_printed_train_example",
                require_flag=True,
                raw_graph=raw_debug_graph,
            )
            loss, acc = self.get_loss_and_acc(yt, yp) # get loss
            aux_loss = self.get_aux_loss(graph)
            weight = self.parameter["attr_regression_weight"]
            total_loss = loss + aux_loss * weight
            total_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            t3 = time.time()
            wandb.log({"step_time": _to_float(t3 - t2)}, step=e)
            wandb.log({"load_time": _to_float(t2 - t1)}, step=e)
            wandb.log(
                {
                    "train_loss": _to_float(loss),
                    "train_acc": _to_float(acc),
                    "train_aux_loss": _to_float(aux_loss),
                    "train_total_loss": _to_float(total_loss),
                },
                step=e,
            )
            t_load += t2 - t1
            t_one_step += t3 - t2
            pbar.set_postfix(
                loss=f"{_to_float(loss):.4f}",
                acc=f"{_to_float(acc):.4f}",
                aux=f"{_to_float(aux_loss):.4f}",
                load=f"{(t2-t1):.2f}s",
                step=f"{(t3-t2):.2f}s",
            )
            # save checkpoint on specific step
            if e % self.checkpoint_step == 0 and e != 0:
                pbar.write(f"[{time.strftime('%H:%M:%S')}] [step {e}] saving checkpoint...")
                self.save_checkpoint(e)

            if e % self.eval_step == 0 and e != 0:
                should_stop = False
                # pbar.write("Evaluating on validation set!")
                with torch.no_grad():
                    self.model.eval()
                    val_loss, val_acc, val_acc_std, val_aux_loss, ranks = self.do_eval(self.val_dataloader, split_name="val", step=e)

                if val_acc >= best_val:
                    best_val = val_acc
                    best_step = e
                    bad_counts = 0
                    self.save_checkpoint(best_step)  # save the best checkpoint
                else:
                    bad_counts += 1
                    pbar.write(f"[{time.strftime('%H:%M:%S')}] [step {e}] val acc did not improve ({bad_counts} checks without improvement)")
                    should_stop = bad_counts >= self.early_stopping_patience

                pbar.write(f"[{time.strftime('%H:%M:%S')}] [step {e}] val  acc={_to_float(val_acc):.4f} ± {_to_float(val_acc_std):.4f}  loss={_to_float(val_loss):.4f}  aux={_to_float(val_aux_loss):.4f}")
                wandb.log({"valid_loss": _to_float(val_loss), "valid_acc": _to_float(val_acc), "valid_aux_loss": _to_float(val_aux_loss)},
                          step=e)

                if self.train_val_dataloader is not None:
                    with torch.no_grad():
                        self.model.eval()
                        tval_loss, tval_acc, tval_acc_std, tval_aux_loss, ranks = self.do_eval(self.train_val_dataloader, split_name="train_val", step=e)
                        wandb.log({"train_val_loss": _to_float(tval_loss), "train_val_acc": _to_float(tval_acc), "train_val_aux_loss": _to_float(tval_aux_loss)}, step=e)

                # Also evaluate on test set
                with torch.no_grad():
                    self.model.eval()
                    test_loss, test_acc, test_acc_std, test_aux_loss, ranks = self.do_eval(self.test_dataloader, split_name="test", step=e)
                    log_dict = {
                        "test_acc": _to_float(test_acc),
                        "test_loss": _to_float(test_loss),
                        "test_aux_loss": _to_float(test_aux_loss),
                        "test_acc_std": _to_float(test_acc_std),
                    }
                    #print("Logging", log_dict)
                    #wandb.log(log_dict, step=e)
                    if ranks is not None:
                        ranks_dict = prefix_dict(ranks, "test_")
                        log_dict.update(ranks_dict)
                    wandb.log(log_dict, step=e)
                    pbar.write(f"[{time.strftime('%H:%M:%S')}] [step {e}] test acc={_to_float(test_acc):.4f} ± {_to_float(test_acc_std):.4f}  loss={_to_float(test_loss):.4f}")
                    best_test_acc = max(best_test_acc, test_acc)
                    if e == best_step:
                        test_acc_on_best_val = test_acc
                        if ranks is not None:
                            other_metrics_on_best = ranks
                if should_stop:
                    pbar.write(f"[{time.strftime('%H:%M:%S')}] Early stopping at step {e}")
                    break
        _log("Training finished")
        print(f"  best step:             {best_step}", flush=True)
        print(f"  best val acc:          {_to_float(best_val):.4f}", flush=True)
        print(f"  best test acc:         {_to_float(best_test_acc):.4f}", flush=True)
        print(f"  test acc @ best val:   {_to_float(test_acc_on_best_val):.4f}", flush=True)
        wandb.run.summary["best_step"] = best_step
        wandb.run.summary["best_test_acc"] = best_test_acc
        wandb.run.summary["test_acc_on_best_val"] = test_acc_on_best_val
        wandb.run.summary["final_validation_acc"] = best_val
        if other_metrics_on_best is not None:
              for key in other_metrics_on_best:
                  wandb.run.summary["final_test_" + key] = other_metrics_on_best[key]
        self.save_best_state_dict(best_step)
        wandb.finish()
        return best_val, test_acc_on_best_val, best_step
        # returns best-val-acc, best-test-acc, best-step

    def do_eval(self, dataloader, eff_len=None, split_name="eval", step=None):
        # calc_ranks: if True, it will calculate MRR, HITS scores etc.
        torch.set_grad_enabled(False)  # disable gradient calculation
        ranks = None
        if self.calc_ranks:
            ranks = []
        ytrueall, ypredall = None, None
        all_aux_loss = []
        acc_all = []
        printed_debug_this_eval = False
        for batch in tqdm(dataloader, leave=False):
            batch = [i.to(self.device) for i in batch]
            raw_debug_graph = self._snapshot_debug_graph(batch)
            yt, yp, graph = self.model(*batch)  # apply the model
            if not printed_debug_this_eval:
                self._maybe_print_debug_example(
                    batch,
                    yt,
                    yp,
                    graph,
                    split_name=split_name,
                    printed_attr="_printed_eval_example",
                    require_flag=False,
                    raw_graph=raw_debug_graph,
                )
                printed_debug_this_eval = True
            if self.calc_ranks:
                assert len(batch) == 10, "Not using the right batch structure; need to include task_mask"
            loss, acc = self.get_loss_and_acc(yt, yp)  # get loss
            acc_all.append(acc)
            aux_loss = self.get_aux_loss(graph)
            if self.calc_ranks:
                task_mask = batch[9]
                query_set_mask = batch[5]
                query_set_mask = torch.where(query_set_mask == 1)[0]
                curr_ranks = self.get_hits(yt, yp, task_mask[query_set_mask])
                ranks.append([curr_ranks, len(task_mask[query_set_mask.unique()])])  # append values and weights

            # If using random sampling as with MultiTaskSplitWay, need to doubly sample labels to avoid shape dim mismatch
            if ytrueall is None:
                ytrueall = yt
                ypredall = yp
            else:
                ytrueall = torch.cat((ytrueall, yt), dim=0)
                ypredall = torch.cat((ypredall, yp), dim=0)
            all_aux_loss.append(aux_loss.item())
        loss_global, acc_global = self.get_loss_and_acc(ytrueall, ypredall)
        eval_metrics = self._compute_eval_metrics(ytrueall, ypredall)
        self._maybe_log_eval_diagnostics(ytrueall, ypredall, split_name=split_name, step=step)
        self._maybe_save_roc_curve(ytrueall, ypredall, split_name=split_name, step=step)
        self._log_eval_metrics(eval_metrics, split_name=split_name, step=step)
        acc_batch_std = np.std(acc_all)
        aux_loss_global = sum(all_aux_loss) / len(all_aux_loss)
        torch.set_grad_enabled(True)
        if ranks is not None:
            ranks = {key: np.average([r[0][key] for r in ranks], weights=[r[1] for r in ranks]) for key in ranks[0][0]}
        return loss_global, acc_global, acc_batch_std, aux_loss_global, ranks
