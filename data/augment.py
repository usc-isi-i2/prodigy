import random
import copy
import torch


class AugBase:
    def __call__(self, graph):
        raise NotImplementedError


class Compose(AugBase):
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, graph):
        for aug in self.augs:
            graph = aug(graph)
        return graph


class Identity(AugBase):
    def __call__(self, graph):
        return graph


class DropNode(AugBase):
    def __init__(self, drop_percent=0.3):
        self.drop_percent = drop_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.drop_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False
        node_mask[0] = True  # center node
        node_mask[-1] = True  # super node

        graph = copy.copy(graph)
        edge_index = graph.edge_index
        edge_mask = (node_mask[edge_index[0]]).logical_and(node_mask[edge_index[1]])
        edge_index = edge_index[:, edge_mask]
        if "edge_attr" in graph and graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[edge_mask]
        graph.edge_index = edge_index
        graph.node_mask = node_mask
        return graph


class ZeroNodeAttr(AugBase):
    def __init__(self, mask_percent=0.3):
        self.mask_percent = mask_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.mask_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False

        graph = copy.copy(graph)
        graph.x_orig = graph.x
        graph.x = graph.x * node_mask.unsqueeze(1)
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        return graph


class RandomNodeAttr(AugBase):
    def __init__(self, distribution, mask_percent=0.3):
        self.distribution = distribution
        self.mask_percent = mask_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.mask_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False

        graph = copy.copy(graph)
        graph.x_orig = graph.x
        graph.x = graph.x.clone()
        random_idx = random.sample(range(self.distribution.size(0)), len(node_drop))
        graph.x[node_drop] = self.distribution[random_idx].float()
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        return graph


class AblateAllFeatures(AugBase):
    """Deterministically remove *all* node-feature information from a subgraph.

    Unlike ``ZeroNodeAttr`` / ``RandomNodeAttr`` (stochastic, per-node, tied to the
    training aug pipeline), this ablates every node's features and is meant as a
    controlled eval-time intervention: comparing intact vs. ablated eval accuracy
    isolates how much a task relies on node features vs. graph topology.

    mode="zero"    -> replace every node feature with zeros.
    mode="permute" -> permute feature rows across nodes within the subgraph. This
                      preserves the feature *distribution* of the subgraph while
                      destroying the node<->feature alignment, so any residual
                      signal must come from topology, not from which node holds
                      which feature.
    """

    def __init__(self, mode="zero"):
        if mode not in ("zero", "permute"):
            raise ValueError(f"AblateAllFeatures mode must be 'zero' or 'permute', got {mode}")
        self.mode = mode

    def __call__(self, graph):
        graph = copy.copy(graph)
        graph.x_orig = graph.x
        if self.mode == "zero":
            graph.x = torch.zeros_like(graph.x)
        else:  # permute
            perm = torch.randperm(graph.x.size(0))
            graph.x = graph.x[perm]
        return graph


def get_aug(aug_spec, node_feature_distribution=None):
    if not aug_spec:
        return Identity()
    augs = []
    for spec in aug_spec.split(","):
        if spec.startswith("ND"):
            augs.append(DropNode(float(spec[2:])))
        elif spec.startswith("NZ"):
            augs.append(ZeroNodeAttr(float(spec[2:])))
        elif spec.startswith("NR"):
            if node_feature_distribution is None:
                raise ValueError(f"node_feature_distribution not defined for RandomNodeAttr")
            augs.append(RandomNodeAttr(node_feature_distribution, float(spec[2:])))
        elif spec == "FZ":  # ablate all features -> zero
            augs.append(AblateAllFeatures("zero"))
        elif spec == "FP":  # ablate all features -> within-subgraph permutation
            augs.append(AblateAllFeatures("permute"))
        else:
            raise ValueError(f"Unknown augmentation {spec}")
    return Compose(augs)
