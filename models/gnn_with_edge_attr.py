'''
    Contains PyG-compatible implementations of various generic GNNs and their extensions (e.g. for edge attributes)
'''
import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing, GATConv, GATv2Conv
from torch_geometric.utils import add_self_loops
from models.layer_classes import BackgroundGNNLayer, MetagraphLayer

def obtain_supernode_embeddings(all_node_emb, supernode_edge_index, supernode_idx, aggr='mean'):
    '''
    A simple aggregator to obtain supernode embeddings.
    :param all_node_emb:
    :param supernode_edge_index:
    :param supernode_idx:
    :param aggr:
    :return:
    '''
    return scatter(src=all_node_emb[supernode_edge_index[0]], index=supernode_edge_index[1], dim=0, reduce=aggr)[
           supernode_idx, :]
    
class NoMessagePassing(torch.nn.Module):
    """
    MLP only on the node features - no message passing
    """

    def __init__(self, x_dim, edge_attr_dim, emb_dim, aggr="add", transform_x=True):
        super().__init__()

        self.x_mlp = torch.nn.Sequential(torch.nn.Linear(x_dim, emb_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(emb_dim, 2*emb_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(2*emb_dim, emb_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_attr=None):
        x = self.x_mlp(x)
        return x



class BipartiteMsgPassingGNN(MessagePassing, BackgroundGNNLayer):
    """
    Very simple aggregation GNN. We are using it for very very simple message passing on the metagraph.

    Args:
        edge_attr_dim (int): dimension of edge attributes (2 for metagraph)
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggregation method. Default: "add".
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_attr_dim, emb_dim, aggr="add"):
        super(BipartiteMsgPassingGNN, self).__init__()
        # multi-layer perceptron
        self.mlp_left = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.mlp_right = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        if edge_attr_dim is None:
            self.mlp_edge_attr = None
        else:
            self.mlp_edge_attr = torch.nn.Sequential(torch.nn.Linear(edge_attr_dim, emb_dim),
                                                     torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
        #                                torch.nn.Linear(2 * emb_dim, emb_dim))
        self.aggr = aggr

    def forward(self, x, edge_index, start_right, edge_attr=None):
        # start_right: starting idx of the right side nodes of the bipartite graph
        x_transformed = torch.cat([self.mlp_left(x[:start_right]), self.mlp_right(x[start_right:])], dim=0)
        if self.mlp_edge_attr is None:
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed)
        else:
            edge_attr_emb = self.mlp_edge_attr(edge_attr)
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed, edge_attr=edge_attr_emb)
        x_msg += x  # original "self-loops"
        return x_msg

    def message(self, x_j, edge_attr=None):
        if self.mlp_edge_attr is None:
            return x_j
        else:
            return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out


class BipartiteGAT(MessagePassing, MetagraphLayer):
    """
    GAT gnn for bipartite graph.
    Args:
        edge_attr_dim (int): dimension of edge attributes (2 for metagraph)
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggregation method. Default: "add".
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_attr_dim, emb_dim, aggr="add"):
        super(BipartiteGAT, self).__init__()
        # multi-layer perceptron
        self.mlp_left = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.mlp_right = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # if edge_attr_dim is None:
        #     self.mlp_edge_attr = None
        # else:
        #     self.mlp_edge_attr = torch.nn.Sequential(torch.nn.Linear(edge_attr_dim, emb_dim),
        #                                              torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
        #                                torch.nn.Linear(2 * emb_dim, emb_dim))
        self.aggr = aggr
        self.gat = GATConv(in_channels=emb_dim, out_channels=emb_dim, heads=2, add_self_loops=False, concat=False,
                           edge_dim=edge_attr_dim)


    def forward(self, x, edge_index, start_right, edge_attr=None):
        # start_right: starting idx of the right side nodes of the bipartite graph
        x_transformed = torch.cat([self.mlp_left(x[:start_right]), self.mlp_right(x[start_right:])], dim=0)
        # if self.mlp_edge_attr is None:
        #     # x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed)
        #     x_msg = self.gat(x=x_transformed, edge_index=edge_index, edge_attr=edge_attr)
        # else:
            # edge_attr_emb = self.mlp_edge_attr(edge_attr)
            # x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed, edge_attr=edge_attr_emb)
        x_msg = self.gat(x=x_transformed, edge_index=edge_index, edge_attr=edge_attr)
        x_msg += x  # original "self-loops"
        return x_msg



class SAGEConvSelfLoops(MessagePassing):
    """
    Extension of SAGE aggregation to incorporate edge information by concatenation.
    Self-loops are accounted for with a different projection.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, x_dim, edge_attr_dim, emb_dim, dropout = 0, aggr="add", transform_x=True, batch_norm=True):
        super(SAGEConvSelfLoops, self).__init__()
        # multi-layer perceptron
        self.transform_x = transform_x
        self.lin_x = torch.nn.Linear(x_dim, emb_dim)
        if transform_x:
            self.lin_self_loops = torch.nn.Linear(x_dim, emb_dim)
        if edge_attr_dim is None:
            #  do not use edge_attr
            self.lin_edge_attr = None
        else:
            self.lin_edge_attr = torch.nn.Linear(edge_attr_dim, emb_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.aggr = aggr

        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.Identity()
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x_transformed = self.lin_x(x)
        if self.lin_edge_attr is None:
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed)
        else:
            edge_attr_emb = self.lin_edge_attr(edge_attr)
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed, edge_attr=edge_attr_emb)
        if self.transform_x:
            x_msg += self.lin_self_loops(x)
        
        if x.shape[1] == x_msg.shape[1]:
            x_msg = self.dropout(x_msg) + x
        x_msg = self.bn(x_msg)

        return x_msg

    def message(self, x_j, edge_attr=None):
        if self.lin_edge_attr is None:
            return x_j
        else:
            return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINConv(MessagePassing):
    # NEED TO ADD SELF-LOOPS!
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, x_dim, edge_attr_dim, emb_dim, aggr="add", transform_x=True):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        # if transform_x:
        self.lin_x = torch.nn.Linear(x_dim, emb_dim)
        if edge_attr_dim is None:
            #  do not use edge_attr
            self.lin_edge_attr = None
        else:
            self.lin_edge_attr = torch.nn.Linear(edge_attr_dim, emb_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin_x(x)
        if self.lin_edge_attr is None:
            return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x)
        else:
            edge_attr_emb = self.lin_edge_attr(edge_attr)
            return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_attr_emb)

    def message(self, x_j, edge_attr=None):
        if self.lin_edge_attr is None:
            return x_j
        else:
            return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3


num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class SimpleMoleculeGNN(MessagePassing):
    """
    From github.com/snap-stanford/pretrain-gnns
    Molecule encoder that works with original molecule categorical features.
    """

    def __init__(self, x_dim=None, edge_attr_dim=None, emb_dim=256, aggr="add"):
        #  Ignore x_dim and edge_attr_dim!!!
        super(SimpleMoleculeGNN, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        self.atom_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.atom_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        x = x.long()
        x = self.atom_embedding1(x[:, 1]) + self.atom_embedding2(x[:, 2])

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4   #  Bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(aggr=self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class SimpleSupernodePoolingGNN:
    # No parameters - very simple supernode pooling (features are projected beforehand with a GNN anyway)
    def __call__(self, x, supernode_edge_index, supernode_idx):
        return obtain_supernode_embeddings(x, supernode_edge_index, supernode_idx)


class GNNWithSupernodePooling(torch.nn.Module, BackgroundGNNLayer):
    '''
        Background GNN and supernode pooling in one step.
        This is a very simple version!
    '''

    def __init__(self, background_gnn, supernode_pooling_gnn=None):
        super().__init__()
        self.background_gnn = background_gnn
        self.supernode_pooling_gnn = supernode_pooling_gnn

    def forward(self, x, edge_index, edge_attr, supernode_edge_index=None, supernode_idx=None):
        x = self.background_gnn(x, edge_index, edge_attr, supernode_edge_index)
        if self.supernode_pooling_gnn is not None:
            x = self.supernode_pooling_gnn(x, supernode_edge_index, supernode_idx)
        return x


class MultiAggSAGE(MessagePassing):
    """E2 (core): SAGEConvSelfLoops with mean-aggregation replaced by
    multi-aggregation (mean + sum + max), so counts / existence / fractions become
    representable — mean alone is count-/degree-blind (the bottleneck the E1
    findings pointed at). The concatenated 3x aggregate is projected back to
    emb_dim; the SAGE self-loop + residual + BatchNorm shell is unchanged. Pairs
    with the multi-readout (mean+sum+max) in MultiLayerGNN.

    (The directed in/out split from the plan's E2 is deferred: these subgraphs are
    sampled with bidirectional=False and carry no reliable per-edge direction flag,
    and E1 already injects directed degree as INPUT features.)
    """

    AGGRS = ["mean", "sum", "max"]

    def __init__(self, x_dim, edge_attr_dim, emb_dim, dropout=0, aggr="mean",
                 transform_x=True, batch_norm=True):
        super().__init__(aggr=self.AGGRS)  # concat of the 3 aggregations -> [N, 3*emb]
        self.transform_x = transform_x
        self.lin_x = torch.nn.Linear(x_dim, emb_dim)
        if transform_x:
            self.lin_self_loops = torch.nn.Linear(x_dim, emb_dim)
        self.lin_edge_attr = None if edge_attr_dim is None else torch.nn.Linear(edge_attr_dim, emb_dim)
        self.lin_agg = torch.nn.Linear(len(self.AGGRS) * emb_dim, emb_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(emb_dim) if batch_norm else torch.nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        x_transformed = self.lin_x(x)
        if self.lin_edge_attr is None or edge_attr is None:
            x_msg = self.propagate(edge_index, x=x_transformed)
        else:
            x_msg = self.propagate(edge_index, x=x_transformed, edge_attr=self.lin_edge_attr(edge_attr))
        x_msg = self.lin_agg(x_msg)  # [N, 3*emb] -> [N, emb]
        if self.transform_x:
            x_msg = x_msg + self.lin_self_loops(x)
        x_msg = self.mlp(x_msg)
        if x.shape[1] == x_msg.shape[1]:
            x_msg = self.dropout(x_msg) + x
        return self.bn(x_msg)

    def message(self, x_j, edge_attr=None):
        return x_j if edge_attr is None else x_j + edge_attr


class GATv2ConvOptionalEdgeAttr(GATv2Conv):
    """GATv2 that ignores supplied edge attributes when edge features are disabled.

    PRODIGY batches retain the graph's ``edge_attr`` tensor even when
    ``use_edge_features=False``. Other background encoders already ignore that
    tensor when constructed with ``edge_attr_dim=None``. PyG's GATv2Conv instead
    asserts if it receives edge attributes without an ``edge_dim`` projection, so
    normalize that case to ``edge_attr=None`` while preserving native behavior
    whenever edge features were explicitly enabled.
    """

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if self.lin_edge is None:
            edge_attr = None
        return super().forward(x, edge_index, edge_attr=edge_attr, **kwargs)


gnn_models = {
    "gin": GINConv,
    "no_msg_passing": NoMessagePassing,
    "sage": SAGEConvSelfLoops,
    "sage_multi": MultiAggSAGE,
    "molecule_sage": SimpleMoleculeGNN,
    "gat": GATv2ConvOptionalEdgeAttr
}
