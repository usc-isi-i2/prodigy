"""Explicit S,U,M construction; production classes and state-dict keys are retained."""
import torch

from .models.general_gnn import SingleLayerGeneralGNN
from .models.gnn_with_edge_attr import SAGEConvSelfLoops
from .models.multilayer_gnn import MultiLayerGNN
from .models.supernode_propagation_layers import BgGraphToSupernodePropagator
from .models.metaGNN import MetaGNN
from .message_content import aggregation_audit

PARAM_KEYS = ('layers', 'emb_dim', 'gnn_type', 'dropout', 'reset_after_layer',
              'has_final_back', 'meta_gnn_pos_only', 'no_bn_metagraph', 'no_bn_encoder',
              'text_features_dropout', 'zero_shot', 'skip_path', 'ignore_label_embeddings',
              'zero_label_embeddings', 'task_name')


def make_model(config, state=None):
    params = config['params']
    if set(params) != set(PARAM_KEYS):
        raise ValueError('Explicit complete inference parameter dictionary required')
    if params['layers'] != 'S,U,M' or params['gnn_type'] != 'sage' or params['task_name'] != 'classification':
        raise ValueError('This release supports the studied S,U,M classification architecture')
    if any(params[k] for k in ('has_final_back', 'zero_shot', 'skip_path', 'ignore_label_embeddings', 'zero_label_embeddings')):
        raise ValueError('Unsupported label interface or reverse/skip pathway')
    if params['reset_after_layer'] not in (None, []):
        raise ValueError('Reset layers are outside this inference contract')
    dim = params['emb_dim']
    conv = SAGEConvSelfLoops(x_dim=config['feature_dim'], edge_attr_dim=None, emb_dim=dim,
                            dropout=params['dropout'], aggr='mean', batch_norm=not params['no_bn_encoder'])
    layers = [MultiLayerGNN(torch.nn.ModuleList([conv]), emb_dim=dim, reset_after_layer=params['reset_after_layer']),
              BgGraphToSupernodePropagator(),
              MetaGNN(emb_dim=dim, edge_attr_dim=2, n_layers=1, heads=8, dropout=params['dropout'],
                      has_final_back=False, msg_pos_only=params['meta_gnn_pos_only'],
                      batch_norm=not params['no_bn_metagraph'], gat_layer=False, use_relu=False)]
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers),
        initial_label_mlp=torch.nn.Linear(config['label_dim'], dim), params=params,
        text_dropout=torch.nn.Dropout(params['text_features_dropout'])).eval()
    if state is not None:
        model.load_state_dict(state, strict=True)
    audit = aggregation_audit(model)
    if audit != [{'layer': 0, 'configured_aggr': 'mean', 'aggregation_module': 'SumAggregation',
                  'unit_message_probe': [1., 2., 0.]}]:
        raise ValueError('Runtime aggregation differs from the evaluated implementation')
    return model
