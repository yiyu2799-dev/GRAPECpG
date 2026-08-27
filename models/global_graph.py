"""GRAPE-based global graph block.

This module contains the edge-enhanced GraphSAGE layer and the stacked global
encoder used by the paper model.  The class/attribute layout intentionally
preserves the legacy V11 state_dict keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


def _get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name in {None, 'none'}:
        return nn.Identity()
    raise ValueError(f'Unsupported activation: {name}')


class EGraphSage(MessagePassing):
    """Edge-enhanced GraphSAGE message passing used by GRAPE-CpG.

    Message: activation(Linear([neighbor_node || edge]))
    Update:  activation(Linear([aggregated_message || current_node]))
    """

    def __init__(self, in_channels, out_channels, edge_channels, activation='relu', normalize_emb=True, aggr='mean'):
        super().__init__(aggr=aggr)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_channels = int(edge_channels)
        self.message_lin = nn.Linear(self.in_channels + self.edge_channels, self.out_channels)
        self.agg_lin = nn.Linear(self.in_channels + self.out_channels, self.out_channels)
        self.message_activation = _get_activation(activation)
        self.update_activation = _get_activation(activation)
        self.normalize_emb = bool(normalize_emb)

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_j, edge_attr):
        message = torch.cat((x_j, edge_attr), dim=-1)
        return self.message_activation(self.message_lin(message))

    def update(self, aggr_out, x):
        out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out


class GNNStack(nn.Module):
    """Repeated EGSAGE layers + edge updates + node post-MLP.

    Attribute names match the prior V11 implementation so compatible legacy
    checkpoints remain loadable after this source-code reorganization.
    """

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        node_dim,
        edge_dim,
        num_layers=3,
        dropout=0.1,
        activation='relu',
        post_mlp_hidden=64,
        normalize_embs=True,
        aggr='mean',
    ):
        super().__init__()
        self.gnn_layer_num = int(num_layers)
        if self.gnn_layer_num <= 0:
            raise ValueError('gnn_layers must be positive.')

        self.convs = nn.ModuleList()
        for layer_idx in range(self.gnn_layer_num):
            in_dim = node_input_dim if layer_idx == 0 else node_dim
            edge_in_dim = edge_input_dim if layer_idx == 0 else edge_dim
            self.convs.append(
                EGraphSage(
                    in_channels=in_dim,
                    out_channels=node_dim,
                    edge_channels=edge_in_dim,
                    activation=activation,
                    normalize_emb=normalize_embs,
                    aggr=aggr,
                )
            )

        # Preserve the V11 state_dict structure: Sequential(Sequential(...), Linear(...)).
        self.node_post_mlp = nn.Sequential(
            nn.Sequential(
                nn.Linear(node_dim, post_mlp_hidden),
                _get_activation(activation),
                nn.Dropout(dropout),
            ),
            nn.Linear(post_mlp_hidden, node_dim),
        )

        self.edge_update_mlps = nn.ModuleList()
        for layer_idx in range(self.gnn_layer_num):
            edge_in_dim = edge_input_dim if layer_idx == 0 else edge_dim
            self.edge_update_mlps.append(
                nn.Sequential(
                    nn.Linear(node_dim + node_dim + edge_in_dim, edge_dim),
                    _get_activation(activation),
                )
            )

    @staticmethod
    def update_edge_attr(x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        return mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))

    def forward(self, x, edge_attr, edge_index):
        for conv, edge_mlp in zip(self.convs, self.edge_update_mlps):
            x = conv(x, edge_attr, edge_index)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, edge_mlp)
        return self.node_post_mlp(x)
