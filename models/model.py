"""Complete GRAPE-CpG paper model wiring."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import parse_int_tuple
from models.dna_encoder import DNAEncoderCNN
from models.global_graph import GNNStack
from models.local_neighbor import MultiScaleLocalCpGAttention


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


class MLPNet(nn.Module):
    """Edge decoder; layout preserves prior V11 state_dict keys."""

    def __init__(self, input_dims, output_dim=1, hidden_layer_sizes=(128, 64), hidden_activation='relu', dropout=0.1):
        super().__init__()
        layers = nn.ModuleList()
        input_dim = int(np.sum(input_dims))
        for hidden_dim in hidden_layer_sizes:
            layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, int(hidden_dim)),
                    _get_activation(hidden_activation),
                    nn.Dropout(dropout),
                )
            )
            input_dim = int(hidden_dim)
        layers.append(nn.Sequential(nn.Linear(input_dim, int(output_dim)), nn.Identity()))
        self.layers = layers

    @property
    def first_linear(self):
        return self.layers[0][0]

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        x = torch.cat(inputs, dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


class GrapeCpGModel(nn.Module):
    """Global GRAPE graph encoder + optional multi-scale local neighbor block."""

    def __init__(self, args):
        super().__init__()
        self.node_dim = int(args.node_dim)
        self.edge_dim = int(args.edge_dim)
        self.dna_window = int(args.dna_window)
        self.use_dna = bool(getattr(args, 'dna', True))
        self.n_cells = int(args.n_cells)
        self.use_cell_embedding = bool(args.cell_embedding)
        self.use_local = bool(args.local)
        self.fourier_dim = int(args.fourier_dim)
        if self.fourier_dim <= 0 or self.fourier_dim % 2 != 0:
            raise ValueError('fourier_dim must be a positive even integer.')

        if self.use_dna:
            self.dna_encoder = DNAEncoderCNN(
                dna_window=self.dna_window,
                vocab_size=5,
                token_dim=4,
                conv1_channels=64,
                conv2_channels=128,
                kernel1=11,
                pool1=4,
                kernel2=3,
                pool2=2,
                hidden_dim=64,
                out_dim=self.node_dim,
                dropout=args.dropout,
            )
            site_input_dim = self.node_dim + self.fourier_dim
        else:
            # DNA ablation: remove the DNA encoder entirely, but preserve genomic-position features.
            self.dna_encoder = None
            site_input_dim = self.fourier_dim
        self.site_fusion = nn.Linear(site_input_dim, self.node_dim)

        self.cell_emb_dim = int(args.cell_emb_dim)
        self.cell_embedding = nn.Embedding(self.n_cells, self.cell_emb_dim)
        self.cell_fusion = nn.Linear(self.node_dim + self.cell_emb_dim, self.node_dim)
        self.edge_encoder = nn.Linear(2, self.edge_dim)

        self.gnn = GNNStack(
            node_input_dim=self.node_dim,
            edge_input_dim=self.edge_dim,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            num_layers=int(args.gnn_layers),
            dropout=float(args.dropout),
            activation=args.gnn_activation,
            post_mlp_hidden=int(args.post_mlp_hidden),
            normalize_embs=True,
            aggr=args.aggr,
        )

        if self.use_local:
            windows = parse_int_tuple(args.local_windows)
            if not windows:
                raise ValueError('local_windows cannot be empty when the local branch is enabled.')
            max_window = max(windows)
            self.local_context = MultiScaleLocalCpGAttention(
                node_dim=self.node_dim,
                max_window=max_window,
                windows=windows,
                meth_dim=int(args.local_meth_dim),
                mask_dim=int(args.local_mask_dim),
                rel_pos_dim=int(args.local_rel_pos_dim),
                hidden_dim=int(args.local_hidden_dim),
                out_dim=int(args.local_context_dim),
                dropout=float(args.dropout),
                aggregation=getattr(args, 'local_aggregation', 'attention'),
            )
            self.local_context_output_dim = self.local_context.output_dim
        else:
            self.local_context = None
            self.local_context_output_dim = 0

        hidden_sizes = parse_int_tuple(args.impute_hiddens)
        decoder_input_dims = [self.node_dim, self.node_dim]
        if self.use_local:
            decoder_input_dims.append(self.local_context_output_dim)
        self.edge_decoder = MLPNet(
            input_dims=decoder_input_dims,
            output_dim=1,
            hidden_layer_sizes=hidden_sizes,
            hidden_activation=args.impute_activation,
            dropout=float(args.dropout),
        )

    def fourier_position_features(self, pos_norm):
        half_dim = self.fourier_dim // 2
        pos_norm = pos_norm.float().clamp(0.0, 1.0).unsqueeze(-1)
        freqs = torch.pow(
            torch.tensor(2.0, device=pos_norm.device),
            torch.arange(half_dim, device=pos_norm.device).float(),
        ).view(1, -1)
        angles = 2.0 * math.pi * pos_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def build_node_features(self, data):
        device = data.pos_norm_seg.device
        n_cells = int(data.n_cells.view(-1)[0].item())
        n_sites = int(data.n_sites.view(-1)[0].item())
        if n_cells != self.n_cells:
            raise RuntimeError(f'Model expects {self.n_cells} cells but batch contains {n_cells}.')

        cell_ones = torch.ones((n_cells, self.node_dim), device=device)
        if self.use_cell_embedding:
            cell_ids = torch.arange(n_cells, device=device)
            cell_x = self.cell_fusion(torch.cat([cell_ones, self.cell_embedding(cell_ids)], dim=1))
        else:
            cell_x = cell_ones

        pos_x = self.fourier_position_features(data.pos_norm_seg.float().view(-1))
        if self.use_dna:
            dna_x = self.dna_encoder(data.dna_seg)
            site_x = self.site_fusion(torch.cat([dna_x, pos_x], dim=1))
        else:
            site_x = self.site_fusion(pos_x)
        if site_x.shape[0] != n_sites:
            raise RuntimeError('Site feature count does not match data.n_sites.')
        return torch.cat([cell_x, site_x], dim=0)

    def build_edge_features(self, edge_attr):
        edge_label = edge_attr.view(-1).long()
        return self.edge_encoder(F.one_hot(edge_label, num_classes=2).float())

    def global_modules(self):
        modules = [
            self.site_fusion,
            self.cell_embedding,
            self.cell_fusion,
            self.edge_encoder,
            self.gnn,
        ]
        if self.dna_encoder is not None:
            modules.insert(0, self.dna_encoder)
        return modules

    def freeze_global_branch(self):
        for module in self.global_modules():
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_global_branch(self):
        for module in self.global_modules():
            for parameter in module.parameters():
                parameter.requires_grad = True

    def global_parameters(self):
        for module in self.global_modules():
            yield from module.parameters()

    def set_global_branch_eval(self):
        for module in self.global_modules():
            module.eval()

    def encode_global(self, data):
        """Compute graph-derived node embeddings for one segment."""
        x = self.build_node_features(data)
        edge_attr = self.build_edge_features(data.edge_attr)
        return self.gnn(x, edge_attr, data.edge_index)

    def decode_targets(self, node_emb, data):
        """Predict the target cell-CpG pairs from precomputed node embeddings."""
        src = data.target_edge_index[0]
        dst = data.target_edge_index[1]
        decoder_inputs = [node_emb[src], node_emb[dst]]
        if self.use_local:
            decoder_inputs.append(
                self.local_context(
                    node_emb=node_emb,
                    src=src,
                    dst=dst,
                    data=data,
                    n_cells=int(data.n_cells.view(-1)[0].item()),
                )
            )
        return self.edge_decoder(decoder_inputs).view(-1)

    def forward(self, data):
        return self.decode_targets(self.encode_global(data), data)
