import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLocalCpGAttention(nn.Module):
    """Target-conditioned multi-scale local CpG attention."""

    def __init__(
        self,
        node_dim,
        max_window=20,
        windows=(5, 20),
        meth_dim=8,
        mask_dim=4,
        rel_pos_dim=8,
        hidden_dim=128,
        out_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.max_window = int(max_window)
        self.windows = tuple(int(w) for w in windows)
        self.out_dim = int(out_dim)
        if not self.windows:
            raise ValueError('local_windows cannot be empty.')
        if min(self.windows) <= 0:
            raise ValueError('local windows must be positive.')
        if max(self.windows) > self.max_window:
            raise ValueError('max(local_windows) cannot exceed max_window.')

        self.meth_embedding = nn.Embedding(3, int(meth_dim))  # 0, 1, unknown/pad
        self.mask_embedding = nn.Embedding(2, int(mask_dim))  # missing/pad, known
        self.rel_pos_embedding = nn.Embedding(2 * self.max_window + 1, int(rel_pos_dim))

        token_in_dim = self.node_dim + int(meth_dim) + int(mask_dim) + int(rel_pos_dim) + 1
        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(),
        )
        self.query_proj = nn.Linear(2 * self.node_dim, int(out_dim))
        self.score_proj = nn.Linear(int(out_dim), 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(int(out_dim) * len(self.windows), int(out_dim) * len(self.windows)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    @property
    def output_dim(self):
        return self.out_dim * len(self.windows)

    def forward(self, node_emb, src, dst, data, n_cells):
        device = node_emb.device
        local_site_idx = data.target_local_site_idx.to(device).long()
        local_meth = data.target_local_meth.to(device).long()
        local_known = data.target_local_known.to(device).long()
        local_valid = data.target_local_valid.to(device).float()
        rel_pos_idx = data.target_local_rel_pos_idx.to(device).long()
        rel_offset = data.target_local_rel_offset.to(device).long()
        rel_dist = data.target_local_rel_dist.to(device).float().unsqueeze(-1)

        safe_site_idx = torch.clamp(local_site_idx, min=0)
        local_node_idx = int(n_cells) + safe_site_idx
        local_site_h = node_emb[local_node_idx] * local_valid.unsqueeze(-1)

        meth_token = torch.where(
            (local_known > 0) & (local_valid > 0),
            local_meth.clamp(min=0, max=1),
            torch.full_like(local_meth, 2),
        )
        known_token = torch.where(
            (local_known > 0) & (local_valid > 0),
            torch.ones_like(local_known),
            torch.zeros_like(local_known),
        )

        token = torch.cat(
            [
                local_site_h,
                self.meth_embedding(meth_token),
                self.mask_embedding(known_token.clamp(min=0, max=1)),
                self.rel_pos_embedding(rel_pos_idx.clamp(min=0, max=2 * self.max_window)),
                rel_dist,
            ],
            dim=-1,
        )
        token_h = self.token_proj(token)
        query = self.query_proj(torch.cat([node_emb[src], node_emb[dst]], dim=-1)).unsqueeze(1)
        score = self.score_proj(torch.tanh(token_h + query)).squeeze(-1)

        contexts = []
        self.last_attention = {}
        for window in self.windows:
            in_scale = (rel_offset.abs() <= int(window)).float() * local_valid
            score_w = score.masked_fill(in_scale <= 0, -1e9)
            attn_w = F.softmax(score_w, dim=1) * in_scale
            attn_w = attn_w / attn_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            contexts.append(torch.sum(attn_w.unsqueeze(-1) * token_h, dim=1))
            self.last_attention[int(window)] = attn_w.detach()

        return self.out_proj(torch.cat(contexts, dim=-1))
