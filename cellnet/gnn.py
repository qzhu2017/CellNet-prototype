"""Graph neural network encoder for molecular crystals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


class MolecularGNN(nn.Module):
    """GIN encoder with edge features; outputs a graph-level embedding."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.3,
        out_dim: int = 512,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "multiscale"}:
            raise ValueError(f"Unknown molecular pooling mode: {pooling}")
        self.pooling = pooling
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        if pooling == "multiscale":
            self.output_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            # Keep the exact legacy layout and state-dict keys.
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.out_dim = out_dim

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.input_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "multiscale":
            # Add/max retain graph size and molecular-extent signals that mean
            # pooling alone discards. The legacy default keeps old checkpoints loadable.
            pooled = torch.cat(
                [
                    global_mean_pool(x, data.batch),
                    global_add_pool(x, data.batch),
                    global_max_pool(x, data.batch),
                ],
                dim=-1,
            )
        else:
            pooled = global_mean_pool(x, data.batch)
        return self.output_proj(pooled)
