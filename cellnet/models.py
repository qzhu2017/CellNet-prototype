"""Conditional lattice-flow GNN for crystal cell prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from cellnet.flow_matching import FlowVelocityHead, integrate_flow, sample_prior, sample_time
from cellnet.gnn import MolecularGNN


class GivenHZConditionalLatticeFlowGNN(nn.Module):
    """
    (SMILES graph, Hall, Z′) → Selling flow, then a flow over [log λ(3), log λ*(3)] given Selling.

    λ is sampled from a dedicated 6-dim flow conditioned on each Selling draw, so every sampled
    metric receives its own direct and reciprocal minima.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        selling_dim: int,
        lambda_dim: int,
        n_halls: int,
        n_zprimes: int,
        hidden_dim: int = 512,
        gnn_layers: int = 4,
        dropout: float = 0.15,
        hall_emb_dim: int = 64,
        zprime_emb_dim: int = 32,
        flow_steps: int = 20,
        gnn_pooling: str = "mean",
        predict_axis_permutation: bool = False,
    ):
        super().__init__()
        self.selling_dim = selling_dim
        self.lambda_dim = lambda_dim
        self.lattice_flow_dim = selling_dim + lambda_dim
        self.flow_steps = flow_steps
        self.encoder = MolecularGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=256,
            n_layers=gnn_layers,
            dropout=dropout,
            out_dim=hidden_dim,
            pooling=gnn_pooling,
        )
        self.hall_emb = nn.Embedding(n_halls, hall_emb_dim)
        self.zprime_emb = nn.Embedding(n_zprimes, zprime_emb_dim)
        self.cond_dim = hidden_dim + hall_emb_dim + zprime_emb_dim
        self.lambda_cond_dim = self.cond_dim + selling_dim
        self.selling_flow_head = FlowVelocityHead(
            cond_dim=self.cond_dim,
            free_dim=selling_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.lambda_flow_head = FlowVelocityHead(
            cond_dim=self.lambda_cond_dim,
            free_dim=lambda_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.density_head = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.axis_permutation_head = (
            nn.Sequential(
                nn.Linear(self.cond_dim + self.lattice_flow_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 6),
            )
            if predict_axis_permutation
            else None
        )

    def _condition(
        self,
        data: Batch,
        hall: torch.Tensor,
        zprime: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(data)
        return torch.cat([h, self.hall_emb(hall), self.zprime_emb(zprime)], dim=-1)

    def _lambda_condition(self, cond: torch.Tensor, selling_norm: torch.Tensor) -> torch.Tensor:
        return torch.cat([cond, selling_norm], dim=-1)

    def predict_selling_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        return self.selling_flow_head(x_t, t, cond)

    def predict_lambda_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, lam_cond: torch.Tensor
    ) -> torch.Tensor:
        return self.lambda_flow_head(x_t, t, lam_cond)

    @torch.no_grad()
    def sample_selling(
        self,
        cond: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        steps = n_steps or self.flow_steps

        def velocity_fn(x, t, c):
            return self.predict_selling_velocity(x, t, c)

        return integrate_flow(velocity_fn, cond, n_steps=steps, target_dim=self.selling_dim)

    @torch.no_grad()
    def sample_lambda(
        self,
        cond: torch.Tensor,
        selling_norm: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        steps = n_steps or self.flow_steps
        lam_cond = self._lambda_condition(cond, selling_norm)

        def velocity_fn(x, t, c):
            return self.predict_lambda_velocity(x, t, c)

        return integrate_flow(velocity_fn, lam_cond, n_steps=steps, target_dim=self.lambda_dim)

    @torch.no_grad()
    def sample_lattice(
        self,
        cond: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """One joint draw: Selling flow, then λ flow conditioned on that Selling."""
        selling = self.sample_selling(cond, n_steps=n_steps)
        lam = self.sample_lambda(cond, selling, n_steps=n_steps)
        return torch.cat([selling, lam], dim=-1)

    def forward(
        self,
        data: Batch,
        hall: torch.Tensor,
        zprime: torch.Tensor,
        x1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hall is None or zprime is None:
            raise ValueError("hall and zprime must be provided for GivenHZConditionalLatticeFlowGNN")
        cond = self._condition(data, hall, zprime)
        log_density = self.density_head(cond).squeeze(-1)
        out: dict[str, torch.Tensor] = {
            "log_density": log_density,
            "hall_idx": hall,
            "zprime_idx": zprime,
        }
        if x1 is not None:
            batch = x1.shape[0]
            device = x1.device
            x1_s = x1[:, : self.selling_dim]
            x1_lam = x1[:, self.selling_dim :]

            x0_s = sample_prior(x1_s.shape, device)
            t_s = sample_time(batch, device)
            t_s_expand = t_s.unsqueeze(-1)
            x_t_s = (1.0 - t_s_expand) * x0_s + t_s_expand * x1_s
            v_target_s = x1_s - x0_s
            v_pred_s = self.predict_selling_velocity(x_t_s, t_s, cond)
            x1_hat_s = x_t_s + (1.0 - t_s_expand) * v_pred_s

            lam_cond = self._lambda_condition(cond, x1_s)
            x0_lam = sample_prior(x1_lam.shape, device)
            t_lam = sample_time(batch, device)
            t_lam_expand = t_lam.unsqueeze(-1)
            x_t_lam = (1.0 - t_lam_expand) * x0_lam + t_lam_expand * x1_lam
            v_target_lam = x1_lam - x0_lam
            v_pred_lam = self.predict_lambda_velocity(x_t_lam, t_lam, lam_cond)
            x1_hat_lam = x_t_lam + (1.0 - t_lam_expand) * v_pred_lam

            out.update({
                "velocity_s": v_pred_s,
                "velocity_target_s": v_target_s,
                "velocity_lambda": v_pred_lam,
                "velocity_target_lambda": v_target_lam,
                "selling": x1_hat_s,
                "lambda_block": x1_hat_lam,
                "lattice_flow": torch.cat([x1_hat_s, x1_hat_lam], dim=-1),
                "t_s": t_s,
                "t_lambda": t_lam,
            })
        else:
            out["lattice_flow"] = self.sample_lattice(cond)
        if self.axis_permutation_head is not None:
            out["axis_permutation_logits"] = self.axis_permutation_head(
                torch.cat([cond, out["lattice_flow"]], dim=-1)
            )
        return out
