"""Auxiliary CellNet model: space-group prediction and a density distribution.

Two heads on one molecular GNN encoder (same encoder design as the lattice flow):

* Space-group head  p(Hall, Z′ | molecule): a softmax over the (Hall, Z′) classes seen
  in training. A molecule can crystallise in several groups, so the training target is
  the set of all its known (Hall, Z′) forms (uniform over that set).
* Density head  p(log ρ | molecule, Hall, Z′): a K-component Gaussian mixture over the
  normalised log density, replacing the flow model's single deterministic value. Draws
  from it give each sampled box its own volume.

It is trained and used separately from the lattice flow, so the flow is unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellnet.gnn import MolecularGNN


class SpaceGroupDensityNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_classes: int,
        n_halls: int,
        n_zprimes: int,
        hidden_dim: int = 512,
        gnn_layers: int = 4,
        dropout: float = 0.2,
        n_mix: int = 3,
        hall_emb_dim: int = 64,
        zprime_emb_dim: int = 32,
    ):
        super().__init__()
        self.n_mix = n_mix
        self.encoder = MolecularGNN(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=256, n_layers=gnn_layers,
            dropout=dropout, out_dim=hidden_dim, pooling="mean",
        )
        self.sg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.hall_emb = nn.Embedding(n_halls, hall_emb_dim)
        self.zprime_emb = nn.Embedding(n_zprimes, zprime_emb_dim)
        cond = hidden_dim + hall_emb_dim + zprime_emb_dim
        self.density_head = nn.Sequential(
            nn.Linear(cond, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3 * n_mix),
        )

    def encode(self, data) -> torch.Tensor:
        return self.encoder(data)

    def sg_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.sg_head(h)

    def density_params(self, h, hall, zprime):
        """Mixture logits, means and log-sigmas (normalised log-density units)."""
        out = self.density_head(torch.cat([h, self.hall_emb(hall), self.zprime_emb(zprime)], -1))
        logit, mu, log_sigma = out.split(self.n_mix, dim=-1)
        return logit, mu, log_sigma.clamp(-5.0, 3.0)

    def forward(self, data, hall, zprime):
        h = self.encode(data)
        return self.sg_logits(h), self.density_params(h, hall, zprime)


def mixture_nll(logit, mu, log_sigma, y) -> torch.Tensor:
    """Per-sample negative log-likelihood of y under the Gaussian mixture."""
    y = y.unsqueeze(-1)
    log_pi = F.log_softmax(logit, dim=-1)
    log_n = -0.5 * ((y - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(log_pi + log_n, dim=-1)


def mixture_cdf(logit, mu, log_sigma, y) -> torch.Tensor:
    """CDF of the mixture at y (for calibration checks)."""
    y = y.unsqueeze(-1)
    pi = F.softmax(logit, dim=-1)
    z = (y - mu) / (log_sigma.exp() * math.sqrt(2.0))
    return (pi * 0.5 * (1.0 + torch.erf(z))).sum(-1)


def mixture_mean_std(logit, mu, log_sigma):
    pi = F.softmax(logit, dim=-1)
    mean = (pi * mu).sum(-1)
    var = (pi * (log_sigma.exp() ** 2 + mu ** 2)).sum(-1) - mean ** 2
    return mean, var.clamp_min(1e-12).sqrt()


@torch.no_grad()
def sample_mixture(logit, mu, log_sigma, n: int, generator: torch.Generator | None = None):
    """n draws per row -> [B, n] (normalised log-density units)."""
    pi = F.softmax(logit, dim=-1)
    comp = torch.multinomial(pi, n, replacement=True, generator=generator)  # [B, n]
    m = torch.gather(mu, 1, comp)
    s = torch.gather(log_sigma.exp(), 1, comp)
    eps = torch.randn(m.shape, generator=generator, device=m.device)
    return m + s * eps
