"""Conditional flow matching for symmetry-reduced cell parameters."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cellnet.sequential import FREE_PARAM_DIM


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow time t in [0, 1]."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class FlowVelocityHead(nn.Module):
    """Predict flow velocity v(x_t, t | condition) for padded free parameters."""

    def __init__(
        self,
        cond_dim: int,
        free_dim: int = FREE_PARAM_DIM,
        hidden_dim: int = 512,
        time_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        in_dim = free_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, free_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        return self.net(torch.cat([x_t, cond, t_emb], dim=-1))


def sample_time(batch: int, device: torch.device, beta_a: float = 1.8, beta_b: float = 1.0) -> torch.Tensor:
    """Beta(1.8, 1) timestep distribution (Clari default). Sampled on CPU for MPS compatibility."""
    dist = torch.distributions.Beta(
        torch.tensor([beta_a]),
        torch.tensor([beta_b]),
    )
    return dist.sample((batch,)).squeeze(-1).clamp(0.0, 1.0).to(device)


def sample_prior(
    shape: tuple[int, ...],
    device: torch.device,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard-normal prior in normalized free-parameter space."""
    x0 = torch.randn(shape, device=device)
    if mask is not None:
        x0 = x0 * mask
    return x0


@torch.no_grad()
def integrate_flow(
    velocity_fn,
    cond: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    n_steps: int = 20,
    target_dim: int | None = None,
) -> torch.Tensor:
    """Euler integration from t=0 prior to t=1."""
    batch = cond.shape[0]
    device = cond.device
    dim = target_dim if target_dim is not None else FREE_PARAM_DIM
    x = sample_prior((batch, dim), device, mask=mask)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((batch,), (step + 0.5) / n_steps, device=device)
        v = velocity_fn(x, t, cond)
        if mask is not None:
            v = v * mask
        x = x + dt * v
    if mask is not None:
        x = x * mask
    return x
