"""Lattice-flow utilities: invariants, normalization, and K-sample helpers."""

from __future__ import annotations

from math import log

import numpy as np
import torch

from cellnet.packing import molecular_weight, packing_density
from cellnet.symmetry import (
    apply_cellpar_constraints,
    cellpar_to_free,
    crystal_system_from_hall,
    free_to_cellpar,
)

FREE_PARAM_DIM = 6
LOG_LAMBDA_DIM = 3
LOG_LAMBDA_RECIP_DIM = 3
LATTICE_LAMBDA_DIM = LOG_LAMBDA_DIM + LOG_LAMBDA_RECIP_DIM


def free_param_mask(hall_number: int) -> np.ndarray:
    """
    Boolean mask over padded free-parameter slots.

    Layout: [log(a), log(b), log(c), alpha, beta, gamma].
    Tetragonal / trigonal / hexagonal use slots 0 (log a) and 1 (log c).
    """
    system = crystal_system_from_hall(hall_number)
    mask = np.zeros(FREE_PARAM_DIM, dtype=bool)
    if system == "triclinic":
        mask[:] = True
    elif system == "monoclinic":
        mask[[0, 1, 2, 4]] = True
    elif system == "orthorhombic":
        mask[[0, 1, 2]] = True
    elif system in ("tetragonal", "trigonal", "hexagonal"):
        mask[[0, 1]] = True
    elif system == "cubic":
        mask[0] = True
    else:
        raise ValueError(f"Unknown crystal system for hall {hall_number}: {system}")
    return mask


def cellpar_to_padded_free(cellpar: np.ndarray, hall_number: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode symmetry-reduced parameters into a fixed 6-slot vector + mask."""
    mask = free_param_mask(hall_number)
    free = cellpar_to_free(cellpar, hall_number)
    padded = np.zeros(FREE_PARAM_DIM, dtype=np.float64)
    padded[mask] = free
    return padded, mask


def padded_free_to_cellpar(padded: np.ndarray, hall_number: int) -> np.ndarray:
    """Decode padded free parameters to direct cell parameters."""
    mask = free_param_mask(hall_number)
    free = np.asarray(padded, dtype=np.float64)[mask]
    return free_to_cellpar(free, hall_number)


def compute_density(
    cellpar: np.ndarray,
    smiles: str,
    zprime: float,
    hall_number: int,
) -> float | None:
    """Crystal density in g/cm³ from cell, composition, Z′, and Hall number."""
    mw = molecular_weight(smiles)
    if mw is None:
        return None
    return packing_density(cellpar, mw, zprime, hall_number)


def compute_log_density(
    cellpar: np.ndarray,
    smiles: str,
    zprime: float,
    hall_number: int,
) -> float:
    """Natural log density; falls back to 0 when molecular weight is unavailable."""
    rho = compute_density(cellpar, smiles, zprime, hall_number)
    if rho is None or rho <= 0:
        return 0.0
    return float(log(rho))


def compute_log_successive_minima(cellpar: np.ndarray) -> np.ndarray:
    """log(λ₁), log(λ₂), log(λ₃) — SL(3,ℤ)-invariant lattice shape invariants (Å)."""
    from cellnet.lattice_invariants import log_successive_minima

    return log_successive_minima(cellpar)


def compute_log_reciprocal_successive_minima(cellpar: np.ndarray) -> np.ndarray:
    """log(λ*₁), log(λ*₂), log(λ*₃) — successive minima on the reciprocal lattice."""
    from cellnet.lattice_invariants import log_reciprocal_successive_minima

    return log_reciprocal_successive_minima(cellpar)


def lattice_flow_slices(stats) -> tuple[slice, slice, slice]:
    """Return (selling, log_λ, log_λ*) slices for a concatenated lattice-flow vector."""
    s1 = int(getattr(stats, "selling_dim", 6))
    l1 = s1 + int(getattr(stats, "log_lambda_dim", LOG_LAMBDA_DIM))
    r1 = l1 + int(getattr(stats, "log_lambda_recip_dim", LOG_LAMBDA_RECIP_DIM))
    return slice(0, s1), slice(s1, l1), slice(l1, r1)


def split_lattice_flow_norm(vec: np.ndarray, stats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split normalized lattice-flow vector into Selling, log λ, log λ* blocks."""
    s_sl, l_sl, r_sl = lattice_flow_slices(stats)
    x = np.asarray(vec, dtype=np.float64)
    if x.ndim == 1:
        return x[s_sl], x[l_sl], x[r_sl]
    return x[..., s_sl], x[..., l_sl], x[..., r_sl]


def compute_selling_parameters(cellpar: np.ndarray) -> np.ndarray:
    """Delaunay-reduced Selling parameters (6,) in Å²."""
    from cellnet.lattice_invariants import selling_parameters

    return selling_parameters(cellpar, delaunay=True)


def signed_log1p(selling: np.ndarray) -> np.ndarray:
    """sign(S) * log1p(|S|) — preserves negative cross terms."""
    s = np.asarray(selling, dtype=np.float64)
    return np.sign(s) * np.log1p(np.abs(s))


def inv_signed_log1p(transformed: np.ndarray) -> np.ndarray:
    """Inverse of signed_log1p."""
    t = np.asarray(transformed, dtype=np.float64)
    return np.sign(t) * np.expm1(np.abs(t))


def denorm_selling_log1p(selling_norm: np.ndarray, stats) -> np.ndarray:
    """Map normalized signed-log Selling back to physical Å²."""
    log_s = np.asarray(selling_norm, dtype=np.float64) * stats.selling_std + stats.selling_mean
    return inv_signed_log1p(log_s)


def norm_selling_log1p(selling: np.ndarray, stats) -> np.ndarray:
    """Normalize physical Selling (Å²) to signed-log z-score space."""
    from cellnet.lattice_invariants import sort_selling_parameters

    s = sort_selling_parameters(np.asarray(selling, dtype=np.float64))
    log_s = signed_log1p(s)
    return ((log_s - stats.selling_mean) / stats.selling_std).astype(np.float64)


def selling_norm_to_phys_torch(
    selling_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Batch denorm: normalized signed-log → physical Selling (Å²)."""
    log_s = selling_norm * std + mean
    return torch.sign(log_s) * torch.expm1(log_s.abs())


def selling_relative_mse(
    pred_phys: torch.Tensor,
    ref_phys: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative MSE matching lattice_invariants.selling_mse."""
    scale = torch.maximum(torch.maximum(pred_phys.abs(), ref_phys.abs()), torch.tensor(eps, device=pred_phys.device))
    return torch.mean(((pred_phys - ref_phys) / scale) ** 2)


def log_lambda_norm_to_phys_torch(
    log_lambda_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Batch denorm: normalized log(λ) → physical log(λ) in Å."""
    return log_lambda_norm * std + mean


def lambda_relative_mse(
    pred_log: torch.Tensor,
    ref_log: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative MSE on linear successive minima λ = exp(log λ)."""
    pred_lin = torch.exp(pred_log)
    ref_lin = torch.exp(ref_log)
    scale = torch.maximum(ref_lin.abs(), torch.tensor(eps, device=pred_log.device))
    return torch.mean(((pred_lin - ref_lin) / scale) ** 2)


def denorm_log_lambda(log_lambda_norm: np.ndarray, stats) -> np.ndarray:
    """Map normalized log(λ₁,λ₂,λ₃) back to physical log-Å."""
    x = np.asarray(log_lambda_norm, dtype=np.float64)
    mean = np.asarray(getattr(stats, "log_lambda_mean", np.zeros(LOG_LAMBDA_DIM)), dtype=np.float64)
    std = np.asarray(getattr(stats, "log_lambda_std", np.ones(LOG_LAMBDA_DIM)), dtype=np.float64)
    if x.ndim == 1:
        return x * std + mean
    return x * std + mean


def denorm_log_reciprocal_lambda(log_lambda_norm: np.ndarray, stats) -> np.ndarray:
    """Map normalized log(λ*₁,λ*₂,λ*₃) back to physical log(Å⁻¹)."""
    x = np.asarray(log_lambda_norm, dtype=np.float64)
    mean = np.asarray(getattr(stats, "log_lambda_recip_mean", np.zeros(LOG_LAMBDA_RECIP_DIM)), dtype=np.float64)
    std = np.asarray(getattr(stats, "log_lambda_recip_std", np.ones(LOG_LAMBDA_RECIP_DIM)), dtype=np.float64)
    if x.ndim == 1:
        return x * std + mean
    return x * std + mean


def norm_log_lambda(log_lambda: np.ndarray, stats) -> np.ndarray:
    """Normalize physical log(λ) to z-score space."""
    x = np.asarray(log_lambda, dtype=np.float64)
    mean = np.asarray(stats.log_lambda_mean, dtype=np.float64)
    std = np.asarray(stats.log_lambda_std, dtype=np.float64)
    return ((x - mean) / std).astype(np.float64)


def delaunay_standardize_cellpar(cellpar: np.ndarray, hall_number: int) -> np.ndarray:
    """Delaunay-reduced unit cell with symmetry constraints applied."""
    from cellnet.lattice_invariants import delaunay_standardize_cellpar as _std

    return _std(cellpar, hall_number)


def decode_free_prediction(
    pred_norm: np.ndarray,
    stats,
    hall_number: int,
) -> np.ndarray:
    """Denormalize padded free prediction and decode to constrained cell parameters."""
    padded = pred_norm * stats.free_std + stats.free_mean
    cellpar = padded_free_to_cellpar(padded, hall_number)
    return apply_cellpar_constraints(cellpar, hall_number)


@torch.no_grad()
def sample_k_given_hz_lattice_flow(
    model,
    x,
    hall: torch.Tensor,
    zprime: torch.Tensor,
    k: int = 5,
    shape_prior: torch.Tensor | None = None,
    shape_prior_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K joint [Selling, log λ, log λ*] vectors; return density."""
    if getattr(model, "shape_head", None) is not None:
        base_cond = model._base_condition(x, hall, zprime)
        logits = model.shape_head(base_cond)
        raw_probs = torch.softmax(logits, dim=-1)
        blended_probs = raw_probs
        if shape_prior is not None and shape_prior_weight > 0.0:
            prior = shape_prior.to(device=raw_probs.device, dtype=raw_probs.dtype)
            if prior.ndim == 1:
                prior = prior.unsqueeze(0).expand_as(raw_probs)
            if prior.shape != raw_probs.shape:
                raise ValueError(
                    f"shape_prior has shape {tuple(prior.shape)}, "
                    f"expected {tuple(raw_probs.shape)}"
                )
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            blend = min(max(float(shape_prior_weight), 0.0), 1.0)
            blended_probs = (1.0 - blend) * raw_probs + blend * prior
        sample_probs = blended_probs
        if model.shape_exploration > 0.0:
            sample_probs = (
                (1.0 - model.shape_exploration) * blended_probs
                + model.shape_exploration / model.n_shape_bins
            )
        pairs = []
        sampled_bins = []
        forced_bins = (
            torch.randperm(model.n_shape_bins, device=base_cond.device)
            if model.shape_exploration > 0.0 and k >= model.n_shape_bins
            else None
        )
        for draw_idx in range(k):
            if forced_bins is not None and draw_idx < model.n_shape_bins:
                shape_bin = forced_bins[draw_idx].expand(base_cond.shape[0])
            else:
                shape_bin = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
            cond, _, _ = model._shape_condition(base_cond, shape_bin)
            pairs.append(model.sample_lattice(cond))
            sampled_bins.append(shape_bin)
        samples = torch.stack(pairs, dim=1)
        log_density = model.density_head(base_cond).squeeze(-1)
        model._last_neural_shape_probabilities = raw_probs.detach()
        model._last_shape_probabilities = blended_probs.detach()
        model._last_sampled_shape_bins = torch.stack(sampled_bins, dim=1).detach()
    else:
        base_cond = model._base_condition(x, hall, zprime)
        cond, _, _ = model._shape_condition(base_cond)
        samples = torch.stack([model.sample_lattice(cond) for _ in range(k)], dim=1)
        log_density = model.density_head(base_cond).squeeze(-1)
    if getattr(model, "axis_permutation_head", None) is not None:
        expanded_cond = base_cond.unsqueeze(1).expand(-1, k, -1)
        model._last_axis_permutation_logits = model.axis_permutation_head(
            torch.cat([expanded_cond, samples], dim=-1)
        ).detach()
    return samples, log_density


@torch.no_grad()
def sample_k_given_hz_conditional_lattice_flow(
    model,
    x,
    hall: torch.Tensor,
    zprime: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample K joint lattice vectors via Selling flow + conditional λ flow.

    Each sample draws Selling first, then (log λ, log λ*) conditioned on that Selling.
    """
    cond = model._condition(x, hall, zprime)
    pairs = []
    for _ in range(k):
        selling = model.sample_selling(cond)
        lam = model.sample_lambda(cond, selling)
        pairs.append(torch.cat([selling, lam], dim=-1))
    samples = torch.stack(pairs, dim=1)
    log_density = model.density_head(cond).squeeze(-1)
    if getattr(model, "axis_permutation_head", None) is not None:
        expanded_cond = cond.unsqueeze(1).expand(-1, k, -1)
        model._last_axis_permutation_logits = model.axis_permutation_head(
            torch.cat([expanded_cond, samples], dim=-1)
        ).detach()
    return samples, log_density
