"""Training losses and evaluation metrics for lattice-flow models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cellnet.symmetry import align_cellpar_to_reference


def _lattice_lambda_endpoint_loss(
    pred_norm: torch.Tensor,
    ref_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode: str = "phys_log",
    relative_weight: float = 0.0,
) -> torch.Tensor:
    """
    Endpoint loss for log(λ) or log(λ*).

    ``ref_norm`` may be [B, D] (scalar mean returned) or [B, K, D] (per-K [B, K] returned).
    """
    from cellnet.sequential import lambda_relative_mse, log_lambda_norm_to_phys_torch

    if ref_norm.ndim == 3:
        if mode == "norm":
            per_k = (pred_norm.unsqueeze(1) - ref_norm).pow(2).mean(dim=-1)
        else:
            pred_log = log_lambda_norm_to_phys_torch(pred_norm, mean, std).unsqueeze(1)
            ref_log = log_lambda_norm_to_phys_torch(ref_norm, mean, std)
            if mode == "phys_log":
                per_k = (pred_log - ref_log).pow(2).mean(dim=-1)
            elif mode == "relative":
                pred_lin = torch.exp(pred_log)
                ref_lin = torch.exp(ref_log)
                scale = ref_lin.abs().clamp(min=1e-12)
                per_k = ((pred_lin - ref_lin) / scale).pow(2).mean(dim=-1)
            else:
                raise ValueError(f"Unknown lambda_loss_mode: {mode}")
            if relative_weight > 0.0 and mode != "relative":
                pred_lin = torch.exp(pred_log)
                ref_lin = torch.exp(ref_log)
                scale = ref_lin.abs().clamp(min=1e-12)
                per_k = per_k + relative_weight * ((pred_lin - ref_lin) / scale).pow(2).mean(dim=-1)
        return per_k

    if mode == "norm":
        loss = F.mse_loss(pred_norm, ref_norm)
    elif mode in ("phys_log", "relative"):
        pred_log = log_lambda_norm_to_phys_torch(pred_norm, mean, std)
        ref_log = log_lambda_norm_to_phys_torch(ref_norm, mean, std)
        if mode == "phys_log":
            loss = F.mse_loss(pred_log, ref_log)
        else:
            loss = lambda_relative_mse(pred_log, ref_log)
        if relative_weight > 0.0 and mode != "relative":
            loss = loss + relative_weight * lambda_relative_mse(pred_log, ref_log)
    else:
        raise ValueError(f"Unknown lambda_loss_mode: {mode}")
    return loss


def _lattice_lambda_product_loss(
    pred_norm: torch.Tensor,
    ref_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Penalize mismatch in log(λ₁λ₂λ₃) to reduce compensating axis errors."""
    from cellnet.sequential import log_lambda_norm_to_phys_torch

    pred_log = log_lambda_norm_to_phys_torch(pred_norm, mean, std)
    ref_log = log_lambda_norm_to_phys_torch(ref_norm, mean, std)
    if ref_log.ndim == 3:
        per_k = (pred_log.sum(dim=-1).unsqueeze(1) - ref_log.sum(dim=-1)).pow(2)
        return per_k.min(dim=1).values.mean()
    return F.mse_loss(pred_log.sum(dim=-1), ref_log.sum(dim=-1))


def _weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean over a per-sample loss vector with stable, mean-one weights."""
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if weights.shape[0] != values.shape[0]:
        raise ValueError(f"Expected {values.shape[0]} sample weights, got {weights.shape[0]}")
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


def given_hz_lattice_flow_loss(
    outputs: dict[str, torch.Tensor],
    target_lattice: torch.Tensor,
    target_log_density: torch.Tensor,
    selling_mean: torch.Tensor,
    selling_std: torch.Tensor,
    log_lambda_mean: torch.Tensor,
    log_lambda_std: torch.Tensor,
    log_lambda_recip_mean: torch.Tensor,
    log_lambda_recip_std: torch.Tensor,
    selling_dim: int = 6,
    log_lambda_dim: int = 3,
    w_flow: float = 1.0,
    w_s: float = 1.0,
    w_lambda: float = 1.0,
    w_lambda_recip: float = 1.0,
    w_density: float = 0.25,
    lambda_loss_mode: str = "phys_log",
    w_lambda_relative: float = 0.0,
    w_lambda_product: float = 0.0,
    polymorph_lattice: torch.Tensor | None = None,
    polymorph_log_density: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    target_shape_bin: torch.Tensor | None = None,
    w_shape_bin: float = 0.0,
    target_axis_permutation_mask: torch.Tensor | None = None,
    w_axis_permutation: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Flow-matching on [Selling, log λ, log λ*] + density regression."""
    from cellnet.sequential import selling_norm_to_phys_torch, selling_relative_mse

    flow_per_sample = (outputs["velocity"] - outputs["velocity_target"]).pow(2).mean(dim=-1)
    flow_loss = _weighted_sample_mean(flow_per_sample, sample_weight)
    pred = outputs["lattice_flow"]
    s_end = selling_dim
    l_end = s_end + log_lambda_dim

    pred_s = pred[:, :s_end]
    pred_lam = pred[:, s_end:l_end]
    pred_lam_r = pred[:, l_end:]

    if polymorph_lattice is not None and polymorph_lattice.ndim == 3:
        ref_s = polymorph_lattice[:, :, :s_end]
        ref_lam = polymorph_lattice[:, :, s_end:l_end]
        ref_lam_r = polymorph_lattice[:, :, l_end:]
        pred_s_phys = selling_norm_to_phys_torch(pred_s, selling_mean, selling_std)
        ref_s_phys = selling_norm_to_phys_torch(ref_s, selling_mean, selling_std)
        eps = torch.tensor(1e-8, device=pred.device)
        scale = torch.maximum(torch.maximum(pred_s_phys.unsqueeze(1).abs(), ref_s_phys.abs()), eps)
        s_per_k = ((pred_s_phys.unsqueeze(1) - ref_s_phys) / scale).pow(2).mean(dim=-1)
        s_loss = s_per_k.min(dim=1).values.mean()
        lam_loss = _lattice_lambda_endpoint_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        ).min(dim=1).values.mean()
        lam_r_loss = _lattice_lambda_endpoint_loss(
            pred_lam_r, ref_lam_r, log_lambda_recip_mean, log_lambda_recip_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        ).min(dim=1).values.mean()
        if polymorph_log_density is not None:
            density_loss = (outputs["log_density"].unsqueeze(1) - polymorph_log_density).pow(2).min(dim=1).values.mean()
        else:
            density_loss = F.mse_loss(outputs["log_density"], target_log_density)
    else:
        ref_s = target_lattice[:, :s_end]
        ref_lam = target_lattice[:, s_end:l_end]
        ref_lam_r = target_lattice[:, l_end:]
        pred_s_phys = selling_norm_to_phys_torch(pred_s, selling_mean, selling_std)
        ref_s_phys = selling_norm_to_phys_torch(ref_s, selling_mean, selling_std)
        s_loss = selling_relative_mse(pred_s_phys, ref_s_phys)
        lam_loss = _lattice_lambda_endpoint_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        )
        lam_r_loss = _lattice_lambda_endpoint_loss(
            pred_lam_r, ref_lam_r, log_lambda_recip_mean, log_lambda_recip_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        )
        density_loss = F.mse_loss(outputs["log_density"], target_log_density)

    lambda_product_loss = torch.tensor(0.0, device=pred.device)
    if w_lambda_product > 0.0:
        lambda_product_loss = _lattice_lambda_product_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
        )
    shape_bin_loss = torch.tensor(0.0, device=pred.device)
    if (
        w_shape_bin > 0.0
        and target_shape_bin is not None
        and "shape_logits" in outputs
    ):
        shape_bin_loss = F.cross_entropy(outputs["shape_logits"], target_shape_bin.long())
    axis_permutation_loss = _axis_permutation_class_loss(
        outputs,
        target_axis_permutation_mask,
        w_axis_permutation,
    )

    total = (
        w_flow * flow_loss
        + w_s * s_loss
        + w_lambda * lam_loss
        + w_lambda_recip * lam_r_loss
        + w_density * density_loss
        + w_lambda_product * lambda_product_loss
        + w_shape_bin * shape_bin_loss
        + w_axis_permutation * axis_permutation_loss
    )
    metrics = {
        "total": total.item(),
        "flow": float(flow_loss.item()),
        "s_loss": float(s_loss.item()),
        "lambda": float(lam_loss.item()),
        "lambda_recip": float(lam_r_loss.item()),
        "density": float(density_loss.item()),
    }
    if w_lambda_product > 0.0:
        metrics["lambda_product"] = float(lambda_product_loss.item())
    if w_shape_bin > 0.0 and "shape_logits" in outputs:
        metrics["shape_bin"] = float(shape_bin_loss.item())
    if w_axis_permutation > 0.0 and "axis_permutation_logits" in outputs:
        metrics["axis_permutation"] = float(axis_permutation_loss.item())
    return total, metrics


def _axis_permutation_class_loss(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor | None,
    weight: float,
) -> torch.Tensor:
    """
    Negative log probability of the Hall-equivalent target permutation set.

    A mask containing all six classes contributes exactly zero, which excludes
    non-orthorhombic samples without a separate validity tensor.
    """
    reference = outputs["log_density"]
    if weight <= 0.0 or target_mask is None or "axis_permutation_logits" not in outputs:
        return torch.tensor(0.0, device=reference.device)
    logits = outputs["axis_permutation_logits"]
    mask = target_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(
            f"axis permutation mask has shape {tuple(mask.shape)}, "
            f"expected {tuple(logits.shape)}"
        )
    allowed_logits = logits.masked_fill(~mask, float("-inf"))
    per_sample = torch.logsumexp(logits, dim=-1) - torch.logsumexp(
        allowed_logits, dim=-1
    )
    return per_sample.mean()


def given_hz_conditional_lattice_flow_loss(
    outputs: dict[str, torch.Tensor],
    target_lattice: torch.Tensor,
    target_log_density: torch.Tensor,
    selling_mean: torch.Tensor,
    selling_std: torch.Tensor,
    log_lambda_mean: torch.Tensor,
    log_lambda_std: torch.Tensor,
    log_lambda_recip_mean: torch.Tensor,
    log_lambda_recip_std: torch.Tensor,
    selling_dim: int = 6,
    log_lambda_dim: int = 3,
    w_flow: float = 1.0,
    w_flow_lambda: float | None = None,
    w_s: float = 1.0,
    w_lambda: float = 1.0,
    w_lambda_recip: float = 1.0,
    w_density: float = 0.25,
    lambda_loss_mode: str = "phys_log",
    w_lambda_relative: float = 0.0,
    w_lambda_product: float = 0.0,
    polymorph_lattice: torch.Tensor | None = None,
    polymorph_log_density: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    target_axis_permutation_mask: torch.Tensor | None = None,
    w_axis_permutation: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Selling flow + conditional λ flow; endpoint losses match joint lattice-flow model."""
    from cellnet.sequential import selling_norm_to_phys_torch, selling_relative_mse

    w_flow_l = w_flow if w_flow_lambda is None else w_flow_lambda
    flow_loss_s = _weighted_sample_mean(
        (outputs["velocity_s"] - outputs["velocity_target_s"]).pow(2).mean(dim=-1),
        sample_weight,
    )
    flow_loss_l = _weighted_sample_mean(
        (outputs["velocity_lambda"] - outputs["velocity_target_lambda"]).pow(2).mean(dim=-1),
        sample_weight,
    )
    flow_loss = flow_loss_s + flow_loss_l

    pred = outputs["lattice_flow"]
    s_end = selling_dim
    l_end = s_end + log_lambda_dim

    pred_s = pred[:, :s_end]
    pred_lam = pred[:, s_end:l_end]
    pred_lam_r = pred[:, l_end:]

    if polymorph_lattice is not None and polymorph_lattice.ndim == 3:
        ref_s = polymorph_lattice[:, :, :s_end]
        ref_lam = polymorph_lattice[:, :, s_end:l_end]
        ref_lam_r = polymorph_lattice[:, :, l_end:]
        pred_s_phys = selling_norm_to_phys_torch(pred_s, selling_mean, selling_std)
        ref_s_phys = selling_norm_to_phys_torch(ref_s, selling_mean, selling_std)
        eps = torch.tensor(1e-8, device=pred.device)
        scale = torch.maximum(torch.maximum(pred_s_phys.unsqueeze(1).abs(), ref_s_phys.abs()), eps)
        s_per_k = ((pred_s_phys.unsqueeze(1) - ref_s_phys) / scale).pow(2).mean(dim=-1)
        s_loss = s_per_k.min(dim=1).values.mean()
        lam_loss = _lattice_lambda_endpoint_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        ).min(dim=1).values.mean()
        lam_r_loss = _lattice_lambda_endpoint_loss(
            pred_lam_r, ref_lam_r, log_lambda_recip_mean, log_lambda_recip_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        ).min(dim=1).values.mean()
        if polymorph_log_density is not None:
            density_loss = (outputs["log_density"].unsqueeze(1) - polymorph_log_density).pow(2).min(dim=1).values.mean()
        else:
            density_loss = F.mse_loss(outputs["log_density"], target_log_density)
    else:
        ref_s = target_lattice[:, :s_end]
        ref_lam = target_lattice[:, s_end:l_end]
        ref_lam_r = target_lattice[:, l_end:]
        pred_s_phys = selling_norm_to_phys_torch(pred_s, selling_mean, selling_std)
        ref_s_phys = selling_norm_to_phys_torch(ref_s, selling_mean, selling_std)
        s_loss = selling_relative_mse(pred_s_phys, ref_s_phys)
        lam_loss = _lattice_lambda_endpoint_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        )
        lam_r_loss = _lattice_lambda_endpoint_loss(
            pred_lam_r, ref_lam_r, log_lambda_recip_mean, log_lambda_recip_std,
            mode=lambda_loss_mode, relative_weight=w_lambda_relative,
        )
        density_loss = F.mse_loss(outputs["log_density"], target_log_density)

    lambda_product_loss = torch.tensor(0.0, device=pred.device)
    if w_lambda_product > 0.0:
        lambda_product_loss = _lattice_lambda_product_loss(
            pred_lam, ref_lam, log_lambda_mean, log_lambda_std,
        )
    axis_permutation_loss = _axis_permutation_class_loss(
        outputs,
        target_axis_permutation_mask,
        w_axis_permutation,
    )

    total = (
        w_flow * flow_loss_s
        + w_flow_l * flow_loss_l
        + w_s * s_loss
        + w_lambda * lam_loss
        + w_lambda_recip * lam_r_loss
        + w_density * density_loss
        + w_lambda_product * lambda_product_loss
        + w_axis_permutation * axis_permutation_loss
    )
    metrics = {
        "total": total.item(),
        "flow": float(flow_loss.item()),
        "flow_s": float(flow_loss_s.item()),
        "flow_lambda": float(flow_loss_l.item()),
        "s_loss": float(s_loss.item()),
        "lambda": float(lam_loss.item()),
        "lambda_recip": float(lam_r_loss.item()),
        "density": float(density_loss.item()),
    }
    if w_lambda_product > 0.0:
        metrics["lambda_product"] = float(lambda_product_loss.item())
    if w_axis_permutation > 0.0 and "axis_permutation_logits" in outputs:
        metrics["axis_permutation"] = float(axis_permutation_loss.item())
    return total, metrics


def evaluate_lattice_flow_k_samples(
    samples_norm: np.ndarray,
    true_norm: np.ndarray,
    component: str = "all",
) -> dict[str, float]:
    """
    Evaluate K flow samples against a single normalized target.

    samples_norm: (K, D) or (B, K, D); true_norm: (D,) or (B, D)
    """
    samp = np.asarray(samples_norm, dtype=np.float64)
    true = np.asarray(true_norm, dtype=np.float64)
    if samp.ndim == 2:
        samp = samp[np.newaxis, ...]
        true = true[np.newaxis, ...]
    per_k = np.mean((samp - true[:, np.newaxis, :]) ** 2, axis=-1)
    best = per_k.min(axis=1)
    return {
        f"{component}_mse_norm_mean": float(np.mean(per_k)),
        f"{component}_mse_norm_best": float(np.mean(best)),
    }


def cellpar_errors(
    pred_cellpar: np.ndarray,
    true_cellpar: np.ndarray,
    hall_numbers: np.ndarray | list[int] | None = None,
    align_axes: bool = False,
) -> dict[str, float]:
    """Compute per-parameter and volume errors."""
    pred = np.asarray(pred_cellpar, dtype=np.float64)
    true = np.asarray(true_cellpar, dtype=np.float64)
    if align_axes:
        if hall_numbers is None:
            raise ValueError("hall_numbers required when align_axes=True")
        pred = np.stack([
            align_cellpar_to_reference(p, t, int(h))
            for p, t, h in zip(pred, true, hall_numbers)
        ])

    names = ["a", "b", "c", "alpha", "beta", "gamma"]
    errors = {}
    for i, name in enumerate(names):
        errors[f"mae_{name}"] = float(np.mean(np.abs(pred[:, i] - true[:, i])))
        if name in ("a", "b", "c"):
            rel = np.abs(pred[:, i] - true[:, i]) / np.clip(true[:, i], 1e-6, None)
            errors[f"mape_{name}"] = float(np.mean(rel) * 100)

    def volume(cp):
        a, b, c, al, be, ga = cp.T
        cos_a, cos_b, cos_g = np.cos(np.radians([al, be, ga]))
        term = 1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g
        return a * b * c * np.sqrt(np.clip(term, 0, None))

    v_pred = volume(pred)
    v_true = volume(true)
    errors["mape_volume"] = float(np.mean(np.abs(v_pred - v_true) / np.clip(v_true, 1e-6, None)) * 100)
    return errors
